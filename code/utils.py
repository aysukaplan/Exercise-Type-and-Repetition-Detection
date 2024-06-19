import cv2
from pytube import YouTube
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from scipy.signal import medfilt

import tensorflow.compat.v2 as tf

def download_youtube_video(url):
    youtube = YouTube(url)
    stream = youtube.streams.get_lowest_resolution()
    download_file = stream.download('yt_videos')
    return download_file

HEIGHT, WIDTH = 64, 64
SEQUENCE_LENGTH = 20
CLASSES = ['frontraise', 'pullups', 'squant', 'front_raise', 'bench_pressing',
       'jump_jack', 'situp', 'benchpressing', 'squat', 'pull_up',
       'push_up', 'jumpjacks', 'pushups', 'others', 'battle_rope',
       'pommelhorse']
CLASSES_INDEX = dict(zip(CLASSES, range(len(CLASSES))))
CLASSES_INDEX_REVERSE = dict(zip(range(len(CLASSES)), CLASSES))

def frame_extraction(video_path):
    frames = []
    optical_flow_frames = []
    video_renderer = cv2.VideoCapture(video_path)
    video_frames_count = int(video_renderer.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)
    
    prev_gray = None
    
    for frame_counter in range(SEQUENCE_LENGTH):
        video_renderer.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_renderer.read()
        if not success:
            break
        frame = cv2.resize(frame, (HEIGHT, WIDTH))
        normalized_frame = frame / 255.0
        frames.append(normalized_frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
        
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        optical_flow_img = np.dstack((magnitude, angle))
        normalized_optical_flow_img = optical_flow_img / np.amax(optical_flow_img)
        optical_flow_frames.append(normalized_optical_flow_img)
        
        prev_gray = gray
        
    video_renderer.release()
    return frames, optical_flow_frames

def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / max(1e-7, (max_v - min_v))
  return query_frame

def read_video(video_filename, width=224, height=224):
  """Read video from file."""
  cap = cv2.VideoCapture(video_filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      frame_rgb = cv2.resize(frame_rgb, (width, height))
      frames.append(frame_rgb)
  frames = np.asarray(frames)
  return frames, fps


def get_score(period_score, within_period_score):
  """Combine the period and periodicity scores."""
  within_period_score = tf.nn.sigmoid(within_period_score)[:, 0]
  per_frame_periods = tf.argmax(period_score, axis=-1) + 1
  pred_period_conf = tf.reduce_max(
      tf.nn.softmax(period_score, axis=-1), axis=-1)
  pred_period_conf = tf.where(
      tf.math.less(per_frame_periods, 3), 0.0, pred_period_conf)
  within_period_score *= pred_period_conf
  within_period_score = np.sqrt(within_period_score)
  pred_score = tf.reduce_mean(within_period_score)
  return pred_score, within_period_score


def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               constant_speed=False,
               median_filter=False,
               fully_periodic=False):
  """Pass frames through model and conver period predictions to count."""
  seq_len = len(frames)
  raw_scores_list = []
  scores = []
  within_period_scores_list = []

  if fully_periodic:
    within_period_threshold = 0.0

  frames = model.preprocess(frames)

  for stride in strides:
    num_batches = int(np.ceil(seq_len/model.num_frames/stride/batch_size))
    raw_scores_per_stride = []
    within_period_score_stride = []
    for batch_idx in range(num_batches):
      idxes = tf.range(batch_idx*batch_size*model.num_frames*stride,
                       (batch_idx+1)*batch_size*model.num_frames*stride,
                       stride)
      idxes = tf.clip_by_value(idxes, 0, seq_len-1)
      curr_frames = tf.gather(frames, idxes)
      curr_frames = tf.reshape(
          curr_frames,
          [batch_size, model.num_frames, model.image_size, model.image_size, 3])

      raw_scores, within_period_scores, _ = model(curr_frames)
      raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                              [-1, model.num_frames//2]))
      within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                   [-1, 1]))
    raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
    raw_scores_list.append(raw_scores_per_stride)
    within_period_score_stride = np.concatenate(
        within_period_score_stride, axis=0)
    pred_score, within_period_score_stride = get_score(
        raw_scores_per_stride, within_period_score_stride)
    scores.append(pred_score)
    within_period_scores_list.append(within_period_score_stride)

  # Stride chooser
  argmax_strides = np.argmax(scores)
  chosen_stride = strides[argmax_strides]
  raw_scores = np.repeat(
      raw_scores_list[argmax_strides], chosen_stride, axis=0)[:seq_len]
  within_period = np.repeat(
      within_period_scores_list[argmax_strides], chosen_stride,
      axis=0)[:seq_len]
  within_period_binary = np.asarray(within_period > within_period_threshold)
  if median_filter:
    within_period_binary = medfilt(np.float32(within_period_binary), 5)
    within_period_binary = within_period_binary.astype(bool)

  # Select Periodic frames
  periodic_idxes = np.where(within_period_binary)[0]

  if constant_speed:
    # Count by averaging predictions. Smoother but
    # assumes constant speed.
    scores = tf.reduce_mean(
        tf.nn.softmax(raw_scores[periodic_idxes], axis=-1), axis=0)
    max_period = np.argmax(scores)
    pred_score = scores[max_period]
    pred_period = chosen_stride * (max_period + 1)
    per_frame_counts = (
        np.asarray(seq_len * [1. / pred_period]) *
        np.asarray(within_period_binary))
  else:
    # Count each frame. More noisy but adapts to changes in speed.
    pred_score = tf.reduce_mean(within_period)
    per_frame_periods = tf.argmax(raw_scores, axis=-1) + 1
    per_frame_counts = tf.where(
        tf.math.less(per_frame_periods, 3),
        0.0,
        tf.math.divide(1.0,
                       tf.cast(chosen_stride * per_frame_periods, tf.float32)),
    )
    if median_filter:
      per_frame_counts = medfilt(per_frame_counts, 5)

    per_frame_counts *= np.asarray(within_period_binary)

    pred_period = seq_len/np.sum(per_frame_counts)

  if pred_score < threshold:
    print('No repetitions detected in video as score '
          '%0.2f is less than threshold %0.2f.'%(pred_score, threshold))
    per_frame_counts = np.asarray(len(per_frame_counts) * [0.])

  return (pred_period, pred_score, within_period,
          per_frame_counts, chosen_stride)

def viz_reps(frames,
             count,
             alpha=1.0,
             pichart=True,
             colormap=plt.cm.spring,
             num_frames=None,
             interval=30,
             tmp_path='output.mp4'):
  """Visualize repetitions."""
  if isinstance(count, list):
    counts = len(frames) * [count/len(frames)]
  else:
    counts = count
  sum_counts = np.cumsum(counts)
  # tmp_path = '/tmp/output.mp4'
  fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5),
                         tight_layout=True,)

  h, w, _ = np.shape(frames[0])
  wedge_x = 95 / 112 * w
  wedge_y = 17 / 112 * h
  wedge_r = 15 / 112 * h
  txt_x = 95 / 112 * w
  txt_y = 19 / 112 * h
  otxt_size = 62 / 112 * h

  im0 = ax.imshow(unnorm(frames[0]))

  if not num_frames:
    num_frames = len(frames)

  wedge1 = matplotlib.patches.Wedge(
      center=(wedge_x, wedge_y),
      r=wedge_r,
      theta1=0,
      theta2=0,
      color=colormap(1.),
      alpha=alpha)
  wedge2 = matplotlib.patches.Wedge(
      center=(wedge_x, wedge_y),
      r=wedge_r,
      theta1=0,
      theta2=0,
      color=colormap(0.5),
      alpha=alpha)

  ax.add_patch(wedge1)
  ax.add_patch(wedge2)
  txt = ax.text(
      txt_x,
      txt_y,
      '0',
      size=35,
      ha='center',
      va='center',
      alpha=0.9,
      color='white',
  )

  def update(i):
    """Update plot with next frame."""
    im0.set_data(unnorm(frames[i]))
    ctr = int(sum_counts[i])
    if pichart:
      if ctr%2 == 0:
        wedge1.set_color(colormap(1.0))
        wedge2.set_color(colormap(0.5))
      else:
        wedge1.set_color(colormap(0.5))
        wedge2.set_color(colormap(1.0))

      wedge1.set_theta1(-90)
      wedge1.set_theta2(-90 - 360 * (1 - sum_counts[i] % 1.0))
      wedge2.set_theta1(-90 - 360 * (1 - sum_counts[i] % 1.0))
      wedge2.set_theta2(-90)

    txt.set_text(int(sum_counts[i]))
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

  anim = FuncAnimation(
      fig,
      update,
      frames=num_frames,
      interval=interval,
      blit=False)
  anim.save(tmp_path, dpi=80)
  plt.close()
  
