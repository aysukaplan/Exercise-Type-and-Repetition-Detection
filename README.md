# Exercise-Type-and-Repetition-Detection
## BBM416 - Fundamentals of Computer Vision Class Project

## Overview
This project aims to develop a computer vision system for workout identification and repetition counting, leveraging deep learning techniques to enhance accuracy and applicability in fitness analysis. The system is designed to help individuals improve their fitness and assist trainers and doctors in monitoring exercises.

## Features
- **Workout Identification**: Classifies different types of workouts using optical flow and LSTM models.
- **Repetition Counting**: Uses Google's RepNet model to count repetitions in workout videos accurately.
- **Real-Time Feedback**: Provides immediate feedback on workout performance.

## Project Structure
- **RepNet Architecture**: Utilizes a modified ResNet50 backbone and transformer architecture to detect repetition patterns in video sequences.
- **Workout Action Recognition Model**: Combines image features and optical flow to classify workout types, leveraging convolutional layers and LSTM for temporal pattern recognition.
- **Preprocessing**: Involves resizing and normalizing video frames for consistency with model requirements.

## Getting Started

### Prerequisites
- Python
- TensorFlow
- Gradio (for GUI)

### Installation
Clone the repository:
   ```
   git clone https://github.com/aysukaplan/Exercise-Type-and-Repetition-Detection.git
   cd Exercise-Type-and-Repetition-Detection
   ```
### Usage
#### Training the Models
1. Ensure you have the Repetition Action Counting Dataset.
2. Run the training script for workout action recognition:
```python
  python train_action_recognition.py
```
3. Train the RepNet model for repetition counting:
```python
python train_repnet.py
```
#### Running the Application
1. Start the Gradio GUI for real-time workout tracking:
```python
python app.py
```
2. Upload a video from Youtube or use your webcam to identify workout types and count repetitions.

## Demo
### Video Demo:  [Youtube](https://www.youtube.com/watch?v=A-GUBdgqG-U)
### Model Weights:  [Google Drive](https://drive.google.com/file/d/1SpzsQcu1HqlhECYaFfR7HLXJ7c2fzUtu/view?usp=sharing)

## Experimental Results
### Action Detection
- Trained on the Repetition Action Counting Dataset.
- Applied grid search for hyperparameter tuning.
- Plots for loss and accuracy are included in the report.
### Repetition Detection and Scoring
- Utilizes RepNet to obtain period predictions and within-period scores.
- Calculates repetition count and confidence score.

## Discussions and Conclusions
The system effectively recognizes actions and counts repetitions, combining CNNs for feature extraction, LSTMs for temporal dependencies, and RepNet for repetition counting.
Future improvements include transfer learning, better data augmentation, and optimization methods.
## Authors
Ataberk Asar (2210356135)  
Aysu Aylin Kaplan (2200356810)  
Department of Computer Engineering, Hacettepe University
## License
This project is licensed under the MIT License.

## Acknowledgements
[RepNet](https://sites.google.com/view/repnet)  
