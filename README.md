 🎧 Deep Learning Based Audio Language Model (ALM)
📌 Project Overview

This project presents a **Deep Learning Based Audio Language Model (ALM)** capable of analyzing both **speech** and **non-speech audio** simultaneously. The system integrates:

- 🎭 Speech Emotion Recognition
- 🔊 Environmental Sound Classification
- 🧠 Context-Aware Audio Understanding

The model can “**Listen, Think, and Understand**” audio environments by combining emotional speech analysis with surrounding background sounds.

Features

✅ Emotion Detection from Speech  
✅ Background Noise Classification  
✅ Integrated Audio Understanding  
✅ Decision-Level Ensembling  
✅ Streamlit-Based Interactive Frontend  
✅ Real-Time Audio File Analysis  


Technologies Used

- Python
- TensorFlow / Keras
- Scikit-learn
- Librosa
- NumPy
- Streamlit

Project Structure

```text
ALM_DEEPLEARNING/
│
├── app.py                 # Streamlit frontend
├── alm.ipynb              # Model training notebook
├── emotion_model.pkl      # Emotion detection model
├── sound_model.h5         # Background sound classification model
├── bc_utils.py            # Utility functions
├── utils.py
├── utils2.py
├── esc50.csv              # ESC-50 metadata
│
├── audio/                 # Emotion dataset folder (provided as link)
├── AudioWAV/              # Audio dataset folder (provided as link)
└── testing/               # Testing audio samples

Datasets Used
🎭 Emotion Recognition Dataset (RAVDESS)

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset was used for training the emotion detection model.

🔗 Dataset Link:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

🔊 Environmental Sound Dataset (ESC-50)

The ESC-50 dataset was used for environmental sound classification.

🔗 Dataset Link:
https://www.kaggle.com/code/salimhammadi07/esc-50-environmental-sound-classification/input

Model Architecture
🎭 Emotion Detection Model
Feature Extraction: MFCC
Algorithm: Random Forest Classifier
Output: Emotion Labels (ANG, SAD, HAP, NEU, etc.)
🔊 Background Sound Classification Model
Feature Extraction: Mel Spectrogram
Algorithm: Convolutional Neural Network (CNN)
Output: Environmental Sound Classes
🧠 Ensembling

Both model outputs are combined using Decision-Level Fusion to generate context-aware audio interpretations.

Applications:
Smart Surveillance Systems
Healthcare Monitoring
Human–Computer Interaction
Intelligent Virtual Assistants
Context-Aware Audio Intelligence
Audio Scene Understanding


Future Enhancements:
Real-Time Audio Streaming
Transformer-Based Audio Models
Multilingual Emotion Recognition
Feature-Level Fusion
Edge AI Deployment



