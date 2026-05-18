import streamlit as st
import numpy as np
import librosa
import pickle
from tensorflow.keras.models import load_model

# ---------------- LOAD MODELS ----------------
emotion_model = pickle.load(open("emotion_model.pkl", "rb"))
sound_model = load_model("sound_model.h5")

# ---------------- LABEL MAPPINGS ----------------
emotion_labels = {
    0: "ANG",
    1: "HAP",
    2: "NEU",
    3: "SAD"
}

sound_labels = {
    0: "dog",
    1: "rain",
    2: "traffic",
    3: "alarm"
}

# ---------------- FEATURE EXTRACTION ----------------
def extract_mfcc(file):
    y, sr = librosa.load(file, duration=3, offset=0.5)

    mfcc = np.mean(
        librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,
        axis=0
    )

    return mfcc


def extract_spec(file):

    y, sr = librosa.load(file, duration=3, offset=0.5)

    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec = librosa.power_to_db(spec)

    if spec.shape[1] < 128:
        spec = np.pad(spec, ((0,0),(0,128-spec.shape[1])))
    else:
        spec = spec[:, :128]

    spec = spec[..., np.newaxis]
    spec = np.expand_dims(spec, axis=0)

    return spec

# ---------------- PREDICTION FUNCTIONS ----------------
def predict_emotion(file):

    mfcc = extract_mfcc(file)

    pred = emotion_model.predict([mfcc])[0]

    return emotion_labels[pred]


def predict_sound(file):

    spec = extract_spec(file)

    pred = sound_model.predict(spec)

    return f"Class {np.argmax(pred)}"

# ---------------- STREAMLIT UI ----------------
st.title("🎧 Audio Language Model")

option = st.radio(
    "Choose Analysis",
    (
        "Emotion Detection",
        "Background Noise Detection",
        "Combined Analysis"
    )
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Analyze"):

        if option == "Emotion Detection":

            emotion = predict_emotion(uploaded_file)

            st.success(f"Predicted Emotion: {emotion}")

        elif option == "Background Noise Detection":

            sound = predict_sound(uploaded_file)

            st.success(f"Detected Sound: {sound}")

        else:

            emotion = predict_emotion(uploaded_file)
            sound = predict_sound(uploaded_file)

            st.success(f"Emotion: {emotion}")
            st.success(f"Background Sound: {sound}")

            if emotion == "ANG" and sound == "traffic":
                st.warning("Angry person detected in traffic environment")

            elif emotion == "SAD" and sound == "rain":
                st.info("Sad emotional state in calm environment")

            else:
                st.write(f"{emotion} emotion detected in {sound} environment")