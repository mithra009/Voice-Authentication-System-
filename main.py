import streamlit as st
import sounddevice as sd
import numpy as np
import wavio
import os
import torch
import librosa
from speechbrain.inference import SpeakerRecognition
from scipy.spatial.distance import cosine
import time
import shutil

# Set Streamlit page configuration
st.set_page_config(page_title="Voice Authentication System", layout="wide")

# Paths for storing user voice data
DATA_DIR = "voice_data"
os.makedirs(DATA_DIR, exist_ok=True)
MODEL_DIR = "tmp_dvector_model"

# Load the model once and cache it
@st.cache_resource()
def load_dvector_model():
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=MODEL_DIR
    )

dvector_model = load_dvector_model()

# Function to record audio
def record_audio(filename, duration=5, samplerate=16000):
    st.write("Recording...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    wavio.write(filename, audio, samplerate, sampwidth=2)
    st.write("Recording complete!")

# Function to extract d-vector embedding
def extract_dvector_embedding(filename):
    signal, sr = librosa.load(filename, sr=16000)
    embedding = dvector_model.encode_batch(torch.tensor(signal).unsqueeze(0))
    return embedding.squeeze().detach().numpy()

# Register user
def register_user(name):
    filename = os.path.join(DATA_DIR, f"{name}.wav")
    record_audio(filename)
    st.success(f"User {name} registered successfully!")

# Verify user
def verify_user():
    test_filename = "test_voice.wav"
    record_audio(test_filename)
    test_embedding = extract_dvector_embedding(test_filename)

    best_match = None
    best_score = float("inf")

    for file in os.listdir(DATA_DIR):
        if file.endswith(".wav"):
            stored_filename = os.path.join(DATA_DIR, file)
            stored_embedding = extract_dvector_embedding(stored_filename)
            similarity = cosine(test_embedding, stored_embedding)

            if similarity < best_score:
                best_score = similarity
                best_match = file.split(".")[0]

    if best_match and best_score < 0.3:
        st.success(f"Authenticated as: {best_match}")
    else:
        st.error("Invalid voice input! User not recognized.")

# Streamlit UI
st.title("Voice Authentication System")

# Create tabs for navigation
tabs = st.tabs(["Register", "Authenticate"])
with tabs[0]:
    name = st.text_input("Enter your name:")
    if st.button("Register") and name:
        register_user(name)

with tabs[1]:
    if st.button("Authenticate"):
        verify_user()
