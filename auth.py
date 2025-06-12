import os
import librosa
import numpy as np
import pickle

# Folder 
ENROLL_FOLDER = "./enrollments/"
os.makedirs(ENROLL_FOLDER, exist_ok=True)

# extract audio features 
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Enroll voice
def enroll_voice(student_id, audio_path):
    features = extract_features(audio_path)
    enroll_path = os.path.join(ENROLL_FOLDER, f"{student_id}.pkl")
    with open(enroll_path, "wb") as f:
        pickle.dump(features, f)

# Verify voice
def verify_voice(student_id, audio_path):
    enroll_path = os.path.join(ENROLL_FOLDER, f"{student_id}.pkl")
    if not os.path.exists(enroll_path):
        return None 

    
    with open(enroll_path, "rb") as f:
        enrolled_features = pickle.load(f)

    
    verify_features = extract_features(audio_path)

    
    similarity = cosine_similarity(enrolled_features, verify_features)
    return similarity


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return np.dot(a, b)
