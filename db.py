# app/db.py

import os
import numpy as np

# Path 
VOICEPRINTS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "voiceprints")


os.makedirs(VOICEPRINTS_FOLDER, exist_ok=True)

def save_voiceprint(student_id: str, embedding: np.ndarray):
    file_path = os.path.join(VOICEPRINTS_FOLDER, f"{student_id}.npy")
    np.save(file_path, embedding)

def load_voiceprint(student_id: str) -> np.ndarray:
    file_path = os.path.join(VOICEPRINTS_FOLDER, f"{student_id}.npy")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Voiceprint for student_id '{student_id}' not found.")
    return np.load(file_path)
