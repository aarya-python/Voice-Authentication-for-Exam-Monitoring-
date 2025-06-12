import librosa
import numpy as np

# Prerecorded detection 
def detect_prerecorded(audio_path: str) -> float:
    y, sr = librosa.load(audio_path, sr=None)
    frame_length = 2048
    hop_length = 512
    
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    
    energy_var = np.var(energy)
    return float(energy_var)

# Background noise detection 
def detect_background_noise(audio_path: str) -> float:
    y, sr = librosa.load(audio_path, sr=None)
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    avg_zcr = np.mean(zcr)
    
    return float(avg_zcr)
