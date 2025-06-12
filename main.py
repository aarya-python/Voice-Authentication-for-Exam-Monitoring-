# main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
import uuid
import os

# logic
from app.auth import enroll_voice, verify_voice
from app.detection import detect_prerecorded, detect_background_noise
from app.models import EnrollmentResponse, VerificationResponse, DetectionResponse

# FastAPI 
app = FastAPI()

# audio files
TEMP_DIR = "./temp_audio/"
os.makedirs(TEMP_DIR, exist_ok=True)


async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    suffix = ".wav"
    tmp_filename = f"{uuid.uuid4()}{suffix}"
    tmp_file_path = os.path.join(TEMP_DIR, tmp_filename)

    contents = await upload_file.read()

    with open(tmp_file_path, "wb") as buffer:
        buffer.write(contents)

    return tmp_file_path

# test endpoint
@app.get("/")
def read_root():
    return {"message": "Hello! FastAPI is working."}

#voice
@app.post("/enroll/{student_id}", response_model=EnrollmentResponse)
async def enroll(student_id: str, audio_file: UploadFile = File(...)):
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Audio must be WAV format")

    audio_path = await save_upload_file_tmp(audio_file)

    enroll_voice(student_id, audio_path)

    os.remove(audio_path)
    return EnrollmentResponse(status="enrolled", student_id=student_id)

# Verify voice
@app.post("/verify/{student_id}", response_model=VerificationResponse)
async def verify(student_id: str, audio_file: UploadFile = File(...)):
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Audio must be WAV format")

    audio_path = await save_upload_file_tmp(audio_file)

    similarity = verify_voice(student_id, audio_path)

    os.remove(audio_path)

    if similarity is None:
        raise HTTPException(status_code=404, detail="Student not enrolled")

    status = "verified" if similarity > 0.85 else "failed"
    return VerificationResponse(status=status, similarity=similarity)

#pre-recorded audio
@app.post("/detect/prerecorded", response_model=DetectionResponse)
async def prerecorded_detection(audio_file: UploadFile = File(...)):
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Audio must be WAV format")

    audio_path = await save_upload_file_tmp(audio_file)

    energy_var = detect_prerecorded(audio_path)

    os.remove(audio_path)

    status = "possible pre-recorded" if energy_var < 0.001 else "live"
    return DetectionResponse(status=status, value=energy_var)

# Detect background noise
@app.post("/detect/background_noise", response_model=DetectionResponse)
async def background_noise_detection(audio_file: UploadFile = File(...)):
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Audio must be WAV format")

    audio_path = await save_upload_file_tmp(audio_file)

    noise_level = detect_background_noise(audio_path)

    os.remove(audio_path)

    status = "background noise detected" if noise_level > 0.3 else "clean"
    return DetectionResponse(status=status, value=noise_level)
