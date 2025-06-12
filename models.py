from pydantic import BaseModel

class EnrollmentResponse(BaseModel):
    status: str
    student_id: str

class VerificationResponse(BaseModel):
    status: str
    similarity: float

class DetectionResponse(BaseModel):
    status: str
    value: float
