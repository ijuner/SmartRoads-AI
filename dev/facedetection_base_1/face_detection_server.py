from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import uuid
from datetime import datetime
import time
import uvicorn
from typing import Optional

app = FastAPI(title="Face Detection API")

# Configure CORS to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load face detection model (using OpenCV's pre-trained Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.post("/detect_face")
async def detect_face(
        image: UploadFile = File(...),
        min_confidence: float = Form(0.7)
):
    # Start timer
    start_time = time.time()

    # Read image file
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000

    # Generate response
    response = {
        "request_id": str(uuid.uuid4()),
        "face_detected": len(faces) > 0,
        "timestamp": datetime.now().isoformat(),
        "count": len(faces),
        "processing_time_ms": processing_time_ms,
        "confidence_level": min_confidence if len(faces) > 0 else 0.0
    }

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)