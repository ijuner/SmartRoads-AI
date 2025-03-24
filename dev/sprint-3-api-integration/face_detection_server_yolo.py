from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import uuid
from datetime import datetime
import time
import uvicorn
from typing import Optional
from ultralytics import YOLO  # Import YOLO from ultralytics
import os

app = FastAPI(title="Face Detection API")

# Configure CORS to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("model/yolo_trained_model_03_06.pt")  # Update this path to where your model is stored

@app.post("/detect_face")
async def detect_face(
        image: UploadFile = File(...),
        min_confidence: float = Form(0.7)
):
    # Start timer
    start_time = time.time()

    min_confidence_env = os.getenv('FACE_DETECTION_MIN_CONFIDENCE')
    print("[SMARTROADS AI][INFO] MINIMUM_CONFIDENCE: ${min_confidence}")
    print("[SMARTROADS AI][INFO] MINIMUM_CONFIDENCE FROM ENVIRONMENT: ${min_confidence_env}")

    min_confidence = min(float(min_confidence), float(min_confidence_env))
    # Read image file
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Save temporary image for YOLO (since it expects a file path)
    temp_img_path = f"temp_{uuid.uuid4()}.jpg"
    cv2.imwrite(temp_img_path, img)

    # Run inference with YOLO
    results = model(temp_img_path)

    # Get detection results
    # YOLO results contain detected objects with bounding boxes and confidence scores
    detections = results[0].boxes

    # Filter detections based on confidence threshold
    face_detections = []
    for detection in detections:
        confidence = float(detection.conf)
        if confidence >= min_confidence:
            face_detections.append({
                "bbox": detection.xyxy.tolist()[0],  # Convert tensor to list
                "confidence": confidence
            })

    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000
    print("[SMARTROADS AI][INFO] PROCESSING TIME : ${processing_time_ms}")

    # Clean up temporary file
    import os
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    # Generate response
    response = {
        "request_id": str(uuid.uuid4()),
        "face_detected": len(face_detections) > 0,
        "timestamp": datetime.now().isoformat(),
        "count": len(face_detections),
        "processing_time_ms": processing_time_ms,
        "confidence_level": max([d["confidence"] for d in face_detections]) if face_detections else 0.0,
        "detections": face_detections
    }
    print("[SMARTROADS AI][INFO] FINAL RESPONSE: ${response}")

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)