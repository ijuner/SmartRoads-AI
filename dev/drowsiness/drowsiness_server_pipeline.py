from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import uuid
from datetime import datetime
import time
import uvicorn
from typing import Optional, List, Dict, Any
import os
import hashlib
import json
from pydantic import BaseModel
import mlflow as mlf
from drowsiness_pipeline import process_batch_images
from dotenv import load_dotenv
load_dotenv()

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'drowsiness_detection')

# Environment variables
FRAMES_COUNT = int(os.getenv('DROWSINESS_FRAMES_COUNT', '8'))
EYE_AR_THRESHOLD = float(os.getenv('EYE_AR_THRESHOLD', '0.2'))  # Eye aspect ratio threshold
EYE_CONSECUTIVE_FRAMES = int(os.getenv('EYE_CONSECUTIVE_FRAMES', '12'))  # Number of consecutive frames to trigger alarm

mlf.set_tracking_uri(MLFLOW_TRACKING_URI)
mlf.set_experiment(MLFLOW_EXPERIMENT_NAME)
print(f"SMARTROADSAI | PIPELINE SERVER RUNNING.py: MLflow tracking at {MLFLOW_TRACKING_URI}, experiment: {MLFLOW_EXPERIMENT_NAME}")

# User credentials (in a real app, you would use a database)
users = {
    "admin": ["admin123"],
    "user": ["user123"]
}

# Store active sessions
active_sessions = {}

app = FastAPI(title="Drowsiness Detection API")

# Configure CORS to allow requests from Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    auth_token: str
    status: str
    username: str


def verify_token(x_auth_token: str = Header(None)):
    """
    Verify the auth token in the request header
    """
    print(f"SMARTROADSAI | server.py | verify_token: Verifying token {x_auth_token}")

    if not x_auth_token or x_auth_token not in active_sessions:
        print(f"SMARTROADSAI | server.py | verify_token: Invalid or missing token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    return active_sessions[x_auth_token]


@app.post("/auth", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and create a session
    """
    print(f"SMARTROADSAI | server.py | login: Login attempt for user {request.username}")

    # Check if the username exists
    if request.username not in users["admin"]:
        print(f"SMARTROADSAI | server.py | login: User {request.username} not found")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate auth token
    auth_token = str(uuid.uuid4())
    active_sessions[auth_token] = request.username

    print(f"SMARTROADSAI | server.py | login: Login successful for user {request.username}")

    return {
        "auth_token": auth_token,
        "status": "success",
        "username": request.username
    }


@app.post("/detect_batch_drowsiness")
async def detect_batch_drowsiness(
        images: List[UploadFile] = File(...),
        username: str = Depends(verify_token)
):
    """
    Process a batch of images to detect drowsiness

    This endpoint handles batches of frames captured from the webcam.
    It analyzes each frame to detect drowsiness and returns a comprehensive result
    including metrics and alert status.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    print(
        f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: Starting batch processing of {len(images)} images")

    # Log environment variables for debugging
    print(
        f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: Environment settings: FRAMES_COUNT={FRAMES_COUNT}, EYE_AR_THRESHOLD={EYE_AR_THRESHOLD}, EYE_CONSECUTIVE_FRAMES={EYE_CONSECUTIVE_FRAMES}")

    # Allow for flexibility in frame count - don't strictly enforce it
    if len(images) < 5:  # Too few frames might not be reliable
        print(
            f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: Too few frames received: {len(images)}")
        raise HTTPException(status_code=400, detail=f"At least 5 frames are required for reliable drowsiness detection")

    # Read all images as bytes first
    image_bytes = []
    for image in images:
        contents = await image.read()
        image_bytes.append(contents)

    # Process the batch of images using the pipeline
    detection_results = []
    try:
        # Use the ML pipeline to process all images
        results = process_batch_images(image_bytes, eye_ar_threshold=EYE_AR_THRESHOLD)

        # Add frame index and processing time to results
        for i, result in enumerate(results):
            frame_start_time = time.time()
            detection_results.append({
                "frame": i,
                "processing_time_ms": (time.time() - frame_start_time) * 1000,
                **result
            })

            if i % 5 == 0:  # Log progress for every 5th frame
                print(
                    f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: Processed {i + 1}/{len(images)} frames")

    except Exception as e:
        print(
            f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: Error processing frames: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing frames: {str(e)}")

    # Calculate statistics from results
    valid_frames = len(detection_results)
    if valid_frames == 0:
        print(
            f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: No valid frames processed")
        raise HTTPException(status_code=400, detail="No valid frames could be processed")

    # Count drowsy frames
    drowsy_frames = sum(1 for result in detection_results if result["is_drowsy"])

    # Calculate drowsiness metrics
    drowsy_percentage = (drowsy_frames / valid_frames) * 100 if valid_frames > 0 else 0

    # Determine if alert is needed - either by consecutive frames or percentage threshold
    consecutive_threshold = min(EYE_CONSECUTIVE_FRAMES, valid_frames // 2)  # Adapt threshold based on frame count
    is_alert_needed = drowsy_frames >= consecutive_threshold

    # Calculate EAR statistics
    ear_values = [result["avg_ear"] for result in detection_results]
    ear_stats = {}
    if ear_values:
        ear_stats = {
            "min_ear": min(ear_values),
            "max_ear": max(ear_values),
            "avg_ear": sum(ear_values) / len(ear_values),
            "threshold": EYE_AR_THRESHOLD
        }

    # Calculate total processing time
    processing_time_ms = (time.time() - start_time) * 1000

    # MLFlow Logging
    log_detection_to_mlflow(
        username=username,
        request_id=request_id,
        ear_stats=ear_stats,
        frame_count=valid_frames,
        drowsy_frames=drowsy_frames,
        drowsy_percentage=drowsy_percentage,
        is_alert=is_alert_needed,
        processing_time=processing_time_ms,
        ear_threshold=EYE_AR_THRESHOLD,
        consecutive_frames=EYE_CONSECUTIVE_FRAMES
    )

    # Generate response
    response = {
        "request_id": request_id,
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "drowsy_frames_count": drowsy_frames,
        "drowsy_percentage": round(drowsy_percentage, 2),
        "total_frames": valid_frames,
        "alert_needed": is_alert_needed,
        "processing_time_ms": round(processing_time_ms, 2),
    }

    # Log detailed result summary
    alert_status = "ALERT TRIGGERED" if is_alert_needed else "No alert needed"
    print(
        f"SMARTROADSAI | server.py | detect_batch_drowsiness | request_id={request_id}: Completed batch processing in {processing_time_ms:.2f}ms. {drowsy_frames}/{valid_frames} drowsy frames ({drowsy_percentage:.2f}%). {alert_status}")

    return response


def log_detection_to_mlflow(username, request_id, ear_stats, frame_count, drowsy_frames, drowsy_percentage,
                            is_alert, processing_time, ear_threshold, consecutive_frames):
    """Log drowsiness detection results to MLflow"""
    try:
        with mlf.start_run(run_name=f"detection_{request_id}"):
            # Log parameters
            mlf.log_params({
                "username": username,
                "eye_ar_threshold": ear_threshold,
                "consecutive_frames_threshold": consecutive_frames,
                "total_frames": frame_count
            })

            # Log metrics
            mlf.log_metrics({
                "drowsy_frames": drowsy_frames,
                "drowsy_percentage": drowsy_percentage,
                "avg_ear": ear_stats.get("avg_ear", 0),
                "min_ear": ear_stats.get("min_ear", 0),
                "max_ear": ear_stats.get("max_ear", 0),
                "processing_time_ms": processing_time
            })

            # Log alert as a tag
            mlf.set_tag("alert_triggered", is_alert)
            mlf.set_tag("request_id", request_id)

        print(f"SMARTROADSAI | server.py: MLflow logging successful for request {request_id}")
    except Exception as e:
        print(f"SMARTROADSAI | server.py: MLflow logging failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)