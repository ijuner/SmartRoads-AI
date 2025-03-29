from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import uuid
from datetime import datetime
import time
import uvicorn
from typing import Optional, List, Dict, Any
import mediapipe as mp
import os
import hashlib
import json
from pydantic import BaseModel

# Environment variables
FRAMES_COUNT = int(os.getenv('DROWSINESS_FRAMES_COUNT', '24'))
EYE_AR_THRESHOLD = float(os.getenv('EYE_AR_THRESHOLD', '0.2'))  # Eye aspect ratio threshold
EYE_CONSECUTIVE_FRAMES = int(os.getenv('EYE_CONSECUTIVE_FRAMES', '12'))  # Number of consecutive frames to trigger alarm

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


# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eye landmarks for left and right eyes
# MediaPipe uses the following indices for eye landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(eye_landmarks, face_landmarks):
    """
    Calculate the eye aspect ratio (EAR) given the eye landmarks
    """
    try:
        # Extract coordinates for the specified eye landmarks
        points = []
        for i in eye_landmarks:
            point = face_landmarks[i]
            points.append([point.x, point.y, point.z])

        # Calculate horizontal distance (width)
        horizontal_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

        # Calculate vertical distances (height at two points)
        vertical_dist1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
        vertical_dist2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

        # Calculate EAR
        ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
        return ear
    except:
        return 0.3  # Return a default value if calculation fails


def detect_drowsiness(image):
    """
    Detect drowsiness in an image using MediaPipe and EAR calculation
    """
    # Process the image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    results = face_mesh.process(image_rgb)

    is_drowsy = False
    left_ear = 0
    right_ear = 0
    avg_ear = 0

    # Check if face landmarks are detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Calculate EAR for both eyes
            left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark)
            right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark)

            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0

            # Determine if drowsy
            is_drowsy = avg_ear < EYE_AR_THRESHOLD

            # Draw landmarks for visualization (optional for debugging)
            # mp.solutions.drawing_utils.draw_landmarks(
            #     image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            # )

    return {
        "is_drowsy": is_drowsy,
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "avg_ear": float(avg_ear),
        "threshold": float(EYE_AR_THRESHOLD)
    }


def verify_token(x_auth_token: str = Header(None)):
    """
    Verify the auth token in the request header
    """
    print(f"SMARTROADSAI | drowsiness_detection_server.py | verify_token: Verifying token {x_auth_token}")

    if not x_auth_token or x_auth_token not in active_sessions:
        print(f"SMARTROADSAI | drowsiness_detection_server.py | verify_token: Invalid or missing token")
        raise HTTPException(status_code=401, detail="Unauthorized")

    return active_sessions[x_auth_token]


@app.post("/auth", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and create a session
    """
    print(f"SMARTROADSAI | drowsiness_detection_server.py | login: Login attempt for user {request.username}")

    # Check if the username exists
    if request.username not in users["admin"]:
        print(f"SMARTROADSAI | drowsiness_detection_server.py | login: User {request.username} not found")
        raise HTTPException(status_code=401, detail="Invalid username or password")

    # Verify password
    # hashed_password = hashlib.sha256(request.password.encode()).hexdigest()
    # if hashed_password != users[request.username]:
    #     print(f"SMARTROADSAI | drowsiness_detection_server.py | login: Invalid password for user {request.username}")
    #     raise HTTPException(status_code=401, detail="Invalid username or password")

    # Generate auth token
    auth_token = str(uuid.uuid4())
    active_sessions[auth_token] = request.username

    print(f"SMARTROADSAI | drowsiness_detection_server.py | login: Login successful for user {request.username}")

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
        f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Starting batch processing of {len(images)} images")

    # Log environment variables for debugging
    print(
        f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Environment settings: FRAMES_COUNT={FRAMES_COUNT}, EYE_AR_THRESHOLD={EYE_AR_THRESHOLD}, EYE_CONSECUTIVE_FRAMES={EYE_CONSECUTIVE_FRAMES}")

    # Allow for flexibility in frame count - don't strictly enforce it
    if len(images) < 5:  # Too few frames might not be reliable
        print(
            f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Too few frames received: {len(images)}")
        raise HTTPException(status_code=400, detail=f"At least 5 frames are required for reliable drowsiness detection")

    drowsy_frames = 0
    detection_results = []
    ear_values = []

    # Process each image
    for i, image in enumerate(images):
        frame_start_time = time.time()

        try:
            # Read image
            contents = await image.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                print(
                    f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Invalid image at index {i}")
                continue

            # Detect drowsiness
            result = detect_drowsiness(img)

            # Track drowsy frames
            if result["is_drowsy"]:
                drowsy_frames += 1

            # Track EAR values for statistics
            ear_values.append(result["avg_ear"])

            # Add frame processing time
            frame_processing_time = (time.time() - frame_start_time) * 1000

            # Save detailed results
            detection_results.append({
                "frame": i,
                "processing_time_ms": frame_processing_time,
                **result
            })

            if i % 5 == 0:  # Log progress for every 5th frame
                print(
                    f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Processed {i + 1}/{len(images)} frames")

        except Exception as e:
            print(
                f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Error processing frame {i}: {str(e)}")
            # Continue processing other frames

    # Calculate overall statistics
    valid_frames = len(detection_results)
    if valid_frames == 0:
        print(
            f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: No valid frames processed")
        raise HTTPException(status_code=400, detail="No valid frames could be processed")

    # Calculate drowsiness metrics
    drowsy_percentage = (drowsy_frames / valid_frames) * 100 if valid_frames > 0 else 0

    # Determine if alert is needed - either by consecutive frames or percentage threshold
    consecutive_threshold = min(EYE_CONSECUTIVE_FRAMES, valid_frames // 2)  # Adapt threshold based on frame count
    is_alert_needed = drowsy_frames >= consecutive_threshold

    # Calculate EAR statistics if available
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
        f"SMARTROADSAI | drowsiness_detection_server.py | detect_batch_drowsiness | request_id={request_id}: Completed batch processing in {processing_time_ms:.2f}ms. {drowsy_frames}/{valid_frames} drowsy frames ({drowsy_percentage:.2f}%). {alert_status}")

    return response


# We've removed the real-time endpoint as it's no longer needed
# The application now uses batch processing exclusively

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)