import streamlit as st
import cv2
import time
import requests
import json
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="SmartRoads - Face Detection", layout="wide")

# App title and description
st.title("Real-time Face Detection")
st.write("This app captures frames from your webcam and detects face every 10 seconds.")

# Server URL
server_url = st.text_input("Server URL", value="http://localhost:8000/detect_face")

# Confidence threshold
min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

# Initialize the webcam
cap = None


# Function to capture and process frames
def process_frames():
    if not cap or not cap.isOpened():
        st.error("Could not access webcam. Please check your camera settings.")
        return

    # Create placeholders for webcam feed and results
    webcam_placeholder = st.empty()
    result_placeholder = st.empty()
    status_placeholder = st.empty()

    last_detection_time = time.time()
    detection_interval = 10  # seconds
    frames_to_capture = 1

    try:
        while True:
            # Capture a frame
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from webcam")
                break

            # Display the live webcam feed
            webcam_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")

            # Check if it's time to detect faces
            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                status_placeholder.info("Capturing frames for face detection...")

                # Capture multiple frames
                detection_results = []
                for i in range(frames_to_capture):
                    # Capture a new frame for detection
                    ret, detection_frame = cap.read()
                    if not ret:
                        continue

                    # Convert frame to JPEG
                    _, buffer = cv2.imencode('.jpg', detection_frame)
                    io_buf = BytesIO(buffer)

                    # Send to server
                    try:
                        files = {'image': ('image.jpg', io_buf, 'image/jpeg')}
                        data = {'min_confidence': min_confidence}

                        response = requests.post(server_url, files=files, data=data)
                        if response.status_code == 200:
                            result = response.json()
                            detection_results.append(result)

                            # Draw rectangles on detected faces for visualization
                            if result["face_detected"]:
                                # Convert frame to grayscale for face detection (just for visualization)
                                gray = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2GRAY)
                                face_cascade = cv2.CascadeClassifier(
                                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                                # Draw rectangles around detected faces
                                for (x, y, w, h) in faces:
                                    cv2.rectangle(detection_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                # Display the frame with face detection
                                webcam_placeholder.image(detection_frame, channels="BGR",
                                                         caption=f"Detected {result['count']} face(s)")
                    except Exception as e:
                        st.error(f"Error communicating with server: {e}")

                    # Small delay between frames
                    time.sleep(0.5)

                # Display detection results
                if detection_results:
                    result_placeholder.json(detection_results)

                    # Count frames with faces
                    frames_with_faces = sum(1 for r in detection_results if r["face_detected"])
                    status_placeholder.success(
                        f"Detection complete: Found faces in {frames_with_faces} of {len(detection_results)} frames")

                # Reset timer
                last_detection_time = current_time

            # Add a small delay to reduce CPU usage
            time.sleep(0.1)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    finally:
        if cap and cap.isOpened():
            cap.release()


# Button to start/stop webcam
if st.button("Start Detection"):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam")
    else:
        process_frames()

# Information about the app
st.markdown("""
### How it works
1. The app continuously displays your webcam feed
2. Every 10 seconds, it captures 5 frames and sends them to the server
3. The server analyzes each frame and returns whether faces were detected
4. Results are displayed below the webcam feed

### Requirements
- Webcam access
- Running backend server (set the correct URL above)
""")