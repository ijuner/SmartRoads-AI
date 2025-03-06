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
st.title("Face Detection App")
st.write("Detect faces using webcam streaming or image upload")

# Server URL
server_url = st.text_input("Server URL", value="http://localhost:8000/detect_face")

# Confidence threshold
min_confidence = st.slider("Minimum Confidence", min_value=0.1, max_value=1.0, value=0.7, step=0.1)

# Add toggle for webcam vs image upload
detection_mode = st.radio("Detection Mode", ["Real-time Webcam", "Image Upload"])

# Initialize the webcam
cap = None


# Function to process uploaded image
def process_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return

    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    # Convert to JPEG for sending to server
    _, buffer = cv2.imencode('.jpg', img_array)
    io_buf = BytesIO(buffer)

    # Create placeholder for results
    result_placeholder = st.empty()
    status_placeholder = st.info("Processing image...")

    try:
        # Send to server
        files = {'image': ('image.jpg', io_buf, 'image/jpeg')}
        data = {'min_confidence': min_confidence}

        response = requests.post(server_url, files=files, data=data)
        if response.status_code == 200:
            result = response.json()

            # Display detection results
            result_placeholder.json(result)

            if result["face_detected"]:
                status_placeholder.success(f"Detection complete: Found {result['count']} face(s)")
            else:
                status_placeholder.warning("No faces detected in the image")
        else:
            status_placeholder.error(f"Error from server: {response.status_code} - {response.text}")

    except Exception as e:
        status_placeholder.error(f"Error communicating with server: {e}")


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


# Display the appropriate interface based on the selected mode
if detection_mode == "Real-time Webcam":
    st.write("This mode captures frames from your webcam and detects faces every 10 seconds.")

    # Button to start/stop webcam
    if st.button("Start Webcam Detection"):
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam")
        else:
            process_frames()

    # Information about webcam mode
    with st.expander("How real-time detection works"):
        st.markdown("""
        1. The app continuously displays your webcam feed
        2. Every 10 seconds, it captures frames and sends them to the server
        3. The server analyzes each frame and returns whether faces were detected
        4. Results are displayed below the webcam feed
        """)

else:  # Image Upload mode
    st.write("Upload an image to detect faces")

    # Create file uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Process button
    if uploaded_file is not None:
        if st.button("Process Image"):
            process_uploaded_image(uploaded_file)

    # Information about upload mode
    with st.expander("How image upload works"):
        st.markdown("""
        1. Upload an image using the file selector above
        2. Click 'Process Image' to send it to the server
        3. The server will analyze the image for faces
        4. Results will be displayed below the image
        """)

# Shared requirements info
st.sidebar.markdown("""
## Requirements
- Backend server running (set the correct URL above)
- For webcam mode: Working camera access
""")