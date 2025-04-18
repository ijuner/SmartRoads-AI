import streamlit as st
import cv2
import time
import requests
import json
import os
import uuid
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np
import base64

# Set page config
st.set_page_config(page_title="SMARTROADS AI - DROWSINESS DETECTION", layout="wide")


# Audio functionality using Streamlit's native audio component
def setup_audio():
    """Set up the audio component for alarms"""
    if 'audio_placeholder' not in st.session_state:
        st.session_state.audio_placeholder = st.empty()
    return st.session_state.audio_placeholder


def autoplay_audio():
    with open("alarm.mp3", "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(
            md,
            unsafe_allow_html=True,
        )

def play_alarm():
    """Play the alarm sound using Streamlit's audio component"""
    try:
        audio_file = open("alarm.mp3", "rb")
        audio_bytes = audio_file.read()
        audio_file.close()
        st.session_state.audio_placeholder.audio(audio_bytes, autoplay=True)
        print("SMARTROADSAI | drowsiness_detection_frontend.py | play_alarm: Playing alarm sound")
    except Exception as e:
        print(f"SMARTROADSAI | drowsiness_detection_frontend.py | play_alarm: Error playing alarm - {str(e)}")


def stop_alarm():
    """Stop the alarm sound by clearing the audio component"""
    try:
        st.session_state.audio_placeholder.empty()
        print("SMARTROADSAI | drowsiness_detection_frontend.py | stop_alarm: Stopping alarm sound")
    except Exception as e:
        print(f"SMARTROADSAI | drowsiness_detection_frontend.py | stop_alarm: Error stopping alarm - {str(e)}")


# Server URL - get from environment variable if available (for Docker)
default_url = os.environ.get("SERVER_URL", "http://localhost:8000")

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'auth_token' not in st.session_state:
    st.session_state.auth_token = None

if 'username' not in st.session_state:
    st.session_state.username = None

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

if 'alarm_active' not in st.session_state:
    st.session_state.alarm_active = False


# Login function
def login(username, password, server_url):
    print(f"SMARTROADSAI | drowsiness_detection_frontend.py | login: Attempting login for user {username}")

    try:
        response = requests.post(
            f"{server_url}/auth",
            json={"username": username, "password": password}
        )

        if response.status_code == 200:
            data = response.json()
            st.session_state.logged_in = True
            st.session_state.auth_token = data["auth_token"]
            st.session_state.username = data["username"]
            return True, "Login successful"
        else:
            error_msg = response.json().get("detail", "Login failed")
            print(f"SMARTROADSAI | drowsiness_detection_frontend.py | login: Failed - {error_msg}")
            return False, error_msg

    except Exception as e:
        print(f"SMARTROADSAI | drowsiness_detection_frontend.py | login: Error - {str(e)}")
        return False, f"Error connecting to server: {str(e)}"


def logout():
    st.session_state.logged_in = False
    st.session_state.auth_token = None
    st.session_state.username = None
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.frame_count = 0
    st.session_state.alarm_active = False


# Login page
def login_page():
    st.title("SMARTROADS AI - DROWSINESS DETECTION")
    st.subheader("Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        server_url_input = st.text_input("Server URL", value=default_url)

        submitted = st.form_submit_button("Login")

        if submitted:
            success, message = login(username, password, server_url_input)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)


# Function to process a batch of uploaded images
def process_batch_images(uploaded_files, server_url):
    if not uploaded_files or len(uploaded_files) == 0:
        return

    print(
        f"SMARTROADSAI | drowsiness_detection_frontend.py | process_batch_images: Processing {len(uploaded_files)} images")

    # Display the first uploaded image as a preview
    first_image = Image.open(uploaded_files[0])
    st.image(first_image, caption="Preview Image", use_column_width=True)

    # Create placeholder for results
    result_placeholder = st.empty()
    status_placeholder = st.info("Processing images for drowsiness detection...")

    try:
        # Prepare the files for sending to server
        files = []
        for i, uploaded_file in enumerate(uploaded_files):
            # Convert the image to bytes
            img_bytes = uploaded_file.getvalue()
            files.append(('images', (f'image_{i}.jpg', img_bytes, 'image/jpeg')))

        # Send to server
        headers = {"X-Auth-Token": st.session_state.auth_token}
        response = requests.post(
            f"{server_url}/detect_batch_drowsiness",
            files=files,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()

            # Display detection results
            result_placeholder.json(result)

            if result["alert_needed"]:
                status_placeholder.error(
                    f"ALERT: Drowsiness detected in {result['drowsy_frames_count']} out of {result['total_frames']} frames!")
                st.session_state.alarm_active = True
                play_alarm()
                #autoplay_audio()
            else:
                status_placeholder.success(
                    f"No drowsiness detected. Only {result['drowsy_frames_count']} out of {result['total_frames']} frames showed signs of drowsiness.")
                st.session_state.alarm_active = False
                stop_alarm()
        else:
            error_msg = response.json().get("detail", "Unknown error")
            status_placeholder.error(f"Error from server: {response.status_code} - {error_msg}")

    except Exception as e:
        status_placeholder.error(f"Error communicating with server: {str(e)}")


# Process batch of frames captured from webcam
def process_webcam_batch(frames, server_url):
    """
    Process a batch of frames captured from the webcam

    Args:
        frames: List of frames captured from webcam
        server_url: URL of the backend server

    Returns:
        dict: Processing result from the server
    """
    print(f"SMARTROADSAI | drowsiness_detection_frontend.py | process_webcam_batch: Processing {len(frames)} frames")

    try:
        # Convert frames to file-like objects
        files = []
        for i, frame in enumerate(frames):
            _, buffer = cv2.imencode('.jpg', frame)
            io_buf = BytesIO(buffer)
            files.append(('images', (f'frame_{i}.jpg', io_buf.getvalue(), 'image/jpeg')))

        # Send to server
        headers = {"X-Auth-Token": st.session_state.auth_token}

        response = requests.post(
            f"{server_url}/detect_batch_drowsiness",
            files=files,
            headers=headers
        )

        if response.status_code == 200:
            return response.json()
        else:
            print(
                f"SMARTROADSAI | drowsiness_detection_frontend.py | process_webcam_batch: Error - Status code {response.status_code}")
            return None

    except Exception as e:
        print(f"SMARTROADSAI | drowsiness_detection_frontend.py | process_webcam_batch: Error - {str(e)}")
        return None


# Main app interface
def main_app():
    st.title("SMARTROADS AI - DROWSINESS DETECTION")

    # User info and logout button in sidebar
    with st.sidebar:
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        if st.button("Logout"):
            logout()
            st.rerun()

    # Server URL input
    server_url = st.text_input("Server URL", value=default_url)

    # Add toggle for webcam vs image upload
    detection_mode = st.radio("Detection Mode", ["Webcam Batch Detection", "Batch Image Upload"])

    # Set up audio for alarm
    audio_placeholder = setup_audio()

    if detection_mode == "Webcam Batch Detection":
        st.write("This mode shows your webcam stream and captures 24 frames every 30 seconds for drowsiness analysis.")

        # Create placeholders
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        timer_placeholder = st.empty()
        results_placeholder = st.empty()

        # Add a slider for capture interval
        capture_interval = st.slider("Capture Interval (seconds)", min_value=10, max_value=60, value=30, step=5)
        frames_to_capture = int(os.environ.get("DROWSINESS_FRAMES_COUNT", "24"))

        # Button to start/stop webcam
        start_button = st.button("Start Webcam Detection")
        stop_button = st.button("Stop Webcam Detection")

        if start_button:
            st.session_state.webcam_running = True
            st.session_state.last_capture_time = time.time() - capture_interval  # Trigger immediate first capture

        if stop_button:
            st.session_state.webcam_running = False
            stop_alarm()
            st.session_state.alarm_active = False

        if st.session_state.get('webcam_running', False):
            # Initialize webcam with more robust error handling
            try:
                # Try different backends and camera indices
                cap = None
                for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW]:  # Try default and DirectShow
                    for camera_idx in range(2):  # Try camera 0 and 1
                        print(
                            f"SMARTROADSAI | drowsiness_detection_frontend.py | Trying camera index {camera_idx} with backend {backend}")
                        try:
                            cap = cv2.VideoCapture(camera_idx, backend)
                            if cap.isOpened():
                                print(
                                    f"SMARTROADSAI | drowsiness_detection_frontend.py | Successfully opened camera {camera_idx} with backend {backend}")
                                break
                        except Exception as e:
                            print(
                                f"SMARTROADSAI | drowsiness_detection_frontend.py | Failed to open camera {camera_idx} with backend {backend}: {str(e)}")
                    if cap is not None and cap.isOpened():
                        break

                if cap is None or not cap.isOpened():
                    # One last attempt with default settings
                    cap = cv2.VideoCapture(0)
            except Exception as e:
                print(f"SMARTROADSAI | drowsiness_detection_frontend.py | Error initializing camera: {str(e)}")
                cap = None

            if cap is None or not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection and permissions.")
                st.session_state.webcam_running = False
            else:
                # Set info message
                info_placeholder.info("Webcam is active. Displaying stream and capturing batches every 30 seconds...")

                try:
                    while st.session_state.get('webcam_running', False):
                        # Capture a frame for display
                        ret, frame = cap.read()

                        # If frame capture fails, try again with a small delay
                        if not ret:
                            print(
                                f"SMARTROADSAI | drowsiness_detection_frontend.py | Frame capture failed, retrying...")
                            time.sleep(0.5)  # Wait a bit
                            ret, frame = cap.read()

                        if not ret:
                            print(f"SMARTROADSAI | drowsiness_detection_frontend.py | Frame capture failed after retry")
                            st.error(
                                "Failed to capture frame from webcam. The camera may be in use by another application.")
                            time.sleep(2)  # Add longer delay before next attempt
                            continue

                        # Display the frame
                        frame_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")

                        # Check if it's time to capture a batch
                        current_time = time.time()
                        time_since_last_capture = current_time - st.session_state.get('last_capture_time', 0)
                        time_to_next_capture = max(0, capture_interval - time_since_last_capture)

                        # Update timer display
                        timer_placeholder.info(f"Next batch capture in: {int(time_to_next_capture)} seconds")

                        # If it's time to capture a batch of frames
                        if time_since_last_capture >= capture_interval:
                            info_placeholder.warning("Capturing batch of frames for drowsiness detection...")

                            # Capture a batch of frames
                            batch_frames = []
                            for i in range(frames_to_capture):
                                ret, batch_frame = cap.read()
                                if ret:
                                    batch_frames.append(batch_frame)
                                    # Brief pause between frames to ensure they're not identical
                                    time.sleep(0.05)

                            # Process the batch if we have enough frames
                            if len(batch_frames) == frames_to_capture:
                                # Convert frames to file-like objects
                                files = []
                                for i, img in enumerate(batch_frames):
                                    _, buffer = cv2.imencode('.jpg', img)
                                    io_buf = BytesIO(buffer)
                                    files.append(('images', (f'frame_{i}.jpg', io_buf.getvalue(), 'image/jpeg')))

                                # Send batch to server
                                try:
                                    headers = {"X-Auth-Token": st.session_state.auth_token}
                                    response = requests.post(
                                        f"{server_url}/detect_batch_drowsiness",
                                        files=files,
                                        headers=headers
                                    )

                                    if response.status_code == 200:
                                        result = response.json()
                                        results_placeholder.json(result)

                                        # Check if alert is needed
                                        if result.get("alert_needed", False):
                                            info_placeholder.error(
                                                f"ALERT: Drowsiness detected in {result['drowsy_frames_count']} of {result['total_frames']} frames!")
                                            if not st.session_state.alarm_active:
                                                st.session_state.alarm_active = True
                                                play_alarm()
                                        else:
                                            info_placeholder.success(
                                                f"No drowsiness detected. Found drowsiness in only {result.get('drowsy_frames_count', 0)} frames.")
                                            if st.session_state.alarm_active:
                                                st.session_state.alarm_active = False
                                                stop_alarm()
                                    else:
                                        info_placeholder.error(f"Error from server: {response.status_code}")
                                except Exception as e:
                                    info_placeholder.error(f"Error sending frames to server: {str(e)}")
                            else:
                                info_placeholder.error(
                                    f"Failed to capture enough frames. Got {len(batch_frames)} of {frames_to_capture}")

                            # Reset capture timer
                            st.session_state.last_capture_time = current_time

                        # Add a small delay to reduce CPU usage
                        time.sleep(0.1)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                finally:
                    if cap and cap.isOpened():
                        cap.release()
                    st.session_state.webcam_running = False

        # Information about webcam batch mode
        with st.expander("How webcam batch detection works"):
            st.markdown(f"""
            1. The app displays a continuous webcam feed locally
            2. Every {capture_interval} seconds, it captures {frames_to_capture} consecutive frames
            3. These frames are sent as a batch to the server for drowsiness analysis
            4. If enough frames show drowsiness signs, an alert will be triggered
            5. This approach is more efficient than sending every frame to the server
            """)

    else:  # Batch Image Upload mode
        st.write("Upload a series of images to detect drowsiness patterns")

        # Create file uploader for multiple images
        uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        # Process button
        if uploaded_files:
            if st.button("Process Images"):
                process_batch_images(uploaded_files, server_url)

        # Information about upload mode
        with st.expander("How batch image processing works"):
            st.markdown("""
            1. Upload a series of images (ideally 24 consecutive frames)
            2. Click 'Process Images' to send them to the server
            3. The server will analyze each image for drowsiness signs
            4. If more than half of the images show drowsiness, an alert will be triggered
            5. Results will be displayed with detailed information about each frame
            """)


# Main app flow
if not st.session_state.logged_in:
    login_page()
else:
    main_app()