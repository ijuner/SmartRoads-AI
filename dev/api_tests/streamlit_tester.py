import streamlit as st
import requests
import json
import os

# API endpoint
local_url = "http://localhost:5000/api/v1/check_drowsiness"
api_env = os.getenv("API_ENV", "local")  # Default to localhost if not set
api_url = "" if api_env == "aws" else local_url

# Streamlit App Title
st.title("Smart Roads")
st.subheader("Check drowsiness, set up alerts, and analyze suspicious messages.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
drowsiness = st.number_input("Choose drowsiness value", min_value=0, max_value=100, value=70)
face_factor = st.number_input("Choose face detection value", min_value=0, max_value=100, value=70)

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", width=150)

    # Convert the image to bytes
    image_bytes = uploaded_file.read()

    # Prepare JSON metadata
    json_metadata = json.dumps({"drowsiness": drowsiness, "face_detect_factor": face_factor})

    # Send POST request to the API
    try:
        files = {
            "image": ("image.jpg", image_bytes, "image/jpeg"),
            "metadata": (None, json_metadata, "application/json")
        }
        response = requests.post(api_url, files=files)

        if response.status_code == 200:
            result = response.json()
            st.subheader("Voila !!!")

            if result.get("drowsiness_factor", 0) < 80:
                st.subheader("⚠️ Alert User !!!!")
            else:
                st.subheader("✅ All Good!")

            if result.get("face_detected"):
                st.subheader("😊 Driver in position!")
            else:
                st.subheader("🚨 Driver NOT detected. Alert!")

        else:
            st.error(f"Error: {response.status_code}")
            st.write(response.text)

    except Exception as e:
        st.error(f"Failed to connect to the API. Error: {str(e)}")
