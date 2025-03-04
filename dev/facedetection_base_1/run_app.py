import subprocess
import sys
import time
import os


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import fastapi
        import streamlit
        import cv2
        import numpy
        import uvicorn
        import requests
        import PIL
        print("All dependencies installed!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False


def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "fastapi", "streamlit", "opencv-python",
                           "numpy", "uvicorn", "requests", "python-multipart", "pillow"])
    print("Dependencies installed successfully!")


def run_app():
    """Run both server and frontend"""
    # Start server in a separate process
    server_process = subprocess.Popen([sys.executable, "face_detection_server.py"])

    # Give the server a moment to start
    time.sleep(2)

    # Start Streamlit frontend
    streamlit_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "streamlit_frontend.py"])

    try:
        # Wait for user to terminate
        print("\nApp is running!")
        print("- FastAPI server is running at http://localhost:8000")
        print("- Streamlit frontend is running (check URL in console output above)")
        print("\nPress Ctrl+C to stop the application")

        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        # Clean up processes
        streamlit_process.terminate()
        server_process.terminate()


if __name__ == "__main__":
    # Create the server and frontend files if they don't exist
    with open("face_detection_server.py", "w") as f:
        # The artifacts might have different file names on Windows
        try:
            # Try the hyphenated filename first
            with open("face-detection-server.py", "r") as source:
                f.write(source.read())
        except FileNotFoundError:
            # If not found, try without hyphens
            with open("face_detection_server.py", "r") as source:
                f.write(source.read())

    with open("streamlit_frontend.py", "w") as f:
        try:
            # Try the hyphenated filename first
            with open("streamlit-frontend.py", "r") as source:
                f.write(source.read())
        except FileNotFoundError:
            # If not found, try without hyphens
            with open("streamlit_frontend.py", "r") as source:
                f.write(source.read())

    # Check and install dependencies if needed
    if not check_dependencies():
        print("Would you like to install the required dependencies? (y/n)")
        choice = input().lower()
        if choice == 'y':
            install_dependencies()
        else:
            print("Please install the required dependencies manually and try again.")
            sys.exit(1)

    # Run the application
    run_app()