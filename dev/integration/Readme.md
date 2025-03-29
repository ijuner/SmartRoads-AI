# Building and Sharing the Face Detection Docker Image

This guide explains how to build your Docker image and share it with your professor.

## Step 1: Create necessary files

1. Create a file named `Dockerfile` with the content from the "All-in-One Dockerfile" above.

2. Create a file named `supervisord.conf` with the content from the "Supervisor Configuration" above.

3. Make sure your project structure is:
```
project-folder/
├── Dockerfile
├── supervisord.conf
├── face_detection_server_yolo.py
├── streamlit_frontend.py
└── model/
    └── yolo_trained_model_03_06.pt
```

## Step 2: Build the Docker image

Open a terminal in your project folder and run:

```bash
docker build -t face-detection-app:latest .
```

This will create a Docker image with all your code, the YOLO model, and dependencies.

## Step 3: Test the image locally

Run the image to make sure everything works correctly:

```bash
docker run -p 8501:8501 -p 8000:8000 face-detection-app:latest
```

Then open your browser and navigate to:
- `http://localhost:8501` - For the Streamlit frontend

## Step 4: Save the Docker image to a file

```bash
docker save face-detection-app:latest -o face-detection-app.tar
```

This creates a file called `face-detection-app.tar` containing your Docker image. This file might be large (could be several GB) because it includes all dependencies, your code, and your ML model.

## Step 5: Share the Docker image with your professor

### Option 1: Share the .tar file directly

Provide the `face-detection-app.tar` file to your professor (via Google Drive, Dropbox, or another file sharing service).

They can load it with:
```bash
docker load -i face-detection-app.tar
```

And then run it with:
```bash
docker run -p 8501:8501 -p 8000:8000 face-detection-app:latest
```

### Option 2: Push to Docker Hub (easier for your professor)

1. Create a free account on [Docker Hub](https://hub.docker.com/)
2. Login to Docker Hub from your terminal:
   ```bash
   docker login
   ```
3. Tag your image with your Docker Hub username:
   ```bash
   docker tag face-detection-app:latest yourusername/face-detection-app:latest
   ```
4. Push the image to Docker Hub:
   ```bash
   docker push yourusername/face-detection-app:latest
   ```
5. Share your Docker Hub image name (`yourusername/face-detection-app:latest`) with your professor

Your professor can then pull and run the image with:
```bash
docker pull yourusername/face-detection-app:latest
docker run -p 8501:8501 -p 8000:8000 yourusername/face-detection-app:latest
```

## Instructions for your professor

To run the application:

1. Open a terminal
2. Run:
   ```bash
   docker run -p 8501:8501 -p 8000:8000 [IMAGE_NAME]
   ```
   (Where [IMAGE_NAME] is either `face-detection-app:latest` or `yourusername/face-detection-app:latest`)
3. Open a web browser and go to `http://localhost:8501`
4. Use the Streamlit interface to either:
   - Upload an image for face detection
   - Use the webcam for real-time face detection

That's it! All the services are running in a single container.