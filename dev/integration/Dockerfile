FROM python:3.11-rc-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY face_detection_server_yolo.py .
COPY streamlit_frontend.py .
COPY requirements-backend.txt .
COPY requirements-frontend.txt .

# Create model directory and copy model file
RUN mkdir -p model
COPY model/yolo_trained_model_03_06.pt ./model/

# Copy startup and config files
COPY backend_start.sh .
RUN chmod +x backend_start.sh

# Expose both ports
EXPOSE 8000 8501

# Set environment variable for frontend to connect to backend
ENV SERVER_URL=http://localhost:8000/detect_face

# Run the startup script directly
CMD ["./backend_start.sh"]