FROM python:3.11-rc-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY drowsiness_detection_frontend.py .
COPY drowsiness_detection_server.py .
COPY drowsiness_server_pipeline.py .
COPY drowsiness_pipeline.py .
COPY backend_requirements.txt .
COPY frontend_requirements.txt .
COPY alarm.mp3 .
# Copy startup and config files
COPY backend_start.sh .
RUN chmod +x backend_start.sh

# Expose both ports
EXPOSE 8000 8501

# Set environment variable for frontend to connect to backend
# Define environment variables with defaults
ENV DROWSINESS_FRAMES_COUNT=48
ENV EYE_AR_THRESHOLD=0.2
ENV EYE_CONSECUTIVE_FRAMES=12

# Run the startup script directly
CMD ["./backend_start.sh"]