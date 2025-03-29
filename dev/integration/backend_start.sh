#!/bin/bash
set -e

# Model file should already be present in the image
echo "Using pre-packaged model file: model/yolo_trained_model_03_06.pt"

# Install Python dependencies at runtime
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements-backend.txt
pip install --no-cache-dir -r requirements-frontend.txt
pip install torch==2.6.0 torchvision==0.21.0
pip install ultralytics==8.3.85

# Apply model loading fix
echo "Applying model loading fix..."
sed -i 's/model = YOLO("model\/yolo_trained_model_03_06.pt")/import torch\ntorch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])\nmodel = YOLO("model\/yolo_trained_model_03_06.pt")/' face_detection_server_yolo.py

# Start the backend in the background
echo "Starting backend service..."
uvicorn face_detection_server_yolo:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 5

# Start the frontend
echo "Starting frontend service..."
streamlit run streamlit_frontend.py --server.port=8501 --server.address=0.0.0.0 &
FRONTEND_PID=$!

# Create a function to handle signals
function handle_sigterm {
  echo "Received SIGTERM, shutting down services..."
  kill $FRONTEND_PID $BACKEND_PID
  wait $FRONTEND_PID $BACKEND_PID
  exit 0
}

# Register the signal handler
trap handle_sigterm SIGTERM SIGINT

# Wait for both processes
echo "Both services started. Monitoring..."
wait $BACKEND_PID $FRONTEND_PID