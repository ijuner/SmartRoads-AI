#!/bin/bash
set -e

# Check if models directory is empty
if [ ! -f model/yolo_trained_model_03_06.pt ]; then
  echo "Downloading model file..."
  # Use curl, wget or any other tool to download your model
  # Example:
  # curl -o model/yolo_trained_model_03_06.pt https://your-storage-url/yolo_trained_model_03_06.pt
fi

# Install Python dependencies at runtime
pip install --no-cache-dir -r requirements-backend.txt
pip install torch==2.6.0 torchvision==0.21.0
pip install ultralytics==8.3.85

# Apply model loading fix
sed -i 's/model = YOLO("model\/yolo_trained_model_03_06.pt")/import torch\ntorch.serialization.add_safe_globals(["ultralytics.nn.tasks.DetectionModel"])\nmodel = YOLO("model\/yolo_trained_model_03_06.pt")/' face_detection_server_yolo.py

# Start the application
exec uvicorn face_detection_server_yolo:app --host 0.0.0.0 --port 8000