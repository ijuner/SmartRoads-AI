#!/bin/bash
set -e

# Display environment variables for debugging
echo "SMARTROADSAI | backend_start.sh | ENV-INFO: DROWSINESS_FRAMES_COUNT=${DROWSINESS_FRAMES_COUNT}"
echo "SMARTROADSAI | backend_start.sh | ENV-INFO: EYE_AR_THRESHOLD=${EYE_AR_THRESHOLD}"
echo "SMARTROADSAI | backend_start.sh | ENV-INFO: EYE_CONSECUTIVE_FRAMES=${EYE_CONSECUTIVE_FRAMES}"


# Install Python dependencies at runtime
echo "SMARTROADSAI | backend_start.sh | SETUP: Installing Python dependencies..."
pip install --no-cache-dir -r backend_requirements.txt
pip install --no-cache-dir -r frontend_requirements.txt

# Start the backend in the background
echo "SMARTROADSAI | backend_start.sh | STARTUP: Starting drowsiness detection backend service...WITH PIPELINE INTEGRATED"
uvicorn drowsiness_server_pipeline:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Give the backend a moment to start
sleep 5

# Start the frontend
echo "SMARTROADSAI | backend_start.sh | STARTUP: Starting drowsiness detection frontend service..."
streamlit run drowsiness_detection_frontend.py --server.port=8501 --server.address=0.0.0.0 &
FRONTEND_PID=$!

sleep 10
echo "SMARTROADSAI | backend_start.sh | STARTUP: Starting drowsiness detection ML flow server..."
mlflow server --host 0.0.0.0 --port 5000 &
MLFLOW_PID=$!

# Create a function to handle signals
function handle_sigterm {
  echo "SMARTROADSAI | backend_start.sh | SHUTDOWN: Received SIGTERM, shutting down services..."
  kill $FRONTEND_PID $BACKEND_PID $MLFLOW_PID
  wait $FRONTEND_PID $BACKEND_PID $MLFLOW_PID
  exit 0
}

# Register the signal handler
trap handle_sigterm SIGTERM SIGINT

# Wait for both processes
echo "SMARTROADSAI | backend_start.sh | STATUS: Both services started. Monitoring..."
wait $BACKEND_PID $FRONTEND_PID