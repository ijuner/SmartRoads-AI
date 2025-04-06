## USage of MLFlow

- Install mlflow ```pip install mlflow```
- Open ```drowsiness_detection_server.py```
- ```MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
MLFLOW_EXPERIMENT_NAME = os.getenv('MLFLOW_EXPERIMENT_NAME', 'drowsiness_detection')```
- We have added the above code for configuring mlflow
- ```mlf.set_tracking_uri(MLFLOW_TRACKING_URI)
mlf.set_experiment(MLFLOW_EXPERIMENT_NAME)```
- The above sets the config to the mlflow instance used in code
- We define a function to log params and metrics to mlflow
- ```def log_detection_to_mlflow(username, request_id, ear_stats, frame_count, drowsy_frames, drowsy_percentage,
                            is_alert, processing_time, ear_threshold, consecutive_frames):
    """Log drowsiness detection results to MLflow"""
    try:
        with mlf.start_run(run_name=f"detection_{request_id}"):
            # Log parameters
            mlf.log_params({
                "username": username,
                "eye_ar_threshold": ear_threshold,
                "consecutive_frames_threshold": consecutive_frames,
                "total_frames": frame_count
            })

            # Log metrics
            mlf.log_metrics({
                "drowsy_frames": drowsy_frames,
                "drowsy_percentage": drowsy_percentage,
                "avg_ear": ear_stats.get("avg_ear", 0),
                "min_ear": ear_stats.get("min_ear", 0),
                "max_ear": ear_stats.get("max_ear", 0),
                "processing_time_ms": processing_time
            })

            # Log alert as a tag
            mlf.set_tag("alert_triggered", is_alert)
            mlf.set_tag("request_id", request_id)

        print(f"SMARTROADSAI | drowsiness_detection_server.py: MLflow logging successful for request {request_id}")
    except Exception as e:
        print(f"SMARTROADSAI | drowsiness_detection_server.py: MLflow logging failed: {str(e)}")```
  
- Now we can run the mlflow server to see the metrics.
- ```mlflow server --host 0.0.0.0 --port 5000```
- Now visit localhost:5000 in your browser to see the logs