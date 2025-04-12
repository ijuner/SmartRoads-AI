import cv2
import numpy as np
import mediapipe as mp
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Environment variables
EYE_AR_THRESHOLD = float(os.getenv('EYE_AR_THRESHOLD', '0.2'))  # Eye aspect ratio threshold

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh

# Define eye landmarks for left and right eyes
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class OpenCVPreprocessor(BaseEstimator, TransformerMixin):
    """
    Step 1: Preprocess images using OpenCV
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform images to RGB format for MediaPipe

        Parameters:
        -----------
        X : list or np.ndarray
            List of images in BGR format or bytes from image files

        Returns:
        --------
        list
            List of preprocessed RGB images
        """
        processed_images = []

        for img in X:
            # Handle case where input is bytes from uploaded file
            if isinstance(img, bytes):
                nparr = np.frombuffer(img, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Make sure image is valid
            if img is None or img.size == 0:
                continue

            # Convert to RGB (MediaPipe requires RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed_images.append(img_rgb)

        return processed_images


class MediaPipeDrowsinessDetector(BaseEstimator, TransformerMixin):
    """
    Step 2: Detect drowsiness using MediaPipe Face Mesh
    """

    def __init__(self, eye_ar_threshold=EYE_AR_THRESHOLD):
        self.eye_ar_threshold = eye_ar_threshold
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Detect drowsiness in each image

        Parameters:
        -----------
        X : list
            List of RGB images

        Returns:
        --------
        list
            List of drowsiness detection results
        """
        results = []

        for image in X:
            # Process the image with MediaPipe
            h, w, _ = image.shape
            mp_results = self.face_mesh.process(image)

            # Default values if no face detected
            is_drowsy = False
            left_ear = 0
            right_ear = 0
            avg_ear = 0

            # Check if face landmarks are detected
            if mp_results.multi_face_landmarks:
                for face_landmarks in mp_results.multi_face_landmarks:
                    # Calculate EAR for both eyes
                    left_ear = self._calculate_ear(LEFT_EYE, face_landmarks.landmark)
                    right_ear = self._calculate_ear(RIGHT_EYE, face_landmarks.landmark)

                    # Average EAR
                    avg_ear = (left_ear + right_ear) / 2.0

                    # Determine if drowsy
                    is_drowsy = avg_ear < self.eye_ar_threshold

            # Store detection results
            results.append({
                "is_drowsy": is_drowsy,
                "left_ear": float(left_ear),
                "right_ear": float(right_ear),
                "avg_ear": float(avg_ear),
                "threshold": float(self.eye_ar_threshold)
            })

        return results

    def _calculate_ear(self, eye_landmarks, face_landmarks):
        """
        Calculate the eye aspect ratio (EAR) given the eye landmarks
        """
        try:
            # Extract coordinates for the specified eye landmarks
            points = []
            for i in eye_landmarks:
                point = face_landmarks[i]
                points.append([point.x, point.y, point.z])

            # Calculate horizontal distance (width)
            horizontal_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

            # Calculate vertical distances (height at two points)
            vertical_dist1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            vertical_dist2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))

            # Calculate EAR
            ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            return ear
        except:
            return 0.3  # Return a default value if calculation fails


# Create the pipeline
def create_drowsiness_pipeline(eye_ar_threshold=EYE_AR_THRESHOLD):
    """
    Create and return the drowsiness detection pipeline

    Parameters:
    -----------
    eye_ar_threshold : float
        Threshold for eye aspect ratio to determine drowsiness

    Returns:
    --------
    Pipeline
        Scikit-learn pipeline for drowsiness detection
    """
    return Pipeline([
        ('preprocess', OpenCVPreprocessor()),
        ('drowsiness_detector', MediaPipeDrowsinessDetector(eye_ar_threshold=eye_ar_threshold))
    ])


# Function to process a batch of images
def process_batch_images(images, eye_ar_threshold=EYE_AR_THRESHOLD):
    """
    Process a batch of images to detect drowsiness

    Parameters:
    -----------
    images : list
        List of images as bytes or numpy arrays
    eye_ar_threshold : float
        Threshold for eye aspect ratio

    Returns:
    --------
    list
        List of drowsiness detection results for each image
    """
    pipeline = create_drowsiness_pipeline(eye_ar_threshold)
    return pipeline.transform(images)