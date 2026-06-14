import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import json
import os
from typing import Dict, List, Tuple

# Global ML service instance
_ml_service_instance = None

class MLService:
    def __init__(self, model_path: str, config_path: str):
        """Initialize ML Service with model and config"""
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.classes = None
        self.input_size = 224
        self.threshold = 0.5
        self.confidence_threshold = 0.65
        self.top2_margin = 0.15
        
        # Load model and config
        self.load_model()
        self.load_config()
    
    def load_model(self):
        """Load the trained Keras model"""
        try:
            print(f"Loading model from: {self.model_path}")
            self.model = keras.models.load_model(self.model_path)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def load_config(self):
        """Load model configuration"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            self.classes = self.config['classes']
            self.input_size = self.config['input_size']
            # Allow threshold to be configured in model config
            self.threshold = float(self.config.get('threshold', self.threshold))
            print(f"✅ Config loaded - Classes: {self.classes}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            raise
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for model prediction
        Same preprocessing as training
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            
            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            img = cv2.resize(img, (self.input_size, self.input_size))
            
            # Normalize (same as training: rescale 1./255)
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            raise
    
    def validate_prediction_confidence(self, confidence: float) -> bool:
        """Check whether the top prediction confidence is sufficient."""
        return confidence >= self.confidence_threshold

    def check_eye_features(self, image_bytes: bytes) -> bool:
        """Detect circular eye features using HoughCircles."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            gray = cv2.medianBlur(gray, 5)
            height, width = gray.shape[:2]
            min_dim = min(height, width)
            min_radius = max(6, min_dim // 30)
            max_radius = max(25, min_dim // 3)
            min_dist = max(10, min_dim // 10)

            search_params = [
                dict(dp=1.0, param1=80, param2=24),
                dict(dp=1.2, param1=100, param2=22),
                dict(dp=1.5, param1=120, param2=20)
            ]

            for params in search_params:
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    dp=params['dp'],
                    minDist=min_dist,
                    param1=params['param1'],
                    param2=params['param2'],
                    minRadius=min_radius,
                    maxRadius=max_radius
                )
                if circles is not None and len(circles[0]) > 0:
                    return True

            # Try a smaller resized image to catch smaller irises in close-up photos
            small_gray = cv2.resize(gray, (max(64, width // 2), max(64, height // 2)))
            small_gray = cv2.medianBlur(small_gray, 5)
            small_height, small_width = small_gray.shape[:2]
            small_min_dim = min(small_height, small_width)
            small_min_radius = max(4, small_min_dim // 30)
            small_max_radius = max(20, small_min_dim // 3)
            small_min_dist = max(8, small_min_dim // 10)

            circles = cv2.HoughCircles(
                small_gray,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=small_min_dist,
                param1=80,
                param2=18,
                minRadius=small_min_radius,
                maxRadius=small_max_radius
            )
            return circles is not None and len(circles[0]) > 0
        except Exception as e:
            print(f"Warning: Eye feature detection failed: {e}")
            return False

    def predict(self, image_bytes: bytes) -> Dict:
        """
        Make prediction on uploaded image
        Includes validation for eye images and confidence checks
        """
        try:
            if not self.check_eye_features(image_bytes):
                return {
                    "error": True,
                    "message": "No eye detected in image. Please upload a clear eye photo.",
                    "predicted_class": "Invalid Image",
                    "confidence": 0.0,
                    "all_probabilities": {},
                    "is_normal": False
                }

            processed_image = self.preprocess_image(image_bytes)
            predictions = self.model.predict(processed_image, verbose=0)
            probabilities = np.array(predictions[0], dtype=float)
            predicted_idx = int(np.argmax(probabilities))
            predicted_class = self.classes[predicted_idx]
            confidence = float(probabilities[predicted_idx])

            all_probabilities = {
                self.classes[i]: float(probabilities[i])
                for i in range(len(self.classes))
            }
            sorted_probs = dict(
                sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)
            )
            top_probs = list(sorted_probs.values())

            if not self.validate_prediction_confidence(confidence):
                return {
                    "error": True,
                    "message": "Cannot confidently diagnose. This may not be one of the 4 trained conditions.",
                    "predicted_class": "Unknown/Unclear Image",
                    "confidence": confidence,
                    "all_probabilities": sorted_probs,
                    "is_normal": False
                }

            if len(top_probs) >= 2 and (top_probs[0] - top_probs[1]) <= self.top2_margin:
                return {
                    "error": True,
                    "message": "Multiple conditions possible. Please consult a doctor.",
                    "predicted_class": "Ambiguous Image",
                    "confidence": confidence,
                    "all_probabilities": sorted_probs,
                    "is_normal": False
                }

            return {
                "error": False,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "all_probabilities": sorted_probs,
                "is_normal": predicted_class.lower() == "normal"
            }
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return {
                "error": True,
                "message": f"Prediction error: {str(e)}",
                "predicted_class": "Error",
                "confidence": 0.0,
                "all_probabilities": {},
                "is_normal": False
            }


def get_ml_service() -> MLService:
    """Get or create ML service instance"""
    global _ml_service_instance
    if _ml_service_instance is None:
        from app.core.config import settings
        _ml_service_instance = MLService(
            model_path=settings.MODEL_PATH,
            config_path=settings.MODEL_CONFIG_PATH
        )
    return _ml_service_instance