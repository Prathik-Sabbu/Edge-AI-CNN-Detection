"""
TFLite CNN classifier for animal detection on Raspberry Pi.
Optimized for tflite-runtime or tensorflow-lite on ARM architectures.
Includes mock fallback when running in testing environments without trained model weights.
"""
import os
import logging
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger("EdgeAI.Classifier")

class AnimalClassifier:
    """Handles TFLite model loading, image preprocessing, and inference."""

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.50,
        enable_roi_crop: bool = True,
        roi_crop_scale: float = 1.0,
        num_threads: int = 4,
    ):
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold
        self.enable_roi_crop = enable_roi_crop
        self.roi_crop_scale = roi_crop_scale
        self.num_threads = num_threads

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.labels: List[str] = []
        self.is_mock_mode = False

        self._load_labels()
        self._load_model()

    def _load_labels(self):
        """Loads class labels from labels.txt."""
        if os.path.exists(self.labels_path):
            try:
                with open(self.labels_path, "r", encoding="utf-8") as f:
                    self.labels = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self.labels)} labels: {self.labels}")
            except Exception as e:
                logger.error(f"Error reading labels file: {e}")
                self.labels = ["Cat", "Dog", "Bird", "Wild Animal"]
        else:
            logger.warning(f"Labels file not found at {self.labels_path}. Using default classes.")
            self.labels = ["Cat", "Dog", "Bird", "Wild Animal"]

    def _load_model(self):
        """Loads TFLite interpreter from tflite-runtime or tensorflow."""
        # Check if file exists and has actual model bytes
        if not os.path.exists(self.model_path) or os.path.getsize(self.model_path) < 1024:
            logger.warning(
                f"Model file at '{self.model_path}' is missing or is an empty placeholder (< 1KB). "
                "Switching to SIMULATION MODE for testing triggers and display."
            )
            self.is_mock_mode = True
            return

        # Attempt to import tflite_runtime or ai_edge_litert first
        interpreter_cls = None
        try:
            from ai_edge_litert.interpreter import Interpreter
            interpreter_cls = Interpreter
            logger.info("Using ai_edge_litert.interpreter.")
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
                interpreter_cls = Interpreter
                logger.info("Using tflite_runtime.interpreter.")
            except ImportError:
                try:
                    import tensorflow as tf
                    interpreter_cls = tf.lite.Interpreter
                    logger.info("Using tensorflow.lite.Interpreter.")
                except ImportError:
                    logger.warning(
                        "Neither tflite-runtime, ai-edge-litert, nor tensorflow is installed. "
                        "Inference will run in SIMULATION MODE."
                    )
                    self.is_mock_mode = True
                    return

        try:
            self.interpreter = interpreter_cls(model_path=self.model_path, num_threads=self.num_threads)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            logger.info(
                f"Successfully loaded TFLite model. "
                f"Input shape: {self.input_details[0]['shape']}, "
                f"Input type: {self.input_details[0]['dtype']}"
            )
        except Exception as e:
            logger.error(f"Failed to load TFLite model from '{self.model_path}': {e}")
            logger.warning("Falling back to SIMULATION MODE.")
            self.is_mock_mode = True

    @staticmethod
    def crop_roi(
        frame: np.ndarray,
        roi_scale: float = 1.0,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops a square Region of Interest (ROI) from the center of the frame.
        Prevents 16:9 to 1:1 squishing distortion and focuses on the active detection zone.

        Returns:
            (cropped_frame, (x1, y1, x2, y2))
        """
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        crop_size = int(min_dim * max(0.1, min(1.0, roi_scale)))

        x1 = max(0, (w - crop_size) // 2)
        y1 = max(0, (h - crop_size) // 2)
        x2 = x1 + crop_size
        y2 = y1 + crop_size

        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts square ROI, resizes, and normalizes the input frame for the model.
        Supports both UINT8 and FLOAT32 input tensors.
        """
        import cv2

        # Apply ROI crop to preserve aspect ratio
        if self.enable_roi_crop:
            cropped_frame, _ = self.crop_roi(frame, self.roi_crop_scale)
        else:
            cropped_frame = frame

        if self.input_details:
            _, height, width, channels = self.input_details[0]["shape"]
            input_dtype = self.input_details[0]["dtype"]
        else:
            height, width, channels = 224, 224, 3
            input_dtype = np.float32

        # Convert BGR to RGB and resize to model input shape
        rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(rgb_frame, (width, height))

        # Add batch dimension
        input_data = np.expand_dims(resized_frame, axis=0)

        if input_dtype == np.float32:
            input_data = input_data.astype(np.float32)
        elif input_dtype == np.uint8:
            input_data = input_data.astype(np.uint8)

        return input_data

    def predict(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Performs inference on a camera frame.
        Returns: (label_name, confidence_score)
        """
        if self.is_mock_mode or self.interpreter is None:
            # Demo inference when model weights are pending
            import random
            mock_class = random.choice(self.labels) if self.labels else "Animal"
            mock_conf = round(random.uniform(0.75, 0.98), 2)
            logger.info(f"[SIMULATION] Predicted: {mock_class} ({mock_conf * 100:.1f}%)")
            return mock_class, mock_conf

        try:
            input_data = self.preprocess_image(frame)
            self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
            self.interpreter.invoke()

            output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
            probabilities = np.squeeze(output_data)

            # Dequantize if model output is quantized
            if self.output_details[0]["dtype"] == np.uint8:
                scale, zero_point = self.output_details[0]["quantization"]
                if scale > 0:
                    probabilities = scale * (probabilities.astype(np.float32) - zero_point)

            top_idx = int(np.argmax(probabilities))
            confidence = float(probabilities[top_idx])

            label = (
                self.labels[top_idx]
                if top_idx < len(self.labels)
                else f"Class_{top_idx}"
            )

            logger.info(f"Prediction: {label} (confidence: {confidence:.2%})")
            return label, confidence

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return "Unknown", 0.0
