"""
Camera module for Raspberry Pi Edge AI system.
Supports both standard USB webcams (OpenCV) and official Raspberry Pi Camera modules (Picamera2).
"""
import time
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("EdgeAI.Camera")

class CameraManager:
    """Manages camera lifecycle, warmups, and frame captures."""

    def __init__(
        self,
        camera_index: int = 0,
        use_picamera: bool = False,
        width: int = 640,
        height: int = 480,
        warmup_frames: int = 5,
        allow_mock: bool = True,
    ):
        self.camera_index = camera_index
        self.use_picamera = use_picamera
        self.width = width
        self.height = height
        self.warmup_frames = warmup_frames
        self.allow_mock = allow_mock

        self._cap = None
        self._picam2 = None
        self._is_initialized = False

    def start(self) -> bool:
        """Initializes the camera and performs exposure warmup."""
        logger.info(
            f"Initializing camera (use_picamera={self.use_picamera}, index={self.camera_index})..."
        )

        if self.use_picamera:
            try:
                from picamera2 import Picamera2
                self._picam2 = Picamera2()
                config = self._picam2.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self._picam2.configure(config)
                self._picam2.start()
                time.sleep(1.0)
                self._is_initialized = True
                logger.info("Picamera2 started successfully.")
                return True
            except Exception as e:
                logger.error(f"Failed to start Picamera2: {e}. Falling back to OpenCV.")
                self.use_picamera = False

        # OpenCV VideoCapture
        try:
            import cv2
            import sys

            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self.camera_index, backend)

            if not self._cap.isOpened():
                logger.warning(
                    f"Camera at index {self.camera_index} could not be opened. "
                    "Make sure a USB webcam or CSI camera is connected."
                )
                self._cap = None
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

            # Warmup frames
            logger.info(f"Running warmup ({self.warmup_frames} frames)...")
            for _ in range(self.warmup_frames):
                ret, _ = self._cap.read()
                if not ret:
                    time.sleep(0.05)

            self._is_initialized = True
            logger.info("OpenCV camera initialized successfully.")
            return True
        except Exception as e:
            logger.error(f"Error opening camera with OpenCV: {e}")
            self._cap = None
            return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Captures a single frame as a BGR/RGB numpy array.
        Returns None if capture fails.
        """
        if not self._is_initialized:
            if not self.start():
                if self.allow_mock:
                    logger.warning("No physical camera available. Generating simulated test frame.")
                    mock_frame = np.full((self.height, self.width, 3), 60, dtype=np.uint8)
                    return mock_frame
                return None

        if self.use_picamera and self._picam2 is not None:
            try:
                frame = self._picam2.capture_array()
                return frame
            except Exception as e:
                logger.error(f"Failed to capture frame from Picamera2: {e}")
                if self.allow_mock:
                    return np.full((self.height, self.width, 3), 60, dtype=np.uint8)
                return None

        if self._cap is not None:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.error("Failed to read frame from camera.")
                if self.allow_mock:
                    return np.full((self.height, self.width, 3), 60, dtype=np.uint8)
                return None
            return frame

        if self.allow_mock:
            return np.full((self.height, self.width, 3), 60, dtype=np.uint8)

        return None

    def release(self):
        """Releases camera hardware resources gracefully."""
        logger.info("Releasing camera hardware...")
        if self._picam2 is not None:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception as e:
                logger.warning(f"Error closing Picamera2: {e}")
            self._picam2 = None

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning(f"Error releasing OpenCV VideoCapture: {e}")
            self._cap = None

        self._is_initialized = False
