"""
Main execution daemon for Raspberry Pi Edge AI CNN Detection System.
Coordinates sensor triggering, camera capture, CNN classification, and display feedback.
"""
import sys
import time
import signal
import logging
from pathlib import Path
import cv2

import config
from vision.camera import CameraManager
from vision.classifier import AnimalClassifier
from io_controllers.serial_controller import ArduinoSerialController
from io_controllers.gpio_controller import RaspberryPiGPIOController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("EdgeAI.Main")

class EdgeAISystem:
    """Coordinates Edge AI hardware loop on Raspberry Pi."""

    def __init__(self):
        self.running = False
        self.captures_dir = config.BASE_DIR / "captures"
        self.captures_dir.mkdir(exist_ok=True)

        logger.info("Initializing Raspberry Pi Edge AI System...")
        logger.info(f"Trigger Mode: {'Arduino USB Serial' if config.USE_ARDUINO_SERIAL else 'Direct Pi GPIO'}")

        # Initialize Vision Components
        self.camera = CameraManager(
            camera_index=config.CAMERA_INDEX,
            use_picamera=config.USE_PICAMERA,
            width=config.FRAME_WIDTH,
            height=config.FRAME_HEIGHT,
            warmup_frames=config.WARMUP_FRAMES,
        )

        self.classifier = AnimalClassifier(
            model_path=config.MODEL_PATH,
            labels_path=config.LABELS_PATH,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
            enable_roi_crop=getattr(config, "ENABLE_ROI_CROP", True),
            roi_crop_scale=getattr(config, "ROI_CROP_SCALE", 1.0),
        )

        # Initialize Trigger Controller
        if config.USE_ARDUINO_SERIAL:
            self.trigger_controller = ArduinoSerialController(
                port=config.SERIAL_PORT,
                fallback_port=config.SERIAL_PORT_FALLBACK,
                baud_rate=config.BAUD_RATE,
                timeout=config.SERIAL_TIMEOUT,
            )
        else:
            self.trigger_controller = RaspberryPiGPIOController(
                trigger_pin=config.GPIO_TRIGGER_PIN,
                echo_pin=config.GPIO_ECHO_PIN,
                distance_threshold_cm=config.TRIGGER_DISTANCE_THRESHOLD_CM,
                min_distance_cm=config.MIN_DISTANCE_THRESHOLD_CM,
                cooldown_seconds=config.COOLDOWN_SECONDS,
            )

    def start(self):
        """Starts hardware interfaces and enters main polling loop."""
        self.running = True

        # Signal handlers for clean termination on Raspberry Pi (e.g. systemd stop)
        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        # Warm up camera
        if not self.camera.start():
            logger.warning("Camera initialization failed. Will retry on demand.")

        # Connect serial if in Arduino mode
        if config.USE_ARDUINO_SERIAL:
            self.trigger_controller.connect()

        logger.info("Edge AI CNN Detection is RUNNING.")
        logger.info("Waiting for sensor triggers...")

        last_reconnect_attempt = time.time()

        try:
            while self.running:
                # Maintain Arduino serial connection if in Arduino mode
                if config.USE_ARDUINO_SERIAL and not self.trigger_controller.is_connected():
                    if time.time() - last_reconnect_attempt > 5.0:
                        logger.info("Attempting to reconnect to Arduino...")
                        self.trigger_controller.connect()
                        last_reconnect_attempt = time.time()
                    time.sleep(0.1)
                    continue

                # Poll for trigger
                if self.trigger_controller.check_for_trigger():
                    self._on_triggered()

                time.sleep(0.05)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop()

    def _on_triggered(self):
        """Callback executed when an object triggers the ultrasonic sensor."""
        logger.info(">>> TRIGGER DETECTED! Capturing camera frame... <<<")

        # Capture snapshot
        frame = self.camera.capture_frame()
        if frame is None:
            logger.error("Could not capture frame from camera.")
            if config.USE_ARDUINO_SERIAL and self.trigger_controller.is_connected():
                self.trigger_controller.send_result_to_lcd("Cam Error", 0.0)
            return

        # Run CNN classification
        label, confidence = self.classifier.predict(frame)
        logger.info(f">>> Result: {label} (Confidence: {confidence:.1%}) <<<")

        # Send feedback to Arduino LCD if connected
        if config.USE_ARDUINO_SERIAL and self.trigger_controller.is_connected():
            self.trigger_controller.send_result_to_lcd(label, confidence)

        # Save capture snapshot for inspection/logging
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_"))
        save_path = self.captures_dir / f"{timestamp}_{safe_label}_{int(confidence*100)}pct.jpg"
        try:
            annotated = frame.copy()
            if getattr(config, "ENABLE_ROI_CROP", True):
                _, (x1, y1, x2, y2) = AnimalClassifier.crop_roi(
                    frame, getattr(config, "ROI_CROP_SCALE", 1.0)
                )
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    annotated,
                    "ROI Cropped Area",
                    (x1 + 10, y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            hud_color = (0, 255, 0) if confidence >= config.CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(
                annotated,
                f"{label}: {confidence:.1%}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                hud_color,
                2,
            )

            cv2.imwrite(str(save_path), annotated)
            logger.info(f"Saved detection snapshot to: {save_path.name}")
        except Exception as e:
            logger.warning(f"Could not save snapshot image: {e}")

    def _handle_exit(self, signum, frame):
        logger.info(f"Exit signal {signum} received. Stopping...")
        self.running = False

    def stop(self):
        """Releases all hardware resources cleanly."""
        logger.info("Shutting down Edge AI System...")
        self.running = False

        if hasattr(self, "camera"):
            self.camera.release()

        if hasattr(self, "trigger_controller"):
            if config.USE_ARDUINO_SERIAL:
                self.trigger_controller.disconnect()
            else:
                self.trigger_controller.cleanup()

        logger.info("All resources released. System shutdown complete.")

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Raspberry Pi Edge AI CNN Detection Daemon"
    )
    parser.add_argument(
        "--test-trigger",
        action="store_true",
        help="Simulate a single sensor trigger event and exit (useful for testing)",
    )
    args = parser.parse_args()

    system = EdgeAISystem()
    if args.test_trigger:
        logger.info("Executing manual test trigger...")
        system._on_triggered()
        system.stop()
    else:
        system.start()

if __name__ == "__main__":
    main()
