"""
Central configuration for Raspberry Pi Edge AI CNN Detection System.
Modify settings here to suit your hardware setup (USB camera vs Pi camera,
Arduino Serial vs Direct Pi GPIO) and model training parameters.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "Data" / "Testing" / "raw-img"
TRAIN_DIR = DATA_DIR
VAL_DIR = DATA_DIR
INF_DIR = PROJECT_ROOT / "Data" / "inf"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = str(MODELS_DIR / "animal_classifier.tflite")
LABELS_PATH = str(MODELS_DIR / "labels.txt")
OUTPUT_TFLITE = MODELS_DIR / "animal_classifier.tflite"
OUTPUT_LABELS = MODELS_DIR / "labels.txt"

# Model & Image Dimensions
INPUT_WIDTH = 224
INPUT_HEIGHT = 224
IMAGE_SIZE = (INPUT_WIDTH, INPUT_HEIGHT)
CONFIDENCE_THRESHOLD = 0.50 
TOP_K = 1

# Training Hyperparameters
BATCH_SIZE = 64
TRAIN_BATCH_SIZE = BATCH_SIZE
EPOCHS = 30
TRAIN_EPOCHS = EPOCHS
LEARNING_RATE = 0.001
TRAIN_LEARNING_RATE = LEARNING_RATE

# Hardware Trigger Mode:
# Set to True if Arduino is connected via USB cable
# Set to False if HC-SR04 ultrasonic sensor is wired directly to Raspberry Pi GPIO
USE_ARDUINO_SERIAL = True

# Serial Settings (Used if USE_ARDUINO_SERIAL is True)
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyACM0")
SERIAL_PORT_FALLBACK = "/dev/ttyUSB0"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1.0

# Direct Raspberry Pi GPIO Settings (Used if USE_ARDUINO_SERIAL is False)
GPIO_TRIGGER_PIN = 23
GPIO_ECHO_PIN = 24
TRIGGER_DISTANCE_THRESHOLD_CM = 40.0
MIN_DISTANCE_THRESHOLD_CM = 2.0

# Cooldown between triggers
COOLDOWN_SECONDS = 3.0

# Camera Settings
CAMERA_INDEX = 0
USE_PICAMERA = False
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WARMUP_FRAMES = 5

# Region of Interest (ROI) & Aspect Ratio Optimization
ENABLE_ROI_CROP = True
ROI_CROP_SCALE = 1.0

