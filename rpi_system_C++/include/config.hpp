#pragma once

#include <filesystem>
#include <string>

namespace Config {
// Paths
inline const std::string MODEL_PATH = "models/animal_classifier.tflite";
inline const std::string LABELS_PATH = "models/labels.txt";
inline const std::string BASE_PATH = std::filesystem::current_path().string();

// Model & Image Dimensions
inline const int INPUT_WIDTH = 224;
inline const int INPUT_HEIGHT = 224;
inline const float CONFIDENCE_THRESHOLD = 0.50f;
inline const int TOP_K = 1;

// Hardware Trigger Mode
inline const bool USE_ARDUINO_SERIAL = true;

// Serial Settings
inline const std::string SERIAL_PORT = "/dev/ttyACM0";
inline const std::string SERIAL_PORT_FALLBACK = "/dev/ttyUSB0";
inline const int BAUD_RATE = 9600;
inline const float SERIAL_TIMEOUT = 1.0f;

// Direct Raspberry Pi GPIO Settings
inline const int GPIO_TRIGGER_PIN = 23;
inline const int GPIO_ECHO_PIN = 24;
inline const float TRIGGER_DISTANCE_THRESHOLD_CM = 40.0f;
inline const float MIN_DISTANCE_THRESHOLD_CM = 2.0f;

// Cooldown
inline const float COOLDOWN_SECONDS = 3.0f;

// Camera Settings
inline const int CAMERA_INDEX = 0;
inline const bool USE_PICAMERA = false;
inline const int FRAME_WIDTH = 1280;
inline const int FRAME_HEIGHT = 720;
inline const int WARMUP_FRAMES = 5;

// ROI
inline const bool ENABLE_ROI_CROP = true;
inline const float ROI_CROP_SCALE = 1.0f;
} // namespace Config
