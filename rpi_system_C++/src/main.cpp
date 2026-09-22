#include <chrono>
#include <csignal>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>

#include "../include/config.hpp"
#include "../include/io_controllers/gpio_controller.hpp"
#include "../include/io_controllers/serial_controller.hpp"
#include "../include/vision/camera.hpp"
#include "../include/vision/classifier.hpp"
#include <spdlog/sinks/stdout_color_sinks.h>

bool running = false;
std::filesystem::path captures_dir =
    std::filesystem::path(Config::BASE_PATH) / "captures";

CameraManager camera;
AnimalClassifier classifier;
std::unique_ptr<ArduinoSerialController> serial_controller;
std::unique_ptr<GPIOController> gpio_controller;

void setup_logging() {
  spdlog::set_level(spdlog::level::info);
  // Format: "YYYY-MM-DD HH:MM:SS [LEVEL] NAME: MESSAGE"
  spdlog::set_pattern("%Y-%m-%d %H:%M:%S [%^%l%$] %n: %v");

  // Create a color logger with the name "EdgeAI.Main"
  auto logger = spdlog::stdout_color_mt("EdgeAI.Main");

  // Set it as the default logger so spdlog::info() uses it globally
  spdlog::set_default_logger(logger);
}

void handle_exit(int signum) {
  spdlog::info("Keyboard interrupt received.");
  running = false;
}

void on_triggered() {
  spdlog::info(">>> TRIGGER DETECTED! Capturing camera frame... <<<");

  auto frame_opt = camera.capture_frame();

  if (!frame_opt) {
    spdlog::error("Could not capture frame from camera.");
    if (Config::USE_ARDUINO_SERIAL && serial_controller &&
        serial_controller->is_connected()) {
      serial_controller->send_result_to_lcd("Cam Error", 0.0f);
    }
    return;
  }

  cv::Mat frame = frame_opt.value();

  auto [label, confidence] = classifier.predict(frame);
  spdlog::info(">>> Result: {} (Confidence: {:.1f}%) <<<", label,
               confidence * 100.0f);

  if (Config::USE_ARDUINO_SERIAL && serial_controller &&
      serial_controller->is_connected()) {
    serial_controller->send_result_to_lcd(label, confidence);
  }

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream filename_stream;
  filename_stream << std::put_time(&tm, "%Y%m%d_%H%M%S") << "_" << label << "_"
                  << static_cast<int>(confidence * 100) << "pct.jpg";
  std::filesystem::path save_path = captures_dir / filename_stream.str();

  cv::Mat annotated = frame.clone();

  cv::Scalar box_color = (confidence >= Config::CONFIDENCE_THRESHOLD)
                             ? cv::Scalar(0, 230, 0)
                             : cv::Scalar(0, 140, 255);

  int x1 = 0, y1 = 0, x2 = frame.cols, y2 = frame.rows;

  cv::rectangle(annotated, cv::Point(x1, y1), cv::Point(x2, y2), box_color, 3);

  std::string label_text =
      label + " " + std::to_string(static_cast<int>(confidence * 100)) + "%";
  cv::putText(annotated, label_text, cv::Point(x1, y1 - 10),
              cv::FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2);

  cv::imwrite(save_path.string(), annotated);
}

void start() {
  running = true;

  std::signal(SIGINT, handle_exit);
  std::signal(SIGTERM, handle_exit);

  if (Config::USE_ARDUINO_SERIAL && serial_controller) {
    serial_controller->connect();
  }

  spdlog::info("Edge AI CNN Detection is RUNNING.");
  spdlog::info("Waiting for sensor triggers...");

  std::chrono::time_point<std::chrono::steady_clock> last_reconnect_attempt =
      std::chrono::steady_clock::now();

  while (running) {
    if (Config::USE_ARDUINO_SERIAL && serial_controller &&
        !serial_controller->is_connected()) {
      auto now = std::chrono::steady_clock::now();
      auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                                 now - last_reconnect_attempt)
                                 .count();

      if (elapsed_seconds > 5) {
        spdlog::info("Attempting to reconnect to Arduino...");
        serial_controller->connect();
        last_reconnect_attempt = now;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      continue;
    }

    bool triggered = false;
    if (Config::USE_ARDUINO_SERIAL && serial_controller) {
      triggered = serial_controller->check_for_trigger();
    } else if (gpio_controller) {
      triggered = gpio_controller->check_for_trigger();
    }

    if (triggered) {
      on_triggered();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  spdlog::info("Shutting down Edge AI System...");
}

void stop() {
  spdlog::info("Shutting down Edge AI System...");
  running = false;

  if (gpio_controller) {
    gpio_controller.reset();
  }
  if (serial_controller) {
    serial_controller.reset();
  }

  spdlog::info("All resources released. System shutdown complete.");
}

int main(int argc, char *argv[]) {
  bool test_trigger = false;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--test-trigger") {
      test_trigger = true;
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Raspberry Pi Edge AI CNN Detection Daemon\n\n";
      std::cout << "Options:\n";
      std::cout << "  --test-trigger    Simulate a single sensor trigger event "
                   "and exit (useful for testing)\n";
      std::cout << "  --help            Show this help message\n";
      return 0;
    }
  }

  setup_logging();

  std::filesystem::create_directories(captures_dir);

  spdlog::info("Initializing Raspberry Pi Edge AI System...");
  spdlog::info("Trigger Mode: {}", Config::USE_ARDUINO_SERIAL
                                       ? "Arduino USB Serial"
                                       : "Direct Pi GPIO");

  if (Config::USE_ARDUINO_SERIAL) {
    serial_controller = std::make_unique<ArduinoSerialController>();
  } else {
    gpio_controller = std::make_unique<GPIOController>();
  }

  if (test_trigger) {
    spdlog::info("Executing manual test trigger...");
    on_triggered();
    stop();
  } else {
    start();
  }

  return 0;
}
