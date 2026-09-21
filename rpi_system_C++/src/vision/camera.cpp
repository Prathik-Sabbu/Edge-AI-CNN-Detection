#include "rpi_system_C++/include/vision/camera.hpp"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <thread>
#include <tuple>


CameraManager::CameraManager() { start(); }
CameraManager::~CameraManager() { release(); }

bool CameraManager::start() {
  spdlog::info("Initializing camera (use_picamera={}, index={})...",
               use_picamera, camera_index);
#ifdef _WIN64
  int backend = cv::CAP_DSHOW;
#else
  int backend = cv::CAP_ANY;
#endif

  try {
    cap = cv::VideoCapture(camera_index, backend);
    if (!cap.isOpened()) {
      spdlog::warn("Camera at index {} could not be opened. Make sure a USB "
                   "webcam or CSI camera is connected.",
                   camera_index);
      cap.release();
      return false;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);

    // warmup frames
    spdlog::info("Running warmup ({} frames)...", warmup_frames);
    for (int i = 0; i < warmup_frames; i++) {
      cv::Mat frame;
      bool ret = cap.read(frame);
      if (!ret) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }

    is_initialized = true;
    spdlog::info("OpenCV camera initialized successfully.");
    return true;
  } catch (const std::exception &e) {
    spdlog::error("Error opening camera with OpenCV: {}", e.what());
    cap.release();
    return false;
  }
}

std::optional<cv::Mat> CameraManager::capture_frame() {
  if (!is_initialized) {
    if (!start()) {
      if (allow_mock) {
        spdlog::warn(
            "No physical camera available. Generating simulated test frame.");
        return cv::Mat(height, width, CV_8UC3, cv::Scalar(60, 60, 60));
      }
      return std::nullopt;
    }
  }

  if (cap.isOpened()) {
    cv::Mat frame;
    bool ret = cap.read(frame);
    if (!ret || frame.empty()) {
      spdlog::error("Failed to read frame from camera.");
      if (allow_mock) {
        return cv::Mat(height, width, CV_8UC3, cv::Scalar(60, 60, 60));
      }
      return std::nullopt;
    }
    return frame;
  }

  if (allow_mock) {
    return cv::Mat(height, width, CV_8UC3, cv::Scalar(60, 60, 60));
  }
  return std::nullopt;
}

void CameraManager::release() {
  spdlog::info("Releasing camera hardware...");
  if (cap.isOpened()) {
    try {
      cap.release();
    } catch (const std::exception &e) {
      spdlog::warn("Error releasing OpenCV VideoCapture: {}", e.what());
    }
  }
  is_initialized = false;
}