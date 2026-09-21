#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <tuple>

class CameraManager {
private:
  int camera_index = 0;
  bool use_picamera = false;
  int width = 640;
  int height = 480;
  int warmup_frames = 5;
  bool allow_mock = true;
  bool is_initialized = false;
  cv::VideoCapture cap;

  bool start();
  void release();

public:
  CameraManager();
  ~CameraManager();
  std::optional<cv::Mat> capture_frame();
};