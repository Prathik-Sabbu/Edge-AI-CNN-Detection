#pragma once

#include <chrono>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <tuple>

class CameraManager {
private:
  int camera_index;
  bool use_picamera;
  int width;
  int height;
  int warmup_frames;
  bool allow_mock;
  bool is_initialized;
  cv::VideoCapture cap;

  bool start();
  void release();

public:
  CameraManager();
  ~CameraManager();
  std::optional<cv::Mat> capture_frame();
};