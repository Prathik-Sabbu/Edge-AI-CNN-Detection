#pragma once

#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/spdlog.h>
#include <string>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tuple>
#include <vector>


class AnimalClassifier {
private:
  std::string model_path;
  std::string labels_path;
  float confidence_threshold = 0.50;
  bool enable_roi_crop = true;
  float roi_crop_scale = 1.0;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::vector<std::string> labels;
  bool is_mock_mode = false;
  std::unique_ptr<tflite::FlatBufferModel> model;

  void load_labels();
  void load_model();
  cv::Mat preprocess_image(const cv::Mat &frame);

public:
  AnimalClassifier(const std::string &model_path,
                   const std::string &labels_path,
                   float confidence_threshold = 0.5f,
                   bool enable_roi_crop = true, float roi_crop_scale = 1.0f);
  ~AnimalClassifier();
  std::tuple<std::string, double> predict(const cv::Mat &frame);
  static std::pair<cv::Mat, cv::Rect> crop_roi(const cv::Mat &frame,
                                               float roi_scale = 1.0f);
};