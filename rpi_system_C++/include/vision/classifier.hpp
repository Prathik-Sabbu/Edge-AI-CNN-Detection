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

#ifdef USE_XNNPACK
#include <tensorflow/lite/tflite_with_xnnpack_optional.h>
#endif


class AnimalClassifier {
private:
  std::string model_path;
  std::string labels_path;
  float confidence_threshold;
  bool enable_roi_crop;
  float roi_crop_scale;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::vector<std::string> labels;
  bool is_mock_mode = false;
  std::unique_ptr<tflite::FlatBufferModel> model;

#ifdef USE_XNNPACK
  decltype(tflite::MaybeCreateXNNPACKDelegate()) xnnpack_delegate = nullptr;
#endif

  void load_labels();
  void load_model();
  cv::Mat preprocess_image(const cv::Mat &frame);

public:
  AnimalClassifier();
  ~AnimalClassifier();
  std::tuple<std::string, double> predict(const cv::Mat &frame);
  static std::pair<cv::Mat, cv::Rect> crop_roi(const cv::Mat &frame,
                                               float roi_scale = 1.0f);
};