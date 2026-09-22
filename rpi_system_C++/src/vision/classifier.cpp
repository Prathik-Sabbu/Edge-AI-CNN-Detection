#include "../../include/vision/classifier.hpp"
#include "../../include/config.hpp"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <optional>
#include <spdlog/fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <string>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tuple>
#include <vector>

AnimalClassifier::AnimalClassifier()
    : model_path(Config::MODEL_PATH), labels_path(Config::LABELS_PATH),
      confidence_threshold(Config::CONFIDENCE_THRESHOLD),
      enable_roi_crop(Config::ENABLE_ROI_CROP), roi_crop_scale(Config::ROI_CROP_SCALE) {
  load_labels();
  load_model();
}

AnimalClassifier::~AnimalClassifier() {}

void AnimalClassifier::load_labels() {
  if (std::filesystem::exists(labels_path)) {
    try {
      std::ifstream file(labels_path);
      std::string line;
      while (std::getline(file, line)) {
        if (!line.empty()) {
          labels.push_back(line);
        }
      }
      spdlog::info("Loaded {} labels: {}", labels.size(), labels);
    } catch (const std::exception &e) {
      spdlog::error("Error reading labels file: {}", e.what());
      labels = {"Cat", "Dog", "Bird", "Wild Animal"};
    }
  } else {
    spdlog::warn("Labels file not found at {}. Using default classes.",
                 labels_path);
    labels = { "Cat", "Dog", "Bird", "Wild Animal" };
  }
}

void AnimalClassifier::load_model() {
  if (!std::filesystem::exists(model_path) ||
      !std::filesystem::is_regular_file(model_path)) {
    spdlog::warn(
        "Model file at '{}' is missing or is an empty placeholder (< 1KB). "
        "Switching to SIMULATION MODE for testing triggers and display.",
        model_path);
    is_mock_mode = true;
    return;
  }

  model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model) {
    spdlog::warn("Failed to load TFLite model from '{}'. Falling back to "
                 "SIMULATION MODE.",
                 model_path);
    is_mock_mode = true;
    return;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  builder(&interpreter);

  if (interpreter) {
    interpreter->SetNumThreads(4); // Utilize all 4 cores on Raspberry Pi
  }

  if (!interpreter) {
    spdlog::error("Failed to construct TFLite interpreter. Falling back to "
                  "SIMULATION MODE.");
    is_mock_mode = true;
    return;
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    spdlog::error(
        "Failed to allocate tensors. Falling back to SIMULATION MODE.");
    is_mock_mode = true;
    return;
  }
  spdlog::info("Successfully loaded TFLite model.");
}

std::pair<cv::Mat, cv::Rect> AnimalClassifier::crop_roi(const cv::Mat &frame,
                                                        float roi_scale) {
  int h = frame.rows;
  int w = frame.cols;

  int min_dim = std::min(h, w);
  int crop_size = int(min_dim * std::max(0.1f, std::min(1.0f, roi_scale)));

  int x1 = std::max(0, (w - crop_size) / 2);
  int y1 = std::max(0, (h - crop_size) / 2);

  cv::Rect roi(x1, y1, crop_size, crop_size);
  cv::Mat cropped = frame(roi);

  return {cropped, roi};
}

cv::Mat AnimalClassifier::preprocess_image(const cv::Mat &frame) {
  cv::Mat cropped_frame;
  if (enable_roi_crop) {
    cropped_frame = crop_roi(frame, roi_crop_scale).first;
  } else {
    cropped_frame = frame;
  }

  TfLiteTensor *input = interpreter->input_tensor(0);
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  TfLiteType input_dtype = input->type;

  cv::Mat rgb_frame;
  cv::cvtColor(cropped_frame, rgb_frame, cv::COLOR_BGR2RGB);

  cv::Mat resized_frame;
  cv::resize(rgb_frame, resized_frame, cv::Size(width, height));

  cv::Mat input_data;
  if (input_dtype == kTfLiteFloat32) {
    resized_frame.convertTo(input_data, CV_32FC3);
  } else {
    resized_frame.convertTo(input_data, CV_8UC3);
  }

  return input_data;
}

std::tuple<std::string, double>
AnimalClassifier::predict(const cv::Mat &frame) {
  if (is_mock_mode || !interpreter) {
    std::string mock_class =
        labels.empty() ? "Animal" : labels[rand() % labels.size()];
    double mock_conf = 0.75 + (rand() / (RAND_MAX / (0.98 - 0.75)));
    spdlog::info("[SIMULATION] Predicted: {} ({:.1f}%)", mock_class,
                 mock_conf * 100);
    return {mock_class, mock_conf};
  }

  try {
    cv::Mat input_data = preprocess_image(frame);
    TfLiteTensor *input_tensor = interpreter->input_tensor(0);

    size_t bytes = input_data.total() * input_data.elemSize();
    if (!input_data.isContinuous() || bytes != input_tensor->bytes) {
      spdlog::error("Input size mismatch: {} vs {}", bytes,
                    input_tensor->bytes);
      return {"Unknown", 0.0};
    }

    memcpy(input_tensor->data.raw, input_data.data, bytes);

    if (interpreter->Invoke() != kTfLiteOk) {
      spdlog::error("Inference error: Invoke failed.");
      return {"Unknown", 0.0};
    }

    TfLiteTensor *output_tensor = interpreter->output_tensor(0);
    if (output_tensor->dims->size < 2) {
      spdlog::error("Unexpected output shape (rank {})",
                    output_tensor->dims->size);
      return {"Unknown", 0.0};
    }
    int num_classes = output_tensor->dims->data[1];

    double max_conf = -1.0;
    int top_idx = 0;

    if (output_tensor->type == kTfLiteFloat32) {
      const float *output_data = output_tensor->data.f;
      for (int i = 0; i < num_classes; i++) {
        if (output_data[i] > max_conf) {
          max_conf = output_data[i];
          top_idx = i;
        }
      }
    } else if (output_tensor->type == kTfLiteUInt8) {
      const uint8_t *output_data = output_tensor->data.uint8;
      float scale = output_tensor->params.scale;
      int zero_point = output_tensor->params.zero_point;

      for (int i = 0; i < num_classes; i++) {
        double conf = scale * (output_data[i] - zero_point);
        if (conf > max_conf) {
          max_conf = conf;
          top_idx = i;
        }
      }
    } else {
      spdlog::error("Unsupported output tensor type: {}",
                    TfLiteTypeGetName(output_tensor->type));
      return {"Unknown", 0.0};
    }

    std::string label = (static_cast<size_t>(top_idx) < labels.size())
                            ? labels[top_idx]
                            : "Class_" + std::to_string(top_idx);
    spdlog::info("Prediction: {} (confidence: {:.2f}%)", label, max_conf * 100);
    return {label, max_conf};

  } catch (const std::exception &e) {
    spdlog::error("Inference error: {}", e.what());
    return {"Unknown", 0.0};
  }
}