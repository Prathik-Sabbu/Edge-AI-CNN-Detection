#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#ifdef __linux__
#include <sys/resource.h>
#endif

#include "../include/config.hpp"
#include "../include/vision/classifier.hpp"

// Utility to get memory usage
double get_peak_memory_mb() {
#ifdef __linux__
  struct rusage usage;
  getrusage(RUSAGE_SELF, &usage);
  return static_cast<double>(usage.ru_maxrss) / 1024.0;
#else
  return 0.0; // Not available on Windows natively without psutil/windows api
#endif
}

std::tuple<long long, double, double>
get_model_size_info(const std::filesystem::path &model_path) {
  long long size_bytes = std::filesystem::file_size(model_path);
  double size_kb = size_bytes / 1024.0;
  double size_mb = size_kb / 1024.0;
  return {size_bytes, size_kb, size_mb};
}

void benchmark_inference_latency(AnimalClassifier &clf,
                                 int num_iterations = 300,
                                 double warmup_seconds = 3.0) {
  std::cout << "\n[3] INFERENCE SPEED & LATENCY (" << num_iterations
            << " Iterations):\n";

  cv::Mat dummy_frame = cv::Mat::zeros(480, 640, CV_8UC3);

  std::cout << "  -> Warming up CPU for " << warmup_seconds << " seconds...\n";
  auto warmup_start = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       warmup_start)
             .count() < warmup_seconds) {
    clf.predict(dummy_frame);
  }

  std::vector<double> e2e_latencies_ms;
  e2e_latencies_ms.reserve(num_iterations);

  for (int i = 0; i < num_iterations; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    clf.predict(dummy_frame);
    auto t1 = std::chrono::steady_clock::now();
    e2e_latencies_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());
  }

  std::sort(e2e_latencies_ms.begin(), e2e_latencies_ms.end());
  double mean_ms =
      std::accumulate(e2e_latencies_ms.begin(), e2e_latencies_ms.end(), 0.0) /
      num_iterations;
  double p99_ms = e2e_latencies_ms[static_cast<int>(num_iterations * 0.99)];
  double fps = 1000.0 / mean_ms;

  std::cout << "  • End-to-End (Mean)     : " << mean_ms << " ms\n";
  std::cout << "  • End-to-End (p99)      : " << p99_ms << " ms\n";
  std::cout << "  • True Throughput       : " << fps << " FPS\n";
}

int main(int argc, char *argv[]) {
  spdlog::set_level(spdlog::level::info);
  spdlog::set_pattern("%Y-%m-%d %H:%M:%S [%^%l%$] %n: %v");
  auto logger = spdlog::stdout_color_mt("Benchmark");
  spdlog::set_default_logger(logger);

  std::cout << "        EDGE AI CNN DETECTION SYSTEM: BENCHMARK & METRICS "
               "PROFILER              \n";

  std::filesystem::path model_path(Config::MODEL_PATH);
  std::filesystem::path labels_path(Config::LABELS_PATH);

  if (!std::filesystem::exists(model_path)) {
    std::cerr << "Error: Model file not found at " << model_path << "\n";
    return 1;
  }

  auto [size_bytes, size_kb, size_mb] = get_model_size_info(model_path);

  std::vector<std::string> labels;
  std::ifstream f(labels_path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty())
      labels.push_back(line);
  }

  std::cout << "\n[1] MODEL ARCHITECTURE PROFILE:\n";

  AnimalClassifier clf;

  std::cout << "  • Base Architecture    : MobileNetV3-Small (Transfer "
               "Learning from ImageNet)\n";
  std::cout << "  • Model Format          : TensorFlow Lite (.tflite)\n";
  std::cout << "  • Number of Classes     : " << labels.size() << " classes\n";
  std::cout << "  • File Size on Disk     : " << size_mb << " MB\n";

  benchmark_inference_latency(clf, 300, 3.0);

  std::cout << "\n[4] MEMORY / RAM FOOTPRINT:\n";
  std::cout << "  • Peak Process RSS      : " << get_peak_memory_mb()
            << " MB\n";

  return 0;
}
