#pragma once

#include <chrono>
#include <iostream>
#include <optional>
#include <pigpio.h>
#include <spdlog/spdlog.h>
#include <wiringPi.h>

class GPIOController {
private:
  int trigger_pin;
  int echo_pin;
  float distance_threshold_cm;
  float min_distance_cm;
  float cooldown_seconds;

  std::chrono::system_clock::time_point last_trigger_time;
  bool is_ready = true;
  bool gpio_available = false;

  void setup_gpio();
  std::optional<float> measure_distance_cm();
  void cleanup();

public:
  GPIOController();
  ~GPIOController();
  bool check_for_trigger();
};
