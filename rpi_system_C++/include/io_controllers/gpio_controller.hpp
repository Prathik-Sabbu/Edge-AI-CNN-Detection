#pragma once

#include <chrono>
#include <iostream>
#include <optional>
#include <pigpio.h>
#include <spdlog/spdlog.h>
#include <wiringPi.h>

class GPIOController {
private:
  int trigger_pin = 23;
  int echo_pin = 24;
  float distance_threashhold_cm = 40.0;
  float min_distance_cm = 2.0;
  float cooldown_seconds = 3.0;

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
