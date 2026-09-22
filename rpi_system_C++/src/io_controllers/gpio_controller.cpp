#include "../../include/io_controllers/gpio_controller.hpp"
#include "../../include/config.hpp"
#include <chrono>
#include <spdlog/spdlog.h>
#include <thread>

#ifdef __arm__
#include <pigpio.h>
#endif

GPIOController::GPIOController()
  : trigger_pin(Config::GPIO_TRIGGER_PIN),
    echo_pin(Config::GPIO_ECHO_PIN),
    distance_threshold_cm(Config::TRIGGER_DISTANCE_THRESHOLD_CM),
    min_distance_cm(Config::MIN_DISTANCE_THRESHOLD_CM),
    cooldown_seconds(Config::COOLDOWN_SECONDS),
    is_ready(true),
    gpio_available(false) { setup_gpio(); }
GPIOController::~GPIOController() { cleanup(); }

void GPIOController::setup_gpio() {
#ifdef __arm__
  if (gpioInitialise() < 0) {
    spdlog::warn("pigpio initialization failed, running in sim mode");
    gpio_available = false;
    return;
  }

  // Initialize the input and output pins
  gpioSetMode(trigger_pin, PI_OUTPUT);
  gpioSetMode(echo_pin, PI_INPUT);

  // set Output pin to 0 so it doesnt output a pulse yet
  gpioWrite(trigger_pin, 0);

  // sensor settling time (500ms)
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  gpio_available = true;
  spdlog::info("Raspberry Pi GPIO initialized. Trig=BCM{}, Echo=BCM{}",
               trigger_pin, echo_pin);

#else
  spdlog::warn(
      "Running on non-Raspberry Pi architecture. GPIO in SIMULATION mode.");
  gpio_available = false;
#endif
}

std::optional<float> GPIOController::measure_distance_cm() {
  if (!gpio_available) {
    return 25.0f;
  }

#ifdef __arm__
  // send a 10 micro second trigger pulse
  gpioWrite(trigger_pin, 1);
  gpioDelay(10); // sleep for 10 micro seconds
  gpioWrite(trigger_pin, 0);

  uint32_t pulse_start = gpioTick();
  uint32_t timeout_start = pulse_start;

  // wait for echo to start
  while (gpioRead(echo_pin) == 0) {
    pulse_start = gpioTick();
    // 40ms timeout (40000 us)
    if ((pulse_start - timeout_start) > 40000) {
      return std::nullopt;
    }
  }
  
  uint32_t pulse_end = pulse_start;
  uint32_t timeout_echo = pulse_start;

  // wait for echo to end
  while (gpioRead(echo_pin) == 1) {
    pulse_end = gpioTick();
    if ((pulse_end - timeout_echo) > 40000) {
      return std::nullopt;
    }
  }

  // calculate distance traveled (speed of sound 34300 cm/s = 0.0343 cm/us)
  float pulse_duration = static_cast<float>(pulse_end - pulse_start);
  float distance = (pulse_duration * 0.0343f) / 2.0f;
  return std::round(distance * 10.0f) / 10.0f;
#else
  return std::nullopt;
#endif
}

bool GPIOController::check_for_trigger() {
  auto now = std::chrono::steady_clock::now();

  // check cooldown
  std::chrono::duration<float> elapsed = now - last_trigger_time;
  if (elapsed.count() < cooldown_seconds) {
    return false;
  }

  // ready message after cooldown
  if (!is_ready) {
    spdlog::info("Sensor ready for next trigger.");
    is_ready = true;
  }

  // measure distance
  std::optional<float> distance = measure_distance_cm();

  // evaluate detection threashold
  if (distance.has_value()) {
    float d = distance.value();
    if ((min_distance_cm <= d) && (d <= distance_threshold_cm)) {
      spdlog::info("Direct GPIO Trigger: Object detected at {} cm!", d);
      last_trigger_time = now;
      is_ready = false;
      return true;
    }
  }
  return false;
}

void GPIOController::cleanup() {
#ifdef __arm__
  if (gpio_available) {
    spdlog::info("Cleaning up GPIO resources");
    // Ensure trigger pin is low
    gpioWrite(trigger_pin, 0);
    // Release hardware resources
    gpioTerminate();
    gpio_available = false;
  }
#else
  gpio_available = false;
#endif
}
