#pragma once

#include <string>

class ArduinoSerialController {
public:
  ArduinoSerialController();
  ~ArduinoSerialController();

  ArduinoSerialController(const ArduinoSerialController &) = delete;
  ArduinoSerialController &operator=(const ArduinoSerialController &) = delete;

  bool connect();
  bool is_connected() const;
  bool check_for_trigger();
  bool send_result_to_lcd(const std::string &label, float confidence);
  void disconnect();

private:
  std::string preferred_port_;
  std::string fallback_port_;
  int baud_rate_;
  float timeout_sec_;
  int serial_fd_ = -1;

  int try_open_port(const std::string &port);
};