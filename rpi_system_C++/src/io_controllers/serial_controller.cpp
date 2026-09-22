#include "../../include/io_controllers/serial_controller.hpp"
#include "../../include/config.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <spdlog/spdlog.h>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

#ifdef __linux__
/// Map a common integer baud rate to the POSIX speed_t constant.
static speed_t baud_to_speed(int baud) {
  switch (baud) {
  case 9600:
    return B9600;
  case 19200:
    return B19200;
  case 38400:
    return B38400;
  case 57600:
    return B57600;
  case 115200:
    return B115200;
  default:
    spdlog::warn("Unsupported baud rate {}. Falling back to 9600.", baud);
    return B9600;
  }
}

/// List /dev/ttyACM* and /dev/ttyUSB* device paths.
static std::vector<std::string> list_serial_ports() {
  std::vector<std::string> ports;
  DIR *dir = opendir("/dev");
  if (!dir)
    return ports;

  struct dirent *entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name = entry->d_name;
    if (name.rfind("ttyACM", 0) == 0 || name.rfind("ttyUSB", 0) == 0) {
      ports.push_back("/dev/" + name);
    }
  }
  closedir(dir);
  std::sort(ports.begin(), ports.end());
  return ports;
}
#endif

ArduinoSerialController::ArduinoSerialController()
    : preferred_port_(Config::SERIAL_PORT), fallback_port_(Config::SERIAL_PORT_FALLBACK),
      baud_rate_(Config::BAUD_RATE), timeout_sec_(Config::SERIAL_TIMEOUT) {}

ArduinoSerialController::~ArduinoSerialController() { disconnect(); }

bool ArduinoSerialController::is_connected() const { return serial_fd_ >= 0; }

bool ArduinoSerialController::connect() {
#ifndef __linux__
  spdlog::warn("Serial communication is only supported on Linux. "
               "Running without serial.");
  return false;
#else
  // Build the candidate port list: preferred -> fallback -> auto-detected
  std::vector<std::string> candidates = {preferred_port_, fallback_port_};

  for (const auto &p : list_serial_ports()) {
    if (std::find(candidates.begin(), candidates.end(), p) ==
        candidates.end()) {
      candidates.push_back(p);
    }
  }

  for (const auto &port : candidates) {
    if (port.empty())
      continue;

    spdlog::info("Attempting connection to Arduino on {} @ {} baud...", port,
                 baud_rate_);

    int fd = try_open_port(port);
    if (fd >= 0) {
      // Allow the Arduino bootloader to finish resetting (~2 s)
      std::this_thread::sleep_for(std::chrono::seconds(2));
      tcflush(fd, TCIFLUSH); // clear any boot garbage
      serial_fd_ = fd;
      spdlog::info("Connected to Arduino on {}.", port);
      return true;
    }
  }

  spdlog::warn("Could not connect to Arduino on any port. "
               "Verify USB connection and user permissions "
               "(e.g. 'sudo usermod -a -G dialout $USER').");
  return false;
#endif
}

void ArduinoSerialController::disconnect() {
#ifdef __linux__
  if (serial_fd_ >= 0) {
    close(serial_fd_);
    serial_fd_ = -1;
    spdlog::info("Serial connection closed.");
  }
#endif
}

int ArduinoSerialController::try_open_port(const std::string &port) {
#ifndef __linux__
  (void)port;
  return -1;
#else
  int fd = open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
  if (fd < 0) {
    spdlog::debug("Could not open {}: {}", port, strerror(errno));
    return -1;
  }

  // Restore blocking mode after open succeeds
  int flags = fcntl(fd, F_GETFL, 0);
  fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);

  struct termios tty{};
  if (tcgetattr(fd, &tty) != 0) {
    spdlog::debug("tcgetattr failed for {}: {}", port, strerror(errno));
    close(fd);
    return -1;
  }

  speed_t speed = baud_to_speed(baud_rate_);
  cfsetispeed(&tty, speed);
  cfsetospeed(&tty, speed);

  // 8N1, no flow control
  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
  tty.c_cflag |= CLOCAL | CREAD;

  // Raw input mode (no echo, no canonical processing)
  tty.c_iflag &= ~(IXON | IXOFF | IXANY | IGNBRK | BRKINT | PARMRK | ISTRIP |
                   INLCR | IGNCR | ICRNL);
  tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
  tty.c_oflag &= ~OPOST;

  // VMIN/VTIME: block for up to timeout_sec_ waiting for at least 1 byte
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = static_cast<cc_t>(
      std::clamp(static_cast<int>(timeout_sec_ * 10), 1, 255));

  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    spdlog::debug("tcsetattr failed for {}: {}", port, strerror(errno));
    close(fd);
    return -1;
  }

  return fd;
#endif
}

bool ArduinoSerialController::check_for_trigger() {
  if (!is_connected())
    return false;

#ifdef __linux__
  // Check how many bytes are waiting
  int bytes_available = 0;
  ioctl(serial_fd_, FIONREAD, &bytes_available);
  if (bytes_available <= 0)
    return false;

  // Read one line (up to newline or buffer limit)
  char buf[256];
  int idx = 0;
  while (idx < static_cast<int>(sizeof(buf) - 1)) {
    ssize_t n = read(serial_fd_, &buf[idx], 1);
    if (n <= 0)
      break;
    if (buf[idx] == '\n') {
      break;
    }
    ++idx;
  }
  buf[idx] = '\0';

  // Trim trailing whitespace
  std::string line(buf);
  while (!line.empty() &&
         (line.back() == '\r' || line.back() == '\n' || line.back() == ' ')) {
    line.pop_back();
  }

  if (!line.empty()) {
    spdlog::info("Arduino -> Pi: '{}'", line);

    // Case-insensitive search for "TRIGGER"
    std::string upper = line;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper.find("TRIGGER") != std::string::npos) {
      return true;
    }
  }
#endif

  return false;
}

bool ArduinoSerialController::send_result_to_lcd(const std::string &label,
                                                 float confidence) {
  if (!is_connected()) {
    spdlog::warn("Cannot send result: Serial connection is not open.");
    return false;
  }

#ifdef __linux__
  // Format: "Label: NN%"  — truncated to 16 chars for a single LCD line
  int conf_percent = static_cast<int>(std::round(confidence * 100.0f));
  std::string message = label + ": " + std::to_string(conf_percent) + "%";
  if (message.size() > 16) {
    message.resize(16);
  }

  message += '\n';

  ssize_t written =
      write(serial_fd_, message.c_str(), static_cast<size_t>(message.size()));
  if (written < 0) {
    spdlog::error("Serial write error: {}", strerror(errno));
    disconnect();
    return false;
  }
  // fsync is not meaningful for ttys; tcdrain waits for output to transmit
  tcdrain(serial_fd_);

  // Log without the trailing newline
  message.pop_back();
  spdlog::info("Pi -> Arduino (LCD): '{}'", message);
  return true;
#else
  (void)label;
  (void)confidence;
  return false;
#endif
}
