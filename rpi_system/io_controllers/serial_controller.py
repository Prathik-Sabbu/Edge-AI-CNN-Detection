"""
Serial controller for communicating between Raspberry Pi and Arduino Uno.
Maintains compatible protocol with arduino_controller.cpp:
- Receives: "TRIGGER"
- Transmits: Detection results formatted for the 16x2 LCD (e.g. "Cat (95%)")
"""
import time
import logging
from typing import Optional
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    serial = None
    SERIAL_AVAILABLE = False

logger = logging.getLogger("EdgeAI.SerialController")

class ArduinoSerialController:
    """Manages USB-Serial communication with the Arduino Uno microcontroller."""

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        fallback_port: str = "/dev/ttyUSB0",
        baud_rate: int = 9600,
        timeout: float = 1.0,
    ):
        self.preferred_port = port
        self.fallback_port = fallback_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial_conn: Optional[serial.Serial] = None

    def connect(self) -> bool:
        """Attempts connection to the Arduino on preferred, fallback, or auto-detected ports."""
        if not SERIAL_AVAILABLE:
            logger.warning(
                "pyserial module is not installed. Please run 'pip install pyserial' or 'setup_rpi.sh'."
            )
            return False

        candidate_ports = [self.preferred_port, self.fallback_port]

        # Check existing available serial ports on the system
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        for p in available_ports:
            if p not in candidate_ports:
                candidate_ports.append(p)

        for port in candidate_ports:
            if not port:
                continue
            try:
                logger.info(f"Attempting connection to Arduino on {port} @ {self.baud_rate} baud...")
                conn = serial.Serial(port, self.baud_rate, timeout=self.timeout)
                time.sleep(2.0)
                conn.reset_input_buffer()
                self.serial_conn = conn
                logger.info(f"Connected to Arduino on {port}.")
                return True
            except (serial.SerialException, FileNotFoundError, PermissionError) as e:
                logger.debug(f"Could not connect to {port}: {e}")

        logger.warning(
            "Could not connect to Arduino on any port. "
            "Verify USB connection and user permissions (e.g. 'sudo usermod -a -G dialout $USER')."
        )
        return False

    def is_connected(self) -> bool:
        """Returns True if the serial connection is open."""
        return self.serial_conn is not None and self.serial_conn.is_open

    def check_for_trigger(self) -> bool:
        """
        Polls the serial buffer for an incoming message.
        Returns True if "TRIGGER" command was received from Arduino.
        """
        if not self.is_connected():
            return False

        try:
            if self.serial_conn.in_waiting > 0:
                raw_line = self.serial_conn.readline()
                try:
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                except Exception:
                    line = ""

                if line:
                    logger.info(f"Arduino -> Pi: '{line}'")
                    if "TRIGGER" in line.upper():
                        return True
        except serial.SerialException as e:
            logger.error(f"Serial read error: {e}. Reconnecting...")
            self.disconnect()

        return False

    def send_result_to_lcd(self, label: str, confidence: float) -> bool:
        """
        Sends formatted prediction back to the Arduino for display on the 16x2 LCD.
        Truncates to 16 characters to fit neatly on a single LCD line.
        """
        if not self.is_connected():
            logger.warning("Cannot send result: Serial connection is not open.")
            return False

        # Format message
        conf_percent = int(round(confidence * 100))
        message = f"{label}: {conf_percent}%"[:16]

        try:
            payload = (message + "\n").encode("utf-8")
            self.serial_conn.write(payload)
            self.serial_conn.flush()
            logger.info(f"Pi -> Arduino (LCD): '{message}'")
            return True
        except serial.SerialException as e:
            logger.error(f"Serial write error: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Closes the serial connection."""
        if self.serial_conn is not None:
            try:
                self.serial_conn.close()
            except Exception:
                pass
            self.serial_conn = None
            logger.info("Serial connection closed.")
