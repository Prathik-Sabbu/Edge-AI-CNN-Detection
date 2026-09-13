"""
Direct GPIO controller for Raspberry Pi.
Manages HC-SR04 ultrasonic distance sensor wired directly to Pi 40-pin header.
Emulates the 3-second non-blocking cooldown logic from arduino_controller.cpp.
"""
import time
import logging
from typing import Optional

logger = logging.getLogger("EdgeAI.GPIOController")

class RaspberryPiGPIOController:
    """Controls HC-SR04 sensor directly using Raspberry Pi GPIO pins."""

    def __init__(
        self,
        trigger_pin: int = 23,
        echo_pin: int = 24,
        distance_threshold_cm: float = 40.0,
        min_distance_cm: float = 2.0,
        cooldown_seconds: float = 3.0,
    ):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.distance_threshold_cm = distance_threshold_cm
        self.min_distance_cm = min_distance_cm
        self.cooldown_seconds = cooldown_seconds

        self.last_trigger_time = 0.0
        self.is_ready = True
        self.gpio_available = False
        self._setup_gpio()

    def _setup_gpio(self):
        """Initializes RPi.GPIO or gpiozero if available."""
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.GPIO.setmode(self.GPIO.BCM)
            self.GPIO.setwarnings(False)
            self.GPIO.setup(self.trigger_pin, self.GPIO.OUT)
            self.GPIO.setup(self.echo_pin, self.GPIO.IN)
            self.GPIO.output(self.trigger_pin, False)
            time.sleep(0.5)
            self.gpio_available = True
            logger.info(
                f"Raspberry Pi GPIO initialized. Trig=BCM{self.trigger_pin}, Echo=BCM{self.echo_pin}"
            )
        except Exception as e:
            logger.warning(
                f"RPi.GPIO not available ({e}). Running in GPIO SIMULATION mode."
            )
            self.gpio_available = False

    def measure_distance_cm(self) -> Optional[float]:
        """
        Triggers an ultrasonic pulse and measures echo duration.
        Returns distance in centimeters, or None if timed out.
        """
        if not self.gpio_available:
            # Return simulated distance for offline testing
            return 25.0

        # Send 10 microsecond trigger pulse
        self.GPIO.output(self.trigger_pin, True)
        time.sleep(0.00001)
        self.GPIO.output(self.trigger_pin, False)

        pulse_start = time.time()
        timeout_start = pulse_start

        # Wait for echo to start
        while self.GPIO.input(self.echo_pin) == 0:
            pulse_start = time.time()
            if pulse_start - timeout_start > 0.04:  # 40ms timeout
                return None

        pulse_end = pulse_start
        timeout_echo = pulse_start

        # Wait for echo to end
        while self.GPIO.input(self.echo_pin) == 1:
            pulse_end = time.time()
            if pulse_end - timeout_echo > 0.04:
                return None

        # Calculate distance
        pulse_duration = pulse_end - pulse_start
        distance = (pulse_duration * 34300) / 2.0
        return round(distance, 1)

    def check_for_trigger(self) -> bool:
        """
        Checks if an object is within the goldilocks zone with non-blocking cooldown.
        Replicates arduino_controller.cpp logic:
          - distance > 2cm and < 40cm
          - cooldown elapsed
        """
        current_time = time.time()

        # Check cooldown
        if (current_time - self.last_trigger_time) < self.cooldown_seconds:
            return False

        if not self.is_ready:
            logger.info("Sensor ready for next trigger.")
            self.is_ready = True

        distance = self.measure_distance_cm()
        if distance is not None:
            if self.min_distance_cm < distance <= self.distance_threshold_cm:
                logger.info(f"Direct GPIO Trigger: Object detected at {distance} cm!")
                self.last_trigger_time = current_time
                self.is_ready = False
                return True

        return False

    def cleanup(self):
        """Cleans up GPIO state."""
        if self.gpio_available:
            try:
                self.GPIO.cleanup()
            except Exception:
                pass
