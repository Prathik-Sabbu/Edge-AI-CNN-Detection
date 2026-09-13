# Raspberry Pi Independent Edge AI CNN Detection System

This directory contains the standalone, edge-deployed vision and inference system for running animal detection on a Raspberry Pi.

---

## 1. System Architecture

The system operates independently on the Raspberry Pi without requiring a desktop computer:

```text
[ Ultrasonic Distance Sensor ] 
              │
         (Object in Range: 2-40cm)
              ▼
   [ Sensor Trigger / Cooldown ]
              │
              ▼
  [ Camera Frame Capture (CSI/USB) ]
              │
              ▼
 [ CNN Classifier (TFLite Model) ]
              │
              ▼
[ Display Output (LCD 16x2 / Serial / Log) ]
```

---

## 2. Hardware Setup Options

### Option A: Raspberry Pi + Arduino Uno (Recommended for Quick Transition)
In this setup, your existing Arduino hardware wiring remains **100% intact**:
- **Arduino Uno**: Keeps HC-SR04 sensor (`Trigger: Pin 11`, `Echo: Pin 12`) and 16x2 LCD (`Pins 7, 6, 5, 4, 3, 2`). Running `arduino_controller.cpp`.
- **Raspberry Pi**: Connects to the Arduino via a standard USB-A to USB-B cable.
- **Camera**: USB Webcam or Raspberry Pi Camera connected directly to the Pi.
- **Configuration**: In `config.py`, set:
  ```python
  USE_ARDUINO_SERIAL = True
  SERIAL_PORT = "/dev/ttyACM0"  # or /dev/ttyUSB0
  ```

### Option B: Standalone Raspberry Pi (All-in-One Direct GPIO)
The Arduino is removed entirely. The HC-SR04 sensor connects directly to the Raspberry Pi 40-pin GPIO header:
- **HC-SR04 VCC** -> Pi Pin 2 or 4 (5V Power)
- **HC-SR04 GND** -> Pi Pin 6 (Ground)
- **HC-SR04 TRIG** -> Pi Pin 16 (`GPIO 23`)
- **HC-SR04 ECHO** -> **Voltage Divider** (1kΩ + 2kΩ resistors to reduce 5V to 3.3V) -> Pi Pin 18 (`GPIO 24`)
  > **CAUTION**: Pi GPIO pins are 3.3V only. Never connect the 5V Echo pin directly to the Pi GPIO without a voltage divider or logic level shifter!
- **Configuration**: In `config.py`, set:
  ```python
  USE_ARDUINO_SERIAL = False
  ```

---

## 3. Installation on Raspberry Pi

### Quick Setup
1. Copy the `rpi_system` folder to your Raspberry Pi (via `scp`, `git`, or USB drive).
2. Open a terminal on your Pi inside `rpi_system`:
   ```bash
   cd ~/rpi_system
   chmod +x scripts/setup_rpi.sh
   ./scripts/setup_rpi.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

---

## 4. Running the System

### Manual Launch
```bash
python main.py
```

You will see live logs indicating:
- Camera warmup and readiness
- Serial connection to Arduino (if using Option A)
- Real-time trigger events, inference latency, top predictions, and saved snapshots in `captures/`.

### Run Automatically on Boot (Headless / Standalone)
To have the Pi start detection whenever powered on (no keyboard or monitor required):
```bash
# Edit username/path in scripts/edge_ai.service if your Pi user is not 'pi'
sudo cp scripts/edge_ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edge_ai.service
sudo systemctl start edge_ai.service
```

Check status or view live logs anytime with:
```bash
sudo systemctl status edge_ai.service
journalctl -u edge_ai.service -f
```

---

## 5. Directory Structure
```text
rpi_system/
├── config.py                 # Central hardware & model settings
├── main.py                   # Master edge AI daemon loop
├── vision/
│   ├── camera.py             # Resilient camera capture (OpenCV / Picamera2)
│   └── classifier.py         # Fast TFLite CNN inference & simulation mode
├── io_controllers/
│   ├── serial_controller.py  # Arduino USB serial protocol & LCD formatter
│   └── gpio_controller.py    # Direct Pi GPIO HC-SR04 driver
├── models/
│   ├── animal_classifier.tflite # Trained TFLite weights
│   └── labels.txt            # Detection classes
├── captures/                 # Automatically saved snapshots on trigger
├── scripts/
│   ├── setup_rpi.sh          # One-click Raspberry Pi OS installer
│   └── edge_ai.service       # Systemd boot service
├── requirements.txt          # Pi-optimized Python dependencies
└── README.md                 # System manual and wiring guide
```
