#!/bin/bash
# ==============================================================================
# Setup script for Raspberry Pi Edge AI CNN Detection System
# Run on your Raspberry Pi: bash scripts/setup_rpi.sh
# ==============================================================================

set -e

echo "=== [1/5] Updating system packages ==="
sudo apt-get update -y
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libatlas-base-dev \
    v4l-utils \
    libgl1-mesa-glx

echo "=== [2/5] Adding user to hardware access groups (dialout, video) ==="
sudo usermod -a -G dialout,video $USER

echo "=== [3/5] Setting up Python virtual environment ==="
cd "$(dirname "$0")/.."
if [ ! -d "venv" ]; then
    python3 -m venv venv --system-site-packages
    echo "Virtual environment created at $(pwd)/venv"
fi

source venv/bin/activate

echo "=== [4/5] Installing Python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

# Attempt to install tflite-runtime wheel if not already installed
if ! python3 -c "import tflite_runtime" 2>/dev/null; then
    echo "Installing tflite-runtime..."
    pip install tflite-runtime || echo "Note: If tflite-runtime fails on your specific Pi OS version, you can install via: pip install tensorflow-lite or tflite-runtime wheels."
fi

echo "=== [5/5] Installation Complete! ==="
echo "To run the system manually:"
echo "    source venv/bin/activate"
echo "    python main.py"
echo ""
echo "To configure auto-start on boot via systemd, run:"
echo "    sudo cp scripts/edge_ai.service /etc/systemd/system/"
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl enable edge_ai.service"
echo "    sudo systemctl start edge_ai.service"
