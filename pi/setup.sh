#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ "$(uname -s)" != Linux ]; then
  echo 'Use python3 launch.py on a computer. This installer is for Raspberry Pi OS.'
  exit 1
fi
sudo apt-get update
sudo apt-get install -y chromium python3 python3-picamera2 rpicam-apps
python3 launch.py --verify-only
python3 doctor.py
echo 'Ready. Start the ribbon-camera player: python3 launch.py --camera csi'
echo 'For a USB webcam: python3 launch.py --camera usb'
