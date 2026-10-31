#!/usr/bin/env bash
# Native (non-Docker) install on the Pi. Run AFTER scripts/pi_gate_check.sh passes.
#
#   bash scripts/pi_setup.sh            # venv + deps + models + calibration reminder
#   bash scripts/pi_setup.sh --service  # additionally install + enable the systemd unit
#
# Why not Docker: one more layer between you and the camera/audio devices, and an
# x86_64 image will not run on ARM64. If you containerise later, BUILD ON THE PI
# (or buildx --platform linux/arm64); never copy an image built on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
DIR=$(pwd)

sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends python3-venv libegl1 libgles2 alsa-utils espeak-ng
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-pi.txt   # headless OpenCV; swap for opencv-python to get --show
bash scripts/fetch_models.sh
B=https://storage.googleapis.com/mediapipe-models
[ -f models/hand_landmarker.task ] || curl -fsL -o models/hand_landmarker.task $B/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
[ -f models/pose_landmarker_lite.task ] || curl -fsL -o models/pose_landmarker_lite.task $B/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
[ -d wavs ] || bash scripts/render_wavs.sh

echo
echo "Next, on THIS Pi and THIS camera:"
echo "  .venv/bin/python realtime.py --calibrate --width 640 --height 480 --fps 15 --no-face"
echo "  .venv/bin/python realtime.py --width 640 --height 480 --fps 15 --threads 3 --no-face"

if [ "${1:-}" = "--service" ]; then
  U=$(id -un)
  sed -e "s#__USER__#$U#g" -e "s#__DIR__#$DIR#g" scripts/signtalk.service | sudo tee /etc/systemd/system/signtalk.service >/dev/null
  sudo usermod -aG audio,video "$U"
  sudo systemctl daemon-reload
  sudo systemctl enable signtalk.service
  echo "service installed; start with: sudo systemctl start signtalk && journalctl -u signtalk -f"
  echo "(log out/in once so the audio,video group change applies)"
fi
