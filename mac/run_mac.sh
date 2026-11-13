#!/bin/bash
# VitalEye on macOS. Usage (Terminal):  bash run_mac.sh  (zip kit)   or   bash mac/run_mac.sh  (GitHub clone)
# First run creates a private Python environment in .venv (~1.5 GB download, 5-10 min). Later runs start
# in seconds. Extra options are passed to the app, e.g.:  bash run_mac.sh --ensemble 4   |   bash run_mac.sh --list_cameras
cd "$(dirname "$0")" || exit 1
# works both from the unzipped kit (scripts/ next to this file) and from a GitHub clone (this file in mac/)
if [ ! -f scripts/live_monitor.py ] && [ -f ../scripts/live_monitor.py ]; then cd ..; fi
if [ ! -f scripts/live_monitor.py ]; then echo "scripts/live_monitor.py not found next to run_mac.sh"; exit 1; fi

PY=""
for c in python3.12 python3.11 python3.13 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python 3 is not installed. Install Python 3.12 from https://www.python.org/downloads/macos/ and run again."
  exit 1
fi
ARCH=$(uname -m)

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up Python environment in .venv using $PY ($ARCH) ..."
  "$PY" -m venv .venv || exit 1
  .venv/bin/python -m pip install --upgrade pip
  if [ "$ARCH" = "arm64" ]; then          # Apple Silicon (M1/M2/M3/M4): current PyTorch with Apple GPU (MPS)
    .venv/bin/python -m pip install torch numpy scipy opencv-python pillow ultralytics || { rm -rf .venv; exit 1; }
  else                                    # Intel Mac: last PyTorch release for Intel macOS (needs Python <= 3.12)
    .venv/bin/python -m pip install "torch==2.2.2" "torchvision==0.17.2" "numpy<2" scipy opencv-python pillow ultralytics \
      || { echo "Install failed. On an Intel Mac use Python 3.11 or 3.12."; rm -rf .venv; exit 1; }
  fi
fi

export PYTORCH_ENABLE_MPS_FALLBACK=1     # any op missing on the Apple GPU runs on the CPU instead of crashing
echo "Starting VitalEye (q = quit, r = reset, s = screenshot, f = fullscreen, h = pulse-region glow)"
.venv/bin/python scripts/live_monitor.py --camera 0 --ensemble 2 --always_show --save_raw auto "$@"
echo "HR log CSV and raw video are in the outputs/ folder."
