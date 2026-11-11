#!/usr/bin/env bash
# VitalEye — One-Command Live Heart Rate Monitor Launcher
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export YOLO_CONFIG_DIR="/tmp/Ultralytics"
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [ -x "/usr/local/bin/python3.12" ]; then
    PYTHON_BIN="/usr/local/bin/python3.12"
else
    PYTHON_BIN="python3"
fi

echo "🚀 Launching VitalEye Live Heart Rate Monitor using $PYTHON_BIN..."
exec "$PYTHON_BIN" "$SCRIPT_DIR/live_hr_monitor.py" "$@"
