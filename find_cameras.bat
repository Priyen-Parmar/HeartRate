@echo off
cd /d "%~dp0"
echo Looking for cameras (each one is opened for a moment, nothing is saved) ...
python scripts\live_monitor.py --list_cameras
echo.
echo Use the line you want, e.g.:  run_vitaleye.bat --camera 1 --backend dshow
pause
