@echo off
cd /d "%~dp0"
echo VitalEye with the PHONE camera (DroidCam must be running and connected).
echo Raw video is saved losslessly to outputs\raw_<time>.mkv for analysis (~0.3-0.6 GB per minute).
echo q = quit, r = reset, s = screenshot
python scripts\live_monitor.py --camera 1 --backend msmf --cam_fps 60 --save_raw auto %*
echo.
echo HR log CSV and raw video are in the outputs\ folder.
pause
