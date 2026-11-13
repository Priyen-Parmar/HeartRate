@echo off
cd /d "%~dp0"
echo VitalEye reading the phone DIRECTLY over WiFi (no DroidCam PC client / virtual webcam needed).
echo 1. Phone and laptop on the same WiFi.  2. Open the DroidCam app on the phone (screen on).
echo 3. CLOSE the DroidCam PC client (the phone serves only one viewer at a time).
set /p IP=Type the "WiFi IP" shown in the DroidCam phone app (e.g. 192.168.1.5): 
python scripts\live_monitor.py --camera http://%IP%:4747/video --save_raw auto %*
echo.
echo HR log CSV and raw video are in the outputs\ folder.
pause
