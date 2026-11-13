echo off
REM ==========================================================================
REM  VitalEye v2 - one-click live heart-rate monitor (webcam, NVIDIA GPU)
REM  Double-click, or from a terminal:  run_vitaleye.bat --camera 1
REM  Any extra arguments are passed to scripts\live_monitor.py
REM ==========================================================================
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0.cache\ultralytics"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
if not exist "models\vitalnet\pix_fold0.pt" (
    echo [!] Trained weights not found in models\vitalnet - run train_pipeline.bat first.
    pause
    exit /b 1
)
echo Starting VitalEye v2 ... (q = quit, r = reset, s = screenshot, f = fullscreen)
python scripts\live_monitor.py %*
echo.
echo HR log CSV files are saved in the outputs\ folder.
pause
