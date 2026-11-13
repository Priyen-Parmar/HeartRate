@echo off
REM ==========================================================================
REM  OptiPulse - one-click live heart-rate monitor (webcam, NVIDIA GPU)
REM  Double-click, or from a terminal:  launchers\windows\run_optipulse.bat --camera 1
REM  Any extra arguments are passed to the app (python -m optipulse --help)
REM ==========================================================================
cd /d "%~dp0..\.."
if not exist "models\vitalnet\pix_fold0.pt" (
    echo [!] Trained weights not found in models\vitalnet - run launchers\windows\train_pipeline.bat first.
    pause
    exit /b 1
)
echo Starting OptiPulse ... (q = quit, r = reset, s = screenshot, f = fullscreen)
python -m optipulse %*
echo.
echo HR log CSV files are saved in the outputs\ folder.
pause
