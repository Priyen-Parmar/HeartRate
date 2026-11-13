@echo off
REM List every camera index / backend that works on this PC.
cd /d "%~dp0..\.."
echo Looking for cameras (each one is opened for a moment, nothing is saved) ...
python -m optipulse --list_cameras
echo.
echo Use the line you want, e.g.:  launchers\windows\run_optipulse.bat --camera 1 --backend dshow
pause
