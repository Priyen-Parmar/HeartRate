@echo off
REM ===========================================================================================
REM  OptiPulse on Windows - one-click live heart-rate monitor.
REM
REM  Double-click this file, or run it from a terminal (any folder):
REM      launchers\windows\run_windows.bat                                   webcam 0
REM      launchers\windows\run_windows.bat --camera 1                        another camera
REM      launchers\windows\run_windows.bat --list_cameras                    list cameras that work
REM      launchers\windows\run_windows.bat --camera 1 --backend msmf --cam_fps 60     phone via DroidCam client
REM      launchers\windows\run_windows.bat --camera http://192.168.1.5:4747/video     phone via DroidCam WiFi
REM      launchers\windows\run_windows.bat --help                            every option
REM  Extra arguments are passed to the app (python -m optipulse) and override the defaults below
REM  (--camera 0 --always_show --save_raw auto, plus --ensemble 2 without an NVIDIA GPU).
REM  --save_raw "" turns off the lossless raw-video recording (~0.3-0.6 GB per minute in outputs\).
REM
REM  First run: finds a 64-bit Python 3.9+ (offers to install Python 3.12 with winget if none),
REM  creates a private environment in .venv and installs PyTorch (CUDA build when an NVIDIA GPU is
REM  present, otherwise CPU) plus everything in requirements.txt. Download: ~3 GB with CUDA, ~0.5 GB
REM  without; 5-15 min. Later runs start in seconds. To reinstall from scratch, delete the .venv folder.
REM ===========================================================================================
setlocal EnableExtensions DisableDelayedExpansion
title OptiPulse

REM ---- find the project root: the folder that holds optipulse\ (two levels up in a clone,
REM      or right next to this file in a zipped kit). pushd also works from a network share.
pushd "%~dp0" || goto :fail_root
set /a _UP=0
:find_root
if exist "optipulse\__main__.py" goto :root_found
set /a _UP+=1
if %_UP% GTR 3 goto :fail_root
cd ..
goto :find_root
:root_found
echo OptiPulse folder: "%CD%"

if not exist "models\vitalnet\*.pt" goto :fail_models
if not exist "models\face_detector\yolov8_face_nose_chest.pt" goto :fail_models

set "VPY=.venv\Scripts\python.exe"
set "READY=.venv\optipulse_setup_ok.txt"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
REM a real 3D convolution on the GPU: fails if PyTorch has no CUDA, or no kernels for this GPU model
set "CUDA_TEST=import torch; x=torch.randn(1,3,4,8,8,device='cuda'); torch.nn.functional.conv3d(x,torch.randn(4,3,3,3,3,device='cuda')).sum().item()"

REM ---- existing environment that works and is fully installed -> start straight away
if not exist "%VPY%" goto :check_foreign_venv
"%VPY%" -c "import sys" >nul 2>&1 || goto :broken_venv
if exist "%READY%" goto :run
goto :install

:check_foreign_venv
REM a .venv without Scripts\python.exe was made on macOS/Linux (or is half-created): rebuild it
if not exist ".venv\" goto :find_python
:broken_venv
echo The existing .venv folder is not a working Windows environment - recreating it ...
rmdir /s /q ".venv"
if exist ".venv\" goto :fail_rmvenv

REM ===========================================================================================
:find_python
set "PYEXE="
for %%V in (3.12 3.11 3.13 3.10 3.14 3) do if not defined PYEXE call :try_python py -%%V
if not defined PYEXE call :try_python python
if not defined PYEXE call :try_python python3
if not defined PYEXE call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE goto :no_python
:have_python
echo First run: creating the Python environment in .venv with:
%PYEXE% --version
%PYEXE% -m venv .venv
if errorlevel 1 goto :fail_venv
if not exist "%VPY%" goto :fail_venv

REM ===========================================================================================
:install
echo.
echo ===== Installing the required packages into .venv (only needed once) =====
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto :fail_pip

call :check_vcredist

REM ---- PyTorch: CUDA build for NVIDIA GPUs, otherwise the CPU build
set "TORCH_OK=0"
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>&1 && set "HAS_NVIDIA=1"
if "%HAS_NVIDIA%"=="0" powershell -NoProfile -Command "if ((Get-CimInstance Win32_VideoController).Name -match 'NVIDIA') { exit 0 } else { exit 1 }" >nul 2>&1 && set "HAS_NVIDIA=1"
if "%HAS_NVIDIA%"=="0" goto :torch_cpu

echo NVIDIA GPU found - installing PyTorch with CUDA support.
REM already installed with working CUDA (e.g. an earlier run stopped later on)? keep it
"%VPY%" -c "%CUDA_TEST%" >nul 2>&1 && set "TORCH_OK=1"
if "%TORCH_OK%"=="1" goto :torch_done
REM CUDA builds the installed driver supports (its CUDA version from nvidia-smi); cu128/cu126 first because
REM they cover the most GPU generations (GTX 900 ... RTX 50), cu130 only if those have no build for this Python
"%VPY%" -c "import re,subprocess; o=subprocess.run('nvidia-smi',shell=True,capture_output=True,text=True).stdout; m=re.search(r'CUDA Version:\s*(\d+)\.(\d+)',o); v=(int(m[1]),int(m[2])) if m else (12,8); print(' '.join(t for a,b,t in [(12,8,'cu128'),(12,6,'cu126'),(13,0,'cu130'),(12,4,'cu124'),(11,8,'cu118')] if (a,b)<=v))" > ".venv\cuda_tags.txt" 2>nul
set "CU_TAGS="
set /p CU_TAGS=<".venv\cuda_tags.txt"
del ".venv\cuda_tags.txt" >nul 2>&1
if not defined CU_TAGS echo [!] The NVIDIA driver is too old for current PyTorch - update it from nvidia.com for GPU speed.
for %%T in (%CU_TAGS%) do call :try_cuda %%T
if "%TORCH_OK%"=="1" goto :torch_done
echo [!] Could not get a working CUDA build of PyTorch - falling back to the CPU build (slower).
echo     Updating the NVIDIA driver and deleting the .venv folder, then running again, usually fixes this.
"%VPY%" -m pip uninstall -y torch torchvision >nul 2>&1

:torch_cpu
if "%HAS_NVIDIA%"=="0" echo No NVIDIA GPU found - installing the CPU build of PyTorch - runs on the processor.
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 "%VPY%" -m pip install torch torchvision
if errorlevel 1 goto :fail_torch

:torch_done
REM ---- everything else (numpy, scipy, opencv, pillow, ultralytics, matplotlib)
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail_reqs

REM ---- final check: every package the app imports must load
"%VPY%" -c "import torch, torchvision, numpy, scipy, cv2, PIL, ultralytics, matplotlib; print('PyTorch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')"
if errorlevel 1 goto :fail_verify
> "%READY%" echo setup completed %DATE% %TIME%
echo ===== Setup complete =====
echo.

REM ===========================================================================================
:run
set "MODE_ARGS=--ensemble 2"
set "MODE_TEXT=CPU, 2 models (no NVIDIA GPU usable)"
"%VPY%" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 goto :start_app
set "MODE_ARGS="
set "MODE_TEXT=NVIDIA GPU, 10 models"
:start_app
echo Starting OptiPulse (%MODE_TEXT%) ...
echo Keys: q = quit, r = reset, s = screenshot, f = fullscreen, h = pulse-region glow, c = camera settings
"%VPY%" -m optipulse --camera 0 --always_show --save_raw auto %MODE_ARGS% %*
set "APP_RC=%ERRORLEVEL%"
echo.
if not "%APP_RC%"=="0" goto :app_failed
echo HR log CSV and raw video are in the outputs\ folder.
popd
pause
exit /b 0

:app_failed
echo [!] OptiPulse stopped with an error (code %APP_RC%) - see the messages above.
echo     Camera not opening? List the cameras that work:  launchers\windows\run_windows.bat --list_cameras
echo     then use one, e.g.:                               launchers\windows\run_windows.bat --camera 1 --backend msmf
popd
pause
exit /b %APP_RC%

REM ===========================================================================================
REM  subroutines
REM ===========================================================================================
:try_python
REM usable = runs, Python 3.9 or newer, 64-bit (PyTorch has no 32-bit builds)
%* -c "import sys,struct; sys.exit(0 if sys.version_info >= (3,9) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PYEXE=%*"
goto :eof

:try_cuda
if "%TORCH_OK%"=="1" goto :eof
echo.
echo --- trying PyTorch built for CUDA %1 ---
"%VPY%" -m pip uninstall -y torch torchvision >nul 2>&1
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/%1
if errorlevel 1 goto :eof
"%VPY%" -c "%CUDA_TEST%" >nul 2>&1
if errorlevel 1 echo     installed, but this build cannot use the GPU - trying the next one
if errorlevel 1 goto :eof
set "TORCH_OK=1"
goto :eof

:check_vcredist
REM PyTorch needs the Microsoft Visual C++ 2015-2022 runtime (missing on some fresh Windows installs)
if exist "%SystemRoot%\System32\msvcp140.dll" if exist "%SystemRoot%\System32\vcruntime140_1.dll" goto :eof
echo Microsoft Visual C++ runtime not found (PyTorch needs it).
where winget >nul 2>&1 || goto :vcredist_manual
echo Installing it with winget (Windows may ask for permission) ...
winget install -e --id Microsoft.VCRedist.2015+.x64 --accept-package-agreements --accept-source-agreements
if exist "%SystemRoot%\System32\msvcp140.dll" goto :eof
:vcredist_manual
echo [!] Please install it from https://aka.ms/vs/17/release/vc_redist.x64.exe - continuing anyway.
goto :eof

REM ===========================================================================================
REM  errors
REM ===========================================================================================
:no_python
echo.
echo No 64-bit Python 3.9 or newer was found on this PC.
where winget >nul 2>&1 || goto :no_python_manual
choice /c YN /m "Install Python 3.12 now (winget, current user only)"
if errorlevel 2 goto :no_python_manual
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :try_python "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYEXE call :try_python py -3.12
if defined PYEXE goto :have_python
echo [!] Python was installed but cannot be found yet - close this window and run the launcher again.
goto :end_fail
:no_python_manual
echo Install Python 3.12 (64-bit) from https://www.python.org/downloads/windows/
echo (tick "Add python.exe to PATH" in the installer), then run this file again.
goto :end_fail

:fail_root
echo [!] OptiPulse project folder (the one containing optipulse\) not found above "%~dp0".
goto :end_fail
:fail_models
echo [!] Model weights missing: need models\vitalnet\*.pt and models\face_detector\yolov8_face_nose_chest.pt
goto :end_fail
:fail_rmvenv
echo [!] Could not delete the old .venv folder - close programs using it (editors, terminals) and try again.
goto :end_fail
:fail_venv
echo [!] Creating the Python environment failed - see the messages above.
if exist ".venv\" rmdir /s /q ".venv"
goto :end_fail
:fail_pip
echo [!] pip could not run - check the internet connection and try again.
goto :end_fail
:fail_torch
echo [!] Installing PyTorch failed - see the messages above. Common causes:
echo     - no internet connection or not enough disk space (needs ~3 GB free);
echo     - a Python version PyTorch does not support yet: install Python 3.12, delete .venv, run again;
echo     - a very long folder path: move the OptiPulse folder closer to the drive root (e.g. C:\OptiPulse)
echo       or enable Windows long paths, then delete .venv and run again.
goto :end_fail
:fail_reqs
echo [!] Installing the packages in requirements.txt failed - see the messages above.
echo     Check the internet connection, then run this file again (finished downloads are reused).
goto :end_fail
:fail_verify
echo [!] The packages were installed but do not load - see the error above.
echo     If it mentions a DLL: install https://aka.ms/vs/17/release/vc_redist.x64.exe, then run again.
echo     Otherwise delete the .venv folder and run this file again.
goto :end_fail

:end_fail
popd
echo.
pause
exit /b 1
