@echo off
REM ============================================================================
REM VitalEyeProNet — One-Click Windows NVIDIA GPU Training Launcher
REM ============================================================================
setlocal EnableDelayedExpansion

cd /d "%~dp0"

echo ============================================================================
echo   VitalEyeProNet - Windows CUDA GPU Training Launcher
echo ============================================================================

REM Enable optimal CUDA memory allocation and TF32 acceleration
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set YOLO_CONFIG_DIR=%TEMP%\Ultralytics

REM Auto-detect dataset path on E:\CS776 Project\
set DATA_DIR=
if exist "E:\CS776 Project\preprocessed_vidformer1\window_0.pt" (
    set "DATA_DIR=E:\CS776 Project\preprocessed_vidformer1"
) else if exist "E:\CS776 Project\preprocessed_vidformer2\window_0.pt" (
    set "DATA_DIR=E:\CS776 Project\preprocessed_vidformer2"
) else if exist "E:\CS776 Project\preprocessed_vidformer\window_0.pt" (
    set "DATA_DIR=E:\CS776 Project\preprocessed_vidformer"
)

if defined DATA_DIR (
    echo [OK] Found Preprocessed UBFC-rPPG Dataset at: "!DATA_DIR!"
    python train_vitaleye_pro.py --data_dir "!DATA_DIR!" %*
) else (
    echo [INFO] Default E:\CS776 Project path not found; passing CLI args directly...
    python train_vitaleye_pro.py %*
)

echo.
echo ============================================================================
echo   Training finished! Best weights saved to:
echo   VitalEye_CS776_Deploy\models\vitaleye_pro_best.pt
echo ============================================================================
pause
