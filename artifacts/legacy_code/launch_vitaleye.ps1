$env:YOLO_CONFIG_DIR = "$env:TEMP\Ultralytics"
$env:PYTORCH_ENABLE_MPS_FALLBACK = "1"
Set-Location "e:\CS776 Project"
Write-Host "Launching VitalEye on RTX 3070 Ti (CUDA)..." -ForegroundColor Green
python live_hr_monitor.py
Write-Host "`nPress any key to close..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

