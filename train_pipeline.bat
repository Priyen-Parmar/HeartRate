@echo off
REM ==========================================================================
REM  VitalEye - reproduce EVERYTHING from the raw UBFC-rPPG videos (RTX GPU)
REM    1. preprocess  (YOLO face crops + aligned contact PPG, 30 Hz)        ~12 min
REM    2. audit       (ground-truth quality report)                         ~10 min
REM    3. train pix   (VitalNet v1, 5-fold subject-wise CV)                 ~1.6 h
REM    4. train mrc   (VitalNet v4 region-MRC + low-light consistency, CV)  ~1.8 h
REM    5. evaluate    (CV system eval: ensemble + causal tracking; plus the
REM                    full deployed ensemble on the 7 never-trained people) ~40 min
REM    6. report      (reports\RESULTS.md + figures)
REM  Weights land in models\vitalnet\ (pix_fold0..4.pt, mrc_fold0..4.pt)
REM ==========================================================================
cd /d "%~dp0"
set "YOLO_CONFIG_DIR=%~dp0.cache\ultralytics"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"
set PYTHONIOENCODING=utf-8
if not exist reports\logs mkdir reports\logs
set EXCL=sub3,sub4,sub11,sub14,sub18,sub20,sub24

python scripts\preprocess_ubfc.py || goto :err
python scripts\audit_dataset.py || goto :err
python scripts\train_cv.py --version 1 --exclude_train %EXCL% --prefix pix_ --report reports\cv_results_pix.json || goto :err
python scripts\train_cv.py --version 4 --pair_w 0.5 --exclude_train %EXCL% --prefix mrc_ --report reports\cv_results_mrc.json || goto :err
python scripts\eval_tracking.py --models "models/vitalnet/pix_*.pt" --tag pix || goto :err
python scripts\eval_tracking.py --models "models/vitalnet/mrc_*.pt" --tag mrc || goto :err
python scripts\eval_tracking.py --models "models/vitalnet/pix_*.pt,models/vitalnet/mrc_*.pt" --tag system || goto :err
python scripts\eval_tracking.py --models "models/vitalnet/pix_*.pt,models/vitalnet/mrc_*.pt" --tag system_never_trained --never_trained %EXCL% || goto :err
python scripts\compare_pipelines.py --which both || goto :err
python scripts\make_report.py || goto :err
echo.
echo All done. Weights: models\vitalnet\   Reports: reports\RESULTS.md
pause
exit /b 0
:err
echo [!] A step failed - see the messages above.
pause
exit /b 1
