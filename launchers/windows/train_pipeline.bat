@echo off
REM ==========================================================================
REM  OptiPulse - reproduce EVERYTHING from the raw UBFC-rPPG videos (RTX GPU)
REM  needs the dataset in data\raw\UBFC-rPPG\
REM    1. preprocess  (YOLO face crops + aligned contact PPG, 30 Hz)        ~12 min
REM    2. audit       (ground-truth quality report)                         ~10 min
REM    3. train pix   (VitalNet v1, 5-fold subject-wise CV)                 ~1.6 h
REM    4. train mrc   (VitalNet v4 region-MRC + low-light consistency, CV)  ~1.8 h
REM    5. evaluate    (CV system eval: ensemble + causal tracking; plus the
REM                    full deployed ensemble on the 7 never-trained people) ~40 min
REM    6. report      (results\RESULTS.md + results\figures\)
REM  Weights land in models\vitalnet\ (pix_fold0..4.pt, mrc_fold0..4.pt),
REM  metrics in results\metrics\
REM ==========================================================================
cd /d "%~dp0..\.."
set PYTHONIOENCODING=utf-8
set EXCL=sub3,sub4,sub11,sub14,sub18,sub20,sub24

python pipeline\preprocess_ubfc.py || goto :err
python pipeline\audit_dataset.py || goto :err
python pipeline\train_cv.py --version 1 --exclude_train %EXCL% --prefix pix_ --report results\metrics\cv_results_pix.json || goto :err
python pipeline\train_cv.py --version 4 --pair_w 0.5 --exclude_train %EXCL% --prefix mrc_ --report results\metrics\cv_results_mrc.json || goto :err
python pipeline\eval_tracking.py --models "models/vitalnet/pix_*.pt" --tag pix || goto :err
python pipeline\eval_tracking.py --models "models/vitalnet/mrc_*.pt" --tag mrc || goto :err
python pipeline\eval_tracking.py --models "models/vitalnet/pix_*.pt,models/vitalnet/mrc_*.pt" --tag system || goto :err
python pipeline\eval_tracking.py --models "models/vitalnet/pix_*.pt,models/vitalnet/mrc_*.pt" --tag system_never_trained --never_trained %EXCL% || goto :err
python pipeline\compare_pipelines.py --which both || goto :err
python pipeline\make_report.py || goto :err
echo.
echo All done. Weights: models\vitalnet\   Results: results\RESULTS.md
pause
exit /b 0
:err
echo [!] A step failed - see the messages above.
pause
exit /b 1
