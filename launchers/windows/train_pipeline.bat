@echo off
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
python pipeline\compare_pipelines.py || goto :err
python pipeline\make_report.py || goto :err
echo.
echo All done. Weights: models\vitalnet\   Results: results\RESULTS.md
pause
exit /b 0
:err
echo [!] A step failed - see the messages above.
pause
exit /b 1
