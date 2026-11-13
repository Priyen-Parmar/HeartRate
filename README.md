# VitalEye v2 — Contactless Heart Rate from a Webcam (CS776)

Real-time heart-rate (BPM) estimation from a normal webcam. A YOLOv8 face tracker feeds a
**10-model ensemble of two complementary rPPG networks**:
* **VitalNet-Pix**: a 3D-CNN with an appearance-attention pixel path;
* **VitalNet-MRC**: adds a linear-first, region-wise maximal-ratio-combining branch with noise-gated
  fusion, trained with cross-illumination consistency.

The two are fused in the spectral domain and decoded by a **causal HMM heart-rate tracker** that uses
no HR prior. Trained on UBFC-rPPG with heavy low-light / motion / heart-rate augmentation. Runs on an
NVIDIA GPU (tested: RTX 3070 Ti Laptop). Method write-up: **[docs/METHOD.md](docs/METHOD.md)**.

## Run it

```bat
run_vitaleye.bat                 :: double-click, webcam 0
run_vitaleye.bat --camera 1      :: another camera
```

Sit ~50–80 cm from the camera with your face lit from the front (a lamp or window in front of you,
not behind). The window shows:

* **left:** the camera. Corner brackets mark the tracked face (blue = measuring, green = stable,
  amber = weak signal). A warm glow marks the skin the network actually reads the pulse from;
  press `h` to toggle it.
* **right:** the panel.
  * Big BPM, with a heart beating at the measured rate.
  * A state chip, signal-quality bars, the 60-s heart-rate trend and the live pulse waveform.
  * Hints when it is too dark or the camera fps is low.

A number appears only once the tracker is confident (usually 8–15 s after you sit down). After that,
the displayed value changes by at most 3 BPM per second, or ~9 BPM/s when the tracker is very sure.
A resting heart cannot go from 50 to 120 in one second, but real changes such as exercise recovery are
followed. If the pulse signal itself is too weak (signal SNR below -1.5 dB, median over 5 s), **no number
is shown** ("No clear pulse"): a confident-looking wrong number is worse than none. On the dataset videos
this hides 5 % of readings and lowers the error of the shown ones from 1.59 to 1.17 BPM.
Keys: `q` quit · `r` reset · `s` screenshot · `f` fullscreen · `h` pulse-region overlay ·
`c` camera driver settings (Windows: turn off auto-exposure / auto white balance).

Every session writes `outputs/hr_log_<time>.csv` (time, HR, raw HR, signal SNR, tracker confidence, effective fps, `shown` = 1 if the number was displayed),
so you can line it up second-by-second against a chest strap / pulse oximeter / treadmill reading.

Other modes:

```bat
python scripts\live_monitor.py --video clip.mp4 --headless --csv out.csv   :: offline file
python scripts\live_monitor.py --ubfc sub42 --weights models\vitalnet\pix_fold0.pt models\vitalnet\mrc_fold0.pt --headless  :: score vs ground truth (fold-0 models never saw sub42)
run_vitaleye.bat --no_track      :: plain per-window estimate instead of the tracker
run_vitaleye.bat --save_raw outputs\me_rest   :: also save the raw camera video losslessly (+ timestamps) for offline diagnosis
run_vitaleye.bat --fourcc YUY2   :: force a camera stream format (default auto: MJPG for dshow, NV12 for msmf)
find_cameras.bat                 :: list every camera index / backend that works
run_phone.bat                    :: phone camera (DroidCam, camera 1 via MSMF) + lossless raw recording
run_phone.bat --always_show      :: always show the tracked value; weak-signal readings in amber ("Low confidence")
python scripts\live_monitor.py --cpu --ensemble 2   :: machine without an NVIDIA GPU (~1.8 s per update)
```

### On a Mac

`dist/VitalEye_mac.zip` (71 MB: code, 10 models, face detector) runs on macOS:

* unzip it and run `bash run_mac.sh`; the first run sets up its own Python environment;
* it uses the Apple GPU when its results match the CPU, otherwise the CPU;
* 2 models by default, which suits 8 GB RAM;
* face detection runs in a background thread.

Instructions: `mac/README_MAC.md`.

## Results (unseen subjects, 5-fold subject-wise cross-validation)

MAE in BPM over 10-s windows, 43 subjects with reliable ground truth. Every subject is scored only by
models that never saw that person.

| Method | normal | low light | motion | low light + motion | 12 fps webcam |
|---|---|---|---|---|---|
| POS (classical) | 20.56 | 25.97 | 29.17 | 27.64 | 22.74 |
| Old PhysFormerX monitor (normal video) | 6.81 | - | - | - | - |
| VitalNet-Pix (single model) | 1.42 | 10.91 | 4.06 | 11.47 | 1.50 |
| **VitalEye full system** | **1.45** | **7.55** | **3.12** | **8.23** | **1.59** |
| **Live app end-to-end on the raw videos** (face detector -> display) | **1.59** | - | - | - | - |

In normal light, 96 % of windows are within 5 BPM. 0 MAE is not physically reachable: the contact-PPG
reference itself is only good to about +-1 BPM. Low light remains the hardest case.

See **[reports/RESULTS.md](reports/RESULTS.md)** for all tables and figures and
**[reports/WORK_REPORT.md](reports/WORK_REPORT.md)** for everything that was done.

## Reproduce from scratch

```bat
train_pipeline.bat
```
= `preprocess_ubfc.py` → `audit_dataset.py` → `train_cv.py` (Pix, then MRC) → `eval_tracking.py`
(CV system + never-trained check) → `compare_pipelines.py` → `make_report.py`  (~5 h on an RTX 3070 Ti)

## Folder layout

```text
CS776 Project/
├── run_vitaleye.bat            one-click live monitor
├── train_pipeline.bat          rebuild data + retrain + evaluate everything
├── requirements.txt
├── vitaleye/                   the library (single source of truth for train AND live)
│   ├── dsp.py                  detrend / bandpass / HR estimator / POS, CHROM, GREEN
│   ├── ubfc.py                 UBFC-rPPG readers (video + contact PPG, timestamp handling)
│   ├── face.py                 YOLOv8 face detector, motion-adaptive box smoother, 1.6x crop
│   ├── model.py                VitalNet v1 (Pix) / v4 (MRC region branch + noise gate) + losses
│   ├── tracking.py             causal spectral HMM heart-rate tracker
│   ├── augment.py              GPU augmentations: low light, noise, motion, HR resampling, fps drop
│   ├── data.py                 in-memory dataset + GPU batch builder
│   ├── evaluation.py           window-level metrics + low-light / motion / low-fps stress tests
│   ├── live.py                 real-time engine (30 Hz resampling, ensemble, spectral fusion, tracking, display lock)
│   └── ui.py                   live window: camera view + instrument panel
├── scripts/
│   ├── preprocess_ubfc.py      raw videos -> data/processed/ubfc_v2/*.npz
│   ├── audit_dataset.py        ground-truth / label quality audit
│   ├── train_cv.py             5-fold CV training on the GPU
│   ├── eval_tracking.py        CV evaluation of model sets with / without tracking
│   ├── compare_pipelines.py    full live pipeline vs legacy monitor on all videos
│   ├── make_report.py          reports/RESULTS.md + figures
│   └── live_monitor.py         webcam / video app
├── models/
│   ├── vitalnet/pix_fold0-4.pt deployed ensemble, pixel models (~2.4 MB each)
│   ├── vitalnet/mrc_fold0-4.pt deployed ensemble, region-MRC models (~2.5 MB each)
│   └── face_detector/          YOLOv8 Face/Nose/Chest detector
├── data/
│   ├── raw/UBFC-rPPG/          50 raw videos + contact-PPG ground truth
│   ├── processed/ubfc_v2/      new 30 Hz face-crop cache (built by preprocess_ubfc.py)
│   └── legacy/                 old preprocessed_vidformer* windows, old crops, old zip
├── reports/                    RESULTS.md, WORK_REPORT.md, dataset audit, CV json, figures, logs
├── docs/                       METHOD.md (method write-up), old notes, old dataset description
├── outputs/                    HR logs, screenshots
├── artifacts/
│   ├── trial_models/           old checkpoints + every VitalNet experiment (run1, run2, v3, v4, mrc)
│   └── legacy_code/            old PhysFormerX / VitalEyePro code + old Gradio deploy (still runnable)
└── other_projects/rideflow/    unrelated Node.js project that was in this folder (moved, untouched)
```

Dataset: UBFC-rPPG (Bobbia et al., 2019). Subjects 21 and 27 of Dataset 2 must not appear in
publications; never publish frames of sub27.
