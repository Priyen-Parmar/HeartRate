# OptiPulse, explained from zero

This document explains the whole repository to someone who has never seen it: what the project does,
the physics behind it, every stage from camera frame to the number on screen, how the models were
trained and tested, what the results mean, what does **not** work well, and answers to the questions
people usually ask in a presentation.

Read sections 1–3 for the big picture (10 minutes). Sections 4–9 cover each part in detail. Section 12 is a
list of likely counter-questions with answers.

---

## Contents

1. [The project in one paragraph](#1-the-project-in-one-paragraph)
2. [The physics: why a camera can see your pulse](#2-the-physics-why-a-camera-can-see-your-pulse)
3. [The big picture: the pipeline at a glance](#3-the-big-picture-the-pipeline-at-a-glance)
4. [Tour of the repository (what every folder/file is)](#4-tour-of-the-repository)
5. [The live app, step by step (frame → number)](#5-the-live-app-step-by-step)
6. [The neural networks (VitalNet-Pix and VitalNet-MRC)](#6-the-neural-networks)
7. [Signal processing: from a waveform to BPM and SNR](#7-signal-processing-from-a-waveform-to-bpm-and-snr)
8. [The tracker, confidence and the honesty gate](#8-the-tracker-confidence-and-the-honesty-gate)
9. [Training: data, augmentation, losses, cross-validation](#9-training)
10. [Results and how to read them](#10-results-and-how-to-read-them)
11. [Real-world behaviour, limitations and the Mac test session](#11-real-world-behaviour-limitations-and-the-mac-test-session)
12. [Questions you will probably be asked (with answers)](#12-questions-you-will-probably-be-asked)
13. [How to run things](#13-how-to-run-things)
14. [Glossary](#14-glossary)
15. [Code index: where to find what](#15-code-index)

---

## 1. The project in one paragraph

**OptiPulse** measures a person's **heart rate (BPM) from an ordinary webcam or phone camera, without
touching them**. The general technique is called **rPPG (remote photoplethysmography)**. It started as a university course project.
A face detector (YOLOv8) finds the face in every frame. The last 10 seconds of face crops go into small
3D convolutional neural networks (**VitalNet**) that output a pulse waveform. The dominant frequency
of that waveform is the heart rate. Several networks are combined (an *ensemble*), and a small
probabilistic **tracker** (an HMM) smooths the estimate over time. A quality check (the **honesty gate**) decides whether
the signal is clean enough to show a number at all. The models were trained on the public
**UBFC-rPPG** dataset (≈50 people filmed with a finger pulse sensor as ground truth). On people the models never saw
during training, the error is **≈1.45 BPM** in normal light.

The 30-second pitch:

> "Each heartbeat pushes blood into the skin of your face, which changes its colour by about a third of a
> percent. That's invisible to us but measurable by a camera. We track the face, feed 10 seconds of video
> into a 3D-CNN trained to turn those colour changes into a pulse wave, take the frequency of that wave, and
> stabilise it with a tracker. On held-out people it's within 5 BPM of a medical finger sensor 96 % of the
> time in normal light. Low light and real webcams with auto-exposure are the hard part, and the app
> tells you when it can't measure instead of showing a wrong number."

---

## 2. The physics: why a camera can see your pulse

* **PPG (photoplethysmography)** is what a pulse oximeter on your finger does. It shines light into the skin and measures
  how much comes back. With every heartbeat the small blood vessels fill with more blood, and blood
  (haemoglobin) absorbs light, especially **green** light. So the reflected light dips slightly with every
  beat.
* **Remote PPG** does the same with ambient light and a camera. The facial skin gets slightly darker/redder
  with every beat. The change is **tiny**: about **0.2–1 % of the pixel brightness** on the UBFC
  dataset (≈0.3 % in our own Mac recording). One 8-bit pixel step is 0.4 % of mid-grey, so in one pixel
  the pulse is smaller than the camera's own noise and rounding.
* That is why we **average many pixels** (noise cancels out, the pulse does not) and **many frames** (we
  look at 10 s at a time). The network learns which pixels to trust: cheeks and forehead help, hair and
  background do not.
* **Heart rate = frequency.** If the skin colour oscillates 1.2 times per second, the heart rate is
  1.2 × 60 = **72 BPM**. We only look at frequencies between **0.65 and 3.2 Hz = 39–192 BPM** (the
  "HR band"). Anything outside cannot be a human resting/exercising heart rate and is filtered out.
* **What competes with the pulse:**
  * **Head motion** moves skin, shadow and background across pixels. These changes are much bigger than 0.3 %.
  * **Lighting changes**: flicker, auto-exposure and auto white-balance changes in the camera, a screen
    reflecting on glasses.
  * **Sensor noise** grows when it's dark.
  * **Video compression** (MPEG/H.264) throws away exactly this kind of tiny detail.

The rest of the project is about pulling a 0.3 % signal out of all that.

---

## 3. The big picture: the pipeline at a glance

```
 camera frame (e.g. 640x480 @ ~30 fps, with its capture timestamp)
        │
        ▼
 ① FACE DETECTION      YOLOv8 face detector (models/face_detector/…pt), runs 4–8x/s in a background thread
        │               → bounding box of the face
        ▼
 ② BOX SMOOTHING       BoxSmoother: removes detector jitter, still follows real head motion
        │
        ▼
 ③ CROP                square crop 1.6x the face box, resized to 80x80 RGB   (optipulse/face.py)
        │
        ▼
 ④ BUFFER              keep the last ~10.5 s of (timestamp, crop)                (optipulse/live.py)
        │   every 0.5 s (once ≥6 s are buffered):
        ▼
 ⑤ RESAMPLE TO 30 Hz   use the REAL timestamps to build a uniform 30 frames/s clip (300 frames for 10 s)
        │               → centre-zoom 80x80 → 72x72  (exactly as in training)
        ▼
 ⑥ NETWORKS            VitalNet ensemble (2 models on the Mac, up to 10 on an NVIDIA GPU)
        │               each outputs a pulse waveform: 300 numbers
        ▼
 ⑦ SPECTRUM            each waveform → detrend → band-pass 39–192 BPM → FFT power spectrum
        │               → raw HR (peak) + SNR (how dominant the peak is)
        ▼
 ⑧ TRACKER             HMM over 39–192 BPM combines this spectrum with the past ones
        │               → tracked HR + confidence
        ▼
 ⑨ DISPLAY LOGIC       lock when confident, limit how fast the number can move,
        │               honesty gate: hide the number if the pulse signal is too weak
        ▼
 ⑩ UI + CSV LOG        window with camera view, BPM, beating heart, signal bars, trend, waveform;
                        one CSV row per update in outputs/
```

The same code (`optipulse/` package) is used for **training, evaluation and the live app**, so the
model sees exactly the same kind of input in all three. This was a deliberate design rule. The old version of the project
broke because preprocessing differed between training and live use.

---

## 4. Tour of the repository

The folder names say what each part is for. You only need `optipulse/` and `models/` to **run** the
system. `pipeline/` and `data/` are needed only to **re-train and re-evaluate** it, and `archive/` is history.

```text
HeartRate/
├── README.md              start here: what it is, how to run it, headline results
├── requirements.txt       Python packages
├── optipulse/              THE SYSTEM: one Python package (library + live app; `python -m optipulse`)
├── models/                trained weights the app loads (10 VitalNet models + YOLO face detector)
├── launchers/             double-click / one-command starters
│   ├── windows/           run_optipulse.bat, run_phone.bat, run_phone_wifi.bat, find_cameras.bat, train_pipeline.bat
│   └── macos/             run_mac.sh
├── pipeline/              offline research pipeline: prepare data → train → evaluate → write report
├── results/               evidence: RESULTS.md (all tables), figures/, metrics/ (raw evaluation JSONs)
├── docs/                  reading material: EXPLAIN.md (this), METHOD.md, WORK_REPORT.md, MAC_GUIDE.md, HANDOFF_CONTEXT.md
├── archive/               NOT used by the current system: the old PhysFormerX system, old docs, dev experiments
├── outputs/               (created when you run the app; git-ignored) HR logs, raw recordings, screenshots
└── data/                  (git-ignored, not on every machine) the UBFC-rPPG dataset, only for training/evaluation
```

### Top level

| Path | What it is |
|---|---|
| `README.md` | User-facing readme: how to run, headline results, folder layout. |
| `requirements.txt` | Python packages: torch, numpy, scipy, opencv-python, ultralytics (YOLO), matplotlib. |
| `optipulse/` | **The system.** All real logic (sections 5–8) plus the live app itself. Run it with `python -m optipulse` from the repository root. |
| `models/` | **The trained weights that the app uses** (deployed models). |
| `launchers/windows/` | `run_optipulse.bat` (webcam), `run_phone.bat` / `run_phone_wifi.bat` (phone as camera via DroidCam, with raw recording), `find_cameras.bat` (which camera indices work), `train_pipeline.bat` (re-does **everything** from the raw dataset, ≈5 h on an RTX 3070 Ti). Each one first changes into the repository root, so it can be double-clicked. |
| `launchers/macos/run_mac.sh` | Mac launcher: creates `.venv` on first run, installs packages, runs the app on the Apple GPU (MPS) with 2 models. |
| `pipeline/` | Offline scripts that build the dataset cache, train, evaluate and write the report (table below). |
| `results/` | `RESULTS.md` (all result tables, generated), `figures/` (plots), `metrics/` (raw JSON results of every evaluation run, read by `make_report.py`). |
| `docs/` | `EXPLAIN.md` (this document), `METHOD.md` (paper-style method description), `WORK_REPORT.md` (diary of everything done and why), `MAC_GUIDE.md` (Mac setup + how to test well), `HANDOFF_CONTEXT.md` (notes for whoever continues the project). |
| `archive/` | Kept for reference only (see `archive/README.md`): `legacy_system/` (the previous PhysFormerX/Vidformer/VitalEyePro system: code, Gradio web demo in `deploy/`, its docs and figures), `experiments/` (one-off ensemble experiments), `trial_models/` (experiment checkpoints, local only). |
| `outputs/` | Created by the app: HR log CSVs, raw recordings, screenshots. **Contains videos of faces → private, never publish, git-ignored.** |
| `data/` | The UBFC dataset (raw ≈31 GB + preprocessed cache). **Not in git and not on the Mac**; needed only for training/evaluation, not for the live app. |

> ⚠️ Everything in `archive/legacy_system/docs/` (`Pipeline_Explanation.md`, `Vidformer_Presentation_Notes.txt`,
> `DATASET_DESCRIPTION_legacy.md`, `model_architecture.png`, `training_curves_*.png`, …) describes the **old**
> PhysFormerX/Vidformer system (a video transformer, MAE ≈6.8 BPM). The current system is VitalNet. Don't
> present those as the current design. `docs/METHOD.md` is the up-to-date method description.

### `optipulse/`: the system (single source of truth for training, evaluation and the live app)

| File | Responsibility |
|---|---|
| `__init__.py` | Global constants: `TARGET_FPS=30`, `CROP_SIZE=80`, `INPUT_SIZE=72`, `BOX_SCALE=1.6`, HR band `0.65–3.2 Hz` (39–192 BPM). |
| `__main__.py` | Makes `python -m optipulse` start the app. |
| `app.py` | **The live app.** Camera capture, face thread, engine, UI, CSV logging, recording, all command-line options. Also replays video files (`--video`) and UBFC subjects (`--ubfc subN`) for testing. |
| `paths.py` | Every folder the project reads or writes (models, results, outputs, data, archive) in one place. |
| `face.py` | `FaceDetector` (YOLOv8, falls back to OpenCV Haar cascade if YOLO is unavailable), `BoxSmoother`, `crop_square`. |
| `dsp.py` | Classic signal processing: detrend, band-pass, spectrum, HR + SNR estimation, uniform resampling, and the classical methods POS / CHROM / GREEN (used as baselines). |
| `model.py` | The neural network `VitalNet` (versions 1, 3, 4) and the two loss functions (`neg_pearson`, `SpectralKL`). |
| `tracking.py` | `HRTracker` (causal HMM) and `band_psd` (ensemble spectrum on a 1-BPM grid). |
| `live.py` | `load_ensemble` (loads checkpoints) and `HREngine`, the real-time engine: buffer → resample → models → HR → tracker → display logic. |
| `ui.py` | `LiveUI`: draws the window (camera view + instrument panel) with OpenCV. |
| `augment.py` | Training-time augmentations simulating bad webcams (low light, noise, motion, fps drops, different heart rates). |
| `data.py` | Loads the preprocessed dataset into memory, builds random training batches on the GPU, `to_model_input` (80→72 view). |
| `ubfc.py` | Readers for the raw UBFC videos and their ground-truth files (incl. timestamp clean-up). |
| `evaluation.py` | Scoring: 10-s windows, metrics (MAE, RMSE, r, …) and the deterministic stress tests (low light, motion, 12 fps). |

### `pipeline/`: offline research pipeline (in the order `launchers/windows/train_pipeline.bat` runs them)

| Script | What it does |
|---|---|
| `preprocess_ubfc.py` | Raw UBFC videos → face crops 80×80 at 30 Hz + aligned contact PPG → `data/processed/ubfc_v2/<subject>.npz`. |
| `audit_dataset.py` | Checks the quality of the ground truth. Flags 7 subjects whose finger-PPG is unreliable (sub3, 4, 11, 14, 18, 20, 24) → `results/metrics/dataset_audit.json`. |
| `train_cv.py` | 5-fold subject-wise cross-validation training. Produces `models/vitalnet/pix_foldK.pt` / `mrc_foldK.pt` and `results/metrics/cv_results_*.json`. |
| `eval_tracking.py` | Evaluates model sets with/without tracking, per condition → `results/metrics/tracking_*.json`. |
| `compare_pipelines.py` | Runs the **complete** live app (incl. face detection) on all UBFC videos, new vs old system → `results/metrics/pipeline_comparison.json`. |
| `make_report.py` | Builds `results/RESULTS.md` and `results/figures/` from the JSONs in `results/metrics/`. |

### `models/`

| File | What |
|---|---|
| `models/vitalnet/pix_fold0..4.pt` | 5 × **VitalNet-Pix** (version 1), ≈600 k parameters, ~2.4 MB each. |
| `models/vitalnet/mrc_fold0..4.pt` | 5 × **VitalNet-MRC** (version 4), ≈618 k parameters, ~2.5 MB each. |
| `models/face_detector/yolov8_face_nose_chest.pt` | YOLOv8 detector (50 MB) with classes face / nose / chest; only "face" is used. |

"fold K" means: this model was trained on 4/5 of the people and tested on the remaining 1/5 (fold K).
Each `.pt` file also stores which subjects it was trained and tested on.
---

## 5. The live app, step by step

Entry point: `python -m optipulse` → `optipulse/app.py` → `run()`. Threads involved:

* **Camera thread** (`Camera`): reads frames as fast as the camera delivers them and stamps each with
  `time.perf_counter()` (a precise clock). It keeps a queue of up to 45 frames and drops the oldest if the
  main loop falls behind.
* **Face thread** (`AsyncFace`): runs YOLO on the newest frame ~4–8 times per second (when not on CUDA),
  so the slow detector never blocks the video.
* **Inference thread** (`HREngine.infer_async`): runs the networks in the background. If the previous
  inference is still running, the next request is skipped.
* **Main thread**: glues everything together and draws the UI (≤20 Hz).

### 5.1 Camera capture and timestamps
Webcams almost never deliver exactly 30 fps. They deliver 24, 27, sometimes bursts with gaps. Our Mac
session got **24 fps average with 137 gaps of 0.1–0.35 s**. The frequency → BPM conversion needs
an accurate time axis. If you *assume* 30 fps while the camera delivers 24, every heart rate comes out
30/24 = 25 % too high. That's why **every frame keeps its real capture timestamp**, and why duplicate
frames (same image delivered twice) are skipped.

### 5.2 Face detection and smoothing (`optipulse/face.py`)
* `FaceDetector`: YOLOv8 at input size 320, confidence threshold 0.30, and the face box with the highest confidence is taken.
  Runs on CUDA/MPS/CPU. If YOLO fails to load, it falls back to OpenCV's Haar cascade.
* Raw detector boxes jitter by a few pixels from frame to frame. A jittering crop shifts skin/background pixels and
  creates fake brightness changes much larger than the pulse. `BoxSmoother` is an **adaptive
  exponential moving average** over (centre x, centre y, size):
  * small displacement (jitter) → smoothing factor 0.12 (very smooth);
  * large displacement (≥8 % of face size, i.e. real head motion) → 0.75 (follows quickly);
  * size changes are limited to factor ≤0.25 (zoom jitter is especially harmful).
* Between two detections the last box is reused, and the smoother still updates every frame.

### 5.3 Crop (`crop_square`)
A **square 1.6× larger than the face box** is cut out and resized to **80×80 RGB** with an affine warp
(replicate border if the face is near the image edge). 1.6× includes forehead, cheeks, chin and a little
hair/background. **No hard skin mask is used.** The old system used a YCbCr skin mask, whose flicker
from frame to frame was itself a big noise source. Instead, the network learns a *soft* skin mask
(attention, section 6).

### 5.4 Buffer (`HREngine.push`)
The engine keeps the last ~10.5 s of `(timestamp, crop)`. If the gap between two frames is > 1 s (face
lost, camera stalled), the buffer is **cleared** and the measurement restarts, because a signal with a
hole in it would give a wrong spectrum.

### 5.5 Resample to 30 Hz (`HREngine.infer`)
Every 0.5 s (`--update`), once ≥6 s are buffered (`--min_window`):
1. Take the last ≤10 s (`--window`).
2. If the camera runs faster than ~39 fps (phones at 60 fps), **average** frames that fall into the
   same 1/30-s slot (`bin_average`) instead of discarding them. Averaging k frames lowers noise by √k.
3. **Linearly interpolate** the crops onto an exact 30 Hz time grid using the real timestamps
   (`dsp.resample_uniform`). Result: a clip of 300 frames × 80×80×3.
4. `to_model_input`: deterministic centre zoom 0.93 → **72×72**. That is the middle of the zoom range used
   during training, so the live input looks like an average training input.
5. Converted to float32 (needed because Apple's MPS GPU cannot hold float64).

### 5.6 Networks
Each loaded model gets the clip `[1, 3, 300, 72, 72]` and returns a waveform `[300]`. Details in
section 6. Ensemble size:
* NVIDIA GPU: default 10 models (5 Pix + 5 MRC).
* Mac/CPU: 2 models (`--ensemble 2` → `mrc_fold0` + `pix_fold0`). `load_ensemble` alternates the
  two families so any small subset stays balanced.
* Speed measured on an Apple M4: ~0.4 s per model per update → 2 models ≈0.9 s, 10 models ≈3.4 s.

The last model also returns its **attention map** (where it "looks"). The UI draws that as the warm glow
on the face (toggle with `h`).

### 5.7 Waveforms → HR (section 7)
`hr_from_waves(waves, 30, "median")`: every waveform is filtered and its spectral peak found. The **raw HR**
is the median over models, and the **SNR** is the median SNR. Separately, `band_psd(waves)` averages the
models' normalised power spectra on a 1-BPM grid and hands that to the tracker.

### 5.8 Tracker and display logic (section 8)
The tracker gives `tracked HR` and `confidence`. The engine then decides what the screen shows:
* **Lock**: no number until confidence ≥ 0.60 (or 20 s have passed).
* **Slew limit**: after locking, the displayed number moves at most **3 BPM per second** (9 BPM/s when
  confidence ≥ 0.75). This is a speed limit, not a pull towards any value.
* **Honesty gate**: the number is shown only if the **median SNR over the last 5 s ≥ −1.5 dB**.
  Otherwise the UI says "Pulse signal too weak – no reading". With `--always_show` (the Mac launcher's
  default), the number is shown anyway in **amber** as "Low confidence reading".

### 5.9 The window (`optipulse/ui.py`, `LiveUI`)
* **Left**: camera image. Corner brackets mark the face: blue = measuring, green = stable, amber = weak.
  The amber glow is the network's attention (which skin it reads the pulse from).
* **Right panel** (420 px):
  * Big BPM number and a heart icon beating at that rate.
  * **State chip**: *no face* / *measuring* / **Stable reading** (green: SNR ≥ 0 dB, confidence ≥ 0.6, 5-s
    SNR ≥ −1.5) / **Low confidence reading** (amber) / **Pulse signal too weak – no reading** (red).
  * **SIGNAL bars** (0–5): the 5-s median SNR compared with thresholds −4, −2, 0, +2, +4 dB, one bar per
    threshold passed. 1–2 bars = SNR between −4 and 0 dB = the pulse peak is barely above the noise.
  * 60-s HR trend, live pulse waveform, hints ("too dark", "low fps", "head is moving…", "light is behind
    you…"), footer with fps / device / number of models.
* Keys: `q` quit · `r` reset · `s` screenshot · `f` fullscreen · `h` attention overlay · `c` camera
  settings (Windows).

### 5.10 Logging and recording
* `outputs/hr_log_<time>.csv`: one row per inference with columns
  `t_s, hr_bpm` (displayed), `raw_hr_bpm` (this window's peak), `snr_db`, `confidence`, `rppg_fps`
  (effective camera fps), `shown` (1 = number was visible), `[gt_hr_bpm` when replaying UBFC`]`,
  `snr_avg_db` (5-s median), `head_motion`, `face_light`, `bg_light`. You can line this up with a
  chest strap / smartwatch / pulse oximeter afterwards.
* `--save_raw auto` writes the **untouched** camera frames losslessly (`raw_<time>.mkv`, FFV1 codec,
  ~100 MB/min) plus `raw_<time>_timestamps.csv`. Such a recording can be replayed exactly with
  `--video outputs/raw_<time>.mkv`, which picks up the real timestamps automatically.
* `SceneMonitor` (added in the Mac session) measures head motion (face-centre movement over 4 s relative to
  face size; "moving" if > 0.10) and backlight (face brightness < 0.65 × background brightness). These
  drive the hints and the extra CSV columns.

---

## 6. The neural networks

File: `optipulse/model.py`. Both deployed families are the same class `VitalNet` with a different `version`.

### 6.1 Input normalisation (inside the network)
`normalize_inputs` turns the RGB clip `[B,3,T,72,72]` in [0,1] into **6 channels**:
* **Motion stream (3 ch)**: *normalised frame difference* `(x[t+1] − x[t]) / (x[t+1] + x[t])`. This is
  the relative change of every pixel from one frame to the next, which is exactly where the pulse lives. Because
  it's a ratio, it doesn't care about overall brightness (×0.1 darker gives the same values), which helps in low
  light. It's then scaled to unit std and clipped to ±6.
* **Appearance stream (3 ch)**: the frames standardised (mean 0, std 1). This tells the network *what* is
  where (skin, eyes, hair, background).

Putting normalisation inside `forward()` means training and live inference **cannot** normalise differently.
(This idea is from DeepPhys, Chen & McDuff 2018, and TS-CAN, Liu et al. 2020.)

### 6.2 VitalNet-Pix (version 1), the "pixel" model
Two parallel 3D-CNN towers (3D = convolutions over time *and* space):

```
motion  [3,T,72,72] ─stem(stride 2)→ [16,T,36,36] ─2×Conv3d 3x3x3→ [32,T,36,36] ─×mask1─pool→ [32,T,18,18]
                                                                                   ▲
appearance [3,T,72,72] ─stem→ [16,T,36,36] ─2×Conv(1x3x3)→ [32,T,36,36] ──Attention──┘ mask1 (soft skin mask)
                                                         ─pool→ ─2×Conv→ [64,T,18,18] ──Attention── mask2
motion ─2×Conv3d→ [64,T,18,18] ─×mask2─pool→ [64,T,9,9] ─2×Conv3d (temporal dilation 2,4)→ spatial mean → [64,T]
        → temporal head: 4 residual 1-D conv blocks (dilation 1,2,4,8) → 1×1 conv → waveform [T]
```

* **Attention** (`Attention` class): a 1×1 conv + sigmoid on the appearance features gives a value in [0,1]
  for every pixel, normalised to mean 0.5. Multiplying the motion features by it means the network learns
  **which pixels are skin with a usable pulse** and ignores the rest. This replaces a hand-made skin mask.
* The **temporal head** keeps full time resolution (no pooling in time), and the dilations let it see
  ~±1 s of context.
* Output is made zero-mean. Its scale is arbitrary; only the *shape/frequency* matters.
* Building blocks: Conv3d + BatchNorm + ELU, dropout 0.2.

### 6.3 VitalNet-MRC (version 4), the "region / maximal-ratio-combining" model
Version 1 **plus** a second path designed for low light:

* **Problem it addresses**: in the dark the pulse in a single pixel is far below the sensor noise (≈40 dB
  below). The pixel path applies non-linear layers (BatchNorm + ELU) *before* enough pixels are averaged, and
  non-linearities applied to noise-dominated values partly destroy the weak pulse.
* **RegionBranch**: first **average linearly** (in float32, because bf16 has only 0.4 % precision, larger than the
  pulse) the pixels inside regions, weighted by the learned skin mask, on a **6×6, 3×3 and 1×1 grid**
  (36 + 9 + 1 = 46 regions). Averaging N pixels reduces noise by √N *before* any non-linearity.
* Each region trace is diff-normalised and divided by its own noise level, and a shared dilated 1-D CNN processes
  each region. A softmax produces **time-varying reliability weights per region**, and the regions are summed with
  those weights. This is a learned version of **maximal-ratio combining** (DistancePPG, Kumar et al. 2015:
  weight each region by how clean it is).
* **Noise-gated fusion**: the network measures how noisy its own input is (pixel noise from the 2nd temporal
  difference; region noise level) and a tiny MLP turns those 2 numbers into 2 gates. These gates weigh the pixel path
  against the region path. In the dark it can lean on the region path.
* Trained with an extra **cross-illumination consistency loss** (section 9.5).

### 6.4 Why two families and an ensemble?
* MRC alone is *not* better than Pix in normal light (1.89 vs 1.42 BPM), but they **fail in different
  windows**. Combining them reduces motion error by 12 % and low-light error by 10 % before tracking.
* The 5 folds of each family were trained on different subsets of people, so they also make somewhat
  different errors, and averaging their spectra reduces variance.
* The ensemble combines **spectra** (or per-model HR estimates), never the raw waveforms: two models can output
  the same pulse with a different phase/sign, and averaging those waveforms would cancel them out.

### 6.5 Version 3 (rejected)
A temporal pre-filter before differencing. It was worse on the development fold (normal light 2.14 vs 1.06), so it was
dropped. It's still in the code because the `version` switch is shared.

---

## 7. Signal processing: from a waveform to BPM and SNR

File: `optipulse/dsp.py`. The **same** functions are used for the network output **and** for the
finger-sensor ground truth, so both are measured with the same ruler.

1. **`detrend`** (smoothness-priors method, Tarvainen 2002, λ=100): removes slow drift (e.g. lighting
   slowly changing).
2. **`bandpass`**: Butterworth filter order 2, 0.65–3.2 Hz, applied forwards and backwards (`filtfilt`, no
   phase shift).
3. z-score (mean 0, std 1).
4. **`spectrum`**: Hann window + FFT with zero-padding to ≥4096 points → power spectrum with
   0.44 BPM bin spacing.
5. **`hr_from_signal`**: the highest peak within 39–192 BPM, refined between bins with parabolic
   interpolation → HR.
6. **SNR** (definition from de Haan & Jeanne 2013): power within ±0.1 Hz (±6 BPM) of the peak **and**
   ±0.2 Hz of its first harmonic (2× frequency), divided by all other power in the band, in dB.
   * **0 dB** = as much energy in the pulse peak as everywhere else.
   * **+5 dB** = clean pulse.
   * **−3 dB** = the "peak" holds only a third of the energy. It could just be the biggest noise bump.
   * Typical UBFC (good conditions): clearly positive. Our backlit Mac recording: −4.5 … +0.5.

**Why 10-s windows?** Frequency resolution is about 1/window length: 10 s → 0.1 Hz = 6 BPM natural resolution.
Peak interpolation and averaging improve on that. Longer windows give finer/steadier estimates but react
slower and hold more motion. 10 s is the usual trade-off in rPPG papers.

**Classical baselines** (also in `dsp.py`): take the mean RGB of a central ellipse of the face and combine
the channels with a fixed formula:
* GREEN: just the green channel.
* CHROM (de Haan 2013): chrominance signals that cancel specular reflections.
* POS (Wang 2017): projects onto the plane orthogonal to skin tone, in 1.6-s windows.

They need no training but are **15–20 BPM MAE on UBFC** (heavily compressed videos). That is why
a learned model is needed.

---

## 8. The tracker, confidence and the honesty gate

### 8.1 Why a tracker?
In a hard 10-s window, a random noise peak is sometimes higher than the pulse peak (in simulated low light
this happens in ~35 % of windows). A single window cannot tell which peak is real. But **the true pulse peak
stays at about the same place window after window, while noise peaks jump around**.

### 8.2 How `HRTracker` works (`optipulse/tracking.py`)
It's a **forward HMM filter** (a Bayes filter) over the 154 possible heart rates 39, 40, …, 192 BPM:
* **Start**: uniform belief over all heart rates. **There is no prior towards 70–80 BPM.** 45 or 170 are
  equally possible.
* **Predict** (every update, time step dt): the belief is blurred by a Gaussian random walk with σ = 4 BPM per
  second. This encodes "heart rate is continuous": it can't jump 40 BPM in a second, but can drift anywhere
  (post-exercise recovery is ~1–2 BPM/s).
* **Update**: multiply by the evidence = the ensemble's normalised spectrum, sharpened (squared, α=2), with
  a 2 % uniform floor so a wrong past belief can always be overturned. Because consecutive 10-s windows overlap
  heavily, the evidence is **down-weighted** when updates come faster than once per second (otherwise the same data
  would be counted many times and the tracker would become falsely sure).
* **Output**: HR = posterior mean within ±3 BPM of the most probable value; **confidence = how much
  of the belief lies within ±3 BPM of it** (0–1).
* It is **causal**: it only uses the past, so it works live.
* Parameters were set from physiology, not tuned on test data. Inspired by TROIKA (Zhang 2015) and HMM trackers for
  wrist PPG.

### 8.3 Confidence vs SNR, an important distinction
* **Confidence** (tracker) = "the estimate is *consistent over time*".
* **SNR** (signal) = "this window *actually contains a clear periodic pulse*".

These are different. A steady noise source (e.g. a flickering light, or a periodic camera artefact) can be
consistent over time and give high confidence without any pulse. Our Mac test showed this: a **patch of
wall/shirt gave the same SNR as the face and even a higher tracker confidence**. That's why the gate uses SNR
and not confidence.

### 8.4 Honesty gate
Show a number only if the 5-s median SNR ≥ −1.5 dB. On the 43 reliable UBFC videos (full live
pipeline) this keeps **95 %** of readings and lowers their error from **1.59 → 1.17 BPM**. The hidden 5 % had an error of
**9.6 BPM**. The argument: *a confident-looking wrong number is worse than no number*. Caveat: the
threshold was chosen looking at UBFC, so 1.17 is somewhat optimistic.

`--always_show` turns the gate into a colour code (amber "Low confidence reading") instead of hiding the number.
That is useful for demos and debugging, but the amber numbers are not trustworthy.

---

## 9. Training

### 9.1 Dataset: UBFC-rPPG (Bobbia et al., 2019)
* 50 videos (Dataset 1: a few videos with "gt" files, Dataset 2: 42 subjects), 640×480, ~1–2 minutes each,
  people sitting indoors in front of a webcam. Some do a maths game to raise their heart rate.
* Ground truth: a **finger pulse oximeter** (CMS50E) recording the PPG waveform + its own HR reading.
* Heavily compressed video, mostly lighter skin tones (Fitzpatrick 2–3), indoor lighting.
* License/ethics rule: **subjects 21 and 27 must not appear in publications; never publish frames of
  sub27.**

### 9.2 Preprocessing (`pipeline/preprocess_ubfc.py`)
For every video: YOLO on every frame → same `BoxSmoother` → same 1.6× crop at 80×80 → read ground truth
(`ubfc.py`; Dataset 2's logged timestamps jitter in an 11/44 ms pattern, so a linear fit is used
instead) → resample frames **and** PPG to a uniform 30 Hz timeline → save as `.npz`
(`frames [N,80,80,3]`, `ppg`, `sensor_hr`, `box`, …). This is exactly what the live app does, but offline.

### 9.3 Ground-truth audit (`pipeline/audit_dataset.py`)
Some finger recordings are bad (sensor slipped, flat lines, oximeter HR disagrees with its own waveform). A subject is
flagged if, in >30 % of 10-s windows, the oximeter HR and the PPG-derived HR disagree by >10 BPM, or the
PPG's SNR is < 2 dB. Flagged: **sub3, sub4, sub11, sub14, sub18, sub20, sub24**. They are **excluded
from training**, and results are reported for the **43 reliable** subjects (flagged ones are listed
separately). Training on wrong labels teaches wrong things, and testing on them punishes correct
predictions.

### 9.4 Augmentation (`optipulse/augment.py`, `data.py`)
Each augmentation targets one real failure mode:

| Augmentation | Why |
|---|---|
| **HR resampling** (60 % of clips) | Time-stretch the clip (×0.55–1.8) so the label HR becomes uniform in 45–180 BPM. UBFC is mostly 60–120. Without this, the model learns "always answer ~75". The old model had exactly this bias. |
| Low light (60 %) | Gain down to 6 %, gamma, **shot + read sensor noise**, 8-bit re-quantisation. |
| Colour cast, side light | Per-channel gain, brightness gradient across the face. |
| Illumination drift, exposure steps | Slow brightness changes (like auto-exposure) and sudden jumps. |
| Motion (70 %) | Smooth random per-frame translation/rotation/zoom, plus occasional jumps (detector re-lock). |
| Zoom / shift / rotation / flip | Variation in framing (zoom 0.86–1.0, so the live/eval view uses 0.93). |
| Blur (30 %) | Out-of-focus or denoised webcams. |
| FPS drop (30 %) | Simulate a 10–25 fps camera with timing jitter, re-interpolated to 30 Hz like the live app. |

### 9.5 Losses
* **Negative Pearson correlation** between predicted waveform and the label PPG: rewards the right *shape*
  regardless of scale/offset.
* **Spectral KL** (weight 0.5): compares the predicted and label power spectra on a 1-BPM grid (39–192). It
  rewards the right *frequency*, which is what we finally need.
* **Cross-illumination consistency** (MRC only, weight 0.5): each clip is shown twice with identical
  motion, once normally lit and once forced into low light. The dark view must reproduce the waveform the
  model itself predicts for the lit view (the lit prediction is the "teacher" and is not trained by this term).
  Why not just the label? The finger PPG arrives later than the face pulse (pulse transit time) and has a
  different shape. The model's own face-derived waveform is a better target for "ignore the
  lighting".

### 9.6 Optimisation (`pipeline/train_cv.py`)
AdamW, lr 1e-3 (1 epoch warm-up, then cosine decay), weight decay 0.02, gradient clipping 5, bf16 mixed
precision, batch 8 clips × 180 frames (6 s), 30 epochs × 150 steps, width 32, dropout 0.2, seed 776.
Fixed schedule, **last epoch is kept**: no early stopping on the test fold. ≈1.5 h per family on an RTX 3070 Ti
Laptop. Training needs CUDA. The live app does not.

### 9.7 Cross-validation, "fold" explained
The 50 subjects are split into 5 groups ("folds"). Model K is trained on the other 4 groups and tested on
group K. **Every subject is tested by a model that has never seen that person.** This is *subject-wise*
splitting: splitting by time within the same person would leak identity/skin/lighting and inflate
results. The deployed ensemble is simply all 10 fold models together. Note that the full ensemble has seen
everyone except the 7 flagged subjects, so testing it on UBFC videos only makes sense on those 7
("never-trained" check) or per fold.

---

## 10. Results and how to read them

Metric: **MAE** (mean absolute error) in BPM over non-overlapping 10-s windows, 43 subjects, 270 windows,
unseen subjects (5-fold CV). Stress conditions are simulated deterministically and identically for every method.

| Method | normal | low light | motion | low light + motion | 12 fps |
|---|---|---|---|---|---|
| POS (classical) | 20.56 | 25.97 | 29.17 | 27.64 | 22.74 |
| Old PhysFormerX monitor | 6.81 | – | – | – | – |
| VitalNet-Pix (1 model) | 1.42 | 10.91 | 4.06 | 11.47 | 1.50 |
| VitalNet-MRC (1 model) | 1.89 | 10.09 | 4.42 | 11.20 | 1.85 |
| Pix + MRC fusion | 1.46 | 9.77 | 3.59 | 10.72 | 1.43 |
| **Full system (+ tracking)** | **1.45** | **7.55** | **3.12** | **8.23** | **1.59** |

* Normal light: **96.3 % of windows within 5 BPM**, Pearson r = 0.948, RMSE 5.28. RMSE is much larger than
  MAE because a few windows are badly wrong (outliers), while most are within ~1 BPM.
* **Full live app end-to-end** on the raw videos (real face detection, real frame timing): **1.59 BPM**.
  With the honesty gate: **1.17** on the 95 % of readings shown.
* **Where the gains come from**: tracking is the biggest effect in hard conditions (Pix low light 10.9 → 8.0).
  The ensemble of the two families helps motion and low light. Normal light stays ~1.4 either way.
* **Why not 0?** The reference itself (finger sensor, 10-s spectral HR) is only good to about ±1 BPM, the
  pulse reaches the finger later than the face, and HR changes within a window. ~1 BPM is at the noise floor of
  the reference.
* Low light (simulated: 10× darker + sensor noise) remains much worse (7.5 BPM).
* Never-trained check (7 flagged subjects, full ensemble): 0.8 BPM on the 4 whose PPG is actually OK. In low light,
  tracking *hurt* on those 4. It's a small sample, so it's reported as such (`--no_track` exists).

Full tables, per-subject results and the development ablation: `results/RESULTS.md`. Diary of every
experiment and decision: `docs/WORK_REPORT.md`.

---

## 11. Real-world behaviour, limitations and the Mac test session

### 11.1 Known limitations (say these openly)
* **One dataset.** Trained and tested only on UBFC-rPPG: 50 mostly light-skinned people, indoor, seated,
  one camera type. Performance on other cameras, darker skin, outdoors or exercise is **not validated**. The next
  step would be cross-dataset tests (PURE, MMPD, UBFC-Phys) and unsupervised adaptation (SiNC-style).
* Low light and motion results are on **simulated** degradations, not real dark recordings.
* Real webcams do **automatic exposure / white balance / gain**. These adjustments change brightness by several percent, and
  the pulse is 0.3 %. UBFC's camera had these effects but trained models only partly learn to ignore them.
* Only heart rate. No HRV, SpO₂, blood pressure or respiration. **Not a medical device.**

### 11.2 The Mac test session (Nov 2026): why "low confidence" and "1–2 signal bars"
A 67-s session on a MacBook (Apple M4, built-in camera, 2-model ensemble) showed HR 76–99, SNR −4.5…+0.5 dB,
confidence 0.39–0.74 and mostly 1–2 signal bars. The analysis (details in `docs/WORK_REPORT.md` §13):

* The camera delivered **24 fps instead of 30**, in bursts (~19 % of frames lost). The app handles this through
  timestamps, but fewer frames means less signal.
* **Backlit**: the face was much darker than the background (skin ≈80 vs background ≈163 on 0–255).
  Auto-exposure was set by the bright background and kept adjusting: frame-to-frame gain swings of **−11 % to +8 %,
  i.e. ~30× larger than the pulse**.
* Glasses reflected the screen, and a beard covered part of the skin.
* **Control test**: running the same networks on a patch of wall/shirt (no pulse possible) gave the
  *same* SNR as the face (−2.4 vs −2.1 dB). So in that recording the pulse was indistinguishable from noise,
  and **"low confidence" was the correct verdict, not a bug**. Head motion was not the main cause: still
  segments were not better. Lighting/camera was.
* Over the whole session, all 10 models and the classical POS agreed on ~79 BPM. That's plausible, but
  no reference device was used, so it's unconfirmed.
* Tried and **rejected** (made things inconsistent or worse): smoothing the frame timestamps,
  compensating gain using the background.

**What was changed in the code** (so the app explains itself instead of silently showing weak readings):
* `SceneMonitor` in `optipulse/app.py`: detects head motion and backlight.
* New UI hints: *"Head is moving: rest your arms, sit back, keep still"* and *"Light is behind you: put a
  lamp/window in FRONT of your face"*.
* Signal bars now use the 5-s median SNR, so they are steadier.
* CSV gets `snr_avg_db, head_motion, face_light, bg_light`.
* `--video` replay uses the real timestamps saved with raw recordings.
* `docs/MAC_GUIDE.md`: lighting-first test instructions.

**How to get a good reading**: light source **in front** of the face (lamp or window behind the camera, not
behind you), glasses off, sit still with arms resting, 50–80 cm from the camera, wait 10–15 s. To *prove* it works,
record with `--save_raw auto` while wearing a smartwatch/pulse oximeter and note its reading every 15 s.

---

## 12. Questions you will probably be asked

**Q: How can a normal camera see the heartbeat at all?**
Blood absorbs light. Each beat increases blood volume in the skin and changes its colour by ~0.3 %. One pixel
can't see that, but averaging thousands of pixels over many frames can (section 2).

**Q: What exactly does the neural network output?**
A pulse waveform: 300 numbers for 10 s at 30 Hz. It does *not* output a BPM directly. BPM is computed from the
waveform's frequency with classical FFT. This keeps the HR estimate interpretable and lets us compute an
SNR.

**Q: Why not predict BPM directly with a regression head?**
Direct regression tends to collapse towards the average HR of the training set ("always ~75"). Predicting a
waveform and taking its frequency has no such pull. HR-resampling augmentation also removes the bias.

**Q: Did you hardcode or bias anything towards a normal heart rate?**
No. The HR band is 39–192 BPM, the tracker starts with a uniform prior, and the training labels are spread
uniformly over 45–180 BPM by augmentation. The display slew limit restricts *speed of change* (3 BPM/s), not
the value.

**Q: Why not just use the classical POS/CHROM algorithms?**
On UBFC's compressed videos they give 15–20 BPM MAE. Fixed formulas can't adapt to compression, which skin
pixels are useful, etc. We still use them as baselines and for sanity checks.

**Q: Why a 3D-CNN and not a transformer?**
The previous version of this project *was* a video transformer (PhysFormerX) and got 6.81 BPM end-to-end.
With only ~50 training videos, a small 3D-CNN (0.6 M parameters) with strong inductive bias (local
convolutions, attention mask) generalises better and runs in real time on a laptop.

**Q: How big are the models, and how fast?**
~0.6 M parameters, ~2.5 MB each. On an RTX 3070 Ti, 10 models run comfortably every 0.5 s. On an Apple M4 ≈0.4 s
per model, so the Mac uses 2 models. On CPU only ≈1.8 s per update with 2 models.

**Q: Is the 1.45 BPM result overfitted?**
Every subject is evaluated only by models that never saw that person (5-fold subject-wise CV). The schedule is fixed
with no early stopping or checkpoint picking on test data, and design decisions were made on fold 0 only. *But*
it's all one dataset, so generalisation to other cameras/people is unproven (section 11).

**Q: What's a "fold"? Why are there 10 model files?**
See 9.7: 5 data splits × 2 model families. The 5 models of a family were trained on different 80 % subsets of
people. In the app they're used together as an ensemble.

**Q: Why resample everything to 30 Hz?**
The networks were trained on 30 Hz clips, and BPM = frequency × 60 needs the true time axis. Cameras give 15–60
fps with jitter. Using real timestamps and interpolating to exactly 30 Hz makes the input match training and the
frequencies correct.

**Q: Why 10-second windows? Why does the first number take 8–15 s?**
You need several heartbeats to measure a frequency: 10 s ≈ 12 beats at 72 BPM. The first estimate is
computed after 6 s, and a number is shown once the tracker is confident (≥0.6), typically 8–15 s.

**Q: What does "confidence" mean? And "SNR"? And the bars?**
Confidence = tracker's belief that the HR is within ±3 BPM of its estimate (temporal consistency).
SNR = how much of the signal's energy sits at the pulse frequency vs elsewhere (is there a pulse at all?). Bars =
5-s median SNR vs −4/−2/0/+2/+4 dB (section 8.3, 5.9).

**Q: Why is the signal low / confidence low on my laptop?**
Usually the lighting: backlight + auto-exposure, too dark, screen reflections on glasses, or a low/unsteady frame rate.
Our test showed that in such conditions a wall patch looks as "pulsy" as the face, so the app is right to say
low confidence (section 11.2). Fix the lighting first.

**Q: What is the honesty gate?**
Hide the number if the 5-s median SNR < −1.5 dB. That removes 5 % of readings on UBFC and lowers error 1.59 → 1.17
(section 8.4).

**Q: What does the tracker do if my heart rate really changes fast (e.g. after running)?**
Its random-walk model allows ~4 BPM/s drift, and the display follows at 3 BPM/s, or 9 BPM/s when very
confident. Post-exercise recovery is about 1–2 BPM/s, so it's followed. A jump of 50 BPM in a second is
physiologically implausible and treated as noise.

**Q: What happens if I move or leave the frame?**
Moderate motion is handled (motion augmentation, box smoother; motion stress test MAE 3.1). Strong
motion makes the reading unreliable, and the app shows a "head is moving" hint. If the face disappears for >1 s the buffer
restarts.

**Q: Does it work for dark skin?**
Unknown. UBFC is mostly light skin. Darker skin absorbs more light, so the pulse signal is weaker. This is a known
rPPG fairness problem and a stated limitation.

**Q: Does it work with glasses / beard / makeup?**
Glasses cause reflections (bad if they reflect a screen), and a beard hides skin. The attention mask lets the model
use forehead/cheeks, but signal quality drops. Makeup also reduces signal.

**Q: What does the warm glow on the face show?**
The attention mask of one network: where it reads the pulse from. Usually cheeks and forehead. It's a
visual sanity check that the model looks at skin, not background.

**Q: Is any data sent anywhere?**
No. Everything runs locally. Logs and optional raw recordings stay in `outputs/` (git-ignored). Raw recordings are
videos of the face and must not be shared.

**Q: Can it measure SpO₂, blood pressure, HRV, breathing?**
Not in this project. SpO₂ needs calibrated multi-wavelength light. HRV needs beat-to-beat timing, which wasn't
validated. Breathing could be extracted in principle but isn't implemented.

**Q: Is it a medical device?**
No. It's a research/course prototype.

**Q: What's new compared to existing work?**
(Section 6, `docs/METHOD.md`.) The individual ideas are borrowed (DeepPhys attention, PhysNet 3D-CNN, DistancePPG
MRC, TROIKA-style tracking). The combination is ours:
- a linear-first region branch with learned maximal-ratio combining using the learned skin mask;
- noise-gated fusion of the two paths;
- cross-illumination self-consistency;
- a heterogeneous spectral ensemble;
- a prior-free causal HMM tracker;
- an SNR honesty gate.

**Q: Which part helped the most?**
Normal light: the learned network itself (POS 20.6 → Pix 1.4). Hard conditions: the tracker (low light
10.9 → 8.0 for Pix). Then the Pix+MRC ensemble (−10 to −12 %).

**Q: Why are 7 subjects excluded?**
Their finger-sensor ground truth is unreliable (section 9.3). They're excluded from training and reported separately,
not hidden.

**Q: What were the main problems with the old version?**
- It assumed fps instead of using timestamps.
- Preprocessing was inconsistent between training and live use.
- A hard skin mask flickered.
- It was biased to ~70–100 BPM.
- It was a transformer on tiny data.
- Result: 6.81 BPM end-to-end vs 1.59 now on the same videos.

**Q: How long does training take and can I reproduce it?**
`launchers/windows/train_pipeline.bat` on Windows with an NVIDIA GPU, ≈5 h total, needs the UBFC dataset in `data/raw/UBFC-rPPG/`.

**Q: What would you do next?**
- Validate against a reference device in real webcam sessions.
- Test on other datasets (PURE, MMPD, UBFC-Phys) and darker skin tones.
- Add unsupervised test-time adaptation (SiNC).
- Lock camera exposure/white balance where the OS allows it.

---

## 13. How to run things

```bash
# Mac (sets up .venv on first run; Apple GPU; 2 models; shows low-confidence readings in amber)
bash launchers/macos/run_mac.sh

# Any OS, manually (always from the repository root)
python -m optipulse --camera 0                 # webcam 0, all defaults
python -m optipulse --ensemble 2 --device cpu  # weak machine
python -m optipulse --list_cameras             # which camera indices work
python -m optipulse --save_raw auto            # also record raw video for later analysis
python -m optipulse --video outputs/raw_XXXX.mkv --headless --csv out.csv   # replay a recording
python -m optipulse --ubfc sub42 --weights models/vitalnet/pix_fold0.pt models/vitalnet/mrc_fold0.pt --headless  # score vs ground truth (needs data/)
python -m optipulse --no_track                 # disable the tracker
python -m optipulse --always_show              # show the number even when the signal is weak (amber)
```

Windows (in `launchers/windows/`, double-click or run from a terminal): `run_optipulse.bat`, `run_phone.bat` /
`run_phone_wifi.bat` (DroidCam phone camera), `find_cameras.bat`, `train_pipeline.bat` (retrain everything).
All options: `python -m optipulse --help`.

Important options: `--window 10` (analysis seconds), `--min_window 6`, `--update 0.5` (seconds between
updates), `--ensemble N`, `--device auto|cuda|mps|cpu`, `--cam_fps`, `--fourcc`, `--headless`, `--record`,
`--csv`.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **PPG / rPPG** | (remote) photoplethysmography: measuring blood-volume pulse optically (with contact / with a camera). |
| **BPM** | Beats per minute. HR (Hz) × 60. |
| **HR band** | 0.65–3.2 Hz = 39–192 BPM, the only frequencies considered. |
| **MAE** | Mean absolute error between predicted and true HR, in BPM. |
| **RMSE** | Root-mean-square error, which punishes big mistakes more. |
| **SNR (dB)** | Pulse-peak energy vs the rest of the band. 10·log10(ratio). 0 dB = equal. |
| **PSD / spectrum** | Power spectral density: how much energy at each frequency. |
| **Detrend / band-pass** | Remove slow drift / keep only the HR band. |
| **Ensemble** | Several models whose outputs are combined. |
| **Fold / cross-validation** | Splitting people into 5 groups. Each group is tested by a model trained on the others. |
| **Subject-wise split** | Train and test people never overlap. |
| **HMM tracker** | Hidden Markov model: a probability distribution over HR, updated over time. |
| **Attention mask** | Learned per-pixel weight (0–1) of where the pulse is read. |
| **Maximal-ratio combining (MRC)** | Combine noisy channels weighted by their quality. Borrowed from radio engineering. |
| **Diff-normalised** | (x[t+1]−x[t])/(x[t+1]+x[t]): relative frame-to-frame change, brightness-invariant. |
| **Auto-exposure** | Camera adjusting brightness automatically. A major noise source for rPPG. |
| **MPS** | Apple's GPU backend for PyTorch (Metal Performance Shaders). |
| **bf16** | 16-bit floating point used for speed on NVIDIA GPUs. Too imprecise for the region averaging. |
| **UBFC-rPPG** | The public dataset used (Bobbia et al., 2019). |
| **POS / CHROM / GREEN** | Classical, non-learned rPPG methods used as baselines. |

---

## 15. Code index

| I want to understand… | Look at |
|---|---|
| constants (fps, crop size, HR band) | `optipulse/__init__.py` |
| the app's main loop, threads, CLI options, CSV | `optipulse/app.py` → `run()`, `Camera`, `AsyncFace`, `SceneMonitor`, `RawRecorder`, `parse()`, `cli()` |
| where files are read/written | `optipulse/paths.py` |
| face detection / smoothing / crop | `optipulse/face.py` → `FaceDetector`, `BoxSmoother`, `crop_square` |
| buffering, resampling, ensemble, lock/slew/gate | `optipulse/live.py` → `HREngine.push`, `.infer`, `.showing`, `load_ensemble`, `bin_average` |
| the network | `optipulse/model.py` → `normalize_inputs`, `Attention`, `RegionBranch`, `VitalNet.forward` |
| losses | `optipulse/model.py` → `neg_pearson`, `SpectralKL`; consistency loss in `pipeline/train_cv.py` → `train_fold` |
| HR & SNR from a waveform | `optipulse/dsp.py` → `preprocess_pulse`, `spectrum`, `hr_from_signal`, `hr_from_waves` |
| tracker | `optipulse/tracking.py` → `HRTracker.update`, `band_psd` |
| drawing the window | `optipulse/ui.py` → `LiveUI.render` |
| augmentations | `optipulse/augment.py`; HR resampling in `optipulse/data.py` → `BatchBuilder.sample` |
| dataset loading / ground truth | `optipulse/ubfc.py`, `pipeline/preprocess_ubfc.py`, `pipeline/audit_dataset.py` |
| training | `pipeline/train_cv.py` (full run: `launchers/windows/train_pipeline.bat`) |
| evaluation & stress tests | `optipulse/evaluation.py`, `pipeline/eval_tracking.py`, `pipeline/compare_pipelines.py` |
| all numbers | `results/RESULTS.md` (raw JSON: `results/metrics/`); history of decisions: `docs/WORK_REPORT.md`; method write-up: `docs/METHOD.md` |
| the old system | `archive/legacy_system/` (see `archive/README.md`) |
