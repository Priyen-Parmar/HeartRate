# VitalEye – complete handoff context for the next Claude session

> **For the next Claude session:** read this whole file before doing anything. It summarises the
> previous long working session (2026-11-12 → 2026-11-13, Windows PC). The user is now continuing on a
> **Mac** with the repo cloned from GitHub. Paths written as `E:\CS776 Project\...` are on the Windows PC.
> On the Mac the repo root is wherever it was cloned (e.g. `~/CS776-Project`), and the **dataset and the
> raw test recordings are NOT there** (see §15).

---

## 0. TL;DR – where things stand

* **Project:** CS776 course project "VitalEye", contactless heart rate (rPPG) from a webcam/phone camera.
  The user wants it very accurate (~1 BPM MAE), robust to low light and motion, and the basis of a paper.
* **Built:** a new library `vitaleye/` with:
  * two 3D-CNN models: **Pix** (VitalNet v1) and **MRC** (VitalNet v4, region branch + noise-gated fusion
    + cross-illumination consistency);
  * a 10-model ensemble with spectral fusion;
  * a causal HMM HR tracker;
  * a live app with a new UI and an honesty gate.

  All of it was trained and evaluated on **UBFC-rPPG** with subject-wise 5-fold CV.
* **Dataset results are excellent:**

  | Test | MAE (BPM) |
  |---|---|
  | CV, normal light | **1.45** |
  | Live app end-to-end on 43 videos | **1.59** |
  | Readings actually shown (after the honesty gate) | **1.17** |
  | Simulated low light | 7.55 |
  | Simulated motion | 3.12 |
  | Old PhysFormerX system | 6.81 |

* **Real-world results on the user's own face are bad.** The pulse signal is weak (SNR ≈ −2 to −3 dB;
  the dataset has a median of +3.75 dB), so the app mostly shows "Pulse signal too weak" or
  low-confidence numbers.
  * Already fixed: speed bugs, frame drops, DroidCam issues, misleading hints, duplicate frames.
  * **Still unknown:** whether the weak signal comes from the camera pipeline (DroidCam compression,
    phone ISP), the distance (face too close), the user's skin tone, or model domain shift. **No
    reference (cardio machine) readings have been received yet.**
* **Current step:** the user is testing on a **Mac (8 GB RAM)** with `dist/VitalEye_mac.zip` /
  `mac/run_mac.sh`.
  * The first Mac run failed: `Cannot convert a MPS Tensor to float64`. **Fixed** (frames cast to float32;
    the startup MPS check now runs the whole engine), but **not yet re-tested on the Mac**.
  * The user starts a **new chat on the Mac** with this file as context.
* **Next priorities** (§13):
  1. Get a Mac (or phone-native) recording **with reference HR readings**.
  2. Diagnose: is a pulse present? Does the network or POS/CHROM get it right?
  3. Then act: camera-path fix, or unsupervised personal/domain adaptation (SiNC-style), plus
     multi-dataset training once MMPD/PURE/UBFC-Phys are obtained.

---

## 1. The user and how to work with them

* Git user **Priyen Parmar**, GitHub repo `https://github.com/Priyen-Parmar/HeartRate.git` (branch `main`).
* Writes in English, Hindi and Hinglish (often voice dictation, Devanagari). Answer in clear, simple
  English; short Hindi phrases are fine.
* Wishes and constraints they stated:
  * "Don't do any faaltu [useless] things, stay on the line", and at the end give a report of what was
    downloaded, created and how much memory/disk was used (this is kept in `reports/WORK_REPORT.md`).
  * Wants MAE ≈ 1 or lower; asked for "0" (explained: impossible, the reference itself is ±1 BPM).
  * **No hardcoding toward ~70 BPM, no overfitting.** They will verify live against a cardio/HR machine
    and expect e.g. ~120 after exercise to be read correctly.
  * Don't blindly implement one paper: **combine and take inspiration so it can become a new paper.**
  * **"Don't ask for too many permissions."** Act autonomously, but don't commit or push unless asked.
  * Wants the app to "give readings every time", but was also upset when wrong numbers (100+ while
    sitting) were shown. The compromise is the honesty gate plus the `--always_show` option (amber
    "low confidence").
  * Gets frustrated by vague answers. Give measured numbers, explain causes plainly, and give concrete
    next steps.
* Privacy: the user's face appears in `Screenshot 2026-11-12 143739.png`, `image.png` and the
  `outputs/raw_*.mkv` recordings (all git-ignored, all on the PC). Never publish them.

---

## 2. Machines

| | Windows PC (main) | Mac (testing now) |
|---|---|---|
| OS | Windows 11 | macOS (chip not confirmed, probably Apple Silicon) |
| GPU | RTX 3070 Ti Laptop 8 GB | Apple GPU (MPS) or CPU |
| RAM | 16 GB | **8 GB** |
| Python | `D:\miniconda3\python.exe` 3.13.5, torch 2.6.0+cu124, ultralytics 8.4.37, Pillow 12.3 | created by `run_mac.sh` in `.venv` |
| Has dataset? | yes (`data/`, 31 GB) | **no** |
| Cameras | built-in "ACER HD User Facing" (cam 0); Vivo X300 phone via **DroidCam** (cam 1, MSMF backend) | FaceTime camera (cam 0), possibly iPhone via Continuity Camera |

PC lessons:
* Run **one GPU job at a time**: two or three in parallel thrashed VRAM (7.8 GB) and were killed.
* PowerShell `*>` writes UTF-16 logs that grep can't read; use the JSON outputs.
* `scripts/live_monitor.py` has a **UTF-8 BOM**: read and write it with `encoding="utf-8-sig"`.

---

## 3. Data: UBFC-rPPG

* Public dataset (Bobbia et al. 2019). Raw: `data/raw/UBFC-rPPG/` with `vid_1_*` (dataset 1:
  `ds1_5gt…ds1_12gt`, `ds1_after_exercise`) and `vid_2_subN` (dataset 2). 50 videos in total, 640×480.
* **Licence/privacy rule:** subjects **21 and 27** of dataset 2 must not appear in publications. Never
  publish frames of sub27.
* The videos are heavily compressed MPEG-4 (~0.8 Mbit/s), which is why classical POS reaches only ~20 BPM
  MAE.
* Preprocessed cache: `data/processed/ubfc_v2/*.npz` (1.9 GB), built by `scripts/preprocess_ubfc.py`:
  * YOLO face box 1.6× enlarged → 80×80 RGB crops, timestamps de-jittered, resampled to 30 Hz;
  * contact PPG aligned to the frames;
  * a face was found in 100 % of frames.
* **Audit** (`scripts/audit_dataset.py` → `reports/dataset_audit.json`): unreliable references in
  **sub3, sub4, sub11, sub14, sub18, sub20, sub24**:
  * sub11/18/20/24: oximeter HR dropouts (their PPG waveform turned out usable);
  * sub3/4/14: noisy PPG.

  These 7 are excluded from training, and results are reported on the **43 reliable** subjects.
* The old `data/legacy/preprocessed_vidformer2` cache has a wrong fps (28.0 vs 28.67).
* UBFC skin types are mostly Fitzpatrick 2–3, indoors, seated, good light. **This narrowness is a prime
  suspect for the real-world failure.**

---

## 4. Repository layout (after reorganisation)

```text
CS776 Project/
├── run_vitaleye.bat        live app, laptop webcam (passes extra args through)
├── run_phone.bat           phone via DroidCam: --camera 1 --backend msmf --cam_fps 60 --save_raw auto
├── run_phone_wifi.bat      phone stream URL directly (http://<phone-ip>:4747/video), no DroidCam PC client
├── find_cameras.bat        lists working cameras and flags flat "no picture" feeds
├── train_pipeline.bat      full reproduction: preprocess → audit → train pix → train mrc → eval → report (~5 h)
├── requirements.txt, README.md, .gitignore
├── vitaleye/               library (single source of truth for training AND the live app)
│   ├── __init__.py         constants: TARGET_FPS 30, CROP_SIZE 80, INPUT_SIZE 72, BOX_SCALE 1.6, band 0.65–3.2 Hz (39–192 BPM)
│   ├── dsp.py              detrend, bandpass, spectrum, hr_from_signal (HR + SNR), hr_from_waves, resample_uniform, POS/CHROM/GREEN
│   ├── ubfc.py             UBFC readers, timestamps, GT interpolation, audit flags
│   ├── face.py             FaceDetector (YOLOv8 face/nose/chest, Haar fallback, MPS→CPU fallback), BoxSmoother, crop_square (returns RGB)
│   ├── model.py            VitalNet v1 (Pix), v3 (rejected), v4 (MRC); losses
│   ├── augment.py          GPU augmentation (low light, noise, colour, drift, motion, fps drop, paired low-light view)
│   ├── data.py             in-memory dataset, BatchBuilder (HR resampling, paired batches), to_model_input (80→72 view), EVAL_SCALE
│   ├── evaluation.py       10-s window eval + deterministic stress tests (lowlight, motion, ll+motion, 12 fps)
│   ├── tracking.py         HRTracker (causal HMM), band_psd, BPM_GRID
│   ├── live.py             load_ensemble (alternates families), bin_average, HREngine (engine, lock/slew, honesty gate)
│   └── ui.py               LiveUI (camera view + side panel)
├── scripts/
│   ├── preprocess_ubfc.py, audit_dataset.py
│   ├── train_cv.py         5-fold subject-wise CV (--version, --pair_w, --prefix, --only_fold …)
│   ├── eval_tracking.py    CV eval of model sets with/without tracking; --never_trained
│   ├── compare_pipelines.py  full live pipeline vs legacy monitor on all 50 raw videos
│   ├── ensemble_check.py, ensemble_check5.py   earlier studies
│   ├── make_report.py      → reports/RESULTS.md + figures
│   └── live_monitor.py     THE APP (webcam / phone / URL / video file / --ubfc scoring)
├── models/
│   ├── vitalnet/pix_fold0-4.pt, mrc_fold0-4.pt   deployed ensemble (~2.5 MB each; versioned in git)
│   └── face_detector/yolov8_face_nose_chest.pt   52 MB YOLOv8 (classes: face=0, nose, chest)
├── mac/run_mac.sh, mac/README_MAC.md            Mac test kit sources
├── dist/VitalEye_mac.zip   built Mac kit (git-ignored; rebuild from mac/ + vitaleye + scripts + models)
├── docs/METHOD.md          paper-style method write-up; this file (HANDOFF_CONTEXT.md); old notes
├── reports/                RESULTS.md, WORK_REPORT.md (sections 1–12 = full log), cv_results_*.json, tracking_*.json, pipeline_comparison*.json, figures/
├── outputs/                live logs hr_log_*.csv, raw_*.mkv recordings (git-ignored), pipeline_eval/
├── artifacts/trial_models/ every experiment's checkpoints (pix = run 1, mrc, run2_partial, v3_dev, v4_dev); git-ignored *.pt
├── artifacts/legacy_code/  old PhysFormerX / VitalEyePro / Gradio deploy (still runnable)
├── data/                   raw, processed, legacy (git-ignored, 31 GB)
└── other_projects/rideflow unrelated Node.js app that had been in the folder (moved, untouched)
```

---

## 5. Method (what the models are)

Full write-up with honest attribution: `docs/METHOD.md`.

**Input.** A 10-s window of 80×80 face crops at 30 Hz goes through `to_model_input`, a deterministic
80→72 view, giving a `[B,3,T,72,72]` tensor. The network outputs a 1-D pulse waveform.

**VitalNet v1 "Pix".** A 3D-CNN with an appearance-attention (soft skin mask) pixel path, DeepPhys/TS-CAN
style.

**VitalNet v4 "MRC"** (v1 plus three additions):
* **`RegionBranch`:**
  * attention-weighted region means on 6×6, 3×3 and 1×1 grids, computed **linearly in float32 before any
    non-linearity** (sensor noise drops by √N);
  * diff-normalised and noise-whitened per region;
  * a shared dilated 1-D CNN per region;
  * softmax reliability weights, i.e. a learned, time-varying maximal-ratio combining (inspired by
    DistancePPG and RhythmNet).
* **Noise-gated fusion:** `pixel_noise_level`, a gate MLP and a fuse conv. The network measures its own
  input noise and weighs the pixel path against the region path.
* **Cross-illumination consistency** (`--pair_w 0.5`): each clip is shown twice with the same geometry and
  motion; the second copy is forced into low light. The extra loss is
  `neg_pearson(pred_dark, pred_light.detach())`.

**Training** (`train_cv.py`, both families):
* settings: 30 epochs × 150 steps, batch 8, clip 180 frames, lr 1e-3, wd 0.02, loss neg-Pearson +
  0.5·spectral KL, width 32, dropout 0.2, seed 776, last epoch kept (no early stopping);
* the 7 audit-flagged subjects are excluded from training;
* augmentation:
  * HR resampling (time-stretch, so the label HR is uniform over 45–180 BPM; this is what removes the
    "always ~70–100" bias);
  * low light, shot/read noise, 8-bit requantisation;
  * colour cast, side light, illumination drift / exposure steps;
  * motion and jumps;
  * 10–25 fps capture re-interpolated to 30 Hz;
* folds are fixed (stored in `reports/cv_results_pix.json["folds"]`);
* run times: Pix 96 min, MRC 87 min; peak VRAM 3.3–3.4 GB.

**Ensemble + tracking (live and eval):**
* 5 Pix + 5 MRC models (one per fold);
* fusion: band-normalised PSD averaging (`band_psd`) plus a median HR across waves (`hr_from_waves`);
* **HRTracker** (causal forward HMM on a 1-BPM grid, 39–192 BPM):
  * uniform prior;
  * Gaussian random walk σ = 4 BPM/s;
  * evidence = PSD^α (α = 2) with a 2 % floor;
  * evidence tempered by `ref_hop_s = 1.0`: weight = clip(dt/1 s, 0.05, 1), so 0.5-s live updates don't
    double-count overlapping windows;
  * returns HR (posterior mean within ±3 BPM of the MAP) and confidence (posterior mass within ±3 BPM).

**Rejected:**
* run 2 (harder low-light augmentation);
* v3 (temporal prefilter stem);
* test-time blur (made low light worse: 14–18);
* the MRC model alone (worse than Pix in normal light);
* ensembling alone did not fix low light.

---

## 6. Results (all on UBFC; MAE in BPM, 10-s windows, 43 reliable subjects)

5-fold CV (`reports/RESULTS.md`, `reports/tracking_*.json`):

| Method | Normal | Low light | Motion | LL+motion | 12 fps |
|---|---|---|---|---|---|
| POS (classical) | 20.56 | 25.97 | 29.17 | 27.64 | 22.74 |
| Pix (v1) | 1.42 | 10.91 | 4.06 | 11.47 | 1.50 |
| MRC (v4 + pair) | 1.89 | 10.09 | 4.42 | 11.20 | 1.85 |
| Pix + tracking | 1.56 | 8.02 | 3.48 | 9.92 | 1.55 |
| MRC + tracking | 1.81 | 7.32 | 3.99 | 9.02 | 1.66 |
| Fusion, no tracking | 1.46 | 9.77 | 3.59 | 10.72 | 1.43 |
| **Full system** | **1.45** | **7.55** | **3.12** | **8.23** | **1.59** |

* Normal light, full system: 96.3 % of windows within 5 BPM, r = 0.948.
* Never-trained check (10-model ensemble on sub11/18/20/24):
  * plain: 0.79 / 11.66 / 1.16 / 15.79 / 0.79;
  * tracked: 1.29 / 15.29 / 1.34 / 16.07 / 1.62;
  * **tracking hurt low light here** (documented caveat).
* Fold-0 dev ablation (clean / lowlight / motion):

  | Setting | Clean | Low light | Motion |
  |---|---|---|---|
  | run1 | 1.06 | 8.61 | 3.76 |
  | v4 | 1.86 | 7.29 | 3.57 |
  | v4pair | 1.98 | 7.92 | 4.14 |
  | Pix + v4pair, tracked | 1.01 | 6.31 | 2.37 |

* Fold-0 low-light diagnosis: dark only 1.59; noise only 2.51; both 8.78. Low-light errors are random
  non-harmonic noise peaks (38 % of windows have error > 10).

**Live app end-to-end** (`scripts/compare_pipelines.py`: face detector → engine → display, frame by frame
on the raw videos, held-out fold models, ensemble of 2):

| Engine version | MAE (43 reliable) | RMSE | Within 5 BPM | All 50 |
|---|---|---|---|---|
| Old PhysFormerX monitor | 6.81 | – | ~63 % | 7.71 |
| New engine v1 | 1.76 | 6.20 | 96 % | – |
| New engine + lock/slew fixes | **1.59** | 4.88 | 95.9 % | 1.72 |
| + honesty gate (shown readings only, offline from logs) | **1.17** (95 % shown; hidden ones had MAE 9.6) | – | – | – |

* Median subject 1.02.
* First reading after a median of 6.5 s.
* Worst subjects: sub37 10.8 (weak pulse, low HR), ds1_6gt 8.2 (0.86 in another run).
* Quick regression commands used after every change:
  * `python scripts/live_monitor.py --ubfc sub42 --weights models/vitalnet/pix_fold0.pt models/vitalnet/mrc_fold0.pt --headless`
    must give **MAE 0.64**;
  * sub37 with fold-4 models must give **2.22** (shown 1.17, 53 % shown).

---

## 7. The live app (`scripts/live_monitor.py` + `vitaleye/live.py` + `vitaleye/ui.py`)

**Pipeline per frame:**
* `Camera` thread (real per-frame timestamps, queue **45 frames** so stalls are caught up, not dropped);
* duplicate-frame skip (DroidCam repeats frames);
* face detection:
  * GPU: every `det_every` frames (auto: 2 at 30 fps, 4 at 60 fps);
  * CPU/Mac: `AsyncFace` background thread at 4–8 Hz;
* `BoxSmoother` → `crop_square` (80×80 RGB) → `HREngine.push`.

**Every 0.5 s** (once ≥ 6 s are buffered), `infer_async` runs in a background thread:
* window 10 s;
* if the effective fps is > 39, `bin_average` averages the frames in each 1/30-s slot;
* `resample_uniform` to 30 Hz, **cast to float32** (MPS needs it);
* `to_model_input` → all models (bf16 autocast on CUDA) → waves;
* `hr_from_waves` (raw HR + SNR) and `HRTracker.update(band_psd(waves))` (tracked HR + confidence).

**Display logic** (HREngine):
* **Lock:** no number until tracker confidence ≥ 0.60 (or after 20 s).
* **Slew limit:** 3 BPM/s, ×3 (9 BPM/s) when confidence ≥ 0.75.
* **Honesty gate:** `showing` = locked AND 5-s median SNR ≥ **−1.5 dB**. Otherwise the UI says "Pulse
  signal too weak – no reading". `--always_show` displays the number anyway, in amber as "Low confidence
  reading".
* **UI states:** noface / measuring / good (green) / weak (amber) / nopulse (red).
* **Hints are measured:**
  * brightness < 60 → "Too dark";
  * face box × 1.6 > 95 % of frame height → "Too close";
  * face < 18 % of height → "Too far";
  * fps < 20;
  * otherwise "Weak pulse: keep head still, don't talk, relax face".

**UI** (`LiveUI`):
* left: camera with corner brackets; an amber glow shows the network's skin attention (computed on a
  96×96 grid, fast);
* right: 420-px panel with BPM, beating heart, state chip, signal bars, 60-s trend and pulse waveform;
* redraw ≤ 20 Hz;
* fonts: Segoe UI on Windows, Arial/Helvetica on macOS.

**Keys:** `q` quit, `r` reset, `s` screenshot, `f` fullscreen, `h` glow on/off, `c` camera driver dialog
(Windows only).

**CSV** `outputs/hr_log_<time>.csv`: `t_s, hr_bpm, raw_hr_bpm, snr_db, confidence, rppg_fps, shown`
(+ `gt_hr_bpm` in `--ubfc` mode). One row per finished inference.

**Main CLI options:**

| Option | What it does |
|---|---|
| `--camera N` or a URL | camera index, or a stream URL |
| `--backend auto/dshow/msmf/any` | camera API |
| `--fourcc auto` | MJPG on dshow, NV12 on msmf (MJPG on DroidCam/MSMF gave only 14 fps), none on macOS |
| `--cam_fps` | frame rate requested from the camera |
| `--device auto/cuda/mps/cpu` | auto: CUDA, else MPS only if a full-engine CPU-vs-MPS check matches within 2 %, else CPU |
| `--cpu` | force CPU |
| `--ensemble N` | number of models (alternating pix/mrc) |
| `--weights …` | explicit checkpoints |
| `--window`, `--min_window`, `--update` | analysis window, first reading, update interval |
| `--no_track` | plain per-window estimate instead of the tracker |
| `--snr_min` | SNR threshold for the "good" state |
| `--det_every` | face-detection interval |
| `--view_h` | camera view height |
| `--headless` | no window |
| `--record` | record the UI |
| `--save_raw auto` | lossless FFV1 `.mkv` + `_timestamps.csv` of the raw frames, written on a background thread; MJPG `.avi` fallback |
| `--csv` | log path |
| `--video` | offline video file |
| `--ubfc subN` | score against ground truth |
| `--always_show` | show low-confidence numbers in amber |
| `--list_cameras` | list working cameras and exit |

**Performance:**
* PC/CUDA: 10-model update ≈ 0.33 s.
* PC CPU: 2 models 1.8–2 s per update; 4 models 3.5–4.2 s; 10 models 8.3 s.
* YOLOv8 (52 MB) on CPU: 147 ms per frame, hence `AsyncFace`.
* Loop throughput after the fixes: **55.7 fps** at a 60 fps camera (recording + 10 models); 30.0 at 30.

---

## 8. Chronology of real-world tests and fixes

1. **Session 1** (laptop webcam, `hr_log_20261112_143647.csv`):
   * The display jumped 72 → 107 → 64 and "50 → 120"; the camera ran at only 15 fps; the user said the
     UI looked bad.
   * Fixes:
     * tracker evidence tempering;
     * confidence lock and slew limit;
     * YOLO every 2nd frame and UI ≤ 20 Hz;
     * a complete UI redesign.
   * Six-video check: sub37 24.3 → 2.2, ds1_6gt 5.7 → 0.9.
2. **Session at 20:15** (laptop, 30 fps):
   * The display showed 75–120 while the user sat. Diagnosis: signal SNR −3.1 dB (dataset +3.75), so the
     models found no pulse but numbers were still shown.
   * **Honesty gate added**, plus `--save_raw` and `--fourcc`.
3. **Phone via DroidCam** (Vivo X300; the user wanted no third-party app, but Android USB-webcam mode
   wasn't available on Vivo, so DroidCam was used):
   * DroidCam appears only via **MSMF as camera 1**.
   * Problems hit, in order:
     * green frames: the DroidCam client wasn't feeding the virtual camera; DroidCam's own fix is File →
       Exit and relaunch, or reboot;
     * error `0xC00D3704` when Google Meet held the camera (one consumer at a time);
     * the camera reopens without a forced format if it refuses one;
     * busy-camera message;
     * `find_cameras.bat` flags flat frames;
     * `run_phone_wifi.bat` URL mode.
4. **Phone session 23:55** (`hr_log_20261112_235507/235731`):
   * The app processed only **7–14 fps** with 0.6–5 s stalls, which restarted the window constantly.
   * Cause: the attention-glow overlay at full size (52 ms per redraw for the 330-px phone face).
   * Fixes: glow on a 96×96 grid (11.6 ms); queue 3 → 45; detection interval auto; buffer-restart gap
     0.6 → 1.0 s; recorder on a background thread.
5. **Phone session 2026-11-13 21:44** (`hr_log_20261113_214407.csv`, 3-min lossless
   `raw_20261113_214407.mkv`, 2 GB, on the PC only):
   * Steady 60 fps, but SNR median −2.7 dB; only 15 % of readings were shown.
   * Offline analysis:
     * face **343 px in a 480-px frame (too close; the 1.6× crop runs off the image; 17 % of frames had
       no face)**;
     * skin brightness 126/255, 0.2 % clipped (light is fine);
     * DroidCam sends ~28 new fps and **54 % of frames are exact duplicates** (now skipped; SNR effect
       only −2.85 → −2.71 dB);
     * no method finds a dominant spectral peak even averaged over 3 min:
       * **classical POS/CHROM/GREEN point to ~52–66 BPM**, the **network to ~80–94 BPM**;
       * the true HR is unknown (no reference readings given);
     * share of exactly unchanged skin pixels frame-to-frame: ≈ UBFC (both compressed), so not conclusive.
   * Changes: measured hints; `--always_show`; duplicate skip.
6. **Mac kit** (`dist/VitalEye_mac.zip`, 71 MB):
   * `run_mac.sh` creates `.venv`:
     * arm64: latest torch, numpy, scipy, opencv-python, pillow, ultralytics;
     * Intel: torch 2.2.2 + torchvision 0.17.2 + numpy<2, Python ≤ 3.12;
   * exports `PYTORCH_ENABLE_MPS_FALLBACK=1` and runs
     `live_monitor.py --camera 0 --ensemble 2 --always_show --save_raw auto`;
   * measured RAM ≈ 1.1 GB (CPU, 2 models + YOLO);
   * **first Mac run:** every inference failed with "Cannot convert a MPS Tensor to float64". Fixed in
     `vitaleye/live.py` (float32 cast). The startup check now runs the full HREngine on CPU and MPS and
     compares. A dispatch-mode scan on CUDA found no float64 GPU ops. **Needs a re-test on the Mac.**

---

## 9. The open problem: weak pulse on the user's face

Facts:
* Dataset videos: median signal SNR +3.75 dB, worst video −1.4 dB.
* User's sessions:

  | Session | Camera | SNR | Shown |
  |---|---|---|---|
  | 14:36 | laptop | −2.0 dB | – |
  | 15:06 | laptop | −2.5 dB | – |
  | 20:15 | laptop | −3.1 dB | 0 % ≥ 0 dB |
  | 21:44 | phone | −2.7 dB | 15 % |

* The tracker "confidence" only measures self-consistency over time, not whether a pulse is present (it
  was 0.8 at −4 dB). The SNR gate is what tells the two apart.

Hypotheses, not yet separated:
1. **Model domain shift:**
   * one dataset, one camera, light skin, bright even light;
   * the user is likely Indian (darker skin returns a weaker pulse; the npj Digital Medicine 2021 study in
     India/Sierra Leone found ~11 BPM errors for existing methods).
2. **Camera pipeline:**
   * DroidCam H.264 compression and frame duplication;
   * phone ISP (noise reduction, beauty, HDR, stabilisation);
   * laptop webcam MJPG and auto exposure/WB.
3. **Geometry:** face too close (phone) and the crop runs off the frame.
4. **Motion / talking** during the sessions (unknown).

**Deciding test:** a recording with **reference HR every ~15 s**, at a proper distance (face ≈ 1/3 of
frame height), Mac video effects off, and ideally also a 1-min **phone-native camera-app recording**
(no DroidCam), run offline via `--video`.
* If POS/CHROM find the reference HR with a decent SNR while the network doesn't → model/domain problem
  (adaptation, more data, or fuse a classical candidate).
* If nothing finds it → capture problem (camera path, light, distance).

The offline analysis scripts used are described in §14 (they lived in a temp scratchpad, **not** in the
repo; recreate them if needed).

---

## 10. Datasets researched for multi-domain training (none downloaded; all need requests)

| Dataset | Content | Why useful | Access |
|---|---|---|---|
| **MMPD** (Tsinghua, EMBC 2023) | 33 people, 11 h, **phone camera**, Fitzpatrick **3–6**, varied light and motion | best match for this user | signed release agreement emailed from an institutional address (tjk19@mails.tsinghua.edu.cn, cc yuntaowang@tsinghua.edu.cn), subject "MMPD Access Request – <institution>"; a 2-person educational subset exists |
| VitalVideos | ~850–900 people, all 6 skin types, PPG + BP | largest and most diverse | requests@vitalvideos.org; terms not public; VV-Medium ~1000 GB |
| UBFC-Phys | 56 people, talking/stress tasks | real motion | "free access", may need a form |
| PURE | 10 people, controlled head motion | standard cross-dataset test | email the authors |
| iBVP | RGB + thermal, PPG quality labels | clean labels | request |
| VIPL-HR | 107 people, 1150 videos, several devices incl. phone | multi-device | signed agreement |
| SCAMPS | 2,800 synthetic videos | pretraining, the only fully public one | GitHub danmcduff/scampsdataset |
| UCLA Diverse-rPPG / VITAL | 54 people, diverse skin tones | skin-tone balance | project page |
| RR-rPPG (arXiv 2606.21511) | 119 recordings, Indian skin tones | closest to the user | unverified release |

* rPPG-Toolbox (github.com/ubicomplab/rPPG-Toolbox) has loaders for most of these.
* The PC has ~99 GB free disk.
* Recommended order: request MMPD + PURE + UBFC-Phys through the user's professor. Train on UBFC + MMPD
  + UBFC-Phys, test on PURE and the user's recordings (cross-dataset = the honest metric).

---

## 11. Paper angle discussed

* The current contributions (`docs/METHOD.md`):
  * linear-first MRC region branch;
  * noise-gated fusion;
  * cross-illumination consistency;
  * heterogeneous spectral ensemble;
  * causal prior-free HMM tracking;
  * HR-resampling augmentation;
  * honesty gate.
* The stronger story going forward: **cross-dataset / cross-device generalisation** plus **unsupervised
  personal or domain adaptation** on the user's own unlabeled video (SiNC-style losses: bandwidth,
  sparsity, variance; or Contrast-Phys). It uses no reference labels, so it can't overfit to the numbers
  being checked.
* Evaluate with reference readings held out.

---

## 12. Mac instructions given to the user (`mac/README_MAC.md`)

* Unzip, then in Terminal: `cd ~/Downloads/VitalEye_mac && bash run_mac.sh`. The first run downloads
  ~1.5 GB into `.venv`; Python 3.11/3.12 is needed. Allow Terminal camera access (System Settings →
  Privacy & Security → Camera).
* **With the repo cloned from GitHub** (what the user is doing now): from the repo root run
  `bash mac/run_mac.sh`. The script detects whether it's in the zip kit or in `mac/` of a clone, and
  `cd`s to the folder holding `scripts/`. The `.venv` is then created in the repo root (git-ignored).
* Test protocol:
  * **Mac video effects OFF** (Portrait, Studio Light, Center Stage);
  * 50–70 cm, face ≈ 1/3 of the frame, lamp in front;
  * sit still, don't talk, 2 min;
  * note the **cardio machine reading + app time every 15 s**;
  * optionally a second run after 30 squats.
* Options: `--list_cameras`, `--camera 1`, `--ensemble 4`, `--device cpu`.
* Send back: `outputs/hr_log_*.csv`, `outputs/raw_*.mkv` (lossless face video, private) and the reference
  readings.

---

## 13. Next steps (prioritised)

1. **Mac re-test** after the float64 fix:
   * check the start-up line `N VitalNet model(s) on mps|cpu`;
   * if `[engine] inference error` appears again, run with `--device cpu` and inspect the traceback.
2. **Get reference readings.** Without them nothing can be scored. Ask for a written list (app time →
   machine HR).
3. **Analyse the recording** (re-create the scripts from §14):
   * per-window network HR/SNR vs POS/CHROM/GREEN;
   * long-window averaged spectra;
   * face size and brightness;
   * duplicate frames;
   * compare against the reference.
4. Depending on the result:
   * **Capture problem** → camera settings (exposure/WB lock), phone-native capture, distance, light.
   * **Model problem** →
     1. a classical-candidate fusion in the tracker (cheap; validate on UBFC so it doesn't hurt);
     2. **unsupervised test-time / personal adaptation** (SiNC-style) on the user's unlabeled
        recordings, validated against the reference readings that were *not* used in training;
     3. **multi-dataset training** once MMPD/PURE/UBFC-Phys arrive.
5. Keep `docs/METHOD.md`, `README.md` and `reports/WORK_REPORT.md` updated (the user wants a complete
   report of everything done, downloaded and used).
6. Optional cleanups:
   * METHOD.md could mention the live lock/slew/gate;
   * `train_cv.py` default `--version` is 4 (the pipeline passes explicit flags);

---

## 14. Offline analysis tools (recreate as needed; they were temporary scripts)

All of these read a raw recording: `<name>.mkv` + `<name>_timestamps.csv` (columns `frame,t_s`).

* **raw_anal:** YOLO + BoxSmoother + `crop_square` per frame with TRUE timestamps → per 10-s window:
  * the network (`HREngine(track=False).infer(list of (t, crop))` → `raw_hr`, `snr`);
  * classical methods: `bin_average` → `resample_uniform` to 30 Hz → `skin_rgb_means` →
    `pos_signal`/`chrom_signal`/`green_signal` → `hr_from_signal` (HR, SNR);
  * also prints face size, skin brightness and the share of clipped pixels.
* **long_psd:** averages `band_psd` over all windows per method and prints the top-3 peaks (×mean power),
  plus the share of exact duplicate frames.
* **compress_test:** share of exactly unchanged skin pixels between frames, periodic jumps, whole-frame
  brightness range (UBFC sub42/sub37 for comparison).
* **bench_loop / bench_prof:** a fake real-time camera serving frames from a recording, monkeypatched into
  `LM.Camera`, with `cv2.imshow`/`waitKey` stubbed. Runs `LM.run` and reports engine fps; it times
  `detect`/`render`/`infer`/`push`/`crop` via wrappers.
* `find_cameras.bat` / `--list_cameras`.

Notes:
* The detector returns boxes as `[x1,y1,x2,y2]`; `BoxSmoother` state is `(cx, cy, size)`.
* `crop_square(frame_bgr, state)` returns RGB.

---

## 15. What is and isn't in GitHub

* `.gitignore` was fixed on 2026-11-13.
  * **In git:** code, docs, reports (JSON/MD/figures), `models/vitalnet/*.pt`,
    `models/face_detector/*.pt`, `mac/`.
  * **Not in git:** `data/` (31 GB, licence), `outputs/*.csv|*.mkv|*.avi|*.png|*.mp4` (user logs and face
    recordings), `artifacts/**/*.pt` (trial checkpoints), `dist/`, `reports/logs/`, root
    `Screenshot*.png` / `image*.png` (the user's face), `.cache/`.
* As of writing, **nothing had been committed or pushed yet.** The user said they'd use GitHub; the
  commands were given to them.
* Consequence for the Mac session:
  * no dataset → no training, no `--ubfc` scoring, no `compare_pipelines`;
  * the user's PC logs and recordings aren't there either. Key numbers are in this file and in
    `reports/WORK_REPORT.md` §8–12.

---

## 16. Compute and disk used so far (for the user's "report everything" request)

* **No pip packages were installed on the PC**; the existing environment was used. Web: searches only,
  plus reading the rPPG-Toolbox paper/config and DroidCam pages.
* **Disk:** ~1.95 GB used (mostly `data/processed`), plus user test recordings in `outputs/` (~4.4 GB of
  raw `.mkv`) and `dist/` 71 MB.
* **GPU time:**

  | Job | Time |
  |---|---|
  | Preprocessing | 11 min |
  | Pix training | 96 min |
  | MRC training | 87 min |
  | Dev runs (run2, v3, v4) | ~60 min |
  | Evaluations | ~45 min |
  | End-to-end runs (50 videos) | ~2.5 h + 37 min |
  | Legacy monitor eval | 98 min |

* Peak VRAM 3.44 GB; RAM ~4 GB per training process; GPU 75–87 °C.
* Full details: `reports/WORK_REPORT.md`.
