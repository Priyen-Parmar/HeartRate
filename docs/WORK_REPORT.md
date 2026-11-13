# Work report - everything that was done

Machine: Windows 11, RTX 3070 Ti Laptop (8 GB), 16 GB RAM, Python 3.13.5 (miniconda), torch 2.6.0+cu124,
ultralytics 8.4.37. Nothing is committed to git: all changes are in the working tree (moves were done with
`git mv`, so `git status` shows them as renames).

> **Paths:** sections 1–13 use the folder names of the time they were written. The repository was
> restructured on 2026-11-14 (section 14), e.g. `scripts/` → `pipeline/`, `reports/` → `results/`,
> `artifacts/legacy_code/` → `archive/legacy_system/`. Section 14 has the full old → new table.

## 1. Downloaded / fetched from the internet

| What | Size | Where it went |
|---|---|---|
| **No pip packages installed**: everything already present was used | 0 | - |
| 5 web searches (rPPG SOTA, UBFC protocols, low-light rPPG) | - | nothing saved |
| arXiv 2210.00716 (rPPG-Toolbox paper PDF) | 1.8 MB | Claude tool-result cache (`~/.claude/projects/...`), not in the project |
| rPPG-Toolbox `PHYSNET` config yaml + `post_process.py` (GitHub, read only) | few KB | not saved |
| Ultralytics settings files (created automatically on first YOLO import) | < 1 KB | `.cache/ultralytics/` |

## 2. Folder reorganisation

| Before | After |
|---|---|
| `Dataset/` (raw UBFC-rPPG) | `data/raw/UBFC-rPPG/` |
| `Dataset2/` (old face/nose/chest crops) | `data/legacy/crops_face_nose_chest/` |
| `preprocessed_vidformer{,1,2}/` | `data/legacy/` |
| `Dataset23.zip` | `data/legacy/Dataset23_crops_backup.zip` |
| all root `*.pt` (trial models) | `artifacts/trial_models/` (`best.pt` = duplicate of the deployed YOLO, renamed `yolov8_face_nose_chest_DUPLICATE.pt`) |
| old code (PhysFormerX, Vidformer1, VitalEyePro, live_hr_monitor, evaluate_model, preprocessor, launch scripts, `archive/`, `*_Deploy/`) | `artifacts/legacy_code/` (still runnable; your uncommitted edit of `live_hr_monitor.py` is preserved there) |
| `DATASET_DESCRIPTION.md` | `docs/DATASET_DESCRIPTION_legacy.md` |
| `rideflow/` (unrelated Node.js app) | `other_projects/rideflow/` (untouched) |
| YOLO face/nose/chest weights | copied to `models/face_detector/yolov8_face_nose_chest.pt` |
| deleted | `__pycache__/`, root `Ultralytics/` config folder, temporary `hud_test.mp4`, `outputs/hud_preview.png` (it showed a subject's face) |

## 3. Code created (new `optipulse` library + scripts)

| File | Purpose |
|---|---|
| `optipulse/__init__.py` | constants (30 Hz, 80->72 crops, 1.6x face box, HR band 39-192 BPM) |
| `optipulse/dsp.py` | detrend, band-pass, HR + SNR estimator, ensemble fusion (wave/psd/median), POS/CHROM/GREEN |
| `optipulse/ubfc.py` | UBFC readers, timestamp de-jitter, GT interpolation, audit flags |
| `optipulse/face.py` | YOLOv8 face detector (+ Haar fallback), motion-adaptive box smoother, crop |
| `optipulse/model.py` | VitalNet v1 (Pix), v3 (tried), **v4 (MRC region branch + noise-gated fusion)**, losses |
| `optipulse/augment.py` | GPU augmentation: low light, sensor noise, colour, drift, motion, fps drop, paired low-light view |
| `optipulse/data.py` | in-memory dataset, batch builder, HR-resampling augmentation, paired batches |
| `optipulse/evaluation.py` | 10-s window evaluation + deterministic stress tests |
| `optipulse/tracking.py` | **causal spectral HMM heart-rate tracker** (uniform prior, continuity only) |
| `optipulse/live.py` | real-time engine: timestamp resampling, 10-model ensemble, spectral fusion, tracking |
| `scripts/preprocess_ubfc.py` | raw videos -> `data/processed/ubfc_v2/*.npz` |
| `scripts/audit_dataset.py` | ground-truth audit (`reports/dataset_audit.json`, figure) |
| `scripts/train_cv.py` | 5-fold subject-wise CV training (`--version`, `--pair_w`, `--prefix`) |
| `scripts/eval_tracking.py` | CV evaluation of model sets with/without tracking; `--never_trained` mode |
| `scripts/compare_pipelines.py` | full live pipeline vs old monitor, frame by frame on the raw videos |
| `scripts/ensemble_check.py`, `ensemble_check5.py` | earlier ensemble studies (results kept in `reports/`) |
| `scripts/make_report.py` | builds `reports/RESULTS.md` + figures |
| `scripts/live_monitor.py` | webcam / video app with HUD and CSV log |
| `run_optipulse.bat` | one-click live monitor |
| `train_pipeline.bat` | one-click full reproduction (~5 h) |
| `requirements.txt`, `README.md`, `.gitignore` | rewritten |
| `docs/METHOD.md` | method write-up (what is new / borrowed, protocol, limitations) |

## 4. Data and models created

| Item | Size |
|---|---|
| `data/processed/ubfc_v2/` - 50 videos as 30 Hz 80x80 face crops + aligned PPG (face found in 100 % of frames) | 1.9 GB |
| `models/vitalnet/pix_fold0-4.pt` (deployed) | 5 x 2.4 MB |
| `models/vitalnet/mrc_fold0-4.pt` (deployed) | 5 x 2.5 MB |
| `artifacts/trial_models/vitalnet_pix/` (= run 1), `vitalnet_mrc/`, `vitalnet_run2_partial/`, `vitalnet_v3_dev/`, `vitalnet_v4_dev/` | ~24 MB |
| `reports/` (JSON results, logs, figures, RESULTS.md, this file) | ~4 MB |
| `outputs/pipeline_eval/*.csv` (per-video HR logs of the end-to-end evaluation), `outputs/test_sub42.csv` | < 1 MB |

**Disk:** 101.42 GiB free at start -> 99.47 GiB free now, i.e. **~1.95 GB used**, almost all by
`data/processed`. (`data/legacy` is 28 GB of *your old* files, only moved.)

## 5. Compute and memory used

| Job | Wall time | Peak GPU memory | Notes |
|---|---|---|---|
| preprocessing (YOLO on 50 videos) | 11.4 min | ~1 GB | |
| dataset audit | ~10 min | - | CPU |
| legacy monitor evaluation (old code, 50 videos) | 97.6 min | ~1 GB | 1.2 GB RAM |
| **Pix** training, 5 folds (run 1) | 96 min | 3.30 GB | ~3.9 GB RAM |
| run 2 (fold 0, then stopped) | 25 min | 3.30 GB | rejected |
| v3 (fold 0) | 17 min | 3.30 GB | rejected |
| v4 (fold 0) | 18.5 min | 3.44 GB | |
| **MRC** training, 5 folds | 87 min | 3.44 GB | |
| two failed parallel launches (VRAM thrashing, killed) | ~25 min | 7.8 GB | lesson: one job at a time |
| evaluations (ensemble checks, tracking, never-trained) | ~45 min | < 2 GB | |
| end-to-end live pipeline on 50 raw videos | ~2.5 h | < 2 GB | |

The GPU ran at 75-87 °C under load. RAM peaked at ~4 GB per training process (whole dataset cached in RAM).

## 6. Main findings

1. **Dataset check.** Video and ground-truth durations match for all 50 subjects.
   * 7 subjects have unreliable references: oximeter HR dropouts in sub11/18/20/24, and noisy PPG in
     sub3/4/14. They were kept out of training. Their PPG waveform was later shown to be usable for
     sub11/18/20/24.
   * The videos are heavily compressed MPEG-4 (0.8 Mbit/s), which is why classical methods fail (POS ~20 BPM).
   * The old `preprocessed_vidformer2` cache has a wrong fps (28.0 instead of 28.67).
2. **Old system:** 6.8 BPM MAE, frame by frame on the 43 reliable videos.
3. **New system (5-fold CV, unseen subjects):** normal light 1.45 BPM; low light 7.55 (single-model
   baseline 10.91); motion 3.12 (4.06); low light + motion 8.23 (11.47); 12 fps webcam 1.59. In normal
   light 96 % of 10-s windows are within 5 BPM.
4. **What worked:**
   * HR-resampling augmentation, so there is no bias towards typical heart rates;
   * the two-family ensemble (Pix + MRC) with spectral fusion;
   * causal tracking.
5. **What did not work** (tried and rejected, numbers in RESULTS.md section 3):
   * harder low-light augmentation (run 2);
   * temporal pre-filter stem (v3);
   * test-time blurring;
   * the MRC model *alone* (worse than Pix in normal light).
6. **0 MAE is impossible.** The reference (finger PPG) is itself only good to about +-1 BPM.

## 7. How to test it yourself against a reference device

1. Double-click `run_optipulse.bat`. Sit 50-80 cm from the camera, face visible, roughly still for the
   first ~6 s.
2. Wear a pulse oximeter or chest strap at the same time, and note its reading with the clock time.
3. Every session writes `outputs/hr_log_<date_time>.csv` (time, HR, raw HR, SNR, fps). Compare it
   second by second with the device. Readings with SNR >= 0 dB are the trustworthy ones (CV error
   1.8 BPM or less).
4. After exercise (e.g. 120+ BPM): nothing limits the reading. The band is 39-192 BPM, there is no prior,
   and the tracker follows changes of up to ~4 BPM/s.

## 8. Live-app fixes after the first real webcam test (2026-11-12, 14:37 session)

Your log `outputs/hr_log_20261112_143647.csv` showed the displayed HR jumping 72 -> 107 -> 64 within
seconds. The causes:

1. **Over-confident tracker.** The live app updates every 0.5 s on overlapping 10-s windows, so each
   piece of video evidence was counted ~20 times. The posterior became over-sure and then flipped
   between peaks. Fix: evidence is tempered by update interval / 1 s. This is exactly the setting that
   was cross-validated, and the offline evaluation numbers are unchanged.
2. **No confidence gate.** Readings were shown even at SNR -4 dB. Fix: a number appears only when the
   tracker posterior holds >= 60 % of its mass within +-3 BPM, or after 20 s, marked "Low signal".
3. **No physiological rate limit.** Fix: the display moves at most 3 BPM/s, or 9 BPM/s when the tracker
   is >= 75 % sure, so a wrong early value is corrected in seconds. Noise peaks never reach that
   confidence, so they cannot cause jumps.
4. **Camera only 15 fps.** The loop ran YOLO on every frame, drew the HUD on every frame, and shared
   the GPU with an evaluation job. Fix: detection every 2nd frame (the box is smoothed in between)
   and UI redraw capped at 20 Hz. Every frame is still used for the pulse.
5. **UI redesign** (`optipulse/ui.py`):
   * camera view with corner brackets, plus a glow over the skin regions the network uses;
   * side panel with BPM, beating heart, state chip, signal bars, 60-s trend and pulse waveform;
   * hints for low light and low fps.

Effect on held-out UBFC videos (same fold models, full live pipeline):

| Video | old engine MAE | new engine MAE |
|---|---|---|
| sub37 (low HR, weak pulse) | 24.31 | 2.22 |
| ds1_6gt | 5.68 | 0.86 |
| ds1_after_exercise (136 -> 88 BPM) | 1.31 | 1.37 |
| sub10 / sub23 / sub42 | 0.88 / 0.68 / 0.64 | 1.09 / 0.78 / 0.64 |

These 6 videos were used to choose the lock and catch-up thresholds, so the full 50-video re-run in
`reports/RESULTS.md` (section 6) is the number to trust. That re-run (37 min) gave:

* 43 reliable subjects: MAE **1.59** BPM (old engine 1.76), RMSE 4.88 (6.20), 95.9 % within 5 BPM.
* Median subject: 1.02 BPM.
* First reading after a median of 6.5 s (max 13.9 s).
* Old PhysFormerX monitor on the same videos: 6.81.
* Worst subjects: sub37 10.8, ds1_6gt 8.2. Both are sensitive to the first lock, and ds1_6gt scored 0.86
  in a separate run, so expect +-a few BPM run-to-run on such weak-signal videos.

## 9. Second webcam test (2026-11-12, 20:15 session): readings of 100+ while sitting

Log `outputs/hr_log_20261112_201555.csv`. The camera ran at 30 fps, but the displayed HR wandered 75-120 BPM.

**Diagnosis: no pulse was found in the video at all.**

| | median signal SNR | share of updates with SNR >= 0 dB |
|---|---|---|
| UBFC videos, live pipeline (50 videos) | +3.75 dB | most |
| worst UBFC video (sub37, MAE 10.8 before the gate) | -1.37 dB | - |
| your sessions 14:36 / 15:06 / 20:15 | -2.0 / -2.5 / -3.1 dB | 17 % / 0 % / 0 % |

At -3 dB the strongest frequency in the model output is noise, so the number was effectively random. The
tracker reported high "confidence" (0.8) anyway, because that value only measures self-consistency over
time, not whether a pulse is present. The frame rate was not the problem.

Likely causes, most likely first (none verified yet, a raw recording is needed):
1. **Domain shift.** All 42 training people come from one dataset with one camera, one room and bright
   even light, and mostly lighter skin tones. Darker skin returns a weaker pulse signal (more melanin
   absorption), and a laptop webcam has a very different sensor and processing.
2. **Camera processing.** Auto exposure / white balance / noise reduction keep changing the image. The app
   also requests a JPEG-compressed (MJPG) stream.
3. **Lighting geometry.** Overhead or back light gives little light on the cheeks and forehead.

Changes:
* **Honesty gate** (`optipulse/live.py`, `ui.py`): no number when the 5-s median SNR is below -1.5 dB; the
  UI shows "No clear pulse". On the 43 reliable UBFC videos (from the existing end-to-end logs), 95 % of
  readings remain and their MAE is 1.17 BPM (all readings: 1.59); the hidden ones had MAE 9.6. sub37
  re-run: 53 % shown, MAE of shown 1.17. The -1.5 dB threshold was picked on these same videos.
  It would have hidden 97 % of your 20:15 session.
* CSV column `shown`. `--ubfc` scoring prints the MAE of the displayed readings.
* `--save_raw <name>`: lossless FFV1 copy of the camera frames + exact timestamps, for offline diagnosis.
* `--fourcc YUY2`: uncompressed camera stream. Key `c`: Windows camera driver dialog (auto exposure / WB).
* CPU timing (no NVIDIA GPU, e.g. a MacBook): 10 models 8.3 s per update, 4 models 3.5 s, 2 models 1.8 s.

## 10. Phone camera (DroidCam) session, 2026-11-12 23:55: no readings

Logs `hr_log_20261112_235507.csv` and `_235731.csv`, with raw videos `raw_*.mkv`.

**Cause: the app loop was too slow, not the camera.** The phone delivered 60 fps, but the app processed
only 7-14 fps, with stalls of 0.6-5 s. Each stall over 0.6 s restarted the 10-s analysis window, so a
reading was almost never possible. Offline, the image itself was fine: face 185-330 px, skin brightness
116/255, 0-0.2 % clipped pixels.

Profiling (replaying the recording as a real-time camera into the full app):

| step | before | after |
|---|---|---|
| UI redraw, close phone face (330 px) | 52 ms | 11.6 ms |
| frames processed, 60 fps camera + raw recording + 10 models | ~11 fps | 55.7 fps |
| frames processed, 30 fps camera | ~10 fps | 30.0 fps |

Fixes:
* **Pulse-region glow computed on a 96x96 grid.** It was full size (~700x700 px for a close face) on
  every redraw.
* **Camera queue 3 -> 45 frames (~1 s).** Short stalls are caught up with true timestamps instead of
  dropped.
* **Face detection interval auto-set:** every 2nd frame at 30 fps, every 4th at 60 fps.
* **Window restarts only after a 1.0 s gap** (was 0.6 s).
* **Raw recording on a background thread.**
* **60 fps input is bin-averaged** to 30 Hz, so every frame is used.
* **Camera handling:**
  * `--camera` also accepts a stream URL (`run_phone_wifi.bat`);
  * `--backend msmf`;
  * the camera is reopened without a forced format if it refuses one;
  * a clear message when a camera is busy (e.g. Google Meet holding DroidCam);
  * `find_cameras.bat` flags flat-colour "no picture" feeds (DroidCam not streaming).

Regression check: sub42 (held-out fold models) still has MAE 0.64 BPM.

## 11. Phone session 2026-11-13 21:44 (after the speed fix): still mostly "no reading"

The loop now ran at a steady 60 fps (min 55), but the signal SNR median was -2.7 dB and only 15 % of
updates were shown. Offline analysis of the 3-min lossless recording `raw_20261113_214407.mkv`:

* **Image quality is fine:** skin brightness 126/255, 0.2 % clipped pixels. But the face is **343 px in a
  480 px frame** (too close): the 1.6x analysis crop runs off the image, and 17 % of frames had no face
  detected.
* **DroidCam sends ~28 new frames/s and repeats frames to fill 60** (54 % exact duplicates). Duplicates
  are now skipped (`[optipulse] skipped N repeated frames`). Effect on SNR: -2.85 -> -2.71 dB, i.e.
  negligible, so the duplicates are not the cause.
* **No method finds a dominant pulse peak,** even with spectra averaged over all 38 windows (top peaks are
  only 2-3x the mean power). Classical POS/CHROM/GREEN favour ~52-66 BPM; the network favours ~80-94.
  Without reference readings it is unknown which is right.

Changes:
* Hints now come from measurements: too dark (brightness), too close / too far (face size), low fps,
  otherwise "Weak pulse: keep head still, don't talk". The old "light face from front" message was shown
  even in good light.
* `--always_show`: the tracked value is always displayed after lock; weak-signal readings are amber
  ("Low confidence reading"). The default still hides them.

Open: whether the weak signal comes from DroidCam's stream compression, the phone's image processing,
distance, or the model (skin tone / camera domain). This needs one recording with reference readings,
plus one recording with the phone's own camera app (no DroidCam).

## 12. Mac test kit (2026-11-13)

`dist/OptiPulse_mac.zip` (71 MB) contains `optipulse/`, `scripts/live_monitor.py`, the 10 models, the YOLO
face detector, `run_mac.sh` and `README_MAC.md`. Source files are in `mac/`.

* `run_mac.sh` creates `.venv` on the first run: current PyTorch on Apple Silicon; torch 2.2.2 +
  numpy<2 on Intel Macs. It then starts the app with `--ensemble 2 --always_show --save_raw auto`.
* `--device auto`: CUDA, else the Apple GPU (MPS) only if a full live-path forward pass (80->72 crop,
  10-s window, attention) matches the CPU within 1 %, else CPU. YOLO falls back to CPU if MPS fails.
* CPU/MPS machines: face detection runs on a background thread at 4-8 Hz (YOLOv8 on CPU costs 147
  ms/frame here), and torch uses all cores but 2.
* Model subsets alternate the two families (2 models = mrc_fold0 + pix_fold0).
* RAM: ~1.1 GB measured (CPU, 2 models, YOLO). The raw-recorder queue is capped at 150 frames.
* macOS fonts are found; the Windows-only camera dialog key is disabled elsewhere.
* Tested on this PC in CPU mode:
  * replayed UBFC video: 30 fps processed, a reading every ~2-3 s;
  * unzipped package run standalone: works, SNR +3 to +7 dB on sub42.
* GPU regression: sub42 still MAE 0.64.
* Fix: on slow machines the CSV got a row per frame while inference was busy; now one row per inference.
* Not tested: a real Mac (no Mac available here).
* **Fix after the first Mac run (2026-11-13):** every inference failed with "Cannot convert a MPS Tensor
  to float64". Frame resampling returns float64; CUDA accepts it, but Apple GPUs do not. The frames are
  now cast to float32 before going to the device.
  * The startup MPS check now runs the complete live engine (uint8 crops -> resampling -> model -> HR) on
    CPU and MPS. It previously ran only the model on float32 input, which is why it missed this.
  * A dispatch-level scan of the engine on CUDA (30 and 60 fps paths) found no float64 GPU ops left.
  * Regression: sub42 0.64, sub37 2.22 (shown 1.17); both unchanged. The zip was rebuilt.

## 13. First real Mac session (2026-11-13 23:44, Apple M4, FaceTime camera): diagnosis

User report: "confidence is low and the signal stays at 1-2 bars". Recording: `outputs/raw_20261113_234414.mkv`
(67 s, 640x480, lossless) + log. **No reference HR readings were taken.**

What was measured (offline, same models/pipeline; scripts lived in the session scratchpad):
* Device: MPS chosen (passes the CPU check). Speed on M4: ~0.3-0.4 s per model per update -> 2 models
  0.9 s, 4 models 1.5 s, 10 models 3.4 s.
* Camera delivery: 1637 frames in 67 s (24 fps of a 30 fps camera, ~19 % lost); arrivals are bursty
  (45 % of intervals < 20 ms, 137 gaps of 0.1-0.35 s). A timestamp de-jitter was tried: +0.5 dB in one test,
  -0.45 dB in the full-app replay -> not a measurable gain, **not kept**.
* Picture: face 139 px (fine), no clipping, but **face skin brightness 80 vs background 163**: bright wall,
  light from the ceiling, nothing in front of the face. Glasses reflect the screen; a beard covers the lower
  cheeks, so the usable skin is forehead + upper cheeks + nose.
* Camera auto-exposure / white balance: the background brightness changes by -11 %..+8 % over the session.
  The pulse changes skin brightness by ~0.34 %, so exposure changes are 20-30x larger than the pulse.
  A background-gain correction was tried: it injected background noise (face, wall and shirt then all gave
  the same fake HR) -> **rejected**.
* **No-pulse control:** the same networks run on wall and T-shirt patches of the same video give the same
  per-window SNR (-2.4 dB) as the face (-2.1 dB) and an even higher tracker confidence. **So in this
  recording, a single 10-s window can't be told apart from a blank wall.** The low-confidence state and
  1-2 signal bars were therefore correct; showing it as "stable" would have meant showing noise as a
  reading.
* There is a weak but real hint of a pulse: averaged over the whole session, all 10 models peak at
  74-83 BPM (ensemble 79) and classical POS over 60 s also gives 79 BPM. Without a reference this can't
  be confirmed.
* Head movement: 18-43 px in 16-46 s; 4-9 px at 50-67 s. The still part was **not** better (SNR -2.8 to
  -3.1 dB in the replay), so lighting / camera, not movement, is the main cause here.

Code changes (diagnostic / UX only; HR estimation and the honesty gate are unchanged, so the UBFC numbers
stay valid):
* `scripts/live_monitor.py`: `SceneMonitor` measures head movement (face-centre range over 4 s / face
  size; flag > 10 %) and backlight (face skin < 0.65 x background brightness). Both are logged in the CSV
  (new columns `snr_avg_db, head_motion, face_light, bg_light`) and drive measured hints: "Head is
  moving..." / "Light is behind you: put a lamp/window in FRONT of your face".
* `--video <raw_*.mkv>` now replays a `--save_raw` recording with its **real** capture timestamps
  (`<name>_timestamps.csv`), so a live session can be reproduced offline exactly.
* `optipulse/ui.py`: signal bars show the 5-s median SNR (the value the gate uses) instead of the
  per-window value, so they no longer flicker.
* Verified: full-app replay of the Mac recording (headless + UI rendering), 102 updates, no errors; the
  backlight hint fires throughout, and the movement hint fires in the moving segments (61-67 % of updates)
  and never in the still ones.
* Not re-run here (no dataset on the Mac): the UBFC regression (sub42 0.64, sub37 2.22). The changed code
  doesn't touch the HR path, but run it on the PC once.

Disk/memory: nothing downloaded; scratch analysis files only in the session temp folder; no new repo files.

## 14. Repository restructure (2026-11-14, Mac)

Goal: a root with only a few self-explaining folders, launchers grouped by OS, and no course code in names.
No code logic, model or result changed; only locations, imports/paths and docs.

| Old | New |
|---|---|
| `run_optipulse.bat`, `run_phone.bat`, `run_phone_wifi.bat`, `find_cameras.bat`, `train_pipeline.bat` | `launchers/windows/` (each now `cd`s to the repo root first) |
| `mac/run_mac.sh` | `launchers/macos/run_mac.sh` |
| `mac/README_MAC.md` | `docs/MAC_GUIDE.md` |
| `scripts/live_monitor.py` | `optipulse/app.py`, started with `python -m optipulse` (new `optipulse/__main__.py`) |
| `scripts/{preprocess_ubfc,audit_dataset,train_cv,eval_tracking,compare_pipelines,make_report}.py` | `pipeline/` |
| `scripts/ensemble_check.py`, `ensemble_check5.py` | `archive/experiments/` |
| `reports/RESULTS.md`, `reports/figures/` | `results/` |
| `reports/*.json` | `results/metrics/` |
| `reports/WORK_REPORT.md` (this file) | `docs/WORK_REPORT.md` |
| `artifacts/legacy_code/` | `archive/legacy_system/` (its `*_Deploy/` → `deploy/`, its `archive/` → `older_versions/`) |
| root `*_Deploy/models/` (old weights, git-ignored) | `archive/legacy_system/deploy/models/` |
| `artifacts/trial_models/` (git-ignored, PC only) | `archive/trial_models/` (move by hand on the PC) |
| old-system docs in `docs/` (`Pipeline_Explanation.md`, `Vidformer_Presentation_Notes.txt`, `DATASET_DESCRIPTION_legacy.md`, architecture / training-curve PNGs, `Paper-3.pdf`) and two old-system images from `outputs/` | `archive/legacy_system/docs/` (`Paper-3.pdf` renamed `reference_Chen2019_respiratory_rate_from_face_videos.pdf`) |

Code changes:
* new `optipulse/paths.py`: every folder the project uses, defined once; the app, the pipeline and the experiments import it;
* `optipulse/app.py` gained `cli()`; argparse name `python -m optipulse`;
* pipeline defaults now write to `results/metrics/` and `results/RESULTS.md`;
* the legacy code's `*_Deploy` path became `deploy`;
* new `.gitattributes` (`*.bat` CRLF, `*.sh` LF);
* `.gitignore` updated (`results/logs/`, `archive/trial_models/`).

New docs: `docs/EXPLAIN.md` (repository tour and commands updated), `archive/README.md`, rewritten
`README.md` (layout and "where to read what").

Verified on the Mac:
* `python -m optipulse --help` works;
* every pipeline/experiment script compiles and all path constants exist;
* the audit loader returns the same 7 flagged subjects;
* `pipeline/make_report.py` regenerates `results/RESULTS.md` identical apart from its timestamp;
* a headless replay of a Mac recording through `python -m optipulse --video …` runs end to end.

Not testable on the Mac (no dataset, no NVIDIA GPU): `launchers/windows/train_pipeline.bat` and the UBFC
regression (sub42 0.64, sub37 2.22). Run them once on the PC after pulling.

Disk/memory: nothing downloaded.
