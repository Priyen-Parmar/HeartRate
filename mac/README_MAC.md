# VitalEye on a Mac - test kit

Contents: the VitalEye app code, the 10 trained heart-rate models, and the face detector (~80 MB in total).
No dataset is included, and none is needed.

## 1. Start (Terminal)

1. Unzip `VitalEye_mac.zip`, e.g. into your Downloads folder.
2. Open **Terminal** (Cmd+Space, type "Terminal") and run:
   ```
   cd ~/Downloads/VitalEye_mac
   bash run_mac.sh
   ```
3. **First run only:** it installs PyTorch, OpenCV and YOLO into a private `.venv` folder: ~1.5 GB download,
   5-10 min, nothing system-wide. You need **Python 3.11 or 3.12** from python.org (or the Python that
   macOS offers to install).
4. macOS asks **"Terminal would like to access the camera"**: click **Allow**. If you clicked "Don't
   allow" before, go to System Settings -> Privacy & Security -> Camera, switch on Terminal, quit
   Terminal and start again.

## 2. Test session

* Sit **50-70 cm** from the Mac. Your face should be about **1/3 of the picture height**: no "Too close"
  hint on screen.
* **Light is the most important part.** Your face must be the brightest thing in the picture: a lamp or a
  window **in front of you** (behind the laptop). Avoid a bright wall, window or ceiling light behind or
  above you: the Mac camera then exposes for the background, your face gets dark, and the camera keeps
  re-adjusting its brightness. That is 20-30x stronger than the pulse (measured on the first Mac session).
  The app tells you "Light is behind you" when it measures this.
* Glasses off if you can (they reflect the screen). Rest your arms and lean back so your head stays still.
* Turn off macOS **video effects** (Control Centre -> Video Effects: Portrait, Studio Light, Center Stage
  OFF). They change the skin pixels and can erase the pulse.
* Sit still, don't talk, **2 minutes**. Every ~15 s write down the **reference device reading and the
  time shown in the app**.
* Press **q** to stop.
* **Optional:** a second run after ~30 squats (heart rate ~110-130).

The number appears ~10 s after your face is found. Amber = low-confidence reading (weak signal); green =
stable. The SIGNAL bars measure how strong the pulse is in the video; only better light and a still head raise
them (dataset videos: +3 to +4 dB = 4 bars; first Mac session: -2 dB = 1-2 bars).

## 3. Send back

From the `outputs/` folder: the `hr_log_<time>.csv` and `raw_<time>.mkv` of each session, plus your list
of reference readings with times. The `.mkv` is lossless video of your face (~0.5-1 GB per minute):
share it only privately.

## Options and notes

| Command | What it does |
|---|---|
| `bash run_mac.sh --list_cameras` | shows the camera numbers (an iPhone via Continuity Camera may be one of them) |
| `bash run_mac.sh --camera 1` | use camera 1 instead of 0 |
| `bash run_mac.sh --ensemble 4` | 4 models instead of 2 (more accurate, slower; fine on M-series with the Apple GPU) |
| `bash run_mac.sh --device cpu` | force CPU if the Apple GPU causes problems |

* **Memory (8 GB Macs):** the app uses ~1-1.5 GB RAM with 2 models. Buffers are capped; close other
  heavy apps.
* **Speed:** on an Apple Silicon Mac the models run on the Apple GPU when its results match the CPU (checked
  at start; the app prints which device it uses), otherwise on the CPU. Face detection runs in the
  background a few times per second, so the camera is never slowed down.
* **Intel Macs** work only on the CPU (PyTorch 2.2.2) and are slower.
