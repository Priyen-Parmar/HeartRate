# 🫀 VitalEye — Live Camera Heart Rate, Respiratory Rate & HRV Monitor

Contactless real-time physiological monitoring system (**CS776 Course Project**) that extracts **Heart Rate (BPM)**, **Respiratory Rate (BrPM)**, and **Heart Rate Variability (HRV)** from a live webcam feed using **YOLOv8** multi-ROI tracking and **PhysFormerX v3** (3D-ResNet with Squeeze-and-Excitation channel attention).

---

## 📂 Organized Repository Structure

```text
HeartRate/
├── live_hr_monitor.py          # Primary real-time webcam HR, RR & HRV monitor
├── run_live_hr.sh              # One-command launcher script (uses Python 3.12 + PyTorch)
├── evaluate_model.py           # Entry-point alias for live_hr_monitor.py
├── Vidformer1.py               # PhysFormerX v3 training script (HR + RR joint objective)
├── PhysFormerX.py              # PhysFormerX architecture & training reference
├── preprocessor.py             # UBFC-rPPG dataset preprocessing pipeline
├── VitalEye_CS776_Deploy/      # Core inference modules, weights & Gradio web app
│   ├── physformer.py           # PhysFormerX model definition & spectral helpers
│   ├── preprocessor.py         # YCbCr skin segmentation & illumination normalization
│   ├── OneEuroFilter.py        # Jitter-free bounding-box stabilization filter
│   ├── sliding_pipeline.py     # Offline video file annotation & causal sliding pipeline
│   ├── pipeline.py             # Multi-ROI video cropping & inference script
│   ├── app.py                  # Gradio web UI for video uploads
│   ├── VitalEye_Group_2_CS776_Project_Report.pdf
│   └── models/
│       ├── vidformer.pt        # Trained PhysFormerX v3 model weights (1.5M params)
│       ├── physformer.pt       # Copy of vidformer.pt for deployment scripts
│       └── yolov8/weights/
│           └── best.pt         # Trained YOLOv8 weights (Classes: Face, Nose, Chest)
├── docs/                       # Reference papers, architecture diagrams & training curves
│   ├── Paper-3.pdf             # Reference paper: RR & HRV estimation from face videos
│   ├── Pipeline_Explanation.md # Detailed stage-by-stage architecture walkthrough
│   └── Vidformer_Presentation_Notes.txt
└── archive/                    # Legacy experimental scripts & Kaggle notebooks
```

---

## 🚀 Quick Start — Running the Live Camera Monitor

From the `HeartRate` directory, run:

```bash
./run_live_hr.sh
```

Or invoke Python directly:

```bash
/usr/local/bin/python3.12 live_hr_monitor.py
```

### Optional Command-Line Arguments

- **Run on a pre-recorded video file**:
  ```bash
  ./run_live_hr.sh --video path/to/video.mp4 --record outputs/annotated_demo.mp4
  ```
- **Select a different camera index**:
  ```bash
  ./run_live_hr.sh --camera 1
  ```
- **Run automated self-test verification**:
  ```bash
  ./run_live_hr.sh --self_test
  ```

### Live Window Keyboard Controls

| Key | Action |
| :--- | :--- |
| **`q` / `ESC`** | Quit the monitor and print final session vitals |
| **`r`** | Reset frame buffers and recalibrate |
| **`m`** | Toggle Multi-ROI bounding boxes (`Face + Nose + Chest` vs. `Face Only`) |
| **`k`** | Toggle live top-right YCbCr Skin-Mask ROI preview |
| **`s`** | Save an annotated screenshot to `outputs/` |
| **`f`** | Toggle fullscreen display |

---

## 🧠 End-to-End Live Pipeline Architecture

1. **Multi-ROI YOLOv8 Detection (`best.pt`)**:
   - Detects `Face` (Class 0), `Nose` (Class 1), and `Chest` (Class 2).
   - Stabilizes bounding box coordinates across frames using `OneEuroFilter` + Exponential Moving Average (`α = 0.80`) to eliminate motion jitter.
2. **Training-Matched Spatiotemporal Preprocessing**:
   - Crops the stabilized `Face` region to `128×128` and isolates skin pixels (forehead and cheeks) in `YCbCr` space (`Cr ∈ [133, 173], Cb ∈ [77, 127]`) with morphological opening/closing.
   - Removes slow ambient lighting and auto-exposure drift via a **30-frame running-mean subtraction** (`uniform_filter1d`), resizes with antialiasing to `64×64`, and stacks RGB with inter-frame differences (`ΔRGB = Frame[t] - Frame[t-1]`) into a `[1, 6, T, 64, 64]` tensor.
3. **Asynchronous PhysFormerX 3D-CNN Inference (`vidformer.pt`)**:
   - Runs in a dedicated background worker thread so the webcam capture loop remains smooth at 30 FPS without dropping frames.
   - Provides a fast initial lock at **150 frames (~5 seconds)** and transitions into a continuous **300-frame (~10 seconds)** sliding window.
4. **Spectral & HRV Vital Sign Extraction**:
   - **Heart Rate (BPM)**: 4096-point Hann-windowed FFT periodogram over `0.75–2.50 Hz` (`45–150 BPM`) using real frame capture timestamps and parabolic peak interpolation, with POS chrominance fallback under low SNR.
   - **Respiratory Rate (BrPM)**: Extracted simultaneously from the `0.15–0.40 Hz` (`9–24 BrPM`) spectral band.
   - **HRV (SDNN in ms)**: Computed via zero-phase cardiac bandpass filtering, quadratic peak interpolation, and 3-sigma Gaussian Inter-Beat Interval (IBI) outlier removal.
