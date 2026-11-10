# Remote Photoplethysmography (rPPG) — Heart Rate Estimation from Video

## Complete Pipeline & Architecture Explanation

---

## 1. What is rPPG?

Every time your heart beats, it pushes blood through tiny blood vessels under your skin. This causes **very subtle color changes** on your face — invisible to the naked eye, but detectable by a camera. Remote Photoplethysmography (rPPG) is a technique that **extracts your heart rate from regular video** by analyzing these micro-color fluctuations.

Our system takes a webcam video of your face as input and outputs:
- A **PPG waveform** (the pulse signal reconstructed from video)
- Your **heart rate in BPM** (beats per minute)

---

## 2. The Complete Pipeline (3 Stages)

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  STAGE 1     │     │  STAGE 2         │     │  STAGE 3           │
│  Preprocessor│ ──► │  Model Training  │ ──► │  Live Evaluation   │
│              │     │  (PhysFormerX)   │     │  (Webcam)          │
└──────────────┘     └──────────────────┘     └────────────────────┘
  preprocessor.py      Vidformer1.py            evaluate_model.py
```

---

## 3. Stage 1 — Preprocessing (`preprocessor.py`)

### What it does
Takes raw video files from the UBFC-rPPG dataset and converts them into clean, structured data chunks that the neural network can learn from.

### Step-by-step pipeline

```
Raw Video (.avi)
    │
    ▼
┌─────────────────────────────┐
│  1. YOLO Face Detection     │  Detect face in every frame using a
│     + EMA Box Smoothing     │  YOLO model. Smooth the bounding box
│                             │  across frames to reduce jitter.
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  2. Crop & Resize           │  Crop the face region and resize
│     to 128×128 pixels       │  to a fixed 128×128 resolution.
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  3. Skin Masking (YCbCr)    │  Convert to YCbCr color space and
│                             │  keep only skin-colored pixels.
│                             │  Zeros out eyes, hair, background.
│                             │  Skin range: Cr∈[133,173], Cb∈[77,127]
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  4. Temporal Normalization   │  Remove slow lighting drift by
│                             │  subtracting a 30-frame running
│                             │  average of mean intensity, then
│                             │  adding back the global mean.
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  5. Ground Truth Processing  │  Read the PPG signal from the
│                             │  ground truth file. Interpolate
│                             │  it to match video frame rate.
│                             │  Apply bandpass filter (0.7–2.5 Hz)
│                             │  to keep only heart-rate frequencies.
│                             │  Z-normalize per chunk.
└─────────────────────────────┘
    │
    ▼
┌─────────────────────────────┐
│  6. Chunking                │  Slide a 300-frame window (10 sec)
│                             │  with stride 150 (50% overlap).
│                             │  Save each chunk as a .pt file
│                             │  containing video + PPG + metadata.
└─────────────────────────────┘
```

### What each chunk contains
| Key          | Shape / Type        | Description                        |
|--------------|---------------------|------------------------------------|
| `video`      | `[3, 300, 128, 128]` uint8 | Face video (RGB, 300 frames)    |
| `ppg`        | `[300]` float32     | Ground-truth PPG waveform          |
| `subject_id` | string              | Subject identifier (for splitting) |
| `fps`        | float               | Video frame rate (usually 30)      |

### Why skin masking?
The rPPG signal comes from **blood flow under the skin**. Areas like eyes, hair, and background introduce noise. By zeroing them out, we force the model to focus on skin regions where the pulse signal is strongest (forehead, cheeks).

### Why temporal normalization?
Room lighting can slowly change over time (clouds passing, lamp flickering). This creates a slow drift that has nothing to do with heart rate. Subtracting the running mean removes this drift while preserving the fast pulse signal.

---

## 4. Stage 2 — Model Architecture (`Vidformer1.py`)

### Overview

```
Input: Face video chunk [3, T, H, W]
                │
    ┌───────────┴───────────┐
    │                       │
  RGB frames          Difference frames
  [3, T, H, W]       diff[t] = frame[t] - frame[t-1]
    │                  [3, T, H, W]
    └───────────┬───────────┘
                │
         Concatenate → [6, T, H, W]
                │
                ▼
        ┌───────────────┐
        │   3D ResNet   │ ← Extracts spatiotemporal features
        │   + SE Blocks │ ← Channel attention (focus on useful channels)
        └───────────────┘
                │
                ▼
         [128, T, 1, 1]  (spatial collapsed)
                │
                ▼
        ┌───────────────┐
        │  Temporal      │ ← Two layers of 1D convolution
        │  Conv Head     │ ← Smooths and refines the signal
        └───────────────┘
                │
                ▼
           PPG waveform [T]
                │
                ▼
         FFT → Peak frequency → Heart Rate (BPM)
```

### Input: 6-Channel Difference Frames

Instead of feeding just RGB frames, we compute **difference frames**:

```
diff[t] = frame[t] - frame[t-1]
```

This is crucial because the heart rate signal is a **temporal change** — the difference between consecutive frames captures exactly those subtle color fluctuations caused by blood flow. The original RGB frames provide the spatial context (where the skin is), while difference frames provide the temporal pulse signal.

Final input: `[6, T, 64, 64]` — 3 channels RGB + 3 channels difference, T frames, 64×64 spatial.

### 3D ResNet Backbone

The backbone is a series of **3D convolutional residual blocks** that process both spatial and temporal dimensions simultaneously.

```
Stem:      6ch → 32ch,  MaxPool spatial /2     → [32, T, 32, 32]
Layer 1:   32ch → 64ch, stride spatial /2      → [64, T, 16, 16]
Layer 2:   64ch → 64ch, identity               → [64, T, 16, 16]
Layer 3:   64ch → 128ch, stride spatial /2     → [128, T, 8, 8]
Layer 4:   128ch → 128ch, identity             → [128, T, 8, 8]
GAP:       AdaptiveAvgPool3d → spatial collapse → [128, T, 1, 1]
```

**Key design**: Spatial dimensions are progressively reduced (64→32→16→8→1), but the **temporal dimension T is preserved** throughout. This is because we need to output a time-series (PPG waveform) with the same number of time steps as the input.

### SE Blocks (Squeeze-and-Excitation)

Each ResBlock includes a **Squeeze-and-Excitation (SE) block** — a channel attention mechanism:

```
Features [B, C, T, H, W]
    │
    ▼
Global Average Pool → [B, C, 1, 1, 1]
    │
    ▼
FC(C → C/8) → ReLU → FC(C/8 → C) → Sigmoid
    │
    ▼
Channel-wise weights [B, C, 1, 1, 1]
    │
    ▼
Multiply with original features
```

**Why?** Not all feature channels are equally useful. The SE block learns to **amplify channels that carry pulse information** and suppress noisy channels. For example, the green channel carries the strongest rPPG signal — the SE block can learn to give it more weight.

### Temporal Convolution Head

After the 3D backbone collapses spatial dimensions, we have a feature vector per time step `[128, T]`. The temporal head processes this into a clean PPG waveform:

```
Conv1d(128→64, kernel=5) → BN → ReLU    ← Smooth features across time
Conv1d(64→64,  kernel=5) → BN → ReLU    ← Further temporal refinement
Dropout(0.1)                              ← Prevent overfitting
Conv1d(64→1,   kernel=1)                  ← Final regression to 1D signal
```

**Kernel size 5** means each output time step looks at 5 neighboring frames (2 before, 2 after), smoothing out frame-level noise. The output is a `[T]` waveform that should match the ground-truth PPG signal.

### Model Size
- **~1.5 million parameters** — small enough to generalize on limited data, large enough to learn the mapping.

---

## 5. Training Strategy

### Loss Function (3 components)

```
Total Loss = NegPearson + 0.5 × FreqCE + 0.1 × SmoothL1
```

| Loss | What it does | Why |
|------|-------------|-----|
| **Negative Pearson** | Measures correlation between predicted and true PPG waveforms. Pearson = 1 means perfect match. Loss = 1 - Pearson. | Forces the predicted waveform **shape** to match the target, regardless of amplitude or offset. |
| **Frequency Cross-Entropy** | Computes FFT of both signals, extracts the HR frequency band (0.75–2.5 Hz), and applies KL-divergence between spectral distributions. | Forces the **spectral peak** (heart rate frequency) to match. This directly optimizes BPM accuracy. |
| **Smooth L1** | Standard regression loss on waveform values. | Provides **amplitude-level supervision** and training stability. |

### Training Details
| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Optimizer | AdamW | Momentum + decoupled weight decay |
| Learning Rate | 3e-4 (max) | OneCycleLR: warmup → peak → cosine anneal |
| Warmup | 10% of training | Prevents early divergence |
| Batch Size | 4 | Small batch = more gradient updates |
| Epochs | 100 (early stop at 15) | Stop when validation MAE plateaus |
| Temporal Crop | 150 frames (training) | Data augmentation via random cropping |
| Full Sequence | 300 frames (val/test) | More signal = better BPM estimation |
| EMA Decay | 0.998 | Validation uses EMA-smoothed weights |
| Grad Clip | 2.0 | Prevents gradient explosion |

### Data Augmentation (gentle)
- **Random horizontal flip** — face is roughly symmetric
- **Mild Gaussian noise** (σ=0.01) — simulates camera noise
- **Brightness jitter** (±10%) — simulates lighting changes
- **Random temporal crop** (150 of 300 frames) — more data variety

### Subject-Level Split
- **70% train / 15% validation / 15% test** — split by **subject ID**, not by chunks
- This prevents data leakage: all chunks from one person are in the same split
- Ensures the model generalizes to **unseen people**, not just unseen chunks of known people

---

## 6. BPM Extraction from PPG

The model outputs a PPG waveform `[T]`. To get heart rate:

```
PPG Signal → Hann Window → FFT → Power Spectral Density
                                        │
                        Filter to HR band (0.75–2.5 Hz = 45–150 BPM)
                                        │
                              Find peak frequency f_peak
                                        │
                              BPM = f_peak × 60
```

**Why FFT?** Counting peaks in a noisy signal is unreliable. FFT converts the signal to the frequency domain where the dominant heart rate appears as a clear spectral peak no matter the signal quality.

---

## 7. Stage 3 — Live Evaluation (`evaluate_model.py`)

### Real-time pipeline

```
Webcam → YOLO Face Detection → Crop 64×64
         (EMA-smoothed box)     │
                                ▼
                         Skin Masking (YCbCr)
                                │
                                ▼
                    Buffer 300 frames (~10 seconds)
                                │
                                ▼
                    Compute Difference Frames
                    Stack RGB + diff → [1, 6, 300, 64, 64]
                                │
                                ▼
                    PhysFormerX Forward Pass
                                │
                                ▼
                    PPG Waveform → FFT → BPM
                                │
                    ┌───────────┴───────────┐
                    │                       │
              Display BPM              Display PPG
              on camera feed           waveform graph
```

### Sliding Window
- Collects 300 frames (~10 seconds at 30fps)
- After each prediction, slides the window by 10 frames
- This gives an updated BPM reading roughly **every 0.3 seconds**
- BPM is smoothed with EMA (α=0.75) to avoid jumpiness

---

## 8. Key Innovations & Design Choices

| Design Choice | Why It Works |
|---------------|-------------|
| **Difference frames** | Directly captures the temporal pulse signal; raw RGB is dominated by static skin color |
| **YCbCr skin masking** | Removes noisy non-skin regions; concentrates on forehead/cheeks where signal is strongest |
| **3D convolutions** | Simultaneously model spatial (where on face) and temporal (pulse pattern) information |
| **SE attention** | Automatically learns that green channel → strongest rPPG signal |
| **Frequency-domain loss** | Directly optimizes the metric we care about (BPM = spectral peak location) |
| **Subject-level split** | Tests generalization to new people, not memorization |
| **EMA model averaging** | Smoother, more stable predictions than raw model weights |

---

## 9. Evaluation Metric

**Mean Absolute Error (MAE) in BPM**:
```
MAE = (1/N) × Σ |predicted_BPM - true_BPM|
```

- Computed per-chunk, averaged across all test chunks
- Target: **< 3 BPM** (clinical-grade accuracy)
- A MAE of 3 means on average the prediction is off by 3 beats per minute

---

## 10. Dataset

**UBFC-rPPG Dataset** — publicly available benchmark for rPPG research.
- ~50 subjects, ~1 minute videos each
- Subjects sit in front of a webcam under varying conditions
- Ground truth: finger pulse oximeter (contact PPG sensor)
- Videos preprocessed with YOLO face detection → `face.mp4` per subject

### Ground Truth Format
- **Line 1**: PPG waveform values (space-separated)
- **Line 2**: Heart rate values (space-separated)
- **Line 3**: Timestamps (space-separated)

---

## 11. Summary Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                                │
│                                                                 │
│  Dataset     Preprocessor      PhysFormerX        Loss          │
│  ───────  →  ─────────────  →  ───────────  →  ──────────      │
│  .avi +      YOLO detect       6ch input        NegPearson      │
│  GT.txt      Skin mask         3D ResNet+SE     + FreqCE        │
│              Temporal norm     Conv1D head       + SmoothL1     │
│              Chunk 300fr       → PPG signal                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                        trained weights
                         (yolo_best.pt)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                               │
│                                                                 │
│  Webcam → YOLO → Skin mask → Buffer 300 → PhysFormerX → BPM   │
│                                                                 │
│  Display: Heart rate + PPG waveform on screen                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
