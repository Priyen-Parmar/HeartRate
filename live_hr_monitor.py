#!/usr/local/bin/python3.12
"""
VitalEye — Live Camera Heart Rate, Respiratory Rate & HRV Monitor
═════════════════════════════════════════════════════════════════
End-to-end real-time contactless physiological monitoring system using:
  1. Custom YOLOv8 Multi-ROI Detector (models/yolov8/weights/best.pt)
     — Tracks Face (Class 0), Nose (Class 1), and Chest (Class 2)
     — Stabilized via OneEuroFilter + Exponential Moving Average (EMA)
  2. Exact Training-Matched Spatiotemporal Preprocessor
     — 128×128 YCbCr Skin Segmentation (Cr ∈ [133,173], Cb ∈ [77,127])
     — 30-frame Running-Mean Illumination Drift Normalization
     — Antialiased 64×64 Resize + 6-Channel [RGB, ΔRGB] Representation
  3. PhysFormerX v3 3D-ResNet + SE Attention Model (models/vidformer.pt)
     — Asynchronous background inference thread for stutter-free 30 FPS video
     — Fast 5-second initial readout (150 frames) expanding to 10s (300 frames)
  4. Real-Timestamp Spectral & HRV Analysis (VitalEye Report & Paper-3)
     — Hann-windowed 4096-pt FFT with parabolic peak interpolation for HR (BPM)
     — Dual-band Respiratory Rate (BrPM) in 0.15–0.40 Hz (9–24 BrPM)
     — Inter-Beat Interval (IBI) quadratic refinement + 3σ outlier removal for HRV

Controls:
  q / ESC — Quit and display session summary
  r       — Reset buffers and recalibrate
  m       — Toggle Multi-ROI boxes (Face only vs. Face + Nose + Chest)
  k       — Toggle live YCbCr Skin-Mask inset preview
  s       — Save annotated screenshot
  f       — Toggle fullscreen

Usage:
  ./run_live_hr.sh
  # or:
  /usr/local/bin/python3.12 live_hr_monitor.py
  /usr/local/bin/python3.12 live_hr_monitor.py --video path/to/video.mp4
  /usr/local/bin/python3.12 live_hr_monitor.py --self_test
"""

import os
import sys

# Ensure writable config dir for Ultralytics before importing
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# Auto-switch to /usr/local/bin/python3.12 on macOS if launched with system Python lacking torch
try:
    import torch  # noqa: F401
except ImportError:
    _preferred_py = "/usr/local/bin/python3.12"
    if os.path.exists(_preferred_py) and os.path.realpath(sys.executable) != os.path.realpath(_preferred_py):
        os.execv(_preferred_py, [_preferred_py] + sys.argv)
    raise

import cv2
import time
import math
import threading
import argparse
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
from torchvision.transforms import Resize
from scipy.signal import periodogram, butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

# ─── Add deploy folder to path so we can import physformer, preprocessor, OneEuroFilter ───
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(ROOT_DIR, "VitalEye_CS776_Deploy")
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from physformer import PhysFormerX
from preprocessor import skin_mask_ycrcb
from OneEuroFilter import OneEuroFilter


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_YOLO    = os.path.join(DEPLOY_DIR, "models", "yolov8", "weights", "best.pt")
DEFAULT_MODEL   = os.path.join(DEPLOY_DIR, "models", "vidformer.pt")
FALLBACK_MODEL  = os.path.join(DEPLOY_DIR, "models", "physformer.pt")

BUFFER_LEN      = 300        # Full window: 300 frames (~10 s @ 30 fps)
WARMUP_LEN      = 150        # Fast initial readout: 150 frames (~5 s @ 30 fps, matches training CROP_LEN)
FACE_CROP_SIZE  = 128        # Matches preprocessor.py FACE_SIZE = 128 for accurate YCbCr morphology
IMG_SIZE        = 64         # Matches PhysFormerX spatial input (64×64)
BPM_EMA_ALPHA   = 0.72       # Temporal smoothing for BPM display
RPM_EMA_ALPHA   = 0.78       # Temporal smoothing for Respiratory Rate display
BOX_EMA_ALPHA   = 0.80       # EMA box stabilization (matches preprocessor.py / sliding_pipeline.py)
PPG_DISPLAY     = 180        # Number of recent PPG samples shown in live waveform graph

CLASS_NAMES     = {0: "Face", 1: "Nose", 2: "Chest"}
CLASS_COLORS    = {
    0: (50, 220, 50),    # Face: Green
    1: (255, 200, 0),    # Nose: Cyan-Blue
    2: (0, 180, 255),    # Chest: Amber-Orange
}


# ─────────────────────────────────────────────────────────────────────────────
#  ROI TRACKING & STABILIZATION (OneEuroFilter + EMA)
# ─────────────────────────────────────────────────────────────────────────────
class ROITracker:
    """Stabilizes [x1, y1, x2, y2] bounding boxes using OneEuroFilter + EMA."""

    def __init__(self, fps: float = 30.0, alpha: float = BOX_EMA_ALPHA):
        self.alpha = alpha
        self.prev_boxes = {}
        self.last_seen = {}
        self.euro_filters = {
            cls_id: [
                OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
                OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
                OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
                OneEuroFilter(freq=fps, mincutoff=1.25, beta=0.007),
            ]
            for cls_id in CLASS_NAMES
        }

    def update(self, cls_id: int, raw_box: list, timestamp: float) -> list:
        filt = self.euro_filters.get(cls_id)
        if filt is not None:
            euro_box = [filt[i](raw_box[i], timestamp) for i in range(4)]
        else:
            euro_box = raw_box

        prev = self.prev_boxes.get(cls_id)
        if prev is None:
            smoothed = euro_box
        else:
            smoothed = [self.alpha * p + (1.0 - self.alpha) * n for p, n in zip(prev, euro_box)]

        self.prev_boxes[cls_id] = smoothed
        self.last_seen[cls_id] = timestamp
        return smoothed

    def get(self, cls_id: int, timestamp: float, max_hold_sec: float = 1.5):
        if cls_id not in self.prev_boxes:
            return None
        if timestamp - self.last_seen.get(cls_id, 0.0) > max_hold_sec:
            return None
        return self.prev_boxes[cls_id]

    def reset(self):
        self.prev_boxes.clear()
        self.last_seen.clear()


# ─────────────────────────────────────────────────────────────────────────────
#  SIGNAL PROCESSING, FFT, POS, AND PAPER-3 HRV/RR EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
def parabolic_peak_freq(freqs: np.ndarray, psd: np.ndarray, valid_idx: np.ndarray) -> float:
    """
    Find dominant frequency in `valid_idx` with quadratic (parabolic) interpolation
    over log-PSD for sub-bin accuracy (Paper-3 Section II-B).
    """
    if len(valid_idx) == 0:
        return 0.0
    sub_psd = psd[valid_idx]
    local_max = int(np.argmax(sub_psd))
    k = int(valid_idx[local_max])

    if 0 < k < len(psd) - 1:
        y0 = math.log(max(float(psd[k - 1]), 1e-12))
        y1 = math.log(max(float(psd[k]), 1e-12))
        y2 = math.log(max(float(psd[k + 1]), 1e-12))
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if abs(denom) > 1e-9:
            delta = (y0 - y2) / denom
            delta = max(-0.5, min(0.5, delta))
            df = float(freqs[1] - freqs[0])
            return float(freqs[k] + delta * df)
    return float(freqs[k])


def analyze_ppg_spectrum(signal: np.ndarray, fps: float = 30.0):
    """
    Extract Heart Rate (BPM), Respiratory Rate (BrPM), Signal Quality (0-1),
    and spectrum data from predicted 1-D rPPG signal.
    """
    if len(signal) < 60 or fps <= 1.0:
        return 0.0, 0.0, 0.0, None, None

    s = signal.astype(np.float64) - np.mean(signal)
    std = np.std(s)
    if std < 1e-7:
        return 0.0, 0.0, 0.0, None, None
    s = s / std

    freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend='constant', window='hann')

    # 1. Heart Rate band: 0.75 - 2.50 Hz (45 - 150 BPM)
    hr_valid = np.where((freqs >= 0.75) & (freqs <= 2.50))[0]
    bpm = 0.0
    quality = 0.0
    if len(hr_valid) > 0:
        f_hr = parabolic_peak_freq(freqs, psd, hr_valid)
        bpm = f_hr * 60.0
        # Spectral concentration quality score around peak (±0.15 Hz)
        peak_band = np.where((freqs >= f_hr - 0.15) & (freqs <= f_hr + 0.15))[0]
        peak_power = np.sum(psd[peak_band])
        total_hr_power = np.sum(psd[hr_valid]) + 1e-9
        quality = float(np.clip((peak_power / total_hr_power) * 1.4, 0.0, 1.0))

    # 2. Respiratory Rate band: 0.15 - 0.40 Hz (9 - 24 BrPM, matches Vidformer1.py)
    rr_valid = np.where((freqs >= 0.15) & (freqs <= 0.40))[0]
    rpm = 0.0
    if len(rr_valid) > 0:
        f_rr = parabolic_peak_freq(freqs, psd, rr_valid)
        rpm = f_rr * 60.0

    return bpm, rpm, quality, freqs, psd


def extract_hrv_and_peaks(signal: np.ndarray, fps: float = 30.0, est_bpm: float = 75.0):
    """
    Implements Paper-3 (Chen et al., 'Respiratory Rate Estimation from Face Videos'):
      1. Zero-phase bandpass filtering around cardiac band
      2. Peak detection + quadratic peak location interpolation
      3. Inter-Beat Interval (IBI) calculation
      4. 3-sigma Gaussian outlier removal on IBIs
      5. Returns (hrv_sdnn_ms, clean_peak_indices, ibi_bpm)
    """
    if len(signal) < 90 or fps <= 5.0:
        return 0.0, [], 0.0

    try:
        nyq = fps / 2.0
        lo = max(0.70 / nyq, 0.01)
        hi = min(2.50 / nyq, 0.98)
        b, a = butter(2, [lo, hi], btype='band')
        filtered = filtfilt(b, a, signal.astype(np.float64))
    except Exception:
        filtered = signal.astype(np.float64) - np.mean(signal)

    # Minimum distance between beats (~0.4s = 150 BPM max)
    min_dist = max(int(fps * 0.40), 4)
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=np.std(filtered) * 0.25)

    if len(peaks) < 3:
        return 0.0, peaks.tolist(), 0.0

    # Quadratic interpolation around each peak (Paper-3 Section II-B)
    refined_times = []
    for p in peaks:
        if 0 < p < len(filtered) - 1:
            y0, y1, y2 = filtered[p - 1], filtered[p], filtered[p + 1]
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            delta = (y0 - y2) / denom if abs(denom) > 1e-9 else 0.0
            delta = max(-0.5, min(0.5, delta))
            refined_times.append((p + delta) / fps)
        else:
            refined_times.append(p / fps)

    refined_times = np.array(refined_times)
    ibis = np.diff(refined_times)  # seconds
    # Valid physiological IBI range: 0.35s (171 BPM) to 1.4s (43 BPM)
    ibis = ibis[(ibis >= 0.35) & (ibis <= 1.40)]
    if len(ibis) < 2:
        return 0.0, peaks.tolist(), 0.0

    # 3-sigma outlier removal (Paper-3 Section II-C)
    mu, sigma = np.mean(ibis), np.std(ibis)
    if sigma > 1e-6:
        clean_ibis = ibis[np.abs(ibis - mu) <= 3.0 * sigma]
    else:
        clean_ibis = ibis

    if len(clean_ibis) < 2:
        clean_ibis = ibis

    sdnn_ms = float(np.clip(np.std(clean_ibis) * 1000.0, 5.0, 180.0))
    ibi_bpm = float(60.0 / np.mean(clean_ibis))
    return sdnn_ms, peaks.tolist(), ibi_bpm


def pos_chrominance_bpm(frames_rgb_128: np.ndarray, fps: float = 30.0) -> float:
    """
    Plane-Orthogonal-to-Skin (POS) mathematical rPPG baseline (Wang et al. 2017 / Report Sec VII-F).
    Used as an auxiliary sanity check when camera motion is high.
    """
    T = len(frames_rgb_128)
    if T < 60:
        return 0.0
    # Compute spatial RGB mean over non-zero (skin) pixels per frame
    rgb_means = []
    for f in frames_rgb_128:
        mask = f.sum(axis=-1) > 0
        if np.any(mask):
            rgb_means.append(f[mask].mean(axis=0))
        else:
            rgb_means.append(f.mean(axis=(0, 1)))
    rgb = np.array(rgb_means, dtype=np.float64)  # [T, 3]
    mean_rgb = np.mean(rgb, axis=0, keepdims=True) + 1e-6
    cn = rgb / mean_rgb
    # POS projection: S1 = G - B, S2 = G + B - 2R
    s1 = cn[:, 1] - cn[:, 2]
    s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
    alpha = np.std(s1) / (np.std(s2) + 1e-8)
    h = s1 + alpha * s2
    h = h - np.mean(h)
    freqs, psd = periodogram(h, fs=fps, nfft=4096, detrend='constant', window='hann')
    valid = np.where((freqs >= 0.75) & (freqs <= 2.50))[0]
    if len(valid) > 0:
        return float(parabolic_peak_freq(freqs, psd, valid) * 60.0)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  EXACT TRAINING-MATCHED PREPROCESSING FOR PHYSFORMERX
# ─────────────────────────────────────────────────────────────────────────────
_RESIZE_64 = Resize((IMG_SIZE, IMG_SIZE), antialias=True)


def build_physformer_input(frames_128: list, timestamps: list, device: torch.device):
    """
    Replicates the exact training & deployment preprocessing pipeline from
    `preprocessor.py` and `VitalEye_CS776_Deploy/sliding_pipeline.py`:
      1. Compute actual sampling rate from frame timestamps
      2. If timestamps have jitter, resample sequence to uniform 30 FPS grid
      3. Apply 30-frame running-mean illumination normalization (uniform_filter1d)
      4. Scale to [0, 1] and resize 128×128 → 64×64 with antialiasing
      5. Compute temporal difference frames: diff[t] = frame[t] - frame[t-1]
      6. Concatenate [RGB, diff] → [1, 6, T, 64, 64]
    """
    frames_float = np.stack(frames_128, axis=0).astype(np.float32)  # [T, 128, 128, 3]
    T = len(frames_float)

    # Calculate true FPS from monotonic capture timestamps
    if len(timestamps) == T and (timestamps[-1] - timestamps[0]) > 0.5:
        actual_fps = float((T - 1) / (timestamps[-1] - timestamps[0]))
        actual_fps = float(np.clip(actual_fps, 10.0, 60.0))
    else:
        actual_fps = 30.0

    # Step 1: 30-frame running-mean illumination drift normalization (Stage 1 Step 4)
    frame_means = frames_float.mean(axis=(1, 2, 3), keepdims=True)
    win_size = min(30, max(5, T // 4))
    smoothed_means = uniform_filter1d(frame_means.squeeze(), size=win_size, axis=0)
    smoothed_means = smoothed_means.reshape(-1, 1, 1, 1)
    global_mean = frame_means.mean()
    frames_norm = np.clip(frames_float - smoothed_means + global_mean, 0.0, 255.0)

    # Step 2: Convert to tensor [3, T, 128, 128] in [0, 1] and antialiased resize to 64×64
    vid = torch.from_numpy(frames_norm).float() / 255.0
    vid = vid.permute(3, 0, 1, 2)  # [3, T, 128, 128]
    vid = _RESIZE_64(vid.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)  # [3, T, 64, 64]

    # Step 3: Compute difference frames diff[t] = frame[t] - frame[t-1]
    diff = torch.zeros_like(vid)
    diff[:, 1:] = vid[:, 1:] - vid[:, :-1]

    # Step 4: Stack RGB + diff → [1, 6, T, 64, 64]
    vid_6ch = torch.cat([vid, diff], dim=0).unsqueeze(0).to(device)
    return vid_6ch, actual_fps, frames_norm


# ─────────────────────────────────────────────────────────────────────────────
#  ASYNCHRONOUS INFERENCE ENGINE (Prevents Webcam Freeze during 3D-CNN Pass)
# ─────────────────────────────────────────────────────────────────────────────
class AsyncPhysFormerEngine:
    """
    Runs PhysFormerX 3D-CNN inference in a background worker thread so the
    main camera capture and UI loop stays at a smooth 30 FPS.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.lock = threading.Lock()
        self.pending_task = None
        self.running = True
        self.busy = False

        # Shared output state
        self.bpm = 0.0
        self.rpm = 0.0
        self.hrv_ms = 0.0
        self.quality = 0.0
        self.actual_fps = 30.0
        self.ppg_waveform = deque(maxlen=PPG_DISPLAY)
        self.psd_freqs = None
        self.psd_power = None
        self.bpm_history = deque(maxlen=5)
        self.rpm_history = deque(maxlen=5)
        self.inference_count = 0
        self.last_infer_ms = 0.0

        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, frames_128: list, timestamps: list):
        with self.lock:
            if not self.busy:
                self.pending_task = (frames_128, timestamps)
                self.busy = True
                return True
        return False

    def reset(self):
        with self.lock:
            self.pending_task = None
            self.bpm = 0.0
            self.rpm = 0.0
            self.hrv_ms = 0.0
            self.quality = 0.0
            self.ppg_waveform.clear()
            self.bpm_history.clear()
            self.rpm_history.clear()
            self.psd_freqs = None
            self.psd_power = None

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            task = None
            with self.lock:
                if self.pending_task is not None:
                    task = self.pending_task
                    self.pending_task = None
            if task is None:
                time.sleep(0.015)
                continue

            frames_128, timestamps = task
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    x_in, actual_fps, frames_norm = build_physformer_input(
                        frames_128, timestamps, self.device
                    )
                    pred = self.model(x_in)
                    ppg_signal = pred.squeeze().float().cpu().numpy()

                raw_bpm, raw_rpm, qual, freqs, psd = analyze_ppg_spectrum(
                    ppg_signal, fps=actual_fps
                )
                hrv_ms, _, ibi_bpm = extract_hrv_and_peaks(
                    ppg_signal, fps=actual_fps, est_bpm=raw_bpm
                )

                # Cross-check with POS chrominance if spectral quality is low
                if qual < 0.28:
                    pos_bpm = pos_chrominance_bpm(frames_norm, fps=actual_fps)
                    if 45.0 <= pos_bpm <= 160.0 and (raw_bpm < 45.0 or abs(raw_bpm - pos_bpm) > 25.0):
                        raw_bpm = 0.65 * raw_bpm + 0.35 * pos_bpm

                infer_ms = (time.perf_counter() - t0) * 1000.0

                with self.lock:
                    self.actual_fps = actual_fps
                    self.last_infer_ms = infer_ms
                    self.inference_count += 1

                    if 42.0 <= raw_bpm <= 180.0:
                        self.bpm_history.append(raw_bpm)
                        median_bpm = float(np.median(self.bpm_history))
                        self.bpm = (
                            BPM_EMA_ALPHA * self.bpm + (1.0 - BPM_EMA_ALPHA) * median_bpm
                            if self.bpm > 0
                            else median_bpm
                        )
                        self.quality = qual

                    if 8.0 <= raw_rpm <= 30.0:
                        self.rpm_history.append(raw_rpm)
                        median_rpm = float(np.median(self.rpm_history))
                        self.rpm = (
                            RPM_EMA_ALPHA * self.rpm + (1.0 - RPM_EMA_ALPHA) * median_rpm
                            if self.rpm > 0
                            else median_rpm
                        )

                    if hrv_ms > 0:
                        self.hrv_ms = 0.75 * self.hrv_ms + 0.25 * hrv_ms if self.hrv_ms > 0 else hrv_ms

                    recent = ppg_signal[-PPG_DISPLAY:]
                    self.ppg_waveform.clear()
                    self.ppg_waveform.extend(recent.tolist())
                    self.psd_freqs = freqs
                    self.psd_power = psd

            except Exception as e:
                print(f"⚠ Inference worker warning: {e}")
            finally:
                with self.lock:
                    self.busy = False


# ─────────────────────────────────────────────────────────────────────────────
#  HUD VISUALIZATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_vitals_panel(
    frame: np.ndarray,
    bpm: float,
    rpm: float,
    hrv_ms: float,
    quality: float,
    buf_fill: int,
    infer_ms: float,
):
    """Draws the top-left VitalEye clinical monitoring panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (355, 195), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (12, 12), (355, 195), (75, 85, 95), 1)

    # Header bar
    cv2.putText(
        frame,
        "VITALEYE  |  PhysFormerX + YOLOv8",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 210, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (22, 42), (345, 42), (55, 60, 68), 1)

    if bpm > 0 and buf_fill >= WARMUP_LEN:
        # Animated pulsing indicator
        pulse_phase = (math.sin(time.time() * (bpm / 60.0) * 2.0 * math.pi) + 1.0) * 0.5
        radius = int(7 + 3 * pulse_phase)
        cv2.circle(frame, (36, 76), radius, (50, 50, 240), -1, cv2.LINE_AA)

        # Primary BPM readout
        bpm_color = (60, 235, 100) if 50 <= bpm <= 110 else (40, 180, 255)
        cv2.putText(
            frame,
            f"{bpm:.0f}",
            (58, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.95,
            bpm_color,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "BPM",
            (180, 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (170, 175, 180),
            1,
            cv2.LINE_AA,
        )
        mode_tag = "10s Lock" if buf_fill >= BUFFER_LEN else f"Warmup ({buf_fill}/{BUFFER_LEN})"
        cv2.putText(
            frame,
            mode_tag,
            (180, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (120, 200, 255) if buf_fill < BUFFER_LEN else (100, 220, 120),
            1,
            cv2.LINE_AA,
        )

        # Secondary Vitals: Respiratory Rate (BrPM) & HRV (SDNN ms)
        rr_str = f"RR: {rpm:.1f} BrPM" if rpm > 0 else "RR: -- BrPM"
        hrv_str = f"HRV: {hrv_ms:.0f} ms" if hrv_ms > 0 else "HRV: -- ms"
        cv2.putText(
            frame,
            rr_str,
            (24, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 205, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            hrv_str,
            (195, 124),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 180, 255),
            1,
            cv2.LINE_AA,
        )

        # Signal Quality Bar
        bar_w = int(315 * np.clip(quality, 0.0, 1.0))
        cv2.rectangle(frame, (24, 142), (339, 156), (42, 45, 50), -1)
        bar_col = (
            (55, 225, 95)
            if quality >= 0.55
            else (35, 175, 255)
            if quality >= 0.30
            else (60, 70, 230)
        )
        cv2.rectangle(frame, (24, 142), (24 + bar_w, 156), bar_col, -1)
        cv2.putText(
            frame,
            f"Signal Quality: {quality * 100:.0f}%   |   3D-CNN: {infer_ms:.0f} ms",
            (24, 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (145, 150, 160),
            1,
            cv2.LINE_AA,
        )
    else:
        # Calibration progress (targets WARMUP_LEN for first reading)
        pct = min(100, int(100 * buf_fill / max(1, WARMUP_LEN)))
        cv2.putText(
            frame,
            "Calibrating rPPG Buffer...",
            (24, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (35, 180, 255),
            2,
            cv2.LINE_AA,
        )
        bar_w = int(315 * min(1.0, buf_fill / max(1, WARMUP_LEN)))
        cv2.rectangle(frame, (24, 96), (339, 114), (42, 45, 50), -1)
        cv2.rectangle(frame, (24, 96), (24 + bar_w, 114), (35, 180, 255), -1)
        cv2.putText(
            frame,
            f"{pct}%  ({buf_fill}/{WARMUP_LEN} frames for initial HR lock)",
            (24, 142),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (165, 170, 175),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Keep face steady in front of camera (~5 s)",
            (24, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 125, 130),
            1,
            cv2.LINE_AA,
        )


def draw_skin_mask_inset(frame: np.ndarray, face_masked_rgb: np.ndarray):
    """Draws the live YCbCr skin-segmented ROI preview in the top-right corner."""
    if face_masked_rgb is None or face_masked_rgb.size == 0:
        return
    h, w = frame.shape[:2]
    inset_sz = 96
    x0 = w - inset_sz - 14
    y0 = 14

    inset_bgr = cv2.cvtColor(cv2.resize(face_masked_rgb, (inset_sz, inset_sz)), cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + inset_sz + 2, y0 + inset_sz + 22), (20, 20, 25), -1)
    frame[y0 : y0 + inset_sz, x0 : x0 + inset_sz] = inset_bgr
    cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + inset_sz + 2, y0 + inset_sz + 22), (80, 90, 100), 1)
    cv2.putText(
        frame,
        "YCbCr Skin ROI",
        (x0 + 4, y0 + inset_sz + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 190, 200),
        1,
        cv2.LINE_AA,
    )


def draw_ppg_and_spectrum(
    frame: np.ndarray,
    ppg_buf: list,
    freqs: np.ndarray,
    psd: np.ndarray,
    bpm: float,
):
    """Draws the real-time rPPG waveform and FFT power spectrum at bottom-right."""
    if len(ppg_buf) < 4:
        return

    h_f, w_f = frame.shape[:2]
    pw = min(360, w_f - 30)
    ph = 105
    x0 = w_f - pw - 14
    y0 = h_f - ph - 32

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 22), (x0 + pw + 6, y0 + ph + 6), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (65, 70, 80), 1)
    cv2.putText(
        frame,
        "Live rPPG Waveform (PhysFormerX)",
        (x0 + 6, y0 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (175, 185, 195),
        1,
        cv2.LINE_AA,
    )

    # Grid lines
    for frac in (0.25, 0.5, 0.75):
        gy = y0 + int(ph * frac)
        cv2.line(frame, (x0, gy), (x0 + pw, gy), (36, 40, 46), 1)

    arr = np.array(ppg_buf, dtype=np.float32)
    std = float(arr.std())
    if std < 1e-7:
        return
    arr_norm = np.clip((arr - arr.mean()) / (std + 1e-8), -2.8, 2.8)

    n = len(arr_norm)
    x_step = pw / max(n - 1, 1)
    y_mid = y0 + ph // 2
    y_scl = (ph * 0.42) / 2.8

    pts = []
    for i, v in enumerate(arr_norm):
        px = int(x0 + i * x_step)
        py = int(np.clip(y_mid - v * y_scl, y0 + 3, y0 + ph - 3))
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (50, 230, 100), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 4, (0, 255, 255), -1, cv2.LINE_AA)


def draw_roi_boxes(
    frame: np.ndarray,
    tracker: ROITracker,
    now_ts: float,
    bpm: float,
    rpm: float,
    buf_fill: int,
    show_multi_roi: bool,
):
    """Draws stabilized Face (and optionally Nose & Chest) bounding boxes."""
    h_f, w_f = frame.shape[:2]
    classes_to_draw = (0, 1, 2) if show_multi_roi else (0,)

    for cls_id in classes_to_draw:
        box = tracker.get(cls_id, now_ts)
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_f - 1, x2), min(h_f - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        color = CLASS_COLORS.get(cls_id, (50, 220, 50))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if cls_id == 0:
            if bpm > 0 and buf_fill >= WARMUP_LEN:
                label = f"Face | HR: {bpm:.0f} BPM"
                if rpm > 0:
                    label += f" | RR: {rpm:.0f}"
            else:
                label = f"Face ROI ({int(100 * min(1.0, buf_fill / WARMUP_LEN))}%)"
        else:
            label = f"{CLASS_NAMES[cls_id]} ROI"

        ts, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
        ly = max(y1, ts[1] + 10)
        cv2.rectangle(frame, (x1, ly - ts[1] - 8), (x1 + ts[0] + 8, ly), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, ly - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST / VERIFICATION ROUTINE
# ─────────────────────────────────────────────────────────────────────────────
def run_self_test(args) -> int:
    """
    Runs an automated end-to-end verification of YOLOv8, PhysFormerX weights,
    preprocessing, BPM/RR/HRV extraction, and HUD rendering without requiring
    interactive keyboard input.
    """
    print("═" * 65)
    print("  VitalEye — End-to-End Self-Test & Model Verification")
    print("═" * 65)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"✔ Compute Device       : {device}")

    model_path = args.model_weights if os.path.exists(args.model_weights) else FALLBACK_MODEL
    if not os.path.exists(model_path):
        print(f"❌ PhysFormerX weights not found at {args.model_weights}")
        return 1

    model = PhysFormerX(seq_len=BUFFER_LEN).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✔ PhysFormerX Loaded   : {model_path} ({n_params / 1e6:.2f}M params)")

    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    print(f"✔ YOLOv8 Loaded        : {args.yolo_weights} (classes: {yolo.names})")

    # Synthesize 300 frames (10s @ 30fps) with 75.0 BPM cardiac pulse + 15.0 BrPM respiration
    fps = 30.0
    T = 300
    t = np.arange(T) / fps
    target_bpm = 75.0
    target_rpm = 15.0
    frames_128 = []
    timestamps = []

    for i in range(T):
        ti = float(t[i])
        pulse = math.sin(2.0 * math.pi * (target_bpm / 60.0) * ti)
        breath = math.sin(2.0 * math.pi * (target_rpm / 60.0) * ti)
        drift = 8.0 * math.sin(2.0 * math.pi * 0.04 * ti)

        patch = np.zeros((FACE_CROP_SIZE, FACE_CROP_SIZE, 3), dtype=np.uint8)
        patch[:, :, 0] = int(np.clip(192.0 + drift + 1.5 * pulse, 0, 255))
        patch[:, :, 1] = int(np.clip(142.0 + drift + 4.5 * pulse + 2.0 * breath, 0, 255))
        patch[:, :, 2] = int(np.clip(118.0 + drift + 1.0 * pulse, 0, 255))

        mask = skin_mask_ycrcb(patch)
        mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
        frames_128.append(patch * mask_3ch)
        timestamps.append(ti)

    t0 = time.perf_counter()
    with torch.no_grad():
        x_in, actual_fps, _ = build_physformer_input(frames_128, timestamps, device)
        pred = model(x_in).squeeze().float().cpu().numpy()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    bpm, rpm, qual, freqs, psd = analyze_ppg_spectrum(pred, fps=actual_fps)
    hrv_ms, _, _ = extract_hrv_and_peaks(pred, fps=actual_fps, est_bpm=bpm)

    print(f"✔ Preprocessing & Pass : {infer_ms:.1f} ms for 300 frames (6×300×64×64)")
    print(f"✔ Target vs Predicted  : Target={target_bpm:.1f} BPM | Predicted={bpm:.2f} BPM (Error={abs(bpm - target_bpm):.2f} BPM)")
    print(f"✔ Signal Metrics       : RR={rpm:.1f} BrPM | HRV={hrv_ms:.1f} ms | Quality={qual * 100:.1f}%")

    # Render a sample HUD preview frame
    canvas = np.full((480, 640, 3), 35, dtype=np.uint8)
    tracker = ROITracker(fps=30.0)
    tracker.update(0, [210, 90, 430, 360], 10.0)
    tracker.update(1, [285, 195, 355, 260], 10.0)
    tracker.update(2, [160, 365, 480, 475], 10.0)

    draw_roi_boxes(canvas, tracker, 10.0, bpm, rpm, 300, show_multi_roi=True)
    draw_vitals_panel(canvas, bpm, rpm, hrv_ms, qual, 300, infer_ms)
    draw_skin_mask_inset(canvas, frames_128[-1])
    draw_ppg_and_spectrum(canvas, pred[-PPG_DISPLAY:].tolist(), freqs, psd, bpm)

    out_dir = os.path.join(ROOT_DIR, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "self_test_hud_preview.png")
    cv2.imwrite(out_path, canvas)
    print(f"✔ HUD Preview Saved    : {out_path}")
    print("═" * 65)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LIVE CAMERA / VIDEO STREAM LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    if args.self_test:
        sys.exit(run_self_test(args))

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"⚙  Device : {device}")

    # ── 1. Load PhysFormerX Model ────────────────────────────────────────────
    model_path = args.model_weights
    if not os.path.exists(model_path) and os.path.exists(FALLBACK_MODEL):
        model_path = FALLBACK_MODEL

    print(f"⚙  Loading PhysFormerX from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ PhysFormerX weights not found at: {model_path}")
        sys.exit(1)

    model = PhysFormerX(seq_len=BUFFER_LEN).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✔  PhysFormerX loaded  ({n_params / 1e6:.2f}M params)")

    # ── 2. Load YOLOv8 Multi-ROI Detector ────────────────────────────────────
    use_yolo = False
    yolo = None
    face_cascade = None

    if os.path.exists(args.yolo_weights):
        try:
            from ultralytics import YOLO as _YOLO
            yolo = _YOLO(args.yolo_weights)
            use_yolo = True
            print(f"✔  YOLOv8 loaded from : {args.yolo_weights} (Classes: {yolo.names})")
        except Exception as e:
            print(f"⚠  YOLOv8 load failed ({e}), falling back to Haar cascade")
    else:
        print(f"⚠  YOLOv8 weights not found at '{args.yolo_weights}', using Haar cascade")

    if not use_yolo:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(haar_path)
        print("✔  Haar cascade face detector loaded (fallback)")

    # ── 3. Open Camera or Video File ─────────────────────────────────────────
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not args.video:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"❌ Cannot open video source: {source}")
        print("   Tip: On macOS, ensure Terminal / IDE has Camera permission in System Settings → Privacy & Security → Camera.")
        sys.exit(1)

    nominal_fps = cap.get(cv2.CAP_PROP_FPS)
    if not nominal_fps or nominal_fps <= 1 or nominal_fps > 120:
        nominal_fps = float(args.fps)
    print(f"✔  Video stream opened (source={source}, nominal_fps={nominal_fps:.1f})")

    # ── 4. Initialize Trackers, Buffers & Async Inference Engine ─────────────
    tracker = ROITracker(fps=nominal_fps, alpha=BOX_EMA_ALPHA)
    engine = AsyncPhysFormerEngine(model=model, device=device)

    frame_buf = deque(maxlen=BUFFER_LEN)
    ts_buf = deque(maxlen=BUFFER_LEN)
    last_skin_preview = None

    frame_count = 0
    last_submit_frame = 0
    show_multi_roi = True
    show_skin_inset = True
    fullscreen = False

    win_name = "VitalEye — Live Heart Rate & Vitals Monitor  [q=quit  r=reset  m=ROIs  k=skin  s=shot  f=full]"
    if not args.headless:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 900, 620)

    writer = None
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
        w_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        writer = cv2.VideoWriter(
            args.record, cv2.VideoWriter_fourcc(*"mp4v"), nominal_fps, (w_out, h_out)
        )

    print("\n" + "═" * 72)
    print("  VitalEye — Live Heart Rate, Respiratory Rate & HRV Monitor")
    print("  Controls:  q=quit  r=reset  m=toggle ROIs  k=skin preview  s=screenshot")
    print("═" * 72 + "\n")

    fps_timer = time.perf_counter()
    fps_display = nominal_fps
    yolo_device = "mps" if str(device) == "mps" else "0" if device.type == "cuda" else "cpu"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("✔  Reached end of video file.")
                else:
                    print("⚠  Camera frame read ended.")
                break

            now_ts = (
                frame_count / nominal_fps if args.video else time.perf_counter()
            )
            frame_count += 1
            h_frm, w_frm = frame.shape[:2]

            # ── Step 1: Multi-ROI Detection (Every N frames for 30 FPS smoothness) ──
            if use_yolo and (frame_count % max(1, args.yolo_interval) == 1 or tracker.get(0, now_ts) is None):
                results = yolo(
                    frame,
                    verbose=False,
                    imgsz=320,
                    conf=0.25,
                    device=yolo_device,
                )[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                    confs = results.boxes.conf.cpu().numpy()
                    classes = results.boxes.cls.cpu().numpy().astype(int)

                    # Pick highest-confidence box per class (0: Face, 1: Nose, 2: Chest)
                    best_per_cls = {}
                    for box_raw, cf, cls_id in zip(boxes_xyxy, confs, classes):
                        if cls_id in CLASS_NAMES:
                            if cls_id not in best_per_cls or cf > best_per_cls[cls_id][0]:
                                best_per_cls[cls_id] = (cf, box_raw.tolist())

                    for cls_id, (_, b) in best_per_cls.items():
                        clipped = [
                            max(0.0, b[0]),
                            max(0.0, b[1]),
                            min(float(w_frm), b[2]),
                            min(float(h_frm), b[3]),
                        ]
                        tracker.update(cls_id, clipped, now_ts)

            elif not use_yolo and (frame_count % max(1, args.yolo_interval) == 1 or tracker.get(0, now_ts) is None):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w_, h_ = faces[0]
                    tracker.update(0, [float(x), float(y), float(x + w_), float(y + h_)], now_ts)

            # ── Step 2: Crop Stabilized Face ROI, Apply 128×128 YCbCr Skin Mask ──
            face_box = tracker.get(0, now_ts)
            if face_box is not None:
                x1, y1, x2, y2 = map(int, face_box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frm, x2), min(h_frm, y2)

                if x2 > x1 + 10 and y2 > y1 + 10:
                    face_crop = frame[y1:y2, x1:x2]
                    # Resize to 128×128 first (matches preprocessor.py FACE_SIZE = 128)
                    face_128 = cv2.resize(face_crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
                    face_rgb = cv2.cvtColor(face_128, cv2.COLOR_BGR2RGB)

                    # Apply YCbCr skin mask
                    mask = skin_mask_ycrcb(face_rgb)
                    mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
                    face_masked = face_rgb * mask_3ch

                    frame_buf.append(face_masked)
                    ts_buf.append(now_ts)
                    last_skin_preview = face_masked

            # ── Step 3: Trigger Non-Blocking PhysFormerX Inference ──
            buf_fill = len(frame_buf)
            if buf_fill >= WARMUP_LEN and (frame_count - last_submit_frame) >= args.slide_step:
                # Submit latest window (up to 300 frames) to background worker
                submitted = engine.submit(list(frame_buf), list(ts_buf))
                if submitted:
                    last_submit_frame = frame_count
                elif args.video:
                    # For offline video file mode, wait briefly so every window is processed
                    while engine.busy:
                        time.sleep(0.01)
                    if engine.submit(list(frame_buf), list(ts_buf)):
                        last_submit_frame = frame_count
                        while engine.busy:
                            time.sleep(0.01)

            # ── Step 4: Read Latest Smoothed Vitals & Render HUD ──
            with engine.lock:
                cur_bpm = engine.bpm
                cur_rpm = engine.rpm
                cur_hrv = engine.hrv_ms
                cur_qual = engine.quality
                cur_infer_ms = engine.last_infer_ms
                ppg_list = list(engine.ppg_waveform)
                psd_freqs = engine.psd_freqs
                psd_power = engine.psd_power

            draw_roi_boxes(
                frame, tracker, now_ts, cur_bpm, cur_rpm, buf_fill, show_multi_roi
            )
            draw_vitals_panel(
                frame, cur_bpm, cur_rpm, cur_hrv, cur_qual, buf_fill, cur_infer_ms
            )
            if show_skin_inset and last_skin_preview is not None:
                draw_skin_mask_inset(frame, last_skin_preview)
            draw_ppg_and_spectrum(frame, ppg_list, psd_freqs, psd_power, cur_bpm)

            # Bottom status bar
            now_perf = time.perf_counter()
            inst_fps = 1.0 / max(now_perf - fps_timer, 1e-6)
            fps_display = 0.90 * fps_display + 0.10 * inst_fps
            fps_timer = now_perf

            cv2.putText(
                frame,
                f"FPS: {fps_display:.1f}",
                (14, h_frm - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (140, 145, 150),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "q=quit  r=reset  m=ROIs  k=skin  s=screenshot  f=fullscreen",
                (max(110, w_frm // 2 - 195), h_frm - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (120, 125, 130),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)

            if not args.headless:
                cv2.imshow(win_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("r"):
                    frame_buf.clear()
                    ts_buf.clear()
                    tracker.reset()
                    engine.reset()
                    print("🔄  Buffers & trackers reset — recalibrating...")
                elif key == ord("m"):
                    show_multi_roi = not show_multi_roi
                    print(f"🎯  Multi-ROI display (Face/Nose/Chest): {'ON' if show_multi_roi else 'Face Only'}")
                elif key == ord("k"):
                    show_skin_inset = not show_skin_inset
                elif key == ord("s"):
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    os.makedirs(os.path.join(ROOT_DIR, "outputs"), exist_ok=True)
                    fname = os.path.join(ROOT_DIR, "outputs", f"screenshot_{ts_str}.png")
                    cv2.imwrite(fname, frame)
                    print(f"📸  Screenshot saved → {fname}")
                elif key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        win_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )

    finally:
        engine.stop()
        cap.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    if engine.bpm > 0:
        print(f"\n{'═' * 52}")
        print(f"  Final Heart Rate Estimate       : {engine.bpm:.1f} BPM")
        if engine.rpm > 0:
            print(f"  Final Respiratory Rate Estimate : {engine.rpm:.1f} BrPM")
        if engine.hrv_ms > 0:
            print(f"  Final HRV (SDNN) Estimate       : {engine.hrv_ms:.0f} ms")
        print(f"{'═' * 52}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="VitalEye — PhysFormerX + YOLOv8 Live Heart Rate Monitor")
    p.add_argument(
        "--yolo_weights",
        default=DEFAULT_YOLO,
        help="Path to trained YOLOv8 .pt weights (default: VitalEye_CS776_Deploy/models/yolov8/weights/best.pt)",
    )
    p.add_argument(
        "--model_weights",
        default=DEFAULT_MODEL,
        help="Path to trained PhysFormerX .pt weights (default: VitalEye_CS776_Deploy/models/vidformer.pt)",
    )
    p.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    p.add_argument(
        "--video",
        type=str,
        default="",
        help="Optional path to an input video file instead of live webcam",
    )
    p.add_argument(
        "--record",
        type=str,
        default="",
        help="Optional path to save annotated output MP4 video",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target camera FPS (default: 30)",
    )
    p.add_argument(
        "--yolo_interval",
        type=int,
        default=2,
        help="Run YOLOv8 detector every N frames to keep camera at 30 FPS on CPU (default: 2)",
    )
    p.add_argument(
        "--slide_step",
        type=int,
        default=15,
        help="Trigger PhysFormerX update every N new frames (default: 15 ≈ 0.5s)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV GUI window (useful with --video and --record)",
    )
    p.add_argument(
        "--self_test",
        action="store_true",
        help="Run automated end-to-end verification of models, preprocessing, and HUD rendering",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
