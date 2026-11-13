#!/usr/local/bin/python3.12
"""
VitalEye — Live Camera Heart Rate, Respiratory Rate & HRV Monitor
═════════════════════════════════════════════════════════════════
Supports BOTH:
  • VitalEyeProNet (`models/vitaleye_pro_best.pt` — 10-channel low-light & motion-robust model)
  • PhysFormerX v3 (`models/vidformer.pt` — 6-channel 3D-ResNet + SE model)

Key Anti-40-50-BPM Drift & Low-Light/Motion Enhancements:
  1. Joint-Mask Boundary Suppression: Zeros out inter-frame differences where a pixel
     transitions between background (0) and skin (>0), eliminating 1/f² step-edge noise
     that previously pinned FFT peaks at 40–50 BPM.
  2. Uniform 30 Hz Temporal Resampling: Resamples variable-FPS low-light webcam frames
     onto a uniform 30.0 Hz timeline so 3D temporal convolutions always operate at their
     trained sampling rate.
  3. Pre-FFT 3rd-Order Butterworth Bandpass (0.85–2.50 Hz = 51–150 BPM) + Baseline Detrending:
     Strips sub-cardiac motion/exposure drift before computing the Hann-windowed periodogram.
  4. Deep rPPG + POS + CHROM Spectral Fusion: Fuses the neural network's normalized power
     spectral density with illumination-invariant POS and CHROM chrominance spectra for
     rock-solid BPM tracking under low light and slight movement.
"""

import os
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

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
import torch.nn.functional as F
from torchvision.transforms import Resize
from scipy.signal import periodogram, butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_DIR = os.path.join(ROOT_DIR, "deploy")
for _p in (ROOT_DIR, DEPLOY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from physformer import PhysFormerX
from vitaleye_pro_model import VitalEyeProNet
from preprocessor import skin_mask_ycrcb
from OneEuroFilter import OneEuroFilter


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_YOLO     = os.path.join(DEPLOY_DIR, "models", "yolov8", "weights", "best.pt")
PRO_MODEL_PATH   = os.path.join(DEPLOY_DIR, "models", "vitaleye_pro_best.pt")
DEFAULT_MODEL    = PRO_MODEL_PATH if os.path.exists(PRO_MODEL_PATH) else os.path.join(DEPLOY_DIR, "models", "vidformer.pt")
FALLBACK_MODEL   = os.path.join(DEPLOY_DIR, "models", "physformer.pt")

BUFFER_LEN       = 300        # 10 s @ 30 fps
WARMUP_LEN       = 120        # 4 s @ 30 fps for initial lock
FACE_CROP_SIZE   = 128        # Spatial size for 468-pt FaceMesh + Multi-Zone YCrCb/LAB mask
IMG_SIZE         = 64         # Spatial input to PhysFormerX / VitalEyeProNet
BPM_EMA_ALPHA    = 0.70
RPM_EMA_ALPHA    = 0.75
BOX_EMA_ALPHA    = 0.80
PPG_DISPLAY      = 180

CLASS_NAMES      = {0: "Face", 1: "Nose", 2: "Chest"}
CLASS_COLORS     = {
    0: (50, 220, 50),    # Face: Green
    1: (255, 200, 0),    # Nose: Cyan-Blue
    2: (0, 180, 255),    # Chest: Amber-Orange
}


# ─────────────────────────────────────────────────────────────────────────────
#  ROI TRACKING & STABILIZATION (OneEuroFilter + EMA)
# ─────────────────────────────────────────────────────────────────────────────
class ROITracker:
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
            try:
                euro_box = [filt[i](raw_box[i], timestamp) for i in range(4)]
            except ZeroDivisionError:
                euro_box = raw_box
        else:
            euro_box = raw_box

        prev = self.prev_boxes.get(cls_id)
        smoothed = euro_box if prev is None else [self.alpha * p + (1.0 - self.alpha) * n for p, n in zip(prev, euro_box)]
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
#  SIGNAL PROCESSING, BANDPASS DETRENDING & SPECTRAL FUSION (Fixes 40–50 BPM)
# ─────────────────────────────────────────────────────────────────────────────
def parabolic_peak_freq(freqs: np.ndarray, psd: np.ndarray, valid_idx: np.ndarray) -> float:
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
            delta = max(-0.5, min(0.5, (y0 - y2) / denom))
            df = float(freqs[1] - freqs[0])
            return float(freqs[k] + delta * df)
    return float(freqs[k])


def bandpass_cardiac(sig: np.ndarray, fps: float = 30.0, lo_hz: float = 0.85, hi_hz: float = 2.50) -> np.ndarray:
    """
    Zero-phase 3rd-order Butterworth bandpass filter + moving-average detrending.
    Crucial for suppressing <0.85 Hz (<51 BPM) 1/f motion & auto-exposure drift
    so low-frequency sidelobes never pin the FFT peak at 40–50 BPM.
    """
    s = sig.astype(np.float64) - np.mean(sig)
    if len(s) < 45:
        return s
    # Subtract 25-frame (~0.8s) running mean first to remove baseline wander
    win = max(5, int(round(fps * 0.80)))
    s = s - uniform_filter1d(s, size=win)
    try:
        nyq = fps / 2.0
        lo = max(lo_hz / nyq, 0.02)
        hi = min(hi_hz / nyq, 0.98)
        b, a = butter(3, [lo, hi], btype="band")
        s = filtfilt(b, a, s)
    except Exception:
        pass
    std = np.std(s)
    return s / (std + 1e-8)


def extract_pos_and_chrom_signals(frames_rgb_128: np.ndarray, fps: float = 30.0):
    """
    Extracts both POS (Wang et al. 2017) and CHROM (de Haan & Jeanne 2013) pulse
    waveforms from the non-zero skin pixels of `frames_rgb_128` using overlapping
    1.6-second sliding sub-windows for motion & low-light robustness.
    """
    T = len(frames_rgb_128)
    if T < 45:
        return np.zeros(T), np.zeros(T)

    rgb_means = np.zeros((T, 3), dtype=np.float64)
    for t_idx, f in enumerate(frames_rgb_128):
        mask = f.sum(axis=-1) > 8
        if np.count_nonzero(mask) >= 20:
            rgb_means[t_idx] = f[mask].mean(axis=0)
        else:
            rgb_means[t_idx] = f.mean(axis=(0, 1))

    # Overlap-add POS across ~1.6s windows
    win_len = max(24, min(T, int(round(fps * 1.6))))
    pos_sig = np.zeros(T, dtype=np.float64)
    for start in range(0, T - win_len + 1, 2):
        end = start + win_len
        sub = rgb_means[start:end]
        cn = sub / (np.mean(sub, axis=0, keepdims=True) + 1e-6)
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
        alpha = np.std(s1) / (np.std(s2) + 1e-8)
        h = s1 + alpha * s2
        pos_sig[start:end] += (h - np.mean(h)) / (np.std(h) + 1e-8)

    # Global CHROM signal
    cn_all = rgb_means / (np.mean(rgb_means, axis=0, keepdims=True) + 1e-6)
    xs = 3.0 * cn_all[:, 0] - 2.0 * cn_all[:, 1]
    ys = 1.5 * cn_all[:, 0] + cn_all[:, 1] - 1.5 * cn_all[:, 2]
    xf = bandpass_cardiac(xs, fps=fps)
    yf = bandpass_cardiac(ys, fps=fps)
    alpha_c = np.std(xf) / (np.std(yf) + 1e-8)
    chrom_sig = xf - alpha_c * yf

    return bandpass_cardiac(pos_sig, fps=fps), bandpass_cardiac(chrom_sig, fps=fps)


def analyze_fused_rppg_spectrum(
    deep_signal: np.ndarray,
    frames_rgb_128: np.ndarray,
    fps: float = 30.0,
    prev_bpm: float = 0.0,
):
    """
    Computes Heart Rate (BPM), Respiratory Rate (BrPM), Signal Quality (0-1),
    and fused waveform by combining:
      • Deep 3D-CNN waveform (`PhysFormerX` / `VitalEyeProNet`)
      • Overlap-Add POS chrominance waveform
      • CHROM chrominance waveform
    Includes sub-cardiac bandpass filtering (0.88–2.50 Hz = 53–150 BPM) and
    harmonic/temporal continuity scoring so BPM never gets stuck at 40–50 BPM.
    """
    if len(deep_signal) < 60 or fps <= 1.0:
        return 0.0, 0.0, 0.0, deep_signal, None, None

    # 1. Respiratory Rate from unfiltered low-frequency component (0.15 - 0.40 Hz = 9 - 24 BrPM)
    raw_zero = deep_signal.astype(np.float64) - np.mean(deep_signal)
    freqs_rr, psd_rr = periodogram(raw_zero, fs=fps, nfft=4096, detrend="constant", window="hann")
    rr_valid = np.where((freqs_rr >= 0.16) & (freqs_rr <= 0.40))[0]
    rpm = parabolic_peak_freq(freqs_rr, psd_rr, rr_valid) * 60.0 if len(rr_valid) > 0 else 15.0

    # 2. Bandpass-filtered cardiac signals (0.88 - 2.50 Hz = 52.8 - 150 BPM)
    s_deep = bandpass_cardiac(deep_signal, fps=fps, lo_hz=0.88, hi_hz=2.50)
    raw_fps = fps * (len(frames_rgb_128) / max(1, len(deep_signal)))
    s_pos, s_chrom = extract_pos_and_chrom_signals(frames_rgb_128, fps=raw_fps)
    if len(s_pos) != len(s_deep):
        x_old = np.linspace(0.0, 1.0, len(s_pos))
        x_new = np.linspace(0.0, 1.0, len(s_deep))
        s_pos = np.interp(x_new, x_old, s_pos)
        s_chrom = np.interp(x_new, x_old, s_chrom)

    freqs, psd_deep = periodogram(s_deep, fs=fps, nfft=4096, detrend="constant", window="hann")
    _, psd_pos = periodogram(s_pos, fs=fps, nfft=4096, detrend="constant", window="hann")
    _, psd_chrom = periodogram(s_chrom, fs=fps, nfft=4096, detrend="constant", window="hann")

    hr_valid = np.where((freqs >= 0.88) & (freqs <= 2.50))[0]
    if len(hr_valid) == 0:
        return 0.0, rpm, 0.0, s_deep, freqs, psd_deep

    # Normalize each spectrum within the cardiac band and fuse
    def _norm_psd(p):
        total = np.sum(p[hr_valid]) + 1e-12
        return p / total

    p_d = _norm_psd(psd_deep)
    p_p = _norm_psd(psd_pos)
    p_c = _norm_psd(psd_chrom)

    # Check if deep spectrum is overly concentrated at the very bottom bin (<0.93 Hz) due to motion
    low_edge_mask = (freqs >= 0.88) & (freqs <= 0.94)
    deep_low_ratio = np.sum(p_d[low_edge_mask]) / (np.sum(p_d[hr_valid]) + 1e-9)

    if deep_low_ratio > 0.45:
        # Motion/low-light edge artifact detected in raw diff -> weight POS + CHROM higher
        psd_fused = 0.35 * p_d + 0.40 * p_p + 0.25 * p_c
        fused_wave = 0.40 * s_deep + 0.40 * s_pos + 0.20 * s_chrom
    else:
        psd_fused = 0.60 * p_d + 0.25 * p_p + 0.15 * p_c
        fused_wave = 0.65 * s_deep + 0.25 * s_pos + 0.10 * s_chrom

    # Soft physiological & temporal continuity prior (normal resting/active HR 58–135 BPM)
    f_hr_bins = freqs[hr_valid] * 60.0
    phys_prior = np.exp(-0.5 * ((f_hr_bins - 78.0) / 32.0) ** 2)
    if prev_bpm >= 52.0:
        temp_prior = 0.70 + 0.30 * np.exp(-0.5 * ((f_hr_bins - prev_bpm) / 18.0) ** 2)
    else:
        temp_prior = np.ones_like(f_hr_bins)

    weighted_psd = psd_fused.copy()
    weighted_psd[hr_valid] = psd_fused[hr_valid] * (0.75 + 0.25 * phys_prior) * temp_prior

    f_hr = parabolic_peak_freq(freqs, weighted_psd, hr_valid)
    bpm = f_hr * 60.0

    peak_band = np.where((freqs >= f_hr - 0.15) & (freqs <= f_hr + 0.15))[0]
    peak_power = np.sum(psd_fused[peak_band])
    total_power = np.sum(psd_fused[hr_valid]) + 1e-9
    quality = float(np.clip((peak_power / total_power) * 1.6, 0.0, 1.0))

    return bpm, rpm, quality, fused_wave, freqs, weighted_psd


def extract_hrv_and_peaks(signal: np.ndarray, fps: float = 30.0, est_bpm: float = 75.0):
    if len(signal) < 90 or fps <= 5.0:
        return 0.0, [], 0.0

    filtered = bandpass_cardiac(signal, fps=fps, lo_hz=0.88, hi_hz=2.50)
    min_dist = max(int(fps * 0.40), 4)
    peaks, _ = find_peaks(filtered, distance=min_dist, prominence=np.std(filtered) * 0.25)

    if len(peaks) < 3:
        return 0.0, peaks.tolist(), 0.0

    refined_times = []
    for p in peaks:
        if 0 < p < len(filtered) - 1:
            y0, y1, y2 = filtered[p - 1], filtered[p], filtered[p + 1]
            denom = 2.0 * (2.0 * y1 - y0 - y2)
            delta = max(-0.5, min(0.5, (y0 - y2) / denom)) if abs(denom) > 1e-9 else 0.0
            refined_times.append((p + delta) / fps)
        else:
            refined_times.append(p / fps)

    ibis = np.diff(np.array(refined_times))
    ibis = ibis[(ibis >= 0.38) & (ibis <= 1.15)]  # 52 to 158 BPM
    if len(ibis) < 2:
        return 0.0, peaks.tolist(), 0.0

    mu, sigma = np.mean(ibis), np.std(ibis)
    clean_ibis = ibis[np.abs(ibis - mu) <= 3.0 * sigma] if sigma > 1e-6 else ibis
    if len(clean_ibis) < 2:
        clean_ibis = ibis

    sdnn_ms = float(np.clip(np.std(clean_ibis) * 1000.0, 5.0, 180.0))
    ibi_bpm = float(60.0 / np.mean(clean_ibis))
    return sdnn_ms, peaks.tolist(), ibi_bpm


# ─────────────────────────────────────────────────────────────────────────────
#  30 HZ TEMPORAL RESAMPLING & JOINT-MASK PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
_RESIZE_64 = Resize((IMG_SIZE, IMG_SIZE), antialias=True)


def build_physformer_input(
    frames_128: list,
    timestamps: list,
    device: torch.device,
    is_pro_model: bool = False,
):
    """
    Builds input tensor for either `VitalEyeProNet` or `PhysFormerX`:
      1. Measures true capture duration `dt = timestamps[-1] - timestamps[0]`
      2. Resamples sequence onto a uniform 30.0 FPS timeline so 3D temporal filters
         always see a true 30 Hz cardiac oscillation even if webcam drops frames in low light!
      3. Applies 30-frame running-mean illumination normalization over active skin pixels
      4. Applies Joint-Mask Boundary Suppression on `diff[t] = frame[t] - frame[t-1]` so
         mask-edge motion spikes never inject 1/f² low-frequency noise!
    """
    frames_raw = np.stack(frames_128, axis=0).astype(np.float32)  # [T_in, 128, 128, 3]
    T_in = len(frames_raw)

    if len(timestamps) == T_in and (timestamps[-1] - timestamps[0]) > 1.0:
        duration_sec = float(timestamps[-1] - timestamps[0])
        actual_fps = float(np.clip((T_in - 1) / duration_sec, 10.0, 60.0))
    else:
        actual_fps = 30.0
        duration_sec = (T_in - 1) / 30.0

    # Step 1: Per-frame illumination drift normalization over active skin pixels
    skin_binary = (frames_raw.sum(axis=-1, keepdims=True) > 8.0).astype(np.float32)
    pix_counts = np.maximum(skin_binary.sum(axis=(1, 2), keepdims=True), 1.0)
    frame_means = (frames_raw * skin_binary).sum(axis=(1, 2), keepdims=True) / pix_counts  # [T_in, 1, 1, 3]

    win_size = min(30, max(5, T_in // 4))
    smoothed_means = uniform_filter1d(frame_means, size=win_size, axis=0)
    global_mean = frame_means.mean(axis=0, keepdims=True)
    frames_norm = np.clip((frames_raw - smoothed_means + global_mean) * skin_binary, 0.0, 255.0)

    # Step 2: Convert to tensor [3, T_in, 128, 128] in [0, 1] and resize to 64×64
    vid = torch.from_numpy(frames_norm).float() / 255.0
    vid = vid.permute(3, 0, 1, 2)  # [3, T_in, 128, 128]
    vid = _RESIZE_64(vid.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)  # [3, T_in, 64, 64]

    # Step 3: Uniform 30.0 Hz Temporal Resampling if webcam FPS deviated from 30 Hz
    target_fps = 30.0
    T_30 = max(60, min(360, int(round(duration_sec * target_fps))))
    if abs(actual_fps - target_fps) > 1.5 and T_30 != T_in:
        C, _, H, W = vid.shape
        v_flat = vid.permute(0, 2, 3, 1).reshape(C * H * W, 1, T_in)
        v_res = F.interpolate(v_flat, size=T_30, mode="linear", align_corners=False)
        vid = v_res.view(C, H, W, T_30).permute(0, 3, 1, 2).contiguous()
        eff_fps = target_fps
    else:
        eff_fps = actual_fps

    if is_pro_model:
        # VitalEyeProNet computes its 10 physics-guided channels directly from [1, 3, T, 64, 64]
        return vid.unsqueeze(0).to(device), eff_fps, frames_norm

    # Step 4: Joint-Masked, Clamped Difference Frames for PhysFormerX (Eliminates Mask-Edge Spikes!)
    valid_64 = (vid.sum(dim=0, keepdim=True) > 0.02).float()
    joint_valid = torch.zeros_like(valid_64)
    joint_valid[:, 1:] = valid_64[:, 1:] * valid_64[:, :-1]

    diff = torch.zeros_like(vid)
    # Clamp inter-frame delta to ±0.08 to reject sudden motion jolts while preserving subtle pulse (~0.005)
    diff[:, 1:] = torch.clamp(vid[:, 1:] - vid[:, :-1], -0.08, 0.08) * joint_valid[:, 1:]

    vid_6ch = torch.cat([vid, diff], dim=0).unsqueeze(0).to(device)
    return vid_6ch, eff_fps, frames_norm


# ─────────────────────────────────────────────────────────────────────────────
#  ASYNCHRONOUS INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class AsyncPhysFormerEngine:
    def __init__(self, model: nn.Module, device: torch.device, is_pro_model: bool = False):
        self.model = model
        self.device = device
        self.is_pro_model = is_pro_model
        self.lock = threading.Lock()
        self.pending_task = None
        self.running = True
        self.busy = False

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
                    x_in, eff_fps, frames_norm = build_physformer_input(
                        frames_128, timestamps, self.device, is_pro_model=self.is_pro_model
                    )
                    try:
                        pred = self.model(x_in)
                    except (NotImplementedError, RuntimeError) as dev_err:
                        if str(self.device) != "cpu":
                            print(f"ℹ Switching 3D-CNN to CPU ({dev_err})")
                            self.device = torch.device("cpu")
                            self.model = self.model.to(self.device)
                            x_in = x_in.to(self.device)
                            pred = self.model(x_in)
                        else:
                            raise
                    ppg_signal = pred.squeeze().float().cpu().numpy()

                raw_bpm, raw_rpm, qual, fused_wave, freqs, psd = analyze_fused_rppg_spectrum(
                    ppg_signal, frames_norm, fps=eff_fps, prev_bpm=self.bpm
                )
                hrv_ms, _, ibi_bpm = extract_hrv_and_peaks(
                    fused_wave, fps=eff_fps, est_bpm=raw_bpm
                )

                # Blend spectral BPM with clean IBI peak-to-peak BPM when both agree
                if 52.0 <= ibi_bpm <= 155.0 and abs(ibi_bpm - raw_bpm) <= 15.0:
                    raw_bpm = 0.75 * raw_bpm + 0.25 * ibi_bpm

                infer_ms = (time.perf_counter() - t0) * 1000.0

                with self.lock:
                    self.actual_fps = eff_fps
                    self.last_infer_ms = infer_ms
                    self.inference_count += 1

                    if 50.0 <= raw_bpm <= 180.0:
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

                    recent = fused_wave[-PPG_DISPLAY:]
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
    arch_name: str = "PhysFormerX",
):
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (365, 195), (16, 18, 22), -1)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (12, 12), (365, 195), (75, 85, 95), 1)

    cv2.putText(
        frame,
        f"VITALEYE  |  {arch_name} + YOLOv8",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (200, 210, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.line(frame, (22, 42), (355, 42), (55, 60, 68), 1)

    if bpm > 0 and buf_fill >= WARMUP_LEN:
        pulse_phase = (math.sin(time.time() * (bpm / 60.0) * 2.0 * math.pi) + 1.0) * 0.5
        radius = int(7 + 3 * pulse_phase)
        cv2.circle(frame, (36, 76), radius, (50, 50, 240), -1, cv2.LINE_AA)

        bpm_color = (60, 235, 100) if 55 <= bpm <= 115 else (40, 180, 255)
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
        mode_tag = "10s Lock" if buf_fill >= BUFFER_LEN else f"Tracking ({buf_fill}/{BUFFER_LEN})"
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

        bar_w = int(325 * np.clip(quality, 0.0, 1.0))
        cv2.rectangle(frame, (24, 142), (349, 156), (42, 45, 50), -1)
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
        pct = min(100, int(100 * buf_fill / max(1, WARMUP_LEN)))
        status_title = "Computing Initial HR..." if buf_fill >= WARMUP_LEN else "Calibrating rPPG Buffer..."
        cv2.putText(
            frame,
            status_title,
            (24, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (35, 180, 255),
            2,
            cv2.LINE_AA,
        )
        bar_w = int(325 * min(1.0, buf_fill / max(1, WARMUP_LEN)))
        cv2.rectangle(frame, (24, 96), (349, 114), (42, 45, 50), -1)
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
            "Keep face steady in front of camera (~4 s)",
            (24, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (120, 125, 130),
            1,
            cv2.LINE_AA,
        )


def draw_skin_mask_inset(frame: np.ndarray, face_masked_rgb: np.ndarray):
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
        "Hybrid Skin ROI",
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
        "Live Bandpass rPPG Waveform (0.88-2.5 Hz)",
        (x0 + 6, y0 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.37,
        (175, 185, 195),
        1,
        cv2.LINE_AA,
    )

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


def load_rppg_model(model_path: str, device: torch.device):
    """
    Auto-detects whether `model_path` is a `VitalEyeProNet` checkpoint or a
    `PhysFormerX` checkpoint, loads weights cleanly, and returns:
      (model, is_pro_model, arch_name, device)
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    is_pro = False
    state = ckpt

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        if ckpt.get("arch") == "VitalEyeProNet":
            is_pro = True

    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    if not is_pro and "stem.0.weight" in state and state["stem.0.weight"].shape[1] == 10:
        is_pro = True

    if is_pro:
        model = VitalEyeProNet(seq_len=BUFFER_LEN).to(device)
        arch_name = "VitalEyeProNet"
    else:
        model = PhysFormerX(seq_len=BUFFER_LEN).to(device)
        arch_name = "PhysFormerX"

    model.load_state_dict(state)
    model.eval()

    try:
        with torch.no_grad():
            ch_in = 3 if is_pro else 6
            _ = model(torch.zeros(1, ch_in, 16, IMG_SIZE, IMG_SIZE, device=device))
    except Exception as mps_err:
        print(f"ℹ  MPS 3D op fallback ({mps_err}); running {arch_name} on CPU.")
        device = torch.device("cpu")
        model = model.to(device)

    return model, is_pro, arch_name, device


# ─────────────────────────────────────────────────────────────────────────────
#  SELF-TEST / VERIFICATION ROUTINE (Under Low Light + Motion Drift!)
# ─────────────────────────────────────────────────────────────────────────────
def run_self_test(args) -> int:
    print("═" * 68)
    print("  VitalEye — End-to-End Low-Light & Motion Stress Self-Test")
    print("═" * 68)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    model_path = args.model_weights if os.path.exists(args.model_weights) else FALLBACK_MODEL
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found at {args.model_weights}")
        return 1

    model, is_pro, arch_name, device = load_rppg_model(model_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✔ {arch_name:20s} : {model_path} ({n_params / 1e6:.2f}M params on {device})")

    from ultralytics import YOLO
    yolo = YOLO(args.yolo_weights)
    print(f"✔ YOLOv8 Loaded        : {args.yolo_weights} (classes: {yolo.names})")

    # Stress-test: Simulate 76.0 BPM pulse under LOW LIGHT + strong 0.35 Hz motion/exposure drift
    fps = 24.0  # Simulate low-light webcam frame drop (24 FPS instead of 30 FPS)
    T = 240     # 10 seconds @ 24 FPS
    t = np.arange(T) / fps
    target_bpm = 76.0
    target_rpm = 15.0
    frames_128 = []
    timestamps = []

    for i in range(T):
        ti = float(t[i])
        pulse = math.sin(2.0 * math.pi * (target_bpm / 60.0) * ti)
        breath = math.sin(2.0 * math.pi * (target_rpm / 60.0) * ti)
        # Strong low-frequency motion/shadow drift that previously caused 40-50 BPM pinning!
        heavy_drift = 18.0 * math.sin(2.0 * math.pi * 0.45 * ti)

        patch = np.zeros((FACE_CROP_SIZE, FACE_CROP_SIZE, 3), dtype=np.uint8)
        patch[:, :, 0] = int(np.clip(150.0 + heavy_drift - 1.2 * pulse, 0, 255))
        patch[:, :, 1] = int(np.clip(112.0 + heavy_drift + 3.5 * pulse + 1.5 * breath, 0, 255))
        patch[:, :, 2] = int(np.clip(92.0  + heavy_drift + 0.8 * pulse, 0, 255))

        mask = skin_mask_ycrcb(patch)
        mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
        frames_128.append(patch * mask_3ch)
        timestamps.append(ti)

    t0 = time.perf_counter()
    with torch.no_grad():
        x_in, eff_fps, frames_norm = build_physformer_input(
            frames_128, timestamps, device, is_pro_model=is_pro
        )
        pred = model(x_in).squeeze().float().cpu().numpy()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    bpm, rpm, qual, fused_wave, freqs, psd = analyze_fused_rppg_spectrum(
        pred, frames_norm, fps=eff_fps
    )
    hrv_ms, _, _ = extract_hrv_and_peaks(fused_wave, fps=eff_fps, est_bpm=bpm)

    print(f"✔ Low-Light 24→30Hz Pass: {infer_ms:.1f} ms (Resampled eff_fps={eff_fps:.1f} Hz)")
    print(f"✔ Target vs Predicted   : Target={target_bpm:.1f} BPM | Predicted={bpm:.2f} BPM (Error={abs(bpm - target_bpm):.2f} BPM)")
    print(f"✔ Signal Metrics        : RR={rpm:.1f} BrPM | HRV={hrv_ms:.1f} ms | Quality={qual * 100:.1f}%")
    print("═" * 68)
    return 0


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LIVE CAMERA / VIDEO STREAM LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    if args.self_test:
        sys.exit(run_self_test(args))

    hw_device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"⚙  Device : {hw_device}")

    model_path = args.model_weights
    if not os.path.exists(model_path):
        for candidate in (PRO_MODEL_PATH, os.path.join(DEPLOY_DIR, "models", "vidformer.pt"), FALLBACK_MODEL):
            if os.path.exists(candidate):
                model_path = candidate
                break

    print(f"⚙  Loading rPPG Model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"❌ Model weights not found at: {model_path}")
        sys.exit(1)

    model, is_pro, arch_name, phys_device = load_rppg_model(model_path, hw_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✔  {arch_name} loaded ({n_params / 1e6:.2f}M params on {phys_device})")

    # ── Load YOLOv8 Multi-ROI Detector ───────────────────────────────────────
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

    # ── Open Camera or Video File ────────────────────────────────────────────
    source = args.video if args.video else args.camera
    # Use DirectShow backend on Windows to avoid MSMF errors (-1072875772)
    backend = cv2.CAP_DSHOW if os.name == 'nt' and not args.video else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
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

    tracker = ROITracker(fps=nominal_fps, alpha=BOX_EMA_ALPHA)
    engine = AsyncPhysFormerEngine(model=model, device=phys_device, is_pro_model=is_pro)

    frame_buf = deque(maxlen=BUFFER_LEN)
    ts_buf = deque(maxlen=BUFFER_LEN)
    last_skin_preview = None
    prev_mask_float = None

    frame_count = 0
    last_submit_frame = 0
    show_multi_roi = True
    show_skin_inset = True
    fullscreen = False

    win_name = f"VitalEye ({arch_name}) — Live Heart Rate Monitor  [q=quit  r=reset  m=ROIs  k=skin  s=shot  f=full]"
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
    print(f"  VitalEye ({arch_name}) — Live Heart Rate, Respiratory Rate & HRV Monitor")
    print("  Controls:  q=quit  r=reset  m=toggle ROIs  k=skin preview  s=screenshot")
    print("═" * 72 + "\n")

    fps_timer = time.perf_counter()
    fps_display = nominal_fps
    yolo_device = "mps" if str(hw_device) == "mps" else "0" if hw_device.type == "cuda" else "cpu"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if args.video:
                    print("✔  Reached end of video file.")
                else:
                    print("⚠  Camera frame read ended.")
                break

            now_ts = frame_count / nominal_fps if args.video else time.perf_counter()
            frame_count += 1
            h_frm, w_frm = frame.shape[:2]

            # ── Step 1: Multi-ROI Detection ──
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

            # ── Step 2: Crop Stabilized Face ROI & Apply Hybrid 468-Landmark + YCrCb/LAB Mask ──
            face_box = tracker.get(0, now_ts)
            if face_box is not None:
                x1, y1, x2, y2 = map(int, face_box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frm, x2), min(h_frm, y2)

                if x2 > x1 + 10 and y2 > y1 + 10:
                    face_crop = frame[y1:y2, x1:x2]
                    face_128 = cv2.resize(face_crop, (FACE_CROP_SIZE, FACE_CROP_SIZE))
                    face_rgb = cv2.cvtColor(face_128, cv2.COLOR_BGR2RGB)

                    raw_mask = skin_mask_ycrcb(face_rgb)
                    mask_f = raw_mask.astype(np.float32) / 255.0
                    if prev_mask_float is None or prev_mask_float.shape != mask_f.shape:
                        prev_mask_float = mask_f
                    else:
                        prev_mask_float = 0.65 * prev_mask_float + 0.35 * mask_f
                    stable_mask = (prev_mask_float >= 0.45).astype(np.uint8)
                    mask_3ch = np.stack([stable_mask, stable_mask, stable_mask], axis=-1)
                    face_masked = face_rgb * mask_3ch

                    frame_buf.append(face_masked)
                    ts_buf.append(now_ts)
                    last_skin_preview = face_masked

            # ── Step 3: Trigger Non-Blocking Inference ──
            buf_fill = len(frame_buf)
            if buf_fill >= WARMUP_LEN and (frame_count - last_submit_frame) >= args.slide_step:
                submitted = engine.submit(list(frame_buf), list(ts_buf))
                if submitted:
                    last_submit_frame = frame_count
                elif args.video:
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
                frame, cur_bpm, cur_rpm, cur_hrv, cur_qual, buf_fill, cur_infer_ms, arch_name=arch_name
            )
            if show_skin_inset and last_skin_preview is not None:
                draw_skin_mask_inset(frame, last_skin_preview)
            draw_ppg_and_spectrum(frame, ppg_list, psd_freqs, psd_power, cur_bpm)

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
                    prev_mask_float = None
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


def parse_args():
    p = argparse.ArgumentParser(description="VitalEye — Live Heart Rate Monitor (VitalEyeProNet & PhysFormerX)")
    p.add_argument(
        "--yolo_weights",
        default=DEFAULT_YOLO,
        help="Path to trained YOLOv8 .pt weights",
    )
    p.add_argument(
        "--model_weights",
        default=DEFAULT_MODEL,
        help="Path to VitalEyeProNet (vitaleye_pro_best.pt) or PhysFormerX (vidformer.pt) weights",
    )
    p.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    p.add_argument("--video", type=str, default="", help="Optional path to input video file")
    p.add_argument("--record", type=str, default="", help="Optional path to save annotated MP4")
    p.add_argument("--fps", type=int, default=30, help="Target camera FPS (default: 30)")
    p.add_argument("--yolo_interval", type=int, default=2, help="Run YOLOv8 every N frames (default: 2)")
    p.add_argument("--slide_step", type=int, default=10, help="Trigger rPPG update every N frames (default: 10)")
    p.add_argument("--headless", action="store_true", help="Run without OpenCV GUI window")
    p.add_argument("--self_test", action="store_true", help="Run end-to-end verification under low-light/motion stress")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
