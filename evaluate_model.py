"""
Real-Time Webcam Heart Rate Monitor using PhysFormerX
═════════════════════════════════════════════════════
Opens your webcam, detects your face with YOLO, runs the trained
PhysFormerX model to extract the rPPG signal, and displays:
  • Live heart rate (BPM) on screen
  • Real-time PPG waveform graph overlay
  • Face bounding box with skin masking indicator

Exactly matches Vidformer1.py preprocessing:
  • Face detected → resized to 64×64 → skin-masked (YCbCr)
  • Difference frames computed: diff[t] = frame[t] - frame[t-1]
  • Stacked to 6-channel [RGB, diff] → model → PPG waveform → BPM

Usage:
  python evaluate_model.py
  Press 'q' to quit.
"""

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from scipy.signal import periodogram, butter, filtfilt

# ─────────────────────────────────────────────────────────
# Import model architecture from Vidformer1.py
# ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Vidformer1_pichla import PhysFormerX

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
MODEL_PATH    = 'E:/CS776 Project/yolo_best.pt'
YOLO_PATH     = 'E:/CS776 Project/best.pt'
IMG_SIZE      = 64          # must match training
BUFFER_LEN    = 300         # frames needed for one prediction (10s @ 30fps)
SLIDE_STEP    = 10          # slide buffer by N frames per prediction
FPS_TARGET    = 30
EMA_ALPHA     = 0.7         # face-box smoothing
BPM_SMOOTH    = 0.75        # BPM EMA smoothing (higher = more stable)

# PPG waveform display
PPG_PLOT_W    = 400         # width of the PPG plot area (pixels)
PPG_PLOT_H    = 120         # height of the PPG plot area (pixels)
PPG_DISPLAY   = 150         # number of recent PPG samples to display


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────
def skin_mask_ycrcb(frame_rgb):
    """Binary skin mask via YCbCr thresholding — same as preprocessor."""
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def smooth_box(prev_box, new_box, alpha=EMA_ALPHA):
    """Exponential moving average of bounding box [x1, y1, x2, y2]."""
    if prev_box is None:
        return new_box
    return [alpha * p + (1 - alpha) * n for p, n in zip(prev_box, new_box)]


def bpm_from_ppg(signal, fps=FPS_TARGET):
    """Compute BPM from PPG signal using periodogram."""
    if len(signal) < 60:
        return 0.0
    s = signal - np.mean(signal)
    freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend='constant',
                             window='hann')
    valid = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
    if len(valid) > 0:
        return float(freqs[valid][np.argmax(psd[valid])] * 60.0)
    return 0.0


def prepare_model_input(frame_buffer, device):
    """
    Convert frame buffer to model input tensor.
    Exactly replicates the preprocessing from rPPGDataset.__getitem__:
      1. frames are already 64×64 RGB float [0,1] with skin masking
      2. Compute difference frames
      3. Stack RGB + diff → 6-channel
      4. Add batch dimension
    """
    # frame_buffer: list of [H, W, 3] float32 arrays in [0, 1]
    frames_np = np.array(frame_buffer)  # [T, H, W, 3]

    # Convert to tensor [3, T, H, W]
    vid = torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2)

    # Compute difference frames: diff[t] = frame[t] - frame[t-1]
    diff = torch.zeros_like(vid)
    diff[:, 1:] = vid[:, 1:] - vid[:, :-1]

    # Stack → 6 channels [6, T, H, W]
    vid_6ch = torch.cat([vid, diff], dim=0)

    # Add batch dimension → [1, 6, T, H, W]
    return vid_6ch.unsqueeze(0).to(device)


def draw_ppg_waveform(frame, ppg_history, x_offset, y_offset, w, h):
    """Draw a real-time PPG waveform graph on the frame."""
    if len(ppg_history) < 2:
        return

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_offset - 5, y_offset - 5),
                  (x_offset + w + 5, y_offset + h + 25),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Draw border
    cv2.rectangle(frame, (x_offset, y_offset),
                  (x_offset + w, y_offset + h),
                  (60, 60, 60), 1)

    # Title
    cv2.putText(frame, "PPG Signal", (x_offset + 5, y_offset - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Normalize PPG to fit in plot area
    ppg_arr = np.array(list(ppg_history))
    if ppg_arr.std() < 1e-6:
        return

    ppg_norm = (ppg_arr - ppg_arr.mean()) / (ppg_arr.std() + 1e-8)
    ppg_norm = np.clip(ppg_norm, -3, 3)

    # Map to pixel coordinates
    n = len(ppg_norm)
    x_step = w / max(n - 1, 1)
    y_mid = y_offset + h // 2
    y_scale = h / 6.0  # ±3 std maps to full height

    points = []
    for i in range(n):
        px = int(x_offset + i * x_step)
        py = int(y_mid - ppg_norm[i] * y_scale)
        py = max(y_offset + 2, min(y_offset + h - 2, py))
        points.append((px, py))

    # Draw waveform line (green, like heart monitor)
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (0, 255, 100), 2,
                 cv2.LINE_AA)

    # Draw a small dot at the latest point
    if points:
        cv2.circle(frame, points[-1], 4, (0, 255, 255), -1)

    # Draw grid lines
    for frac in [0.25, 0.5, 0.75]:
        gy = y_offset + int(h * frac)
        cv2.line(frame, (x_offset, gy), (x_offset + w, gy),
                 (40, 40, 40), 1)


def draw_hr_display(frame, bpm, confidence, buffer_fill):
    """Draw a premium heart rate display on the frame."""
    h, w = frame.shape[:2]

    # Background panel (top-left)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 130), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (320, 130), (60, 60, 60), 1)

    if bpm > 0 and buffer_fill >= BUFFER_LEN:
        # Heart icon (simple)
        heart_color = (0, 0, 255)
        cv2.putText(frame, chr(9829), (20, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, heart_color, 3, cv2.LINE_AA)

        # BPM value
        bpm_text = f"{bpm:.0f}"
        cv2.putText(frame, bpm_text, (60, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    1.8, (0, 255, 100), 3, cv2.LINE_AA)

        # BPM label
        cv2.putText(frame, "BPM", (210, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (180, 180, 180), 2, cv2.LINE_AA)

        # Confidence bar
        bar_w = int(280 * min(confidence, 1.0))
        cv2.rectangle(frame, (20, 80), (300, 90), (40, 40, 40), -1)
        bar_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        cv2.rectangle(frame, (20, 80), (20 + bar_w, 90), bar_color, -1)
        cv2.putText(frame, "Signal Quality", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
    else:
        # Calibrating
        pct = int(100 * buffer_fill / BUFFER_LEN)
        cv2.putText(frame, "Calibrating...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # Progress bar
        bar_w = int(280 * buffer_fill / BUFFER_LEN)
        cv2.rectangle(frame, (20, 70), (300, 85), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 70), (20 + bar_w, 85), (0, 165, 255), -1)
        cv2.putText(frame, f"{pct}% ({buffer_fill}/{BUFFER_LEN} frames)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (140, 140, 140), 1)


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────
def run_live():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Load PhysFormerX model ──────────────────────────────
    model = PhysFormerX(seq_len=300).to(device)
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device,
                                weights_only=True)
        # Handle DataParallel wrapper
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v
                          for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✔ Model loaded from {MODEL_PATH}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params/1e6:.2f}M")
    except Exception as e:
        print(f"✘ Error loading model: {e}")
        print(f"  Make sure {MODEL_PATH} exists (train with Vidformer1.py first)")
        return

    # ── Load YOLO face detector ─────────────────────────────
    try:
        from ultralytics import YOLO
        yolo = YOLO(YOLO_PATH, task='detect')
        use_yolo = True
        print(f"✔ YOLO loaded from {YOLO_PATH}")
    except Exception as e:
        print(f"⚠ YOLO not available ({e}), using Haar cascade fallback")
        use_yolo = False
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # ── Open webcam ─────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("✘ Cannot open webcam!")
        return

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or FPS_TARGET
    print(f"✔ Webcam opened (fps={actual_fps:.0f})")

    # ── State ───────────────────────────────────────────────
    frame_buffer = deque(maxlen=BUFFER_LEN)
    ppg_history = deque(maxlen=PPG_DISPLAY)
    bpm_history = deque(maxlen=10)  # for averaging
    current_bpm = 0.0
    prev_box = None
    frame_count = 0
    signal_quality = 0.0

    print("\n" + "=" * 55)
    print("  PhysFormerX Live Heart Rate Monitor")
    print("  Press 'q' to quit")
    print("=" * 55 + "\n")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            h_frame, w_frame = frame.shape[:2]

            # ── Face detection ─────────────────────────────
            face_found = False
            face_rgb = None

            if use_yolo:
                results = yolo(frame, classes=[0], verbose=False)
                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0].xyxy[0].cpu().numpy().tolist()
                    box = [max(0, box[0]), max(0, box[1]),
                           min(w_frame, box[2]), min(h_frame, box[3])]
                    prev_box = smooth_box(prev_box, box)
                    face_found = True
                elif prev_box is not None:
                    face_found = True
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if len(faces) > 0:
                    x, y, w, hh = faces[0]
                    box = [x, y, x + w, y + hh]
                    prev_box = smooth_box(prev_box, box)
                    face_found = True
                elif prev_box is not None:
                    face_found = True

            if face_found and prev_box is not None:
                x1, y1, x2, y2 = [int(c) for c in prev_box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame, x2), min(h_frame, y2)

                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]

                    # Draw face box
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)

                    # Process face: resize → RGB → skin mask → normalize
                    face_resized = cv2.resize(face_crop,
                                              (IMG_SIZE, IMG_SIZE))
                    face_rgb_full = cv2.cvtColor(face_resized,
                                                 cv2.COLOR_BGR2RGB)

                    # Apply skin mask (same as preprocessor)
                    mask = skin_mask_ycrcb(face_rgb_full)
                    mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
                    face_masked = face_rgb_full * mask_3ch

                    # Normalize to [0, 1] float
                    face_float = face_masked.astype(np.float32) / 255.0
                    frame_buffer.append(face_float)

            # ── Run model when buffer is full ──────────────
            buf_len = len(frame_buffer)
            if buf_len >= BUFFER_LEN and frame_count % SLIDE_STEP == 0:
                # Build 6-channel input (RGB + diff)
                input_tensor = prepare_model_input(
                    list(frame_buffer), device)

                # Forward pass
                pred = model(input_tensor)
                ppg_signal = pred.squeeze().cpu().numpy()

                # Compute BPM
                bpm = bpm_from_ppg(ppg_signal, fps=FPS_TARGET)

                if bpm > 40 and bpm < 180:
                    bpm_history.append(bpm)
                    # Smoothed BPM
                    if current_bpm > 0:
                        current_bpm = (BPM_SMOOTH * current_bpm +
                                       (1 - BPM_SMOOTH) * bpm)
                    else:
                        current_bpm = bpm

                    # Signal quality from spectral peak sharpness
                    s = ppg_signal - np.mean(ppg_signal)
                    freqs, psd = periodogram(s, fs=FPS_TARGET, nfft=4096,
                                             window='hann')
                    valid = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
                    if len(valid) > 0:
                        peak = psd[valid].max()
                        total = psd[valid].sum() + 1e-8
                        signal_quality = min(1.0, (peak / total) * 5.0)

                # Update PPG display history (last portion of signal)
                recent = ppg_signal[-PPG_DISPLAY:]
                ppg_history.clear()
                ppg_history.extend(recent)

                # Slide buffer
                for _ in range(SLIDE_STEP):
                    if frame_buffer:
                        frame_buffer.popleft()

            # ── Draw UI ────────────────────────────────────
            # HR display (top-left)
            draw_hr_display(frame, current_bpm, signal_quality, buf_len)

            # PPG waveform (bottom-right)
            ppg_x = w_frame - PPG_PLOT_W - 20
            ppg_y = h_frame - PPG_PLOT_H - 40
            draw_ppg_waveform(frame, ppg_history, ppg_x, ppg_y,
                              PPG_PLOT_W, PPG_PLOT_H)

            # FPS counter (bottom-left)
            cv2.putText(frame, f"Frame: {frame_count}",
                        (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (100, 100, 100), 1)

            # Show
            cv2.imshow('PhysFormerX - Live Heart Rate Monitor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    if current_bpm > 0:
        print(f"\n{'='*40}")
        print(f"  Final Heart Rate: {current_bpm:.0f} BPM")
        print(f"{'='*40}")


if __name__ == '__main__':
    run_live()