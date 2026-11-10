"""
Live Heart Rate Monitor — PhysFormerX + YOLOv8
═══════════════════════════════════════════════
Reads from your webcam, detects your face with the trained YOLOv8 model,
buffers 300 frames (~10 s @ 30 fps), runs the PhysFormerX rPPG model, and
displays:

  • Heart rate (BPM) with a live confidence bar
  • Real-time PPG waveform overlay
  • YOLOv8 face bounding box + skin-mask preview

Controls:
  q  — quit
  r  — reset / recalibrate buffer
  s  — save a screenshot
  f  — toggle fullscreen

Usage:
  cd "HeartRate"
  python live_hr_monitor.py

  Optional flags:
    --yolo_weights   path to YOLOv8 .pt  (default: VitalEye_CS776_Deploy/models/yolov8/weights/best.pt)
    --model_weights  path to PhysFormerX  (default: VitalEye_CS776_Deploy/models/vidformer.pt)
    --camera         camera index         (default: 0)
    --fps            target fps           (default: 30)
"""

import os
import sys
import cv2
import time
import argparse
import numpy as np
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn

from scipy.signal import periodogram

# ─── Add deploy folder to path so we can import physformer & preprocessor ───
DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "VitalEye_CS776_Deploy")
sys.path.insert(0, DEPLOY_DIR)

from physformer import PhysFormerX
from preprocessor import skin_mask_ycrcb


# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_YOLO    = os.path.join(DEPLOY_DIR, "models", "yolov8", "weights", "best.pt")
DEFAULT_MODEL   = os.path.join(DEPLOY_DIR, "models", "vidformer.pt")
BUFFER_LEN      = 300        # frames before first prediction (10 s @ 30 fps)
SLIDE_STEP      = 10         # slide window every N new frames  → update ~3×/sec
IMG_SIZE        = 64         # PhysFormerX spatial input (matches training)
BPM_EMA_ALPHA   = 0.75       # higher = more stable / slower to react
BOX_EMA_ALPHA   = 0.80       # face-box smoothing
PPG_DISPLAY     = 150        # number of recent PPG samples shown in graph

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def smooth_box(prev, new, alpha=BOX_EMA_ALPHA):
    """EMA smoothing of [x1,y1,x2,y2] bounding box."""
    if prev is None:
        return new
    return [alpha * p + (1 - alpha) * n for p, n in zip(prev, new)]


def bpm_from_ppg(signal: np.ndarray, fps: float = 30.0) -> float:
    """Periodogram-based BPM from a 1-D PPG array."""
    if len(signal) < 60:
        return 0.0
    s = signal - np.mean(signal)
    freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend='constant', window='hann')
    valid = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
    if len(valid) > 0:
        return float(freqs[valid][np.argmax(psd[valid])] * 60.0)
    return 0.0


def signal_quality_score(signal: np.ndarray, fps: float = 30.0) -> float:
    """Return 0–1 score: ratio of HR-band spectral peak to total band power."""
    if len(signal) < 60:
        return 0.0
    s = signal - np.mean(signal)
    freqs, psd = periodogram(s, fs=fps, nfft=4096, detrend='constant', window='hann')
    valid = np.where((freqs >= 0.75) & (freqs <= 2.5))[0]
    if len(valid) > 0:
        peak  = psd[valid].max()
        total = psd[valid].sum() + 1e-9
        return float(min(1.0, (peak / total) * 5.0))
    return 0.0


def build_model_input(frame_buf: list, device: torch.device) -> torch.Tensor:
    """
    Convert list of [H,W,3] float32 [0,1] face frames → model input.
    Exactly replicates training preprocessing:
      RGB frames  +  difference frames  →  [1, 6, T, 64, 64]
    """
    frames_np = np.stack(frame_buf, axis=0)             # [T, H, W, 3]
    vid = torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2)  # [3, T, H, W]
    diff = torch.zeros_like(vid)
    diff[:, 1:] = vid[:, 1:] - vid[:, :-1]
    vid_6ch = torch.cat([vid, diff], dim=0).unsqueeze(0)  # [1, 6, T, H, W]
    return vid_6ch.to(device)


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def draw_hr_panel(frame: np.ndarray, bpm: float, quality: float, buf_fill: int):
    """Top-left heart-rate / calibration panel."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (340, 150), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (340, 150), (60, 60, 60), 1)

    if bpm > 0 and buf_fill >= BUFFER_LEN:
        # ── HEART ICON (simple drawn heart) ──
        heart_color = (60, 60, 220)
        cv2.putText(frame, "HR", (22, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, heart_color, 2, cv2.LINE_AA)

        # ── BPM VALUE ──
        bpm_color = (50, 220, 50) if 50 <= bpm <= 110 else (30, 165, 255)
        cv2.putText(frame, f"{bpm:.0f}", (90, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, bpm_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (260, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2, cv2.LINE_AA)

        # ── QUALITY BAR ──
        bar_w = int(300 * quality)
        cv2.rectangle(frame, (20, 95), (320, 112), (40, 40, 40), -1)
        bar_col = (50, 220, 50) if quality > 0.65 else \
                  (30, 165, 255) if quality > 0.3 else (40, 60, 220)
        cv2.rectangle(frame, (20, 95), (20 + bar_w, 112), bar_col, -1)
        cv2.putText(frame, f"Signal quality  {quality*100:.0f}%",
                    (20, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)

    else:
        # ── CALIBRATING ──
        pct = int(100 * buf_fill / BUFFER_LEN)
        cv2.putText(frame, "Calibrating...", (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 165, 255), 2, cv2.LINE_AA)
        bar_w = int(300 * buf_fill / BUFFER_LEN)
        cv2.rectangle(frame, (20, 75), (320, 95), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 75), (20 + bar_w, 95), (30, 165, 255), -1)
        cv2.putText(frame, f"{pct}%  ({buf_fill}/{BUFFER_LEN} frames collected)",
                    (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
        cv2.putText(frame, "Keep face still for ~10 s",
                    (20, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1)


def draw_ppg_waveform(frame: np.ndarray, ppg_buf: deque,
                      x0: int, y0: int, pw: int, ph: int):
    """Bottom-right PPG waveform graph."""
    if len(ppg_buf) < 2:
        return

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 20), (x0 + pw + 6, y0 + ph + 10), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (55, 55, 55), 1)
    cv2.putText(frame, "rPPG Signal", (x0 + 4, y0 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    arr = np.array(list(ppg_buf), dtype=np.float32)
    std = arr.std()
    if std < 1e-7:
        return
    arr_norm = np.clip((arr - arr.mean()) / (std + 1e-8), -3.0, 3.0)

    n = len(arr_norm)
    x_step = pw / max(n - 1, 1)
    y_mid  = y0 + ph // 2
    y_scl  = ph / 6.5

    pts = []
    for i, v in enumerate(arr_norm):
        px = int(x0 + i * x_step)
        py = int(np.clip(y_mid - v * y_scl, y0 + 2, y0 + ph - 2))
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (30, 220, 90), 2, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 4, (0, 255, 230), -1)

    # Horizontal grid lines
    for frac in (0.25, 0.5, 0.75):
        gy = y0 + int(ph * frac)
        cv2.line(frame, (x0, gy), (x0 + pw, gy), (38, 38, 38), 1)


def draw_face_box(frame: np.ndarray, box: list, bpm: float, buf_fill: int):
    """Annotated face bounding box."""
    if box is None:
        return
    x1, y1, x2, y2 = map(int, box)
    h_f, w_f = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_f, x2), min(h_f, y2)

    color = (50, 220, 50) if bpm > 0 else (30, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if bpm > 0 and buf_fill >= BUFFER_LEN:
        label = f"HR: {bpm:.0f} BPM"
    else:
        label = "Detecting face..."

    ts, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - ts[1] - 10), (x1 + ts[0] + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙  Device : {device}")

    # ── 1. Load PhysFormerX ────────────────────────────────────────────────
    print(f"⚙  Loading PhysFormerX from: {args.model_weights}")
    if not os.path.exists(args.model_weights):
        print(f"❌ PhysFormerX weights not found: {args.model_weights}")
        sys.exit(1)

    model = PhysFormerX(seq_len=BUFFER_LEN).to(device)
    state = torch.load(args.model_weights, map_location=device, weights_only=True)
    # Strip DataParallel prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✔  PhysFormerX loaded  ({n_params/1e6:.2f}M params)")

    # ── 2. Load YOLO face detector ─────────────────────────────────────────
    use_yolo = False
    yolo = None
    face_cascade = None

    if os.path.exists(args.yolo_weights):
        try:
            from ultralytics import YOLO as _YOLO
            yolo = _YOLO(args.yolo_weights)
            use_yolo = True
            print(f"✔  YOLOv8 loaded from: {args.yolo_weights}")
        except Exception as e:
            print(f"⚠  YOLOv8 load failed ({e}), falling back to Haar cascade")
    else:
        print(f"⚠  YOLOv8 weights not found at '{args.yolo_weights}', using Haar cascade")

    if not use_yolo:
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(haar_path)
        print("✔  Haar cascade face detector loaded (fallback)")

    # ── 3. Open webcam ─────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FPS,          args.fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        sys.exit(1)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    print(f"✔  Camera opened  (fps={actual_fps:.0f})")

    # ── 4. State ───────────────────────────────────────────────────────────
    frame_buf   = deque(maxlen=BUFFER_LEN)
    ppg_history = deque(maxlen=PPG_DISPLAY)
    current_bpm = 0.0
    signal_q    = 0.0
    prev_box    = None
    frame_count = 0
    fullscreen  = False
    win_name    = "PhysFormerX — Live Heart Rate Monitor   [q=quit  r=reset  s=screenshot  f=fullscreen]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 500)

    print("\n" + "═" * 60)
    print("  PhysFormerX — Live Heart Rate Monitor")
    print("  Controls:  q=quit   r=reset   s=screenshot   f=fullscreen")
    print("═" * 60 + "\n")

    fps_timer   = time.time()
    fps_display = 0.0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠  Camera read failed.")
                break

            frame_count += 1
            h_frm, w_frm = frame.shape[:2]

            # ── Face detection ──────────────────────────────────────────
            face_found = False

            if use_yolo:
                # Detect only class 0 = "face" (trained model) or use standard person
                results = yolo(frame, verbose=False, imgsz=320,
                               device="mps" if str(device) == "mps" else
                                      "0"   if device.type == "cuda" else "cpu")
                # Try class 0 first (custom YOLO trained on face/nose/chest)
                if len(results[0].boxes) > 0:
                    # Pick highest confidence box
                    idx = int(results[0].boxes.conf.cpu().argmax())
                    box_raw = results[0].boxes[idx].xyxy[0].cpu().numpy().tolist()
                    box_clipped = [
                        max(0, box_raw[0]), max(0, box_raw[1]),
                        min(w_frm, box_raw[2]), min(h_frm, box_raw[3])
                    ]
                    prev_box   = smooth_box(prev_box, box_clipped)
                    face_found = True
                elif prev_box is not None:
                    face_found = True  # use last known box
            else:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    x, y, w_, h_ = faces[0]
                    prev_box   = smooth_box(prev_box, [x, y, x + w_, y + h_])
                    face_found = True
                elif prev_box is not None:
                    face_found = True

            # ── Crop, preprocess, buffer ────────────────────────────────
            if face_found and prev_box is not None:
                x1, y1, x2, y2 = [int(c) for c in prev_box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frm, x2), min(h_frm, y2)

                if x2 > x1 and y2 > y1:
                    face_crop = frame[y1:y2, x1:x2]
                    face_res  = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
                    face_rgb  = cv2.cvtColor(face_res, cv2.COLOR_BGR2RGB)

                    # Skin mask (matches training preprocessing exactly)
                    mask     = skin_mask_ycrcb(face_rgb)
                    mask_3ch = np.stack([mask, mask, mask], axis=-1) // 255
                    face_msk = face_rgb * mask_3ch

                    face_float = face_msk.astype(np.float32) / 255.0
                    frame_buf.append(face_float)

                    draw_face_box(frame, prev_box, current_bpm, len(frame_buf))

            # ── Run model when buffer is full ───────────────────────────
            buf_fill = len(frame_buf)
            if buf_fill >= BUFFER_LEN and frame_count % SLIDE_STEP == 0:
                x_in       = build_model_input(list(frame_buf), device)
                pred       = model(x_in)
                ppg_signal = pred.squeeze().cpu().numpy()

                # Slide window
                for _ in range(SLIDE_STEP):
                    if frame_buf:
                        frame_buf.popleft()

                # BPM
                bpm = bpm_from_ppg(ppg_signal, fps=actual_fps)
                if 40.0 < bpm < 200.0:
                    current_bpm = (BPM_EMA_ALPHA * current_bpm +
                                   (1 - BPM_EMA_ALPHA) * bpm) if current_bpm > 0 else bpm
                    signal_q   = signal_quality_score(ppg_signal, fps=actual_fps)

                # Update PPG display
                recent = ppg_signal[-PPG_DISPLAY:]
                ppg_history.clear()
                ppg_history.extend(recent)

            # ── HUD drawing ─────────────────────────────────────────────
            draw_hr_panel(frame, current_bpm, signal_q, buf_fill)

            ppg_x = w_frm - PPG_DISPLAY * 2 - 30
            ppg_y = h_frm - 155
            draw_ppg_waveform(frame, ppg_history, ppg_x, ppg_y,
                              PPG_DISPLAY * 2, 120)

            # FPS counter
            now = time.time()
            fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - fps_timer, 1e-6))
            fps_timer = now
            cv2.putText(frame, f"FPS: {fps_display:.1f}",
                        (10, h_frm - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (80, 80, 80), 1)

            # Instruction hint
            cv2.putText(frame, "q=quit  r=reset  s=screenshot  f=fullscreen",
                        (w_frm // 2 - 160, h_frm - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 70), 1)

            cv2.imshow(win_name, frame)

            # ── Keyboard ────────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                frame_buf.clear()
                ppg_history.clear()
                current_bpm = 0.0
                signal_q    = 0.0
                prev_box    = None
                print("🔄  Buffer reset — recalibrating...")
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"screenshot_{ts}.png"
                cv2.imwrite(fname, frame)
                print(f"📸  Screenshot saved → {fname}")
            elif key == ord('f'):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    win_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                )

    cap.release()
    cv2.destroyAllWindows()

    if current_bpm > 0:
        print(f"\n{'═'*45}")
        print(f"  Final HR estimate : {current_bpm:.0f} BPM")
        print(f"{'═'*45}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="PhysFormerX Live Heart Rate Monitor")
    p.add_argument("--yolo_weights",  default=DEFAULT_YOLO,
                   help="Path to trained YOLOv8 .pt weights")
    p.add_argument("--model_weights", default=DEFAULT_MODEL,
                   help="Path to trained PhysFormerX .pt weights")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device index (default 0)")
    p.add_argument("--fps",    type=int, default=30,
                   help="Target camera FPS (default 30)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

