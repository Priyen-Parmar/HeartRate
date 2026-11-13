"""Build the VitalEye v2 training cache from raw UBFC-rPPG videos.

For every video:
  1. YOLOv8 face detection on every frame (GPU, batched)
  2. causal motion-adaptive box smoothing (same BoxSmoother as the live app)
  3. 1.6x enlarged square crop -> 80x80 RGB (no hard skin mask: avoids mask flicker noise)
  4. contact PPG aligned to frame timestamps (true per-video frame rate, 23-30 fps)
  5. video + PPG resampled to a uniform 30 Hz timeline (the live app does the same)
Output: data/processed/ubfc_v2/<subject>.npz
"""

import os
import sys
import time
import argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".cache", "ultralytics"))

from vitaleye import TARGET_FPS, CROP_SIZE
from vitaleye.ubfc import list_videos, video_info, read_frames, load_ground_truth
from vitaleye.face import FaceDetector, BoxSmoother, crop_square
from vitaleye.dsp import resample_uniform


def process_video(det, sid, vpath, gpath, out_dir, batch=32):
    info = video_info(vpath)
    frames = list(read_frames(vpath))
    n = len(frames)
    gt = load_ground_truth(gpath, n, info["fps"])
    n = min(n, len(gt["frame_t"]))
    frames = frames[:n]

    boxes = []
    for i in range(0, n, batch):
        boxes.extend(det.detect(frames[i:i + batch]))
    found = np.array([b is not None for b in boxes])

    # fill misses with nearest previous detection (or next one at the start)
    first = int(np.argmax(found)) if found.any() else None
    if first is None:
        raise RuntimeError(f"no face found in {vpath}")
    sm = BoxSmoother()
    crops = np.empty((n, CROP_SIZE, CROP_SIZE, 3), np.uint8)
    states = np.empty((n, 3), np.float32)
    last = boxes[first]
    for i in range(n):
        if boxes[i] is not None:
            last = boxes[i]
        st = sm.update(last)
        states[i] = st
        crops[i] = crop_square(frames[i], st)

    t = gt["frame_t"][:n]
    t30, crops30 = resample_uniform(t, crops, TARGET_FPS)
    _, ppg30 = resample_uniform(t, gt["ppg"][:n], TARGET_FPS)
    _, hr30 = resample_uniform(t, gt["sensor_hr"][:n], TARGET_FPS)
    _, st30 = resample_uniform(t, states, TARGET_FPS)

    np.savez(
        os.path.join(out_dir, f"{sid}.npz"),
        frames=np.clip(np.rint(crops30), 0, 255).astype(np.uint8),
        ppg=ppg30.astype(np.float32),
        sensor_hr=hr30.astype(np.float32),
        box=st30.astype(np.float32),
        src_fps=np.float32((n - 1) / max(t[-1], 1e-6)),
        n_src_frames=np.int32(n),
        det_rate=np.float32(found.mean()),
    )
    return dict(sid=sid, n=n, n30=len(t30), src_fps=(n - 1) / t[-1], det_rate=float(found.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=os.path.join(ROOT, "data", "raw", "UBFC-rPPG"))
    ap.add_argument("--out_dir", default=os.path.join(ROOT, "data", "processed", "ubfc_v2"))
    ap.add_argument("--yolo", default=os.path.join(ROOT, "models", "face_detector", "yolov8_face_nose_chest.pt"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    import torch
    det = FaceDetector(a.yolo, device=0 if torch.cuda.is_available() else "cpu", imgsz=416)
    vids = list_videos(a.raw_dir)
    print(f"{len(vids)} videos found")
    t0 = time.time()
    for sid, v, g in vids:
        out = os.path.join(a.out_dir, f"{sid}.npz")
        if os.path.exists(out) and not a.overwrite:
            continue
        ts = time.time()
        r = process_video(det, sid, v, g, a.out_dir)
        print(f"{sid:22s} frames={r['n']:5d} -> {r['n30']:5d}@30Hz  src_fps={r['src_fps']:.2f}  "
              f"face_det={100*r['det_rate']:.1f}%  ({time.time()-ts:.1f}s)", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
