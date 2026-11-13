"""VitalEye v2 - live webcam heart-rate monitor (or offline on a video file).

  python scripts/live_monitor.py                      # webcam 0
  python scripts/live_monitor.py --camera 1
  python scripts/live_monitor.py --video some.mp4 --headless --csv out.csv
  python scripts/live_monitor.py --ubfc sub23 --weights models/vitalnet/pix_fold2.pt models/vitalnet/mrc_fold2.pt   # check vs ground truth

Keys: q/ESC quit | r reset | s screenshot | f fullscreen | h pulse-region heat overlay
A CSV log (time, HR, raw HR, SNR, fps) is written to outputs/ so you can compare it
second-by-second with a chest strap / pulse oximeter / treadmill reading.
"""

import os
import sys
import csv
import time
import queue
import argparse
import threading
from datetime import datetime

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(ROOT, ".cache", "ultralytics"))
os.makedirs(os.environ["YOLO_CONFIG_DIR"], exist_ok=True)

import torch

from vitaleye.face import FaceDetector, BoxSmoother, crop_square
from vitaleye.live import HREngine, load_ensemble
from vitaleye.ui import LiveUI

YOLO_W = os.path.join(ROOT, "models", "face_detector", "yolov8_face_nose_chest.pt")
MODEL_DIR = os.path.join(ROOT, "models", "vitalnet")


BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}
WINDOWS = os.name == "nt"


def pick_device(want="auto"):
    """cuda (NVIDIA) > mps (Apple Silicon GPU, only if it gives the same numbers as the CPU) > cpu."""
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if want in ("auto", "mps") and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            # run the COMPLETE live engine (uint8 crops -> resampling -> model -> HR) on CPU and on MPS
            import copy
            from vitaleye.model import VitalNet
            torch.manual_seed(0)
            m = VitalNet(c=32, version=4).eval()
            rng = np.random.default_rng(0)
            frames = [(i / 30.0, rng.integers(0, 255, (80, 80, 3), dtype=np.uint8)) for i in range(310)]
            out = []
            for d in ("cpu", "mps"):
                e = HREngine([copy.deepcopy(m).to(d)], torch.device(d))
                e.infer(frames)
                out.append(np.asarray(e.wave, np.float64))
            ref, got = out
            if len(ref) and len(ref) == len(got) and np.abs(ref - got).max() < 2e-2 * (np.abs(ref).max() + 1e-6):
                return torch.device("mps")
            print("[vitaleye] Apple GPU (MPS) gives different results -> using CPU")
        except Exception as e:
            print(f"[vitaleye] Apple GPU (MPS) not usable ({type(e).__name__}) -> using CPU")
    return torch.device("cpu")


def list_cameras(max_index=6):
    """Open each camera index once (nothing is saved) and print what works. Phone webcams
    (DroidCam / Iriun / Windows 'connected camera') usually appear as index 1 or 2; the Windows
    connected camera may only be visible through the MSMF backend."""
    found = 0
    for name in (("dshow", "msmf") if WINDOWS else ("any",)):
        for i in range(max_index):
            cap = cv2.VideoCapture(i, BACKENDS[name])
            if cap.isOpened():
                ok, fr = cap.read()
                for _ in range(10):                      # skip start-up frames
                    ok2, f2 = cap.read()
                    if ok2:
                        fr = f2
                if ok and fr is not None:
                    found += 1
                    flat = fr.reshape(-1, 3).std(0).max() < 4     # one colour everywhere = no real picture
                    note = ("  <-- NO PICTURE (flat " + ("green" if fr[..., 1].mean() > fr[..., 0].mean() + 40 else "colour")
                            + "): phone app not streaming to the PC client") if flat else "  OK, real picture"
                    print(f"  --camera {i} --backend {name}:  {fr.shape[1]}x{fr.shape[0]}  "
                          f"(reported fps {cap.get(cv2.CAP_PROP_FPS):.0f}){note}")
            cap.release()
    if not found:
        print("  no working camera found")


class AsyncFace:
    """Face detection on a background thread, at most max_hz times per second (CPU / Apple machines,
    where the detector is slow): the camera loop never waits for it and uses the latest smoothed box."""

    def __init__(self, det, smoother, max_hz=6.0):
        self.det, self.sm, self.min_dt = det, smoother, 1.0 / max_hz
        self.box, self.last_det_ts, self.last_submit, self.busy = None, -1e9, -1e9, False
        self.lock = threading.Lock()

    def submit(self, ts, frame):
        if self.busy or ts - self.last_submit < self.min_dt:
            return
        self.busy, self.last_submit = True, ts

        def run():
            try:
                b = self.det.detect([frame])[0]
                with self.lock:
                    if b is not None:
                        self.box = self.sm.update(b); self.last_det_ts = ts
                    elif ts - self.last_det_ts > 1.0:
                        self.box = None; self.sm.reset()
            finally:
                self.busy = False
        threading.Thread(target=run, daemon=True).start()

    def get(self):
        with self.lock:
            return self.box

    def reset(self):
        with self.lock:
            self.box = None; self.sm.reset()


class RawRecorder:
    """Lossless (FFV1) recording on a background thread so encoding never slows the pulse loop."""

    def __init__(self, path, fps, size):
        self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, size)
        if not self.w.isOpened():   # OpenCV build without FFV1: fall back to near-lossless MJPG
            path = os.path.splitext(path)[0] + ".avi"
            self.w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
            print(f"[vitaleye] FFV1 not available, raw video -> {path} (MJPG)")
        # bounded RAM: at most ~150 frames waiting (~140 MB at 640x480, ~410 MB at 1280x720)
        self.q = queue.Queue(maxsize=150)
        self.dropped = 0
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    def _loop(self):
        while True:
            fr = self.q.get()
            if fr is None:
                break
            self.w.write(fr)

    def write(self, fr):
        try:
            self.q.put_nowait(fr)
        except queue.Full:
            self.dropped += 1

    def release(self):
        self.q.put(None); self.th.join(); self.w.release()
        if self.dropped:
            print(f"[vitaleye] raw recorder dropped {self.dropped} frames (disk/CPU too slow)")


class Camera:
    """Threaded capture with real per-frame timestamps (never feeds stale frames)."""

    def __init__(self, index, width=640, height=480, fps=30, fourcc="auto", backend_name="auto"):
        index = int(index) if str(index).isdigit() else str(index)
        self.is_url = isinstance(index, str)
        if self.is_url:                      # network stream (phone app): format is set by the sender
            self.cap = cv2.VideoCapture(index, cv2.CAP_FFMPEG)
            self.q = queue.Queue(maxsize=45)   # ~1 s: short stalls are caught up, no pulse samples lost; self.ok = self.cap.isOpened(); self.running = True
            threading.Thread(target=self._loop, daemon=True).start()
            return
        backend = BACKENDS.get(backend_name, cv2.CAP_DSHOW if WINDOWS else cv2.CAP_ANY)
        if fourcc == "auto" and not WINDOWS:
            fourcc = ""        # macOS/Linux: let the camera pick its native format
        if fourcc == "auto":   # DirectShow webcams need MJPG for 30 fps; MSMF phone cameras run 30 fps uncompressed (MJPG -> 14 fps)
            fourcc = "NV12" if backend == cv2.CAP_MSMF else "MJPG"
        self.ok = False
        # 1st try: our preferred format; 2nd: let the camera choose (some virtual cameras, e.g. DroidCam,
        # refuse formats/fps they are not currently producing)
        for attempt, force in enumerate((True, False)):
            self.cap = cv2.VideoCapture(index, backend)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(index)
            if force:
                if fourcc:
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, fps)
            t0 = time.perf_counter()
            while self.cap.isOpened() and time.perf_counter() - t0 < 4.0:
                ok, fr = self.cap.read()
                if ok and fr is not None:
                    self.ok = True
                    break
            if self.ok:
                if not force:
                    print(f"[vitaleye] camera {index}: using its own default format "
                          f"({int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
                break
            self.cap.release()
        if not self.ok:
            print(f"[vitaleye] camera {index} opened but sends no frames. It is probably BUSY: close Google Meet / "
                  f"Zoom / Teams / the Windows Camera app / browser tabs using the camera / other VitalEye windows, "
                  f"then try again. For DroidCam: the phone picture must be visible in the DroidCam PC client.")
        self.q = queue.Queue(maxsize=45)   # ~1 s: short stalls are caught up, no pulse samples lost
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, fr = self.cap.read()
            ts = time.perf_counter()
            if not ok:
                self.ok = False
                break
            if self.q.full():
                try:
                    self.q.get_nowait()   # drop oldest; timestamps stay truthful
                except queue.Empty:
                    pass
            self.q.put((ts, fr))

    def read(self, timeout=2.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        self.cap.release()


def run(args):
    dev = pick_device("cpu" if args.cpu else getattr(args, "device", "auto"))
    models, paths = load_ensemble(MODEL_DIR, dev, args.ensemble, weights=args.weights or None)
    print(f"[vitaleye] {len(models)} VitalNet model(s) on {dev}: {[os.path.basename(p) for p in paths]}")
    det = FaceDetector(args.yolo if os.path.exists(args.yolo) else None,
                       device=0 if dev.type == "cuda" else ("mps" if dev.type == "mps" else "cpu"), imgsz=320)
    if dev.type == "cpu":
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))   # leave cores for camera + UI
    eng = HREngine(models, dev, window_s=args.window, min_s=args.min_window, snr_min=args.snr_min,
                   track=not args.no_track, always_show=getattr(args, "always_show", False))
    smoother = BoxSmoother()
    ui = LiveUI(view_h=getattr(args, "view_h", 640))

    gt = None
    if args.ubfc:
        from vitaleye.ubfc import video_info, load_ground_truth
        raw = os.path.join(ROOT, "data", "raw", "UBFC-rPPG")
        name = args.ubfc.replace("sub", "2_sub") if args.ubfc.startswith("sub") else "1_" + args.ubfc.replace("ds1_", "")
        args.video = os.path.join(raw, f"vid_{name}.avi")
        inf = video_info(args.video)
        gt = load_ground_truth(os.path.join(raw, f"ground_truth_{name}.txt"), inf["n_frames"], inf["fps"])

    offline = bool(args.video)
    if offline:
        cap = cv2.VideoCapture(args.video)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cam = None
    else:
        cam = Camera(args.camera, fps=getattr(args, "cam_fps", 30), fourcc=getattr(args, "fourcc", "auto"), backend_name=getattr(args, "backend", "auto"))
        if not cam.ok:
            cam.release(); print(f"cannot use camera {args.camera} (see message above)"); return None

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.csv or os.path.join(ROOT, "outputs", f"hr_log_{stamp}.csv")
    fcsv = open(csv_path, "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["t_s", "hr_bpm", "raw_hr_bpm", "snr_db", "confidence", "rppg_fps", "shown"] + (["gt_hr_bpm"] if gt else []))

    # --save_raw: keep the untouched camera frames (+ exact timestamps) so a session can be re-analysed
    # offline (signal diagnosis, other methods, unsupervised adaptation). ~100 MB per minute.
    raw_writer, raw_ts = None, None
    if args.save_raw == "auto":
        args.save_raw = os.path.join(ROOT, "outputs", f"raw_{stamp}")
    if args.save_raw and not offline:
        base = os.path.splitext(args.save_raw)[0]
        raw_ts = open(base + "_timestamps.csv", "w", newline="")
        raw_ts.write("frame,t_s" + chr(10))

    writer = None
    win = "VitalEye - live heart rate"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, int(args.view_h * 4 / 3) + LiveUI.PANEL_W, args.view_h)

    # face detection ~15-30 times/s is plenty (the box is smoothed in between); at 60 fps detect every 4th frame
    det_every = args.det_every if args.det_every > 0 else max(2, round(getattr(args, "cam_fps", 30) / 15))
    idx, t_first, last_upd, last_det_ts = 0, None, -1e9, -1e9
    prev_small, n_dup = None, 0
    last_logged = -1
    async_face = AsyncFace(det, smoother, max_hz=8.0 if dev.type == "mps" else 4.0) if (dev.type != "cuda" and not offline) else None
    fps_cam, t_prev = 0.0, None
    box = None
    canvas, last_draw = None, 0.0
    fullscreen = False
    errors = []
    try:
        while True:
            if offline:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = idx / vfps if gt is None else float(gt["frame_t"][min(idx, len(gt["frame_t"]) - 1)])
            else:
                item = cam.read()
                if item is None:
                    print("camera stopped"); break
                ts, frame = item
            if t_first is None:
                t_first = ts
            t_rel = ts - t_first
            if t_prev is not None and ts > t_prev:
                fps_cam = 0.9 * fps_cam + 0.1 / (ts - t_prev) if fps_cam else 1 / (ts - t_prev)
            t_prev = ts
            if raw_ts is not None:
                if raw_writer is None:
                    raw_writer = RawRecorder(os.path.splitext(args.save_raw)[0] + ".mkv",   # lossless
                                             getattr(args, "cam_fps", 30), (frame.shape[1], frame.shape[0]))
                raw_writer.write(frame); raw_ts.write(f"{idx},{t_rel:.4f}" + chr(10))
            idx += 1

            if async_face is not None:          # Mac / CPU: detector never blocks the camera loop
                async_face.submit(ts, frame); box = async_face.get()
            elif det_every <= 1 or idx % det_every == 1 or box is None:
                b = det.detect([frame])[0]
                if b is not None:
                    box = smoother.update(b); last_det_ts = ts
                elif ts - last_det_ts > 1.0:
                    box = None; smoother.reset()
            # virtual cameras (DroidCam) repeat the last frame to fill 60 fps: a repeated frame carries a
            # wrong timestamp and no new information, so it is not fed to the pulse engine
            small = frame[::8, ::8]
            dup = prev_small is not None and np.array_equal(small, prev_small)
            prev_small = small.copy()
            if dup:
                n_dup += 1
            if box is not None:
                if not dup:
                    eng.push(ts, crop_square(frame, box))
            elif eng.buf:
                eng.buf.clear()

            if box is not None and ts - last_upd >= args.update and eng.buffered_seconds() >= args.min_window:
                if offline:
                    eng.infer(); last_upd = ts
                elif eng.infer_async():
                    last_upd = ts
                if eng.hr > 0 and eng.n_updates != last_logged:   # one row per finished inference
                    last_logged = eng.n_updates
                    row = [f"{t_rel:.2f}", f"{eng.hr:.1f}", f"{eng.raw_hr:.1f}", f"{eng.snr:.2f}", f"{eng.conf:.2f}", f"{eng.eff_fps:.1f}", int(eng.showing)]
                    if gt is not None:
                        from vitaleye.dsp import hr_from_signal
                        i1 = idx; i0 = max(0, i1 - int(args.window * vfps))
                        g = hr_from_signal(gt["ppg"][i0:i1], vfps)[0]
                        row.append(f"{g:.1f}"); errors.append((abs(eng.hr - g), eng.showing))
                    wcsv.writerow(row)

            # every frame is processed for the pulse; the (heavier) UI is redrawn at most ~20 times/s so a
            # slow laptop never drops camera frames because of drawing
            redraw = args.record or (time.perf_counter() - last_draw >= 0.05) or canvas is None
            if (not args.headless or args.record) and redraw:
                last_draw = time.perf_counter()
                brightness = float(frame[::4, ::4].mean())
                canvas = ui.render(frame, eng, box, eng.buffered_seconds(), args.min_window, fps_cam, brightness)
                if args.record:
                    if writer is None:
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), 30, (canvas.shape[1], canvas.shape[0]))
                    writer.write(canvas)
            if not args.headless and redraw:
                cv2.imshow(win, canvas)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                if k == ord("r"):
                    eng.reset(); smoother.reset(); box = None; print("reset")
                    if async_face is not None:
                        async_face.reset()
                if k == ord("s"):
                    p = os.path.join(ROOT, "outputs", f"screenshot_{datetime.now():%Y%m%d_%H%M%S}.png")
                    cv2.imwrite(p, canvas); print("saved", p)
                if k == ord("c") and cam is not None and not cam.is_url and WINDOWS:
                    cam.cap.set(cv2.CAP_PROP_SETTINGS, 1)   # Windows driver dialog: turn auto exposure / WB off
                if k == ord("h"):
                    ui.show_heat = not ui.show_heat
                if k == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        fcsv.close()
        if cam: cam.release()
        if offline: cap.release()
        if writer: writer.release()
        if raw_writer: raw_writer.release()
        if raw_ts:
            raw_ts.close(); print(f"[vitaleye] raw video -> {os.path.splitext(args.save_raw)[0]}.mkv")
        if not args.headless: cv2.destroyAllWindows()
    if n_dup:
        print(f"[vitaleye] skipped {n_dup} repeated frames ({100 * n_dup / max(idx, 1):.0f}%) sent by the camera driver")
    print(f"[vitaleye] HR log -> {csv_path}")
    if errors:
        e = np.array([x for x, _ in errors]); sh = np.array([v for _, v in errors], bool)
        print(f"[vitaleye] vs contact-PPG ground truth: MAE={e.mean():.2f} BPM  RMSE={np.sqrt((e**2).mean()):.2f}  "
              f"within 5 BPM: {100*np.mean(e<=5):.1f}%  (n={len(e)} updates)")
        if sh.any():
            print(f"[vitaleye] readings actually displayed ({100*sh.mean():.0f}% of updates): MAE={e[sh].mean():.2f} BPM")
    return dict(errors=errors, csv=csv_path)


def parse():
    p = argparse.ArgumentParser(description="VitalEye v2 live heart-rate monitor")
    p.add_argument("--camera", default="0",
                   help="camera index (0, 1, ...) OR a stream URL, e.g. the DroidCam phone app directly: http://<phone-ip>:4747/video")
    p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"],
                   help="camera API (Windows default: dshow; try msmf for the Windows 'connected camera' phone feature)")
    p.add_argument("--cam_fps", type=int, default=30, help="frame rate requested from the camera (phone via DroidCam: up to 60; "
                   "frames above 30 fps are averaged in 1/30-s slots, so every frame is used)")
    p.add_argument("--always_show", action="store_true",
                   help="always display the tracked value after lock; weak-signal readings are shown in amber as 'low confidence' "
                        "(default: hidden, because at SNR < -1.5 dB they are often wrong)")
    p.add_argument("--list_cameras", action="store_true", help="print every camera index that works, then exit")
    p.add_argument("--video", default="")
    p.add_argument("--ubfc", default="", help="run on a UBFC subject (e.g. sub23) and score vs ground truth")
    p.add_argument("--weights", nargs="*", default=None, help="explicit VitalNet checkpoint(s)")
    p.add_argument("--ensemble", type=int, default=10, help="max number of models loaded from models/vitalnet")
    p.add_argument("--yolo", default=YOLO_W)
    p.add_argument("--window", type=float, default=10.0, help="analysis window (s)")
    p.add_argument("--min_window", type=float, default=6.0, help="seconds before first reading")
    p.add_argument("--update", type=float, default=0.5, help="seconds between HR updates")
    p.add_argument("--no_track", action="store_true", help="disable causal HR tracking (plain per-window peak + median)")
    p.add_argument("--snr_min", type=float, default=0.0, help="confidence gate (dB); CV: >=0 dB -> MAE<=2.2, <0 dB -> MAE~10")
    p.add_argument("--det_every", type=int, default=0, help="run face detector every N frames (0 = auto: 2 at 30 fps, 4 at 60 fps)")
    p.add_argument("--view_h", type=int, default=640, help="height of the camera view in the window")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--record", default="")
    p.add_argument("--csv", default="")
    p.add_argument("--save_raw", default="", help="also save the raw camera frames losslessly to <name>.mkv (+ <name>_timestamps.csv)")
    p.add_argument("--fourcc", default="auto", help="camera stream format: auto (MJPG for dshow, NV12 for msmf), MJPG, YUY2, NV12")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto: NVIDIA GPU, else Apple GPU (MPS, verified against CPU), else CPU")
    return p.parse_args()


if __name__ == "__main__":
    _a = parse()
    if _a.list_cameras:
        print("[vitaleye] cameras that work on this PC:"); list_cameras(); sys.exit(0)
    sys.exit(0 if run(_a) is not None else 1)

