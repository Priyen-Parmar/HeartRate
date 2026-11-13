"""Real-time heart-rate engine (shared by the webcam app and offline video evaluation).

Pipeline per frame : face detect (YOLOv8) -> BoxSmoother -> 1.6x square crop 80x80   (== training)
Every `update_s`   : last <=10 s of crops -> resample to uniform 30 Hz using REAL capture
                     timestamps (handles 10-30 fps / dropped frames) -> to_model_input (== eval)
                     -> 10-model VitalNet ensemble (5 pixel v1 + 5 region-MRC v4) -> averaged band spectrum.
Output decoding    : causal HMM tracker over the ensemble's averaged spectrum (vitaleye/tracking.py):
                     uniform prior over 39-192 BPM, continuity only (sigma 4 BPM/s). No physiological
                     prior, no clamp towards "normal" values: 45 or 170 BPM are reported as measured.
"""

import os
import glob
import threading
from collections import deque

import numpy as np
import torch

from . import TARGET_FPS
from .model import VitalNet
from .data import to_model_input
from .dsp import hr_from_waves, resample_uniform
from .tracking import HRTracker, band_psd


def load_ensemble(model_dir: str, device, max_models: int = 10, weights=None):
    if weights:
        paths = weights
    else:   # alternate model families (pix_fold0, mrc_fold0, pix_fold1, ...) so a small subset stays balanced
        allp = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
        fams = {}
        for p in allp:
            fams.setdefault(os.path.basename(p).split("_fold")[0], []).append(p)
        order = [g[i] for i in range(max(map(len, fams.values()), default=0)) for g in fams.values() if i < len(g)]
        paths = order[:max_models]
    if not paths:
        raise FileNotFoundError(f"no VitalNet weights in {model_dir} (run scripts/train_cv.py)")
    models = []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = VitalNet(c=ck.get("width", 32), drop=ck.get("dropout", 0.2), version=ck.get("version", 1))
        m.load_state_dict(ck["state_dict"])
        models.append(m.to(device).eval())
    return models, paths


def bin_average(ts: np.ndarray, fr: np.ndarray, fs: float):
    """Average all frames falling into the same 1/fs-second slot (for cameras faster than fs).
    Returns (mean timestamp per slot, mean frame per slot). Averaging k frames lowers per-frame sensor
    noise by ~sqrt(k) instead of discarding frames as plain interpolation would."""
    b = np.floor((ts - ts[0]) * fs).astype(np.int64)            # non-decreasing (ts sorted)
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    cnt = np.diff(np.r_[starts, len(b)]).astype(np.float32)
    n = len(b)
    acc = fr[starts].astype(np.float32)
    for j in range(1, int(cnt.max())):                  # slots hold 1-3 frames: a few vectorised gathers
        m = starts + j < np.r_[starts[1:], n]
        acc[m] += fr[starts[m] + j]
    acc /= cnt.reshape((-1,) + (1,) * (fr.ndim - 1))
    return np.add.reduceat(ts, starts) / cnt, acc


class HREngine:
    def __init__(self, models, device, window_s=10.0, min_s=6.0, snr_min=0.0, hist=6, track=True,
                 slew_bpm_s=3.0, lock_conf=0.60, lock_timeout_s=20.0, fast_conf=0.75, fast_factor=3.0,
                 snr_show=-1.5, snr_avg_s=5.0, always_show=False):
        self.models, self.device = models, device
        self.window_s, self.min_s = window_s, min_s
        self.snr_min = snr_min
        self.buf = deque()                     # (timestamp, crop80 RGB uint8)
        self.hist = deque(maxlen=hist)         # (hr, snr)
        self.hr = 0.0                          # displayed heart rate
        self.snr = float("-inf")
        self.raw_hr = 0.0
        self.wave = np.zeros(0)
        self.eff_fps = 0.0
        self.lock = threading.Lock()
        self.busy = False
        self.tracker = HRTracker() if track else None
        # display: the number on screen only appears once the tracker is confident (or after
        # lock_timeout_s, flagged as low confidence) and then changes at most slew_bpm_s BPM per second.
        # A resting heart cannot jump 50 -> 120 in one second; real changes (exercise recovery,
        # ~1-2 BPM/s) are followed. This is a rate limit, NOT a pull towards any particular value.
        self.slew, self.lock_conf, self.lock_timeout = slew_bpm_s, lock_conf, lock_timeout_s
        # when the tracker is very sure (noise peaks never are), catch up faster (fixes a wrong early lock)
        self.fast_conf, self.fast_factor = fast_conf, fast_factor
        self.locked = False
        self.conf = 0.0                        # tracker confidence (posterior mass within +-3 BPM)
        self.t_disp = None
        self.t_start = None
        self.trend = deque(maxlen=2400)        # (t, displayed hr, confident?)
        self.attn = None                       # [36,36] mean skin-attention of one model (for the UI)
        self.n_updates = 0
        # honesty gate: the tracker's confidence only says the estimate is self-consistent, not that a
        # pulse is present. If the signal SNR (median over the last snr_avg_s seconds) is below snr_show
        # dB, NO number is displayed. On the 43 reliable UBFC videos (live pipeline) this keeps 95 % of
        # readings and lowers their MAE 1.59 -> 1.17; the hidden ones had MAE 9.6.
        self.snr_show, self.snr_avg_s = snr_show, snr_avg_s
        self.always_show = always_show
        self.snr_hist = deque()                # (t, snr)
        self.snr_avg = float("-inf")

    # ---------------------------------------------------------------- buffer
    def push(self, ts: float, crop80: np.ndarray):
        if self.buf and ts - self.buf[-1][0] > 1.0:   # face lost / stream stalled -> restart window
            self.buf.clear()
        self.buf.append((ts, crop80))
        while self.buf and ts - self.buf[0][0] > self.window_s + 0.5:
            self.buf.popleft()

    def buffered_seconds(self):
        return 0.0 if len(self.buf) < 2 else self.buf[-1][0] - self.buf[0][0]

    def reset(self):
        with self.lock:
            self.buf.clear(); self.hist.clear()
            if self.tracker is not None:
                self.tracker.reset()
            self.hr, self.raw_hr, self.snr, self.wave = 0.0, 0.0, float("-inf"), np.zeros(0)
            self.locked, self.conf, self.t_disp, self.t_start, self.attn = False, 0.0, None, None, None
            self.trend.clear(); self.snr_hist.clear(); self.snr_avg = float("-inf")

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def infer(self, snapshot=None):
        snap = list(self.buf) if snapshot is None else snapshot
        if len(snap) < 10:
            return None
        ts = np.array([s[0] for s in snap]); fr = np.stack([s[1] for s in snap])
        if ts[-1] - ts[0] < self.min_s:
            return None
        t_abs = float(ts[-1])
        ts = ts - ts[0]
        t_end = ts[-1]; t_start = max(0.0, t_end - self.window_s)
        keep = ts >= t_start - 1e-6
        ts, fr = ts[keep], fr[keep]
        self.eff_fps = (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6)
        if self.eff_fps > 1.3 * TARGET_FPS:
            ts, fr = bin_average(ts, fr, TARGET_FPS)   # 60 fps camera: use EVERY frame (2-frame mean ~ 1/30 s exposure)
        _, f30 = resample_uniform(ts, fr, TARGET_FPS)                    # [T,80,80,3] float
        # float32 on purpose: resampling returns float64, which Apple GPUs (MPS) cannot hold
        x = torch.from_numpy(np.ascontiguousarray(f30, dtype=np.float32)).to(self.device).permute(3, 0, 1, 2).unsqueeze(0) / 255.0
        x = to_model_input(x)
        waves, attn = [], None
        for i, m in enumerate(self.models):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(self.device.type == "cuda")):
                if i == len(self.models) - 1:
                    y, g = m(x, return_attention=True)
                    attn = g.float().mean(dim=(0, 1, 2)).cpu().numpy()
                else:
                    y = m(x)
            waves.append(y.float().squeeze(0).cpu().numpy())
        hr, snr, fused = hr_from_waves(waves, TARGET_FPS, "median")
        tracked, conf = (self.tracker.update(band_psd(waves, TARGET_FPS), t_abs)
                         if self.tracker is not None else (None, 0.0))
        with self.lock:
            self.raw_hr, self.snr, self.conf = hr, snr, conf
            self.wave = fused
            self.attn = attn
            self.n_updates += 1
            self.hist.append((hr, snr))
            self.snr_hist.append((t_abs, snr))
            while self.snr_hist and t_abs - self.snr_hist[0][0] > self.snr_avg_s:
                self.snr_hist.popleft()
            self.snr_avg = float(np.median([v for _, v in self.snr_hist]))
            if self.t_start is None:
                self.t_start = t_abs
            if tracked is not None:
                if not self.locked:
                    if conf >= self.lock_conf or t_abs - self.t_start >= self.lock_timeout:
                        self.locked, self.hr, self.t_disp = True, tracked, t_abs
                else:
                    dt = max(t_abs - self.t_disp, 0.0)
                    rate = self.slew * (self.fast_factor if conf >= self.fast_conf else 1.0)
                    self.hr += float(np.clip(tracked - self.hr, -rate * dt, rate * dt))
                    self.t_disp = t_abs
                if self.showing:
                    self.trend.append((t_abs, self.hr, conf >= self.lock_conf))
            else:
                good = [h for h, s in self.hist if s >= self.snr_min]
                if good:
                    self.hr = float(np.median(good))
                elif self.hr == 0.0:
                    self.hr = float(np.median([h for h, _ in self.hist]))
        return hr, snr

    @property
    def showing(self):
        """True when a number may be displayed (locked AND the recent signal really contains a pulse)."""
        ok = self.hr > 0 and (self.locked or self.tracker is None)
        return ok and (self.always_show or self.snr_avg >= self.snr_show)

    def infer_async(self):
        if self.busy:
            return False
        self.busy = True
        snap = list(self.buf)

        def run():
            try:
                self.infer(snap)
            except Exception as e:  # keep the UI alive
                print(f"[engine] inference error: {e}")
            finally:
                self.busy = False
        threading.Thread(target=run, daemon=True).start()
        return True

