"""VitalEye live UI: camera view on the left, instrument panel on the right.

Camera view : face corner-brackets (colour = reading state) + optional heat overlay showing WHERE the
              network reads the pulse (its learned skin attention, averaged over the analysis window).
Panel       : big BPM with a heart that beats at the measured rate, state chip, signal-quality bars,
              60-s heart-rate trend, live pulse waveform, hints (light / fps / motion), key help.
Text is rendered with Pillow (Segoe UI) for clean typography; shapes and curves with anti-aliased OpenCV.
"""

import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import BOX_SCALE
from .data import EVAL_SCALE

# palette (BGR for OpenCV, RGB for PIL)
BG = (21, 17, 15)
CARD = (34, 29, 26)
GRID = (58, 52, 48)
WHITE = (240, 240, 240)
MUTED = (150, 146, 140)
GREEN = (120, 220, 80)      # BGR
AMBER = (40, 180, 250)
RED = (80, 80, 235)
BLUE = (235, 170, 70)
ACCENT = (250, 190, 90)
HEAT = (60, 170, 255)       # warm amber glow for the pulse-region overlay


def _rgb(c):
    return (c[2], c[1], c[0])


def _font(size, bold=False):
    names = ((["seguisb.ttf", "segoeuib.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf"] if bold else
              ["segoeui.ttf", "Arial.ttf", "Helvetica.ttc"]) + ["arial.ttf", "DejaVuSans.ttf"])
    dirs = (os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),                 # Windows
            "/System/Library/Fonts/Supplemental", "/System/Library/Fonts", "/Library/Fonts",  # macOS
            "/usr/share/fonts/truetype/dejavu", "")
    for n in names:
        for d in dirs:
            try:
                return ImageFont.truetype(os.path.join(d, n) if d else n, size)
            except OSError:
                continue
    return ImageFont.load_default()


def rounded_rect(img, x0, y0, x1, y1, r, color, thickness=-1):
    r = int(max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x0 + r, y0), (x1 - r, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0 + r), (x1, y1 - r), color, -1, cv2.LINE_AA)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
    else:
        for (cx, cy), a in (((x0 + r, y0 + r), 180), ((x1 - r, y0 + r), 270), ((x1 - r, y1 - r), 0), ((x0 + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y0), (x1 - r, y0), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0 + r, y1), (x1 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x0, y0 + r), (x0, y1 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y0 + r), (x1, y1 - r), color, thickness, cv2.LINE_AA)


class LiveUI:
    PANEL_W = 420

    def __init__(self, view_h=640):
        self.view_h = view_h
        self.f_title = _font(26, True)
        self.f_sub = _font(14)
        self.f_big = _font(118, True)
        self.f_unit = _font(24, True)
        self.f_chip = _font(16, True)
        self.f_small = _font(13)
        self.f_label = _font(14, True)
        self.phase = 0.0
        self.t_prev = time.perf_counter()
        self.wave_scale = None
        self.attn_ema = None
        self.show_heat = True
        self._oval = None

    # ------------------------------------------------------------------ state
    @staticmethod
    def state(eng, face, buf_s, min_s):
        if face is None:
            return "noface"
        if not eng.locked and eng.tracker is not None:
            return "measuring"
        if not eng.showing:
            return "nopulse"
        good = eng.snr >= eng.snr_min and eng.conf >= eng.lock_conf and eng.snr_avg >= eng.snr_show
        return "good" if good else "weak"

    # ------------------------------------------------------------------ camera view
    def _camera(self, frame, eng, box, st):
        h, w = frame.shape[:2]
        s = self.view_h / h
        view = cv2.resize(frame, (int(round(w * s)), self.view_h), interpolation=cv2.INTER_LINEAR)
        if box is None:
            return view
        cx, cy, sz = box[0] * s, box[1] * s, box[2] * s
        # heat overlay of the network's skin attention (where the pulse is read)
        if self.show_heat and eng.attn is not None:
            a = eng.attn.astype(np.float32)
            self.attn_ema = a if self.attn_ema is None or self.attn_ema.shape != a.shape else 0.8 * self.attn_ema + 0.2 * a
            a = self.attn_ema
            a = (a - a.min()) / (a.max() - a.min() + 1e-6)
            L = int(sz * BOX_SCALE * EVAL_SCALE)
            if L > 8:
                # the glow is smooth, so it is computed on a small 96x96 grid and scaled up once
                # (full-size maths cost ~45 ms per frame for a close phone-camera face and starved the pulse loop)
                n = 96
                if self._oval is None:
                    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - n / 2
                    ell = np.clip(1.25 - ((xx / (n * 0.36)) ** 2 + (yy / (n * 0.46)) ** 2), 0, 1)   # soft oval window
                    self._oval = np.clip(ell * 1.6, 0, 1)
                a = cv2.GaussianBlur(cv2.resize(a, (n, n), interpolation=cv2.INTER_CUBIC), (0, 0), n / 40) * self._oval
                alpha_n = np.clip((a - 0.30) / 0.60, 0, 1).astype(np.float32) * 0.50
                x0, y0 = int(cx - L / 2), int(cy - L / 2)
                xs0, ys0 = max(0, x0), max(0, y0)
                xs1, ys1 = min(view.shape[1], x0 + L), min(view.shape[0], y0 + L)
                if xs1 > xs0 and ys1 > ys0:
                    alpha = cv2.resize(alpha_n, (L, L), interpolation=cv2.INTER_LINEAR)[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
                    alpha = np.ascontiguousarray(alpha)
                    tint = np.empty((ys1 - ys0, xs1 - xs0, 3), np.uint8); tint[:] = HEAT
                    roi = np.ascontiguousarray(view[ys0:ys1, xs0:xs1])
                    view[ys0:ys1, xs0:xs1] = cv2.blendLinear(roi, tint, 1.0 - alpha, alpha)
        col = {"good": GREEN, "weak": AMBER, "measuring": BLUE}.get(st, MUTED)
        x0, y0, x1, y1 = int(cx - sz / 2), int(cy - sz / 2), int(cx + sz / 2), int(cy + sz / 2)
        k = int(sz * 0.18)
        for (px, py, dx, dy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(view, (px, py), (px + dx * k, py), col, 3, cv2.LINE_AA)
            cv2.line(view, (px, py), (px, py + dy * k), col, 3, cv2.LINE_AA)
        if self.show_heat and eng.attn is not None:
            cv2.circle(view, (x0 + 6, y1 + 18), 5, HEAT, -1, cv2.LINE_AA)
            cv2.putText(view, "pulse read from highlighted skin", (x0 + 18, y1 + 23), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (230, 230, 230), 1, cv2.LINE_AA)
        return view

    # ------------------------------------------------------------------ panel
    def _heart(self, img, cx, cy, size, color):
        t = np.linspace(0, 2 * np.pi, 80)
        x = 16 * np.sin(t) ** 3
        y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        pts = np.stack([cx + x * size / 34, cy - y * size / 34], 1).astype(np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

    def render(self, frame, eng, box, buf_s, min_s, fps_cam, brightness):
        now = time.perf_counter(); dt = now - self.t_prev; self.t_prev = now
        st = self.state(eng, box, buf_s, min_s)
        view = self._camera(frame, eng, box, st)
        P = np.zeros((self.view_h, self.PANEL_W, 3), np.uint8); P[:] = BG
        W, H = self.PANEL_W, self.view_h
        texts = []   # (xy, text, font, rgb, anchor)

        # header
        texts.append(((24, 20), "VitalEye", self.f_title, _rgb(WHITE), "la"))
        texts.append(((24, 54), f"contactless heart rate  ·  {len(eng.models)}-model ensemble", self.f_sub, _rgb(MUTED), "la"))

        # main reading card
        rounded_rect(P, 16, 86, W - 16, 300, 16, CARD)
        hr = eng.hr if eng.showing else 0.0
        if hr > 0:
            self.phase += dt * hr / 60.0
        beat = math.exp(-((self.phase % 1.0) / 0.10) ** 2) if hr > 0 else 0.0
        hcol = {"good": (90, 90, 245), "weak": AMBER}.get(st, GRID)
        self._heart(P, 62, 150, 40 * (1 + 0.18 * beat), hcol)
        num_col = {"good": WHITE, "weak": (190, 215, 245)}.get(st, MUTED)
        texts.append(((104, 92), f"{hr:.0f}" if hr > 0 else "--", self.f_big, _rgb(num_col), "la"))
        texts.append(((W - 36, 214), "BPM", self.f_unit, _rgb(MUTED), "ra"))
        # state chip
        if st == "noface":
            chip, ccol = "No face found - look at the camera", RED
        elif st == "measuring":
            remaining = max(0.0, min_s - buf_s)
            chip, ccol = (f"Measuring...  {remaining:.0f} s" if remaining > 0.5 else "Measuring...  locking on"), BLUE
        elif st == "good":
            chip, ccol = "Stable reading", GREEN
        elif st == "nopulse":
            chip, ccol = "Pulse signal too weak - no reading", RED
        else:
            chip, ccol = "Low confidence reading", AMBER
        rounded_rect(P, 32, 252, W - 32, 284, 16, tuple(int(c * 0.28) for c in ccol))
        cv2.circle(P, (50, 268), 6, ccol, -1, cv2.LINE_AA)
        texts.append(((64, 258), chip, self.f_chip, _rgb(ccol), "la"))
        if st == "measuring":
            frac = min(1.0, buf_s / max(min_s, 1e-6))
            cv2.line(P, (40, 293), (40 + int((W - 80) * frac), 293), BLUE, 3, cv2.LINE_AA)

        # signal quality bars
        texts.append(((24, 314), "SIGNAL", self.f_label, _rgb(MUTED), "la"))
        q = 0 if not np.isfinite(eng.snr) else int(np.sum(eng.snr > np.array([-4, -2, 0, 2, 4])))
        for i in range(5):
            x = 100 + i * 22; hh = 6 + i * 3
            cv2.rectangle(P, (x, 330 - hh), (x + 14, 330), (GREEN if q >= 4 else AMBER if q >= 2 else RED) if i < q else GRID, -1)
        texts.append(((216, 314), f"{eng.snr:+.1f} dB" if np.isfinite(eng.snr) else "--", self.f_small, _rgb(MUTED), "la"))

        # trend chart (last 60 s)
        cx0, cy0, cx1, cy1 = 24, 372, W - 24, 468
        texts.append(((24, 342), "TREND  (last 60 s)", self.f_label, _rgb(MUTED), "la"))
        tr = list(eng.trend)
        if tr:
            t_end = tr[-1][0]
            tr = [p for p in tr if p[0] >= t_end - 60]
        if len(tr) >= 2:
            vals = np.array([p[1] for p in tr]); lo, hi = vals.min() - 5, vals.max() + 5
            if hi - lo < 20:
                m = (hi + lo) / 2; lo, hi = m - 10, m + 10
        else:
            lo, hi = 50, 110
        for gv in np.linspace(lo, hi, 3):
            y = int(cy1 - (gv - lo) / (hi - lo) * (cy1 - cy0))
            cv2.line(P, (cx0 + 34, y), (cx1, y), GRID, 1, cv2.LINE_AA)
            texts.append(((cx0 + 26, y), f"{gv:.0f}", self.f_small, _rgb(MUTED), "rm"))
        if len(tr) >= 2:
            t_end = tr[-1][0]
            pts = np.array([[cx1 - (t_end - t) / 60 * (cx1 - cx0 - 34), cy1 - (v - lo) / (hi - lo) * (cy1 - cy0)] for t, v, _ in tr], np.int32)
            ok = np.array([c for _, _, c in tr])
            cv2.polylines(P, [pts], False, GRID, 2, cv2.LINE_AA)
            for i in range(1, len(pts)):
                if ok[i] and ok[i - 1]:
                    cv2.line(P, tuple(pts[i - 1]), tuple(pts[i]), (110, 110, 250), 2, cv2.LINE_AA)
            cv2.circle(P, tuple(pts[-1]), 4, (110, 110, 250), -1, cv2.LINE_AA)

        # pulse waveform (last 6 s)
        texts.append(((24, 486), "PULSE WAVE", self.f_label, _rgb(MUTED), "la"))
        wx0, wy0, wx1, wy1 = 24, 506, W - 24, 586
        rounded_rect(P, wx0, wy0, wx1, wy1, 10, CARD)
        wv = np.asarray(eng.wave[-180:], np.float64)
        if len(wv) > 20 and eng.showing:
            wv = np.convolve(wv, np.ones(3) / 3, mode="same")
            sc = np.percentile(np.abs(wv), 95) + 1e-6
            self.wave_scale = sc if self.wave_scale is None else 0.9 * self.wave_scale + 0.1 * sc
            v = np.clip(wv / self.wave_scale, -1.2, 1.2)
            xs = np.linspace(wx0 + 8, wx1 - 8, len(v)); ys = (wy0 + wy1) / 2 - v * (wy1 - wy0) * 0.38
            cv2.polylines(P, [np.stack([xs, ys], 1).astype(np.int32)], False,
                          GREEN if st == "good" else AMBER, 2, cv2.LINE_AA)
        else:
            texts.append((((wx0 + wx1) // 2, (wy0 + wy1) // 2), "waiting for a stable signal", self.f_small, _rgb(MUTED), "mm"))

        # hints + footer
        # hints are chosen from what is actually measured (brightness, face size, fps), not guessed
        hint = ""
        fh = frame.shape[0]
        too_close = box is not None and box[2] * BOX_SCALE > fh * 0.95
        too_far = box is not None and box[2] < fh * 0.18
        if brightness < 60:
            hint = "Too dark: face a lamp or window"
        elif too_close:
            hint = "Too close: move back until your face is ~1/3 of the picture"
        elif too_far:
            hint = "Too far: move closer to the camera"
        elif fps_cam and fps_cam < 20:
            hint = f"Camera only {fps_cam:.0f} fps"
        elif st in ("nopulse", "weak"):
            hint = "Weak pulse: keep head still, don't talk, relax face"
        if hint:
            texts.append(((24, 596), hint, self.f_small, _rgb(AMBER), "la"))
        texts.append(((24, H - 22), f"cam {fps_cam:4.1f} fps  ·  light {brightness:3.0f}  ·  q quit  r reset  s save  h heat  c cam",
                      self.f_small, _rgb(MUTED), "la"))

        im = Image.fromarray(cv2.cvtColor(P, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(im)
        for xy, t, f, c, anc in texts:
            d.text(xy, t, font=f, fill=c, anchor=anc)
        P = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
        return np.hstack([view, P])
