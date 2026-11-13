"""Face detection + temporally stable, enlarged square face crop.

The exact same code path is used for building the training set and in the live
monitor, so the network always sees crops produced the same way.
"""

import numpy as np
import cv2

from . import BOX_SCALE, CROP_SIZE


class BoxSmoother:
    """Causal, motion-adaptive box smoother.

    Small displacements (detector jitter) are smoothed strongly; large displacements
    (real head movement) are followed quickly so the face never leaves the crop.
    State is (cx, cy, size) of a square box.
    """

    def __init__(self, slow: float = 0.12, fast: float = 0.75):
        self.slow, self.fast = slow, fast
        self.state = None

    def reset(self):
        self.state = None

    def update(self, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        size = max(x2 - x1, y2 - y1)
        meas = np.array([cx, cy, size], dtype=np.float64)
        if self.state is None:
            self.state = meas
            return self.state.copy()
        # displacement relative to the face size
        rel = np.linalg.norm(meas[:2] - self.state[:2]) / max(self.state[2], 1.0)
        a = float(np.clip(self.slow + (rel / 0.08) * (self.fast - self.slow), self.slow, self.fast))
        a_size = min(a, 0.25)
        self.state[:2] = (1 - a) * self.state[:2] + a * meas[:2]
        self.state[2] = (1 - a_size) * self.state[2] + a_size * meas[2]
        return self.state.copy()


def crop_square(frame_bgr: np.ndarray, state, scale: float = BOX_SCALE, out: int = CROP_SIZE) -> np.ndarray:
    """Crop a square of side scale*size around (cx,cy) with replicate-padding, resize to out x out RGB."""
    cx, cy, size = state
    side = max(8.0, size * scale)
    # affine warp handles borders (replicate) and sub-pixel positioning in one step
    s = out / side
    M = np.array([[s, 0, out / 2.0 - s * cx], [0, s, out / 2.0 - s * cy]], dtype=np.float32)
    crop = cv2.warpAffine(frame_bgr, M, (out, out), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


class FaceDetector:
    """YOLOv8 (project-trained Face/Nose/Chest model) with Haar-cascade fallback."""

    def __init__(self, weights: str = None, device: str = None, imgsz: int = 320, conf: float = 0.30):
        self.yolo = None
        self.imgsz, self.conf = imgsz, conf
        self.device = device
        if weights:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(weights)
            except Exception as e:  # pragma: no cover
                print(f"[face] YOLO load failed ({e}); using Haar cascade")
        if self.yolo is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect(self, frames_bgr):
        """frames_bgr: list of BGR frames -> list of face boxes [x1,y1,x2,y2] or None."""
        out = []
        if self.yolo is not None:
            try:
                res = self.yolo(frames_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False, device=self.device)
            except Exception as e:      # e.g. Apple MPS op missing -> CPU from now on
                if self.device in (None, "cpu"):
                    raise
                print(f"[face] detector on {self.device} failed ({type(e).__name__}); using CPU")
                self.device = "cpu"
                res = self.yolo(frames_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False, device="cpu")
            for r in res:
                best = None
                if r.boxes is not None and len(r.boxes):
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    cf = r.boxes.conf.cpu().numpy()
                    xy = r.boxes.xyxy.cpu().numpy()
                    faces = np.where(cls == 0)[0]
                    if len(faces):
                        best = xy[faces[np.argmax(cf[faces])]].tolist()
                out.append(best)
        else:
            for fr in frames_bgr:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                f = self.haar.detectMultiScale(g, 1.2, 5, minSize=(60, 60))
                if len(f):
                    x, y, w, h = max(f, key=lambda r: r[2] * r[3])
                    out.append([float(x), float(y), float(x + w), float(y + h)])
                else:
                    out.append(None)
        return out
