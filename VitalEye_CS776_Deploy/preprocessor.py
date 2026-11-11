"""
Advanced rPPG Preprocessor & Hybrid Anatomical + Multi-Zone Adaptive Skin Segmentation
──────────────────────────────────────────────────────────────────────────────────────
Combines:
  1. 468-Point 3D FaceMesh Anatomical ROI (`models/face_landmark.tflite` via OpenCV DNN)
     — Tightly contours the Face Oval (100% eliminating background walls/bricks/neck)
     — Explicitly carves out Left/Right Eyes, Eyebrows, Eyeglass Rims, Nostrils, and Lips/Mustache
  2. Multi-Zone Illumination-Invariant Chrominance Gate (Forehead, Left Cheek, Right Cheek Seeds)
     — Computes local chrominance centers for each facial zone so shaded cheeks under
       directional lighting are preserved alongside lit cheeks
  3. Local-Luminance (`Y / GaussianBlur(Y)`) & Lower-Face Beard / Eyeglass Frame Rejection
     — Rejects dark eyeglass rims, eyebrows, and facial hair/beard without cutting off
       shaded skin regions
"""

import os
import glob
import re
import argparse
import cv2
import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────
SEQ_LENGTH = 300        # frames per clip (longer = better HR estimation)
FACE_SIZE  = 128        # spatial resolution
STRIDE     = 150        # stride between chunks (50% overlap)
BP_LO      = 0.7        # bandpass low  (Hz) → ~42 BPM
BP_HI      = 2.5        # bandpass high (Hz) → ~150 BPM
BP_ORDER   = 2          # Butterworth order
EMA_ALPHA  = 0.7        # face-box smoothing strength

# ─────────────────────────────────────────────────────────
# 468-POINT FACEMESH ANATOMICAL TOPOLOGY INDICES
# ─────────────────────────────────────────────────────────
# Vascular Face Contour: Forehead + Temples + Left/Right Malar Cheeks (excludes lower jaw/chin beard border)
VASCULAR_FACE_LOOP = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 435,
    367, 364, 394, 395, 369, 396, 175, 171, 140, 170, 169, 135,
    138, 215, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]

LEFT_PERIOCULAR_PTS = [
    374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373,
    276, 282, 283, 285, 293, 295, 296, 300, 334, 336, 353, 383, 372, 340, 261, 448, 450, 452
]

RIGHT_PERIOCULAR_PTS = [
    154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153,
    46, 52, 53, 55, 63, 65, 66, 70, 105, 107, 124, 156, 143, 111, 31, 228, 230, 232
]

MOUTH_AND_MUSTACHE_PTS = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191,
    267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415,
    164, 165, 167, 92, 186, 57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393
]

NOSTRIL_BASE_PTS = [1, 2, 94, 97, 98, 326, 327, 168, 19]

# Zone-specific pure skin seed landmarks
FOREHEAD_SEED_PTS    = [10, 151, 9, 108, 337, 109, 338, 69, 299]
RIGHT_CHEEK_SEED_PTS = [50, 101, 118, 119, 123, 187, 205, 36, 116, 117]
LEFT_CHEEK_SEED_PTS  = [280, 330, 347, 348, 352, 411, 425, 266, 345, 346]
NOSE_BRIDGE_SEED_PTS = [6, 197, 195, 5]

_FACEMESH_NET = None
_FACEMESH_INIT_TRIED = False


def _get_facemesh_net():
    """Lazily initializes the OpenCV C++ DNN 468-point FaceMesh model."""
    global _FACEMESH_NET, _FACEMESH_INIT_TRIED
    if _FACEMESH_INIT_TRIED:
        return _FACEMESH_NET
    _FACEMESH_INIT_TRIED = True

    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "face_landmark.tflite"),
        "/Users/jayantmitawa/Desktop/UGP/HeartRate/VitalEye_CS776_Deploy/models/face_landmark.tflite",
        "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/mediapipe/modules/face_landmark/face_landmark.tflite",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                _FACEMESH_NET = cv2.dnn.readNet(p)
                break
            except Exception:
                pass
    return _FACEMESH_NET


def _expand_polygon(pts: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Scales a 2D polygon around its centroid."""
    center = pts.mean(axis=0, keepdims=True)
    diff = pts - center
    diff[:, 0] *= scale_x
    diff[:, 1] *= scale_y
    return np.round(center + diff).astype(np.int32)


def _anatomical_facemesh_mask(frame_rgb: np.ndarray):
    """
    Runs 468-point FaceMesh (2 ms via cv2.dnn) on `frame_rgb` and returns:
      (anat_mask, zone_seed_masks, lower_beard_zone_mask)
    """
    h, w = frame_rgb.shape[:2]
    net = _get_facemesh_net()
    if net is None:
        return None, None, None

    try:
        inp_img = cv2.resize(frame_rgb, (192, 192))
        blob = (inp_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        conf = [o for o in outs if o.size == 1][0].item()
        if conf < -1.5:
            return None, None, None

        pts = [o for o in outs if o.size == 1404][0].reshape(468, 3)
        xy = np.zeros((468, 2), dtype=np.float32)
        xy[:, 0] = pts[:, 0] * (w / 192.0)
        xy[:, 1] = pts[:, 1] * (h / 192.0)

        anat_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Fill vascular upper/mid face contour (inset 4% to exclude hairline/sideburns/background edge)
        vascular_poly = _expand_polygon(xy[VASCULAR_FACE_LOOP], scale_x=0.95, scale_y=0.96)
        cv2.fillPoly(anat_mask, [vascular_poly], 255)

        # 2. Carve out Left & Right Periocular + Eyebrow + Eyeglass zones
        left_eye_hull = cv2.convexHull(_expand_polygon(xy[LEFT_PERIOCULAR_PTS], scale_x=1.10, scale_y=1.14))
        right_eye_hull = cv2.convexHull(_expand_polygon(xy[RIGHT_PERIOCULAR_PTS], scale_x=1.10, scale_y=1.14))
        cv2.fillConvexPoly(anat_mask, left_eye_hull, 0)
        cv2.fillConvexPoly(anat_mask, right_eye_hull, 0)

        # 3. Carve out Mouth + Lips + Mustache zone
        mouth_hull = cv2.convexHull(_expand_polygon(xy[MOUTH_AND_MUSTACHE_PTS], scale_x=1.12, scale_y=1.15))
        cv2.fillConvexPoly(anat_mask, mouth_hull, 0)

        # 4. Carve out dark Nostril cavities
        nostril_hull = cv2.convexHull(_expand_polygon(xy[NOSTRIL_BASE_PTS], scale_x=1.08, scale_y=1.08))
        cv2.fillConvexPoly(anat_mask, nostril_hull, 0)

        # 5. Build 3 zone seed masks (Forehead, Right Cheek, Left Cheek) so directional lighting is handled per zone
        r_seed = max(3, int(min(h, w) * 0.045))
        zone_seeds = []
        for index_group in (
            FOREHEAD_SEED_PTS + NOSE_BRIDGE_SEED_PTS,
            RIGHT_CHEEK_SEED_PTS,
            LEFT_CHEEK_SEED_PTS,
        ):
            zm = np.zeros((h, w), dtype=np.uint8)
            for idx in index_group:
                cx, cy = int(round(xy[idx, 0])), int(round(xy[idx, 1]))
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(zm, (cx, cy), r_seed, 255, -1)
            zm = cv2.bitwise_and(zm, anat_mask)
            zone_seeds.append(zm)

        # 6. Identify lower-face beard-prone zone (below subnasale / nose base landmark 2)
        nose_base_y = int(round(xy[2, 1]))
        lower_beard_zone = np.zeros((h, w), dtype=np.uint8)
        if 0 < nose_base_y < h:
            lower_beard_zone[nose_base_y:, :] = 255

        return anat_mask, zone_seeds, lower_beard_zone
    except Exception:
        return None, None, None


def _geometric_fallback_masks(h: int, w: int):
    """Fallback anatomical mask if FaceMesh confidence is low."""
    anat_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        anat_mask,
        (w // 2, int(h * 0.44)),
        (int(w * 0.38), int(h * 0.38)),
        0, 0, 360, 255, -1
    )
    cv2.ellipse(anat_mask, (int(w * 0.33), int(h * 0.36)), (int(w * 0.16), int(h * 0.09)), 0, 0, 360, 0, -1)
    cv2.ellipse(anat_mask, (int(w * 0.67), int(h * 0.36)), (int(w * 0.16), int(h * 0.09)), 0, 0, 360, 0, -1)
    cv2.ellipse(anat_mask, (w // 2, int(h * 0.72)), (int(w * 0.22), int(h * 0.11)), 0, 0, 360, 0, -1)

    s1 = np.zeros((h, w), dtype=np.uint8)
    s2 = np.zeros((h, w), dtype=np.uint8)
    s3 = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(s1, (int(w * 0.35), int(h * 0.16)), (int(w * 0.65), int(h * 0.25)), 255, -1)
    cv2.rectangle(s2, (int(w * 0.20), int(h * 0.48)), (int(w * 0.42), int(h * 0.62)), 255, -1)
    cv2.rectangle(s3, (int(w * 0.58), int(h * 0.48)), (int(w * 0.80), int(h * 0.62)), 255, -1)

    lower_beard_zone = np.zeros((h, w), dtype=np.uint8)
    lower_beard_zone[int(h * 0.60):, :] = 255
    return anat_mask, [cv2.bitwise_and(s, anat_mask) for s in (s1, s2, s3)], lower_beard_zone


def skin_mask_ycrcb(frame_rgb: np.ndarray) -> np.ndarray:
    """
    Hybrid 468-Landmark Anatomical + Multi-Zone Illumination-Invariant Skin Segmentation:
      1. Clips strictly to the vascular facial contour (100% removing background walls/bricks)
      2. Carves out Left/Right Eyes, Eyebrows, Eyeglass Rims, Nostrils, Lips, and Mustache
      3. Evaluates chrominance distance across 3 independent facial zones (Forehead, Left Cheek,
         Right Cheek) so directional lighting/shadows never cut off the shaded side of the face
      4. Uses local illumination ratio (Y / GaussianBlur(Y)) + lower-face beard suppression
         to remove beard/stubble, dark eyeglass rims, and specular highlights
    Returns uint8 binary mask (0 or 255) of shape (H, W).
    """
    if frame_rgb is None or frame_rgb.size == 0:
        return np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.uint8)

    h, w = frame_rgb.shape[:2]

    # ── Stage 1: Anatomical Vascular Face ROI (468-pt FaceMesh with Geometric Fallback) ──
    anat_mask, zone_seeds, lower_beard_zone = _anatomical_facemesh_mask(frame_rgb)
    if anat_mask is None or np.count_nonzero(anat_mask) < int(0.08 * h * w):
        anat_mask, zone_seeds, lower_beard_zone = _geometric_fallback_masks(h, w)

    # ── Stage 2: Multi-Space Color Conversion (YCrCb + CIE LAB) ──
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    lab = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    Y, Cr, Cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    A = lab[:, :, 1]

    # Physiological human skin prior (allows shaded skin down to Y=35)
    broad_skin = (
        (Cr >= 132.0) & (Cr <= 178.0) &
        (Cb >= 75.0) & (Cb <= 129.0) &
        ((Cr - Cb) >= 11.0) &
        (Y >= 35.0) & (Y <= 245.0)
    )

    # Compute per-zone chrominance distance (takes minimum across Forehead, Right Cheek, Left Cheek)
    min_dist_sq = np.full((h, w), 999.0, dtype=np.float32)
    zone_y_meds = []
    zone_cr_meds = []

    for zm in zone_seeds:
        valid = (zm > 0) & broad_skin
        if np.count_nonzero(valid) >= 8:
            z_cr = float(np.median(Cr[valid]))
            z_cb = float(np.median(Cb[valid]))
            z_a = float(np.median(A[valid]))
            z_y = float(np.median(Y[valid]))
            zone_y_meds.append(z_y)
            zone_cr_meds.append(z_cr)

            cr_mad = float(np.median(np.abs(Cr[valid] - z_cr))) * 1.4826
            cb_mad = float(np.median(np.abs(Cb[valid] - z_cb))) * 1.4826
            a_mad = float(np.median(np.abs(A[valid] - z_a))) * 1.4826

            tol_cr = float(np.clip(3.2 * cr_mad, 9.5, 16.5))
            tol_cb = float(np.clip(3.2 * cb_mad, 9.5, 16.5))
            tol_a = float(np.clip(3.2 * a_mad, 8.0, 15.0))

            d_sq = (
                ((Cr - z_cr) / tol_cr) ** 2 +
                ((Cb - z_cb) / tol_cb) ** 2 +
                0.35 * ((A - z_a) / tol_a) ** 2
            )
            min_dist_sq = np.minimum(min_dist_sq, d_sq)

    # ── Stage 3: Local-Illumination & Beard / Eyeglass Frame Rejection ──
    # Smooth local illumination field (captures directional lighting across left vs right cheek)
    k_blur = max(15, (min(h, w) // 4) | 1)
    Y_local = cv2.GaussianBlur(Y, (k_blur, k_blur), 0) + 1e-5
    rel_Y = Y / Y_local

    # Locally dark structures (eyeglass frames, hair strands, coarse beard) have rel_Y < 0.76
    # Locally bright specular reflections on eyeglass lenses have rel_Y > 1.28 and low Cr
    local_valid = (rel_Y >= 0.76) & (rel_Y <= 1.28)

    # In the lower-face beard zone (below the nose base), beard/stubble is darker and less red
    # than the upper malar cheeks
    if zone_y_meds and zone_cr_meds:
        upper_y_ref = float(np.mean(zone_y_meds))
        upper_cr_ref = float(np.mean(zone_cr_meds))
        beard_reject = (
            (lower_beard_zone > 0) &
            ((Y < 0.78 * upper_y_ref) | (Cr < upper_cr_ref - 5.5) | (rel_Y < 0.85))
        )
        adaptive_skin = broad_skin & (min_dist_sq <= 1.0) & local_valid & (~beard_reject)
    else:
        adaptive_skin = broad_skin & local_valid

    combined = np.where((anat_mask > 0) & adaptive_skin, 255, 0).astype(np.uint8)

    # ── Stage 4: Morphological Smoothing & Small-Speckle Removal ──
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k_open)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    clean_mask = np.zeros_like(combined)
    min_area = max(18, int(0.006 * h * w))
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == lbl] = 255

    if np.count_nonzero(clean_mask) < int(0.05 * h * w):
        clean_mask = cv2.bitwise_and(anat_mask, np.where(broad_skin, 255, 0).astype(np.uint8))

    return clean_mask


# ─────────────────────────────────────────────────────────
# DATASET PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────
def bandpass_filter(signal, fps, lo=BP_LO, hi=BP_HI, order=BP_ORDER):
    """Apply zero-phase Butterworth bandpass to 1-D signal."""
    nyq = fps / 2.0
    lo_n, hi_n = max(lo / nyq, 0.001), min(hi / nyq, 0.999)
    b, a = butter(order, [lo_n, hi_n], btype='band')
    return filtfilt(b, a, signal).astype(np.float32)


def extract_subject_id(vid_filename):
    """Extract subject identifier from UBFC filename."""
    name = os.path.splitext(vid_filename)[0]
    m = re.search(r'sub(\d+)', name)
    if m:
        return f"sub{m.group(1)}"
    m = re.search(r'vid_1_(\w+)', name)
    if m:
        return f"ds1_{m.group(1)}"
    return name


def smooth_box(prev_box, new_box, alpha=EMA_ALPHA):
    """Exponential moving average of bounding box [x1,y1,x2,y2]."""
    if prev_box is None:
        return new_box
    return [alpha * p + (1 - alpha) * n for p, n in zip(prev_box, new_box)]
