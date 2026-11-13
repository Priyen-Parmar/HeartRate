"""
Root wrapper for preprocessor.py — imports the upgraded Hybrid Anatomical FaceMesh +
Subject-Adaptive YCrCb/LAB skin segmentation from deploy/preprocessor.py.
"""

import os
import sys

DEPLOY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deploy")
if DEPLOY_DIR not in sys.path:
    sys.path.insert(0, DEPLOY_DIR)

from preprocessor import (  # noqa: F401
    skin_mask_ycrcb,
    bandpass_filter,
    extract_subject_id,
    smooth_box,
    SEQ_LENGTH,
    FACE_SIZE,
    STRIDE,
    BP_LO,
    BP_HI,
    BP_ORDER,
    EMA_ALPHA,
)
