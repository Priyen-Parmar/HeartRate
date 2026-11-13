"""Every folder the project reads from or writes to, in one place.

Scripts anywhere in the repository import these instead of building paths themselves, so the folder
layout can change without touching the rest of the code.
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# deployed models (versioned in git)
MODELS_DIR = os.path.join(ROOT, "models")
VITALNET_DIR = os.path.join(MODELS_DIR, "vitalnet")
FACE_DETECTOR = os.path.join(MODELS_DIR, "face_detector", "yolov8_face_nose_chest.pt")

# evaluation results (versioned): RESULTS.md, figures/, metrics/*.json
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATASET_AUDIT = os.path.join(METRICS_DIR, "dataset_audit.json")

# runtime outputs of the live app: HR logs, raw recordings, screenshots (git-ignored, private)
OUTPUTS_DIR = os.path.join(ROOT, "outputs")

# UBFC-rPPG dataset (git-ignored, only needed for training / evaluation)
DATA_DIR = os.path.join(ROOT, "data")
UBFC_RAW_DIR = os.path.join(DATA_DIR, "raw", "UBFC-rPPG")
UBFC_PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "ubfc_v2")
LEGACY_DATA_DIR = os.path.join(DATA_DIR, "legacy")

# old system and experiment checkpoints (not used by the current app)
ARCHIVE_DIR = os.path.join(ROOT, "archive")
LEGACY_SYSTEM_DIR = os.path.join(ARCHIVE_DIR, "legacy_system")
TRIAL_MODELS_DIR = os.path.join(ARCHIVE_DIR, "trial_models")

# Ultralytics (YOLO) writes a settings file; keep it inside the project
YOLO_CONFIG_DIR = os.path.join(ROOT, ".cache", "ultralytics")
