import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(ROOT, "models")
VITALNET_DIR = os.path.join(MODELS_DIR, "vitalnet")
FACE_DETECTOR = os.path.join(MODELS_DIR, "face_detector", "yolov8_face_nose_chest.pt")

RESULTS_DIR = os.path.join(ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATASET_AUDIT = os.path.join(METRICS_DIR, "dataset_audit.json")

OUTPUTS_DIR = os.path.join(ROOT, "outputs")

DATA_DIR = os.path.join(ROOT, "data")
UBFC_RAW_DIR = os.path.join(DATA_DIR, "raw", "UBFC-rPPG")
UBFC_PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "ubfc_v2")
LEGACY_DATA_DIR = os.path.join(DATA_DIR, "legacy")

YOLO_CONFIG_DIR = os.path.join(ROOT, ".cache", "ultralytics")
