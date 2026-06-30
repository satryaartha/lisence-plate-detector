"""
Central configuration for the Smart License Plate Expiration Detection system.

All tunable parameters live here so experiments stay reproducible and the
notebook / scripts read from a single source of truth.
"""
from __future__ import annotations

import os
import string

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")          # Kaggle dataset extracted here
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")   # generated crops / splits
MODELS_DIR = os.path.join(ROOT_DIR, "models")         # trained .joblib files
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")        # visualisations, reports

for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

PLATE_SVM_PATH = os.path.join(MODELS_DIR, "plate_svm.joblib")
OCR_SVM_PATH = os.path.join(MODELS_DIR, "ocr_svm.joblib")

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
GAUSSIAN_KERNEL = (5, 5)      # Gaussian blur kernel (must be odd)
GAUSSIAN_SIGMA = 0            # 0 -> derived from kernel size

# --------------------------------------------------------------------------- #
# Canny edge detection + contour filtering (plate localisation)
# --------------------------------------------------------------------------- #
CANNY_LOW = 50
CANNY_HIGH = 150
# A region is a plate *candidate* only if its geometry is plate-like:
PLATE_MIN_AREA_RATIO = 0.001   # fraction of full image area
PLATE_MAX_AREA_RATIO = 0.30
PLATE_MIN_ASPECT = 1.8         # width / height
PLATE_MAX_ASPECT = 6.0
APPROX_POLY_EPS = 0.04         # epsilon factor for cv2.approxPolyDP

# --------------------------------------------------------------------------- #
# HOG descriptor parameters (shared helper, sizes differ per task)
# --------------------------------------------------------------------------- #
# For the plate / non-plate classifier
PLATE_HOG_SIZE = (64, 128)     # (height, width) candidate is resized to this
# For the OCR character classifier
CHAR_HOG_SIZE = (32, 32)

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
)

# --------------------------------------------------------------------------- #
# SVM
# --------------------------------------------------------------------------- #
PLATE_SVM_C = 1.0
PLATE_SVM_KERNEL = "rbf"
OCR_SVM_C = 5.0
OCR_SVM_KERNEL = "rbf"
RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Character segmentation
# --------------------------------------------------------------------------- #
CHAR_MIN_AREA_RATIO = 0.004    # of plate area
CHAR_MIN_HEIGHT_RATIO = 0.16   # char height vs plate height (bottom row is smaller)
CHAR_MAX_WIDTH_RATIO = 0.30    # a single char shouldn't span most of the plate
# Characters used by the OCR classifier (Indonesian plates: digits + A-Z)
OCR_CLASSES = list(string.digits + string.ascii_uppercase)

# --------------------------------------------------------------------------- #
# Evaluation success criteria (from the proposal)
# --------------------------------------------------------------------------- #
SUCCESS_CRITERIA = {
    "accuracy": 0.85,
    "precision": 0.85,
    "recall": 0.85,
    "f1": 0.80,
    "iou": 0.70,
    "map": 0.90,
    "char_accuracy": 0.90,
    "word_accuracy": 0.75,
    "wer": 0.15,          # upper bound (lower is better)
    "end_to_end": 0.75,
    "preprocess_time_s": 1.0,
}
IOU_MATCH_THRESHOLD = 0.5      # IoU above which a detection counts as a hit
