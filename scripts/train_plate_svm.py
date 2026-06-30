"""
scripts/train_plate_svm.py

Train the HOG + SVM plate / non-plate classifier on the crops produced by
prepare_dataset.py, evaluate on a held-out split, and save the model.

Run:
  python scripts/train_plate_svm.py
"""
from __future__ import annotations

import glob
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PLATE_SVM_PATH, PROCESSED_DIR, RANDOM_STATE, SUCCESS_CRITERIA
from src.evaluation import classification_metrics, report
from src.features import plate_hog
from src.plate_classifier import PlateClassifier


def _load(folder: str) -> list[np.ndarray]:
    imgs = []
    for ext in ("png", "jpg", "jpeg"):
        for p in glob.glob(os.path.join(folder, f"*.{ext}")):
            im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if im is not None:
                imgs.append(im)
    return imgs


def main():
    pos = _load(os.path.join(PROCESSED_DIR, "positives"))
    neg = _load(os.path.join(PROCESSED_DIR, "negatives"))
    if not pos or not neg:
        raise SystemExit(
            "No crops found. Run scripts/prepare_dataset.py first."
        )
    print(f"Loaded {len(pos)} positives, {len(neg)} negatives")

    X = np.vstack([plate_hog(im) for im in pos + neg])
    y = np.array([1] * len(pos) + [0] * len(neg))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    clf = PlateClassifier().fit(Xtr, ytr)
    preds = clf.predict(Xte)

    m = classification_metrics(yte, preds, positive=1)
    print("\nPlate classifier (held-out test):")
    print(report(m.as_dict(), SUCCESS_CRITERIA))

    clf.save()
    print(f"\nSaved model -> {PLATE_SVM_PATH}")


if __name__ == "__main__":
    main()
