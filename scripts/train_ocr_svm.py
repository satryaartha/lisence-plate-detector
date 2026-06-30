"""
scripts/train_ocr_svm.py

Train the HOG + SVM character recogniser (0-9, A-Z).

Two data sources are supported:

  --source synthetic   (default) render glyphs from system fonts. Self-contained
                       and good enough to demonstrate the full pipeline.
  --source folders --char-dir PATH
                       a labelled character dataset where each class has its own
                       sub-folder (PATH/A/*.png, PATH/B/*.png, ...). Use this
                       (e.g. Chars74K or a cropped-plate-character set) for the
                       best real-world OCR accuracy.

Run:
  python scripts/train_ocr_svm.py --source synthetic --per-class 200
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OCR_CLASSES, OCR_SVM_PATH, RANDOM_STATE, SUCCESS_CRITERIA
from src.evaluation import character_accuracy
from src.ocr import OCRClassifier, build_synthetic_dataset


def _load_folders(char_dir: str):
    """Load labelled character images.

    Robust to nesting: walks `char_dir` recursively and labels each image by the
    name of its immediate parent folder when that name is a single character in
    OCR_CLASSES (handles DatasetCharacter/A/.., DatasetCharacter/DatasetCharacter/A/.., etc.).
    """
    images, labels = [], []
    classes = set(OCR_CLASSES)
    for root, _dirs, files in os.walk(char_dir):
        label = os.path.basename(root).strip().upper()
        if label not in classes:
            continue
        for fn in files:
            if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            im = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                images.append(im)
                labels.append(label)
    return images, np.array(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "folders"], default="synthetic")
    ap.add_argument("--char-dir", default=None)
    ap.add_argument("--per-class", type=int, default=200)
    args = ap.parse_args()

    if args.source == "folders":
        if not args.char_dir:
            raise SystemExit("--char-dir is required with --source folders")
        images, labels = _load_folders(args.char_dir)
    else:
        images, labels = build_synthetic_dataset(args.per_class)
    print(f"Loaded {len(images)} character samples across {len(set(labels))} classes")

    idx = np.arange(len(images))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels,
                              random_state=RANDOM_STATE)
    ocr = OCRClassifier().fit([images[i] for i in tr], labels[tr])

    preds = [ocr.predict_char(images[i]) for i in te]
    truth = [labels[i] for i in te]
    acc = character_accuracy(preds, list(truth))
    crit = SUCCESS_CRITERIA["char_accuracy"]
    print(f"\nCharacter accuracy (held-out): {acc:.3f} "
          f"(target >= {crit:.2f})  {'PASS' if acc >= crit else 'FAIL'}")

    ocr.save()
    print(f"Saved model -> {OCR_SVM_PATH}")


if __name__ == "__main__":
    main()
