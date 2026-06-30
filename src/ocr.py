"""
Stage 6 - OCR with HOG + SVM.

A multi-class SVM recognises individual segmented characters (0-9, A-Z) from
their HOG descriptors. This keeps the whole system within the classical
HOG + SVM family chosen in the proposal (no deep-learning OCR).

The Kaggle detection dataset does not ship per-character labels, so this module
also provides a *synthetic* character generator: it renders glyphs from system
fonts with light augmentation, which is enough to bootstrap and demonstrate the
OCR classifier. Swap in a real labelled character set if you have one.
"""
from __future__ import annotations

import glob
import random

import cv2
import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    CHAR_HOG_SIZE,
    OCR_CLASSES,
    OCR_SVM_C,
    OCR_SVM_KERNEL,
    OCR_SVM_PATH,
    RANDOM_STATE,
)
from src.features import char_hog, normalize_char

_FONT_GLOBS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono*.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-*.ttf",
    "/usr/share/fonts/**/*.ttf",
]


def _find_fonts(max_fonts: int = 6) -> list[str]:
    found: list[str] = []
    for pat in _FONT_GLOBS:
        for f in glob.glob(pat, recursive=True):
            if f not in found:
                found.append(f)
        if len(found) >= max_fonts:
            break
    return found[:max_fonts] or [None]  # None -> PIL default font


def render_glyph(ch: str, font_path: str | None, size: int = 48,
                 jitter: bool = True) -> np.ndarray:
    """Render a single character to a binarised (white-on-black) glyph image."""
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(font_path, int(size * 0.8)) if font_path \
            else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), ch, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    ox = (size - tw) / 2 - bbox[0]
    oy = (size - th) / 2 - bbox[1]
    if jitter:
        ox += random.randint(-2, 2)
        oy += random.randint(-2, 2)
    draw.text((ox, oy), ch, fill=255, font=font)
    arr = np.array(canvas)
    if jitter:
        angle = random.uniform(-8, 8)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        arr = cv2.warpAffine(arr, M, (size, size))
        if random.random() < 0.5:
            arr = cv2.GaussianBlur(arr, (3, 3), 0)
    return arr


def build_synthetic_dataset(samples_per_class: int = 40):
    """Generate (images, labels) for all OCR_CLASSES using system fonts."""
    fonts = _find_fonts()
    images, labels = [], []
    for ch in OCR_CLASSES:
        for _ in range(samples_per_class):
            font = random.choice(fonts)
            images.append(render_glyph(ch, font))
            labels.append(ch)
    return images, np.array(labels)


class OCRClassifier:
    """HOG + multi-class SVM character recogniser."""

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=OCR_SVM_C,
                kernel=OCR_SVM_KERNEL,
                probability=False,
                decision_function_shape="ovr",
                random_state=RANDOM_STATE,
            )),
        ])

    def fit(self, images: list[np.ndarray], labels: np.ndarray) -> "OCRClassifier":
        X = np.vstack([char_hog(normalize_char(im)) for im in images])
        self.model.fit(X, labels)
        return self

    def predict_char(self, glyph: np.ndarray) -> str:
        feats = char_hog(normalize_char(glyph)).reshape(1, -1)
        return str(self.model.predict(feats)[0])

    def predict_char_restricted(self, glyph: np.ndarray, allowed: list[str]) -> str:
        """Predict but only among `allowed` classes (e.g. digits for expiry).

        Uses the SVM decision scores and picks the best-scoring allowed class.
        """
        feats = char_hog(normalize_char(glyph)).reshape(1, -1)
        scaler = self.model.named_steps["scaler"]
        svm = self.model.named_steps["svm"]
        scores = svm.decision_function(scaler.transform(feats))[0]
        classes = list(svm.classes_)
        best, best_score = None, -1e18
        for c in allowed:
            if c in classes:
                s = scores[classes.index(c)]
                if s > best_score:
                    best, best_score = c, s
        return best if best is not None else self.predict_char(glyph)

    def read(self, char_boxes) -> str:
        """Concatenate predictions for an ordered list of CharBox."""
        return "".join(self.predict_char(cb.image) for cb in char_boxes)

    def read_digits(self, char_boxes) -> str:
        """Read an ordered list of CharBox restricted to digit classes."""
        digits = list("0123456789")
        return "".join(self.predict_char_restricted(cb.image, digits)
                       for cb in char_boxes)

    def save(self, path: str = OCR_SVM_PATH) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str = OCR_SVM_PATH) -> "OCRClassifier":
        obj = cls()
        obj.model = joblib.load(path)
        return obj
