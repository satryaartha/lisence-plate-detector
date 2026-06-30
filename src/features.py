"""
Stage 3 - Histogram of Oriented Gradients (HOG) feature extraction.

HOG describes local shape via the distribution of gradient orientations. The
same descriptor is reused for two tasks at different input sizes:
  * plate / non-plate classification  -> PLATE_HOG_SIZE
  * single-character OCR               -> CHAR_HOG_SIZE
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog

from config import CHAR_HOG_SIZE, HOG_PARAMS, PLATE_HOG_SIZE


def _resize_gray(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Resize to (height, width) and ensure single-channel uint8."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = size_hw
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def normalize_char(glyph: np.ndarray, out_size: int = 40,
                   margin_ratio: float = 0.18) -> np.ndarray:
    """Center a character glyph on a square canvas with uniform margin.

    Segmented characters are tight crops while rendered training glyphs have
    padding; normalising both the same way removes that domain gap so the OCR
    SVM sees consistent inputs.
    """
    g = glyph
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:                       # ensure white glyph on black
        b = cv2.bitwise_not(b)
    ys, xs = np.where(b > 0)
    if xs.size == 0:
        return cv2.resize(b, (out_size, out_size))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = b[y0:y1 + 1, x0:x1 + 1]
    ch, cw = crop.shape
    inner = int(out_size * (1 - 2 * margin_ratio))
    scale = inner / max(ch, cw)
    nh, nw = max(1, int(ch * scale)), max(1, int(cw * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), np.uint8)
    oy, ox = (out_size - nh) // 2, (out_size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = crop
    return canvas


def hog_features(img: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    """Compute a flat HOG feature vector for a grayscale image."""
    resized = _resize_gray(img, size_hw)
    feats = hog(resized, feature_vector=True, **HOG_PARAMS)
    return feats.astype(np.float32)


def plate_hog(img: np.ndarray) -> np.ndarray:
    """HOG vector for a plate-candidate region."""
    return hog_features(img, PLATE_HOG_SIZE)


def char_hog(img: np.ndarray) -> np.ndarray:
    """HOG vector for a single segmented character."""
    return hog_features(img, CHAR_HOG_SIZE)


def hog_visualization(img: np.ndarray,
                      size_hw: tuple[int, int] = PLATE_HOG_SIZE) -> np.ndarray:
    """Return a renderable HOG visualisation image (for the notebook/report)."""
    resized = _resize_gray(img, size_hw)
    _, hog_img = hog(resized, visualize=True, **HOG_PARAMS)
    return hog_img


def batch_hog(images: list[np.ndarray], size_hw: tuple[int, int]) -> np.ndarray:
    """Stack HOG vectors for a list of images into a (N, D) matrix."""
    return np.vstack([hog_features(im, size_hw) for im in images])
