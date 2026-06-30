"""
Stage 1 - Preprocessing.

Implements the first step of the proposed method: convert the input image to
grayscale and apply a Gaussian filter to suppress noise before edge detection.
"""
from __future__ import annotations

import cv2
import numpy as np

from config import GAUSSIAN_KERNEL, GAUSSIAN_SIGMA


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR (or already-gray) image to a single-channel gray image."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def gaussian_blur(gray: np.ndarray,
                  kernel: tuple[int, int] = GAUSSIAN_KERNEL,
                  sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """Apply a Gaussian filter to reduce noise / false edges."""
    return cv2.GaussianBlur(gray, kernel, sigma)


def preprocess(image: np.ndarray) -> np.ndarray:
    """Full preprocessing: grayscale then Gaussian blur.

    Parameters
    ----------
    image : np.ndarray
        BGR or grayscale image as loaded by cv2.imread.

    Returns
    -------
    np.ndarray
        Blurred grayscale image, dtype uint8.
    """
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray)
    return blurred


def read_image(path: str) -> np.ndarray:
    """Load an image from disk, raising a clear error if it is missing."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
