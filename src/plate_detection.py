"""
Stage 2 - Plate localisation with Canny Edge Detection + contour filtering.

The blurred grayscale image is passed through the Canny operator, contours are
extracted from the edge map, and only contours whose geometry is plate-like
(rectangular, correct area, correct aspect ratio) are kept as candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    APPROX_POLY_EPS,
    CANNY_HIGH,
    CANNY_LOW,
    PLATE_MAX_AREA_RATIO,
    PLATE_MAX_ASPECT,
    PLATE_MIN_AREA_RATIO,
    PLATE_MIN_ASPECT,
)


@dataclass
class Candidate:
    """A rectangular region that might be a license plate.

    Attributes
    ----------
    box : (x, y, w, h) bounding box in pixels
    crop : the cropped grayscale region
    score : geometric "rectangularity" score, higher = more plate-like
    """
    box: tuple[int, int, int, int]
    crop: np.ndarray
    score: float


def canny_edges(gray_blurred: np.ndarray,
                low: int = CANNY_LOW,
                high: int = CANNY_HIGH) -> np.ndarray:
    """Run the Canny edge detector (noise reduction already done upstream)."""
    return cv2.Canny(gray_blurred, low, high)


def _is_plate_like(w: int, h: int, img_area: int) -> bool:
    if h == 0:
        return False
    area = w * h
    aspect = w / float(h)
    area_ratio = area / float(img_area)
    return (
        PLATE_MIN_AREA_RATIO <= area_ratio <= PLATE_MAX_AREA_RATIO
        and PLATE_MIN_ASPECT <= aspect <= PLATE_MAX_ASPECT
    )


def find_candidates(gray_blurred: np.ndarray,
                    max_candidates: int = 10) -> list[Candidate]:
    """Return plate candidates ordered by how rectangular they are.

    A candidate's geometric score rewards 4-corner rectangles and a contour
    area that fills its bounding box (true plates are solid rectangles).
    """
    edges = canny_edges(gray_blurred)
    # Dilate to close small gaps in plate borders before contour finding.
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = gray_blurred.shape[0] * gray_blurred.shape[1]

    candidates: list[Candidate] = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, APPROX_POLY_EPS * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        if not _is_plate_like(w, h, img_area):
            continue
        # rectangularity: contour area / bounding-box area (1.0 = perfect rect)
        rect_fill = cv2.contourArea(c) / float(w * h + 1e-6)
        corner_bonus = 1.0 if len(approx) == 4 else 0.6
        score = rect_fill * corner_bonus
        crop = gray_blurred[y:y + h, x:x + w]
        candidates.append(Candidate(box=(x, y, w, h), crop=crop, score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:max_candidates]


def draw_boxes(image: np.ndarray,
               boxes: list[tuple[int, int, int, int]],
               color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy of `image` with the given boxes drawn (for visualisation)."""
    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out
