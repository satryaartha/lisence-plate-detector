"""
Stage 5 - Character segmentation.

The selected plate region is binarised (Otsu) and connected components that are
character-like (right height, not too wide) are extracted. Each character is
returned with its bounding box so the OCR stage can read it and the geometry can
later be used to split the plate-number row from the expiry-date row.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import (
    CHAR_MAX_WIDTH_RATIO,
    CHAR_MIN_AREA_RATIO,
    CHAR_MIN_HEIGHT_RATIO,
)


@dataclass
class CharBox:
    box: tuple[int, int, int, int]   # (x, y, w, h) within the plate crop
    image: np.ndarray                # binarised character glyph (white on black)


def _binarise(plate_gray: np.ndarray) -> np.ndarray:
    """Otsu threshold so that characters end up white on a black background.

    Plates come in both polarities (black text on white, or white text on a
    black plate). Characters are always the *minority* of pixels, so after Otsu
    we invert whenever white dominates, guaranteeing white characters.
    """
    _, th = cv2.threshold(plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.mean() > 127:           # white dominates -> characters are the dark part
        th = cv2.bitwise_not(th)
    return th


def segment_characters(plate_gray: np.ndarray) -> list[CharBox]:
    """Return character boxes sorted left-to-right, top-to-bottom.

    Sorting is row-aware so that a two-row Indonesian plate (number row on top,
    expiry/validity row below) keeps a sensible reading order.
    """
    # Trim a small inner margin so the plate's border frame does not become a
    # single enclosing contour that hides the characters inside it.
    mh = max(1, int(plate_gray.shape[0] * 0.06))
    mw = max(1, int(plate_gray.shape[1] * 0.03))
    plate_gray = plate_gray[mh:-mh, mw:-mw]

    ph, pw = plate_gray.shape[:2]
    plate_area = ph * pw
    binary = _binarise(plate_gray)

    # Light morphological opening to remove specks.
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # RETR_LIST returns every contour (not just outermost) so characters are not
    # suppressed when a residual frame ring survives the margin trim.
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    chars: list[CharBox] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area_ratio = (w * h) / float(plate_area)
        height_ratio = h / float(ph)
        width_ratio = w / float(pw)
        if (
            area_ratio >= CHAR_MIN_AREA_RATIO
            and height_ratio >= CHAR_MIN_HEIGHT_RATIO
            and width_ratio <= CHAR_MAX_WIDTH_RATIO
            and h > w * 0.7  # characters are taller than wide-ish
        ):
            glyph = binary[y:y + h, x:x + w]
            chars.append(CharBox(box=(x, y, w, h), image=glyph))

    return _sort_reading_order(chars, ph)


def _sort_reading_order(chars: list[CharBox], plate_h: int) -> list[CharBox]:
    """Group characters into rows then sort each row left-to-right."""
    if not chars:
        return []
    # Cluster by vertical centre: anything in the top ~55% is row 0.
    rows: dict[int, list[CharBox]] = {0: [], 1: []}
    for ch in chars:
        _, y, _, h = ch.box
        cy = y + h / 2.0
        row = 0 if cy < plate_h * 0.55 else 1
        rows[row].append(ch)
    ordered: list[CharBox] = []
    for r in (0, 1):
        ordered.extend(sorted(rows[r], key=lambda c: c.box[0]))
    return ordered


def split_rows(chars: list[CharBox], plate_h: int) -> tuple[list[CharBox], list[CharBox]]:
    """Split characters into (plate-number row, expiry/validity row)."""
    top, bottom = [], []
    for ch in chars:
        _, y, _, h = ch.box
        cy = y + h / 2.0
        (top if cy < plate_h * 0.55 else bottom).append(ch)
    return (sorted(top, key=lambda c: c.box[0]),
            sorted(bottom, key=lambda c: c.box[0]))
