"""
Synthetic data generation (testing & demo helper).

Renders Indonesian-style plates (white-on-black or black-on-white) with a
two-row layout: the registration number on top and an MM.YY validity below.
Optionally composites a plate onto a noisy "scene" background so the full
detection -> classification -> OCR -> expiry pipeline can be exercised without
the Kaggle dataset.

This is for development/demo only; train final models on the real dataset.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ocr import _find_fonts


def make_plate(number: str, expiry: str,
               width: int = 380, height: int = 150,
               dark: bool = True) -> np.ndarray:
    """Render a plate image (BGR) with `number` on top and `expiry` below."""
    bg = (30, 30, 30) if dark else (235, 235, 235)
    fg = (235, 235, 235) if dark else (20, 20, 20)
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font_path = (_find_fonts() or [None])[0]
    try:
        big = ImageFont.truetype(font_path, 66) if font_path else ImageFont.load_default()
        small = ImageFont.truetype(font_path, 40) if font_path else ImageFont.load_default()
    except Exception:
        big = small = ImageFont.load_default()

    def _centered(text, font, cy):
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((width - tw) / 2 - bbox[0], cy), text, fill=fg, font=font)

    _centered(number, big, int(height * 0.10))
    _centered(expiry, small, int(height * 0.66))
    cv2.rectangle  # noqa: keep import usage explicit
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(arr, (3, 3), (width - 4, height - 4), fg, 3)
    return arr


def make_scene(plate_img: np.ndarray,
               canvas: tuple[int, int] = (640, 480),
               seed: int | None = None):
    """Place a plate on a noisy background; return (scene_bgr, gt_box)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    cw, ch = canvas
    scene = np.random.randint(60, 160, (ch, cw, 3), dtype=np.uint8)
    scene = cv2.GaussianBlur(scene, (7, 7), 0)
    # add a few distractor rectangles
    for _ in range(4):
        x1, y1 = random.randint(0, cw - 60), random.randint(0, ch - 40)
        cv2.rectangle(scene, (x1, y1), (x1 + random.randint(30, 90),
                      y1 + random.randint(20, 60)),
                      tuple(int(v) for v in np.random.randint(0, 255, 3)), -1)

    ph, pw = plate_img.shape[:2]
    scale = random.uniform(0.7, 1.0)
    pw, ph = int(pw * scale), int(ph * scale)
    plate_img = cv2.resize(plate_img, (pw, ph))
    x = random.randint(10, cw - pw - 10)
    y = random.randint(10, ch - ph - 10)
    scene[y:y + ph, x:x + pw] = plate_img
    return scene, (x, y, pw, ph)


def random_plate_text() -> tuple[str, str, str]:
    """Return (number, expiry_str, full_truth) for a random Indonesian plate."""
    region = random.choice(["B", "D", "DK", "AB", "L", "N"])
    nums = "".join(random.choice("0123456789") for _ in range(4))
    suffix = "".join(random.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
    number = f"{region}{nums}{suffix}"
    month = random.randint(1, 12)
    year = random.randint(24, 29)
    expiry = f"{month:02d}.{year:02d}"
    return number, expiry, number
