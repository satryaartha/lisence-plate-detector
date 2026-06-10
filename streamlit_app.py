"""
app/streamlit_app.py  (detection + recognition, improved)

Upload a full motorcycle photo; the app localises the plate, segments both rows
(number + MM.YY expiry), reads them, and shows an EXPIRED / VALID verdict.

Self-contained: the plate localiser and segmentation are inline so this file
does not depend on other updated modules. Only models/ocr_svm.joblib is needed.

Improvements in this version:
  * tilt-tolerant character-row detection (follows a sloped row)
  * rejects small-text noise such as the dataset watermark (size-weighted)
  * tighter crops via character-height uniformity
  * row-clustering segmentation that also captures the smaller expiry row

Run from the repo root:
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OCR_SVM_PATH
from src.expiry import check_expiry
from src.features import normalize_char
from src.ocr import OCRClassifier


# --------------------------------------------------------------------------- #
# Plate localisation (no training): find a tilted, uniform row of large chars.
# --------------------------------------------------------------------------- #
def detect_plate(img_bgr):
    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)
    bs = max(31, (min(H, W) // 12) | 1)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, bs, -10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    n, _lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    chars = []
    for i in range(1, n):
        x, y, ww, hh, _a = stats[i]
        if hh == 0:
            continue
        ar, hr = ww / float(hh), hh / float(H)
        if 0.025 <= hr <= 0.16 and 0.1 <= ar <= 1.2:
            chars.append((x, y, ww, hh, cent[i][0], cent[i][1]))
    if len(chars) < 3:
        return None
    chars.sort(key=lambda c: c[4])

    def make_run(seed):
        run, cur = [seed], seed
        for o in chars:
            if o[4] <= cur[4]:
                continue
            med = float(np.median([g[3] for g in run]))
            gap = o[0] - (cur[0] + cur[2])
            dy = abs(o[5] - cur[5])
            if (dy <= 0.6 * cur[3] and 0.65 * med <= o[3] <= 1.5 * med
                    and -0.35 * cur[3] <= gap <= 1.2 * cur[3]):
                run.append(o); cur = o
        return run

    best, best_score = None, -1
    for seed in chars:
        run = make_run(seed)
        if len(run) < 3:
            continue
        xs0 = min(g[0] for g in run); ys0 = min(g[1] for g in run)
        xs1 = max(g[0] + g[2] for g in run); ys1 = max(g[1] + g[3] for g in run)
        gw, gh = xs1 - xs0, ys1 - ys0
        ar = gw / float(gh + 1e-6)
        avg_h = float(np.mean([g[3] for g in run]))
        avg_hr = avg_h / float(H)
        if avg_hr < 0.035 or gw < 0.10 * W or not (1.6 <= ar <= 10.0):
            continue
        score = len(run) + avg_hr * 40.0 + min(ar / 5.0, 1.0)
        if score > best_score:
            best, best_score = (xs0, ys0, xs1, ys1, avg_h), score
    if best is None:
        return None
    xs0, ys0, xs1, ys1, avg_h = best
    px = int(0.06 * (xs1 - xs0))
    x = max(0, xs0 - px)
    w = min(W - x, (xs1 - xs0) + 2 * px)
    y = max(0, ys0 - int(0.35 * avg_h))
    h = min(H - y, int((ys1 - ys0) + 1.4 * avg_h))
    return (x, y, w, h)


# --------------------------------------------------------------------------- #
# Segmentation: CLAHE + adaptive threshold + row clustering (number + expiry).
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Segmentation: find the number row as a uniform aligned run of characters
# (rejects stray noise blobs), then the expiry row as the smaller run just
# below it and within the plate's width (avoids watermark/caption text).
# --------------------------------------------------------------------------- #
def _best_run(blobs):
    """Longest contiguous left-to-right run of uniform-height, aligned blobs."""
    if not blobs:
        return []
    blobs = sorted(blobs, key=lambda c: c[4])
    best = []
    for seed in blobs:
        run, cur = [seed], seed
        for o in blobs:
            if o[4] <= cur[4]:
                continue
            med = np.median([g[3] for g in run])
            gap = o[0] - (cur[0] + cur[2])
            if (abs(o[5] - cur[5]) <= 0.6 * cur[3] and 0.6 * med <= o[3] <= 1.6 * med
                    and -0.4 * cur[3] <= gap <= 1.4 * cur[3]):
                run.append(o); cur = o
        if len(run) > len(best):
            best = run
    return best


def segment_plate(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    s = 240.0 / max(1, g.shape[0])
    g = cv2.resize(g, (max(1, int(g.shape[1] * s)), 240))
    g = cv2.createCLAHE(3.0, (8, 8)).apply(g)
    g = cv2.bilateralFilter(g, 7, 40, 40)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, -9)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.medianBlur(bw, 3)
    H, W = bw.shape
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    blobs = []
    for i in range(1, n):
        x, y, ww, hh, a = stats[i]
        if hh == 0:
            continue
        ar, hr = ww / float(hh), hh / float(H)
        if 0.12 <= hr <= 0.95 and 0.08 <= ar <= 1.1 and a >= 0.15 * ww * hh and ww < 0.3 * W:
            blobs.append((x, y, ww, hh, cent[i][0], cent[i][1], bw[y:y + hh, x:x + ww]))
    if not blobs:
        return [], []

    top = _best_run(blobs)
    if not top:
        return [], []
    th = float(np.median([c[3] for c in top]))
    base = max(c[1] + c[3] for c in top)
    tcy = np.mean([c[5] for c in top])
    tx0 = min(c[0] for c in top); tx1 = max(c[0] + c[2] for c in top)
    # expiry: close below the number row, not larger than it, within plate width
    cand = [c for c in blobs
            if c[5] > tcy + 0.3 * th and (c[1] - base) < 1.8 * th and c[3] <= 1.5 * th
            and c[4] > tx0 - 0.15 * (tx1 - tx0) and c[4] < tx1 + 0.15 * (tx1 - tx0)]
    bottom = _best_run(cand)
    return [c[6] for c in top], [c[6] for c in bottom]


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Plate Expiry Detection", page_icon="🚗")
st.title("🚗 Smart License Plate Expiration Detection")
st.caption("detect plate → segment → HOG+SVM OCR → parse MM.YY → EXPIRED / VALID")

if not os.path.exists(OCR_SVM_PATH):
    st.error("OCR model not found. Train it first, e.g.\n\n"
             "`python scripts/train_ocr_real.py --char-dir data/raw/DatasetCharacter`")
    st.stop()


@st.cache_resource
def load_ocr():
    return OCRClassifier.load(OCR_SVM_PATH)


ocr = load_ocr()
mode = st.sidebar.radio("Input type",
                        ["Full photo (detect plate)", "Cropped plate (skip detection)"])

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    box = None
    if mode.startswith("Full"):
        box = detect_plate(img)
        if box is None:
            st.warning("No plate could be localised in this photo.")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input")
            st.stop()
        x, y, w, h = box
        crop = img[y:y + h, x:x + w]
    else:
        crop = img

    c1, c2 = st.columns(2)
    with c1:
        vis = img.copy()
        if box:
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                 caption="Detected plate" if box else "Input")
    with c2:
        st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), caption="Plate crop")

    top, bottom = segment_plate(crop)
    if not top and not bottom:
        st.warning("Plate found but characters could not be segmented.")
        st.stop()

    st.write(f"**Segmented characters** (number row: {len(top)}, expiry row: {len(bottom)})")
    allg = top + bottom
    cols = st.columns(min(len(allg), 12) or 1)
    for c, g in zip(cols, allg):
        c.image(normalize_char(g), width=38, clamp=True)

    number = "".join(ocr.predict_char(g) for g in top)
    digits = list("0123456789")
    if hasattr(ocr, "predict_char_restricted"):
        expiry_text = "".join(ocr.predict_char_restricted(g, digits) for g in bottom)
    else:
        expiry_text = "".join(ocr.predict_char(g) for g in bottom)

    st.subheader(f"Plate number: `{number or '—'}`")
    exp = check_expiry(expiry_text)
    st.write(f"Expiry text read: `{exp.raw_text or '—'}`")
    if exp.is_expired is None:
        st.info("Expiry could not be parsed (no clear MM.YY).")
    elif exp.is_expired:
        st.error(f"❌ EXPIRED — valid until {exp.month:02d}/{exp.year}")
    else:
        st.success(f"✅ VALID — valid until {exp.month:02d}/{exp.year}")