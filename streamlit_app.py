"""
streamlit_app.py  (self-contained, deploy-ready)

Smart License Plate Expiration Detection.
Everything needed at runtime is in this one file plus the trained model at
models/ocr_svm.joblib. No imports from config/ or src/, so deployment only needs:
  streamlit_app.py, requirements.txt, packages.txt, models/ocr_svm.joblib

Run locally:   streamlit run streamlit_app.py
"""
from __future__ import annotations

import os
import re
from datetime import date

import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.feature import hog

# --------------------------------------------------------------------------- #
# Model + feature config (must match training)
# --------------------------------------------------------------------------- #
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "ocr_svm.joblib")
CHAR_HOG_SIZE = (32, 32)
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                  block_norm="L2-Hys", transform_sqrt=True)
DIGITS = list("0123456789")


def normalize_char(glyph, out_size=40, margin_ratio=0.18):
    g = glyph
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:
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


def char_hog(glyph):
    g = glyph if glyph.ndim == 2 else cv2.cvtColor(glyph, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(g, (CHAR_HOG_SIZE[1], CHAR_HOG_SIZE[0]),
                         interpolation=cv2.INTER_AREA)
    return hog(resized, feature_vector=True, **HOG_PARAMS).astype(np.float32)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict_char(model, glyph):
    feats = char_hog(normalize_char(glyph)).reshape(1, -1)
    return str(model.predict(feats)[0])


def predict_digit(model, glyph):
    feats = char_hog(normalize_char(glyph)).reshape(1, -1)
    try:
        scores = model.decision_function(feats)[0]
        classes = list(model.classes_)
        best, bs = None, -1e18
        for d in DIGITS:
            if d in classes and scores[classes.index(d)] > bs:
                best, bs = d, scores[classes.index(d)]
        return best if best is not None else predict_char(model, glyph)
    except Exception:
        return predict_char(model, glyph)


# --------------------------------------------------------------------------- #
# Expiry parsing
# --------------------------------------------------------------------------- #
def check_expiry(text, today=None):
    today = today or date.today()
    digits = re.sub(r"[^0-9]", "", text)
    if len(digits) < 4:
        return None, None, None
    mm, yy = int(digits[-4:][:2]), int(digits[-4:][2:])
    if not (1 <= mm <= 12):
        return None, None, None
    year = 2000 + yy
    return mm, year, (year, mm) < (today.year, today.month)


# --------------------------------------------------------------------------- #
# Plate localisation (tilt-tolerant, watermark-rejecting)
# --------------------------------------------------------------------------- #
def detect_plate(img_bgr):
    H, W = img_bgr.shape[:2]
    gray = cv2.bilateralFilter(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 9, 50, 50)
    bs = max(31, (min(H, W) // 12) | 1)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, bs, -10)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    n, _l, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
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

    def run_from(seed):
        run, cur = [seed], seed
        for o in chars:
            if o[4] <= cur[4]:
                continue
            med = float(np.median([g[3] for g in run]))
            gap = o[0] - (cur[0] + cur[2])
            if (abs(o[5] - cur[5]) <= 0.6 * cur[3] and 0.65 * med <= o[3] <= 1.5 * med
                    and -0.35 * cur[3] <= gap <= 1.2 * cur[3]):
                run.append(o); cur = o
        return run

    best, bsc = None, -1
    for seed in chars:
        run = run_from(seed)
        if len(run) < 3:
            continue
        xs0 = min(g[0] for g in run); ys0 = min(g[1] for g in run)
        xs1 = max(g[0] + g[2] for g in run); ys1 = max(g[1] + g[3] for g in run)
        gw, gh = xs1 - xs0, ys1 - ys0
        ar = gw / float(gh + 1e-6)
        avg_h = float(np.mean([g[3] for g in run]))
        if avg_h / H < 0.035 or gw < 0.10 * W or not (1.6 <= ar <= 10.0):
            continue
        score = len(run) + (avg_h / H) * 40.0 + min(ar / 5.0, 1.0)
        if score > bsc:
            best, bsc = (xs0, ys0, xs1, ys1, avg_h), score
    if best is None:
        return None
    xs0, ys0, xs1, ys1, avg_h = best
    px = int(0.06 * (xs1 - xs0))
    x = max(0, xs0 - px); w = min(W - x, (xs1 - xs0) + 2 * px)
    y = max(0, ys0 - int(0.35 * avg_h)); h = min(H - y, int((ys1 - ys0) + 1.4 * avg_h))
    return (x, y, w, h)


# --------------------------------------------------------------------------- #
# Segmentation: number row = uniform aligned run; expiry = smaller run below.
# --------------------------------------------------------------------------- #
def _best_run(blobs):
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
    g = cv2.bilateralFilter(cv2.createCLAHE(3.0, (8, 8)).apply(g), 7, 40, 40)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, -9)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.medianBlur(bw, 3)
    H, W = bw.shape
    n, _l, st_, ct = cv2.connectedComponentsWithStats(bw, 8)
    blobs = []
    for i in range(1, n):
        x, y, ww, hh, a = st_[i]
        if hh == 0:
            continue
        ar, hr = ww / float(hh), hh / float(H)
        if 0.12 <= hr <= 0.95 and 0.08 <= ar <= 1.1 and a >= 0.15 * ww * hh and ww < 0.3 * W:
            blobs.append((x, y, ww, hh, ct[i][0], ct[i][1], bw[y:y + hh, x:x + ww]))
    if not blobs:
        return [], []
    top = _best_run(blobs)
    if not top:
        return [], []
    th = float(np.median([c[3] for c in top]))
    base = max(c[1] + c[3] for c in top)
    tcy = np.mean([c[5] for c in top])
    tx0 = min(c[0] for c in top); tx1 = max(c[0] + c[2] for c in top)
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

if not os.path.exists(MODEL_PATH):
    st.error("Model not found at models/ocr_svm.joblib. Commit the trained model "
             "to the repo (git add -f models/ocr_svm.joblib).")
    st.stop()

model = load_model()
file = st.file_uploader("Upload a vehicle photo", type=["jpg", "jpeg", "png", "bmp"])
if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    box = detect_plate(img)
    if box is None:
        st.warning("No plate could be localised in this photo.")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Input")
        st.stop()
    x, y, w, h = box
    crop = img[y:y + h, x:x + w]

    c1, c2 = st.columns(2)
    with c1:
        vis = img.copy()
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Detected plate")
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

    number = "".join(predict_char(model, g) for g in top)
    expiry_text = "".join(predict_digit(model, g) for g in bottom)

    st.subheader(f"Plate number: `{number or '—'}`")
    mm, yy, expired = check_expiry(expiry_text)
    st.write(f"Expiry text read: `{expiry_text or '—'}`")
    if expired is None:
        st.info("Expiry could not be parsed (no clear MM.YY).")
    elif expired:
        st.error(f"❌ EXPIRED — valid until {mm:02d}/{yy}")
    else:
        st.success(f"✅ VALID — valid until {mm:02d}/{yy}")
