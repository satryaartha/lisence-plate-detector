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
# Look for the model next to this file (repo root) OR inside a models/ folder.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [os.path.join(_HERE, "ocr_svm.joblib"),
               os.path.join(_HERE, "models", "ocr_svm.joblib")]
MODEL_PATH = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
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
# Plate localisation — works on both black-white AND white-red plates,
# prefers runs in the lower half of the image (where plates usually are),
# and rejects small-text noise / objects in corners.
# --------------------------------------------------------------------------- #
def detect_plate(img_bgr):
    H, W = img_bgr.shape[:2]

    def _find_runs(bw):
        """Find the best character-run box in a binary image."""
        n, _l, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
        chars = []
        for i in range(1, n):
            x, y, ww, hh, _a = stats[i]
            if hh == 0:
                continue
            ar, hr = ww / float(hh), hh / float(H)
            if 0.025 <= hr <= 0.18 and 0.08 <= ar <= 1.3:
                chars.append((x, y, ww, hh, float(cent[i][0]), float(cent[i][1])))
        if len(chars) < 3:
            return None, -1
        chars.sort(key=lambda c: c[4])

        def run_from(seed):
            run, cur = [seed], seed
            for o in chars:
                if o[4] <= cur[4]:
                    continue
                med = float(np.median([g[3] for g in run]))
                gap = o[0] - (cur[0] + cur[2])
                if (abs(o[5] - cur[5]) <= 0.65 * cur[3]
                        and 0.6 * med <= o[3] <= 1.6 * med
                        and -0.4 * cur[3] <= gap <= 1.3 * cur[3]):
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
            avg_hr = avg_h / H
            if avg_hr < 0.03 or gw < 0.08 * W or not (1.5 <= ar <= 11.0):
                continue
            # prefer runs in the lower 2/3 of the image (plates are low)
            cy_norm = float(np.mean([g[5] for g in run])) / H
            position_bonus = max(0.0, cy_norm - 0.3) * 10.0
            score = len(run) + avg_hr * 40.0 + min(ar / 5.0, 1.0) + position_bonus
            if score > bsc:
                best, bsc = (xs0, ys0, xs1, ys1, avg_h), score
        return best, bsc

    # Try both polarities: black-white plate (bw1) and white-red plate (bw2)
    gray = cv2.bilateralFilter(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 9, 50, 50)
    bs = max(31, (min(H, W) // 12) | 1)
    bw1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY, bs, -10)
    bw2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, bs, -10)
    for bw in (bw1, bw2):
        bw[:] = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    best1, sc1 = _find_runs(bw1)
    best2, sc2 = _find_runs(bw2)
    best = best1 if sc1 >= sc2 else best2
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
def _deskew(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                            minLineLength=g.shape[1] // 4, maxLineGap=20)
    if lines is None:
        return crop_bgr
    angles = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(a) < 30:
            angles.append(a)
    if not angles:
        return crop_bgr
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return crop_bgr
    H, W = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    return cv2.warpAffine(crop_bgr, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


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
            med = float(np.median([g[3] for g in run]))
            gap = o[0] - (cur[0] + cur[2])
            if (abs(o[5] - cur[5]) <= 0.55 * cur[3] and 0.55 * med <= o[3] <= 1.65 * med
                    and -0.4 * cur[3] <= gap <= 1.5 * cur[3]):
                run.append(o); cur = o
        if len(run) > len(best):
            best = run
    return best


def segment_plate(crop_bgr):
    crop = _deskew(crop_bgr)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    s = 280.0 / max(1, g.shape[0])
    g = cv2.resize(g, (max(1, int(g.shape[1] * s)), 280))
    g = cv2.bilateralFilter(cv2.createCLAHE(3.0, (8, 8)).apply(g), 9, 50, 50)
    bw1 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, -8)
    bw2 = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, -8)
    bw = bw1 if float(bw1.mean()) < float(bw2.mean()) else bw2
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    bw = cv2.medianBlur(bw, 3)
    H, W = bw.shape
    hproj = np.convolve(bw.sum(axis=1).astype(float), np.ones(9) / 9, mode='same')
    thr = float(hproj.max()) * 0.15
    bands, s0 = [], None
    for i in range(H):
        v = float(hproj[i])
        if v > thr and s0 is None:
            s0 = i
        elif v <= thr and s0 is not None:
            if i - s0 > H * 0.10:
                bands.append((s0, i))
            s0 = None
    if s0:
        bands.append((s0, H))
    rows_out = []
    for (r0, r1) in bands[:3]:  # up to 3 bands; we'll pick the right two
        strip = bw[r0:r1, :]; SH, SW = strip.shape
        n, _l, st, ct = cv2.connectedComponentsWithStats(strip, 8)
        blobs = []
        for i in range(1, n):
            x2, y2, ww, hh, a = int(st[i][0]),int(st[i][1]),int(st[i][2]),int(st[i][3]),int(st[i][4])
            cx, cy = float(ct[i][0]), float(ct[i][1])
            if hh == 0:
                continue
            if (0.15 <= hh / float(SH) <= 0.98 and 0.05 <= ww / float(hh) <= 1.3
                    and a >= 0.08 * ww * hh and ww < 0.25 * SW):
                blobs.append([x2, y2, ww, hh, cx, cy, strip[y2:y2 + hh, x2:x2 + ww]])
        run = _best_run(blobs)
        if run and len(run) >= 2:
            med_h = float(np.median([c[3] for c in run]))
            rows_out.append((r0, med_h, [c[6] for c in sorted(run, key=lambda c: c[0])]))

    if not rows_out:
        return [], []
    # number row = the band with the tallest median character height
    rows_out.sort(key=lambda r: r[1], reverse=True)
    num_row = rows_out[0]
    # expiry row = any other band that starts BELOW the number row
    exp_row = next((r for r in rows_out[1:] if r[0] > num_row[0]), None)
    top    = num_row[2]
    bottom = exp_row[2] if exp_row else []
    return top, bottom


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Plate Expiry Detection", page_icon="🚗")
st.title("🚗 Smart License Plate Expiration Detection")
st.caption("detect plate → segment → HOG+SVM OCR → parse MM.YY → EXPIRED / VALID")

if not os.path.exists(MODEL_PATH):
    st.error("Model file `ocr_svm.joblib` not found. Upload it to the repo "
             "(either at the root, next to streamlit_app.py, or in a models/ folder).")
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
