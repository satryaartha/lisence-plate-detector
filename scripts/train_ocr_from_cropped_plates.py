"""
scripts/train_ocr_from_cropped_plates.py

Train the OCR SVM using:
  1. PlateTrainingDataset/  — already-cropped plate images (filename = label)
     Segments characters using vertical projection (optimized for close-up plates)
  2. data/raw/DatasetCharacter/  — individual character images (full class coverage)
  3. Augmentation (rotation, noise, blur, contrast) on real plate characters

Run:
  python scripts/train_ocr_from_cropped_plates.py \
      --plates-dir data/PlateTrainingDataset \
      --char-dir data/raw/DatasetCharacter \
      --augment-factor 5
"""
from __future__ import annotations
import argparse, glob, os, re, sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OCR_CLASSES, OCR_SVM_PATH, RANDOM_STATE, SUCCESS_CRITERIA
from src.evaluation import character_accuracy
from src.ocr import OCRClassifier

EXT  = (".png", ".jpg", ".jpeg", ".bmp")
RNG  = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def augment(img: np.ndarray, n: int) -> list[np.ndarray]:
    out = []
    for _ in range(n):
        g = img.copy().astype(np.float32)
        angle = float(RNG.uniform(-8, 8))
        h, w = g.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (w, h), borderValue=0)
        alpha = float(RNG.uniform(0.75, 1.25))
        beta  = float(RNG.uniform(-20, 20))
        g = np.clip(g * alpha + beta, 0, 255)
        noise = RNG.normal(0, float(RNG.uniform(0, 15)), g.shape)
        g = np.clip(g + noise, 0, 255)
        if RNG.random() < 0.4:
            ks = int(RNG.choice([3, 5]))
            g = cv2.GaussianBlur(g, (ks, ks), 0)
        out.append(g.astype(np.uint8))
    return out


# --------------------------------------------------------------------------- #
# Segmentation for already-cropped plate images
# --------------------------------------------------------------------------- #
def _binarize(gray):
    """Try both polarities, pick the one that gives more dark background."""
    bw1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 35, -10)
    bw2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, -10)
    bw1 = cv2.morphologyEx(bw1, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    bw2 = cv2.morphologyEx(bw2, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    for bw in (bw1, bw2):
        bw[:] = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        bw[:] = cv2.medianBlur(bw, 3)
    # white-on-black: lower mean
    return bw1 if float(bw1.mean()) < float(bw2.mean()) else bw2


def segment_cropped_plate(img_bgr: np.ndarray) -> list[np.ndarray]:
    """
    Segment characters from an already-cropped plate image.
    Tries multiple binarization strategies and picks best result.
    Returns a list of grayscale glyph images, left-to-right.
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    H, W = g.shape
    s = 200.0 / H
    g = cv2.resize(g, (max(1, int(W * s)), 200))
    BH, BW = g.shape
    # light border trim — don't trim too much or chars at edge get cut
    mx, my = int(BW * 0.03), int(BH * 0.05)
    g = g[my:BH - my, mx:BW - mx]
    BH, BW = g.shape
    g = cv2.bilateralFilter(cv2.createCLAHE(2.0, (8, 8)).apply(g), 5, 30, 30)

    def _try_segment(bw):
        """Run hproj+CC on a binary image, return list of char glyphs."""
        bw2 = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # horizontal projection → find tallest row band
        hproj = np.convolve(bw2.sum(axis=1).astype(float), np.ones(3) / 3, mode='same')
        thr = float(hproj.max()) * 0.30
        bands, s0 = [], None
        for i in range(BH):
            v = float(hproj[i])
            if v > thr and s0 is None: s0 = i
            elif v <= thr and s0 is not None:
                if i - s0 > BH * 0.12: bands.append((s0, i))
                s0 = None
        if s0: bands.append((s0, BH))
        if not bands: return []
        r0, r1 = max(bands, key=lambda b: b[1] - b[0])
        r0 = max(0, r0 - 2); r1 = min(BH, r1 + 2)
        strip = bw2[r0:r1, :]; SH, SW = strip.shape
        # CC in strip
        n, _l, st, ct = cv2.connectedComponentsWithStats(strip, 8)
        blobs = []
        for i in range(1, n):
            x2, y2, ww, hh, a = int(st[i][0]),int(st[i][1]),int(st[i][2]),int(st[i][3]),int(st[i][4])
            cx, cy = float(ct[i][0]), float(ct[i][1])
            if hh == 0: continue
            hr, ar = hh / float(SH), ww / float(hh)
            if hr < 0.35 or hr > 0.99: continue
            if ar < 0.08 or ar > 1.5: continue
            if a < 0.04 * ww * hh: continue
            blobs.append([x2, y2, ww, hh, cx, cy, strip[y2:y2 + hh, x2:x2 + ww]])
        if not blobs: return []
        blobs.sort(key=lambda c: c[4])
        best = []
        for seed in blobs:
            run, cur = [seed], seed
            for o in blobs:
                if o[4] <= cur[4]: continue
                med = float(np.median([g2[3] for g2 in run]))
                gap = o[0] - (cur[0] + cur[2])
                if (abs(o[5] - cur[5]) <= 0.4 * cur[3] and 0.5 * med <= o[3] <= 1.6 * med
                        and -0.3 * cur[3] <= gap <= 2.5 * cur[3]):
                    run.append(o); cur = o
            if len(run) > len(best): best = run
        return [c[6] for c in sorted(best, key=lambda c: c[0])]

    # Try multiple binarization strategies, pick the one with most chars (5-9)
    candidates = []
    # 1. Otsu
    _, bw_o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_oi = cv2.bitwise_not(bw_o)
    for bw in (bw_o, bw_oi):
        r = _try_segment(bw)
        if r: candidates.append(r)
    # 2. Adaptive Gaussian both polarities
    for inv in (False, True):
        mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, 31, -8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        bw = cv2.medianBlur(bw, 3)
        r = _try_segment(bw)
        if r: candidates.append(r)
    if not candidates: return []
    # pick result closest to typical plate char count (7-8), prefer more chars
    def score(c): return -abs(len(c) - 7.5)
    return max(candidates, key=score)


def label_from_name(path: str) -> str:
    stem = re.sub(r"[^A-Z0-9]", "", os.path.splitext(os.path.basename(path))[0].upper())
    m = re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,4}", stem)
    return m.group(0) if m else ""


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
def load_from_cropped_plates(plates_dir: str, augment_factor: int):
    files = [os.path.join(plates_dir, f)
             for f in os.listdir(plates_dir)
             if f.lower().endswith(EXT)]
    images, labels, used, skipped = [], [], 0, 0
    for p in sorted(files):
        label = label_from_name(p)
        if not (5 <= len(label) <= 9):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        glyphs = segment_cropped_plate(img)
        if len(glyphs) == len(label):
            for glyph, ch in zip(glyphs, label):
                if ch in OCR_CLASSES:
                    images.append(glyph)
                    labels.append(ch)
                    # augment each real char
                    for aug in augment(glyph, augment_factor):
                        images.append(aug)
                        labels.append(ch)
            used += 1
        else:
            skipped += 1
    print(f"  Cropped plates aligned: {used}/{used+skipped} "
          f"-> {len(images)} chars (incl. {augment_factor}x augment)")
    return images, labels


def load_char_dir(char_dir: str, max_per_class: int):
    classes = set(OCR_CLASSES)
    images, labels, seen = [], [], {}
    for root, _d, files in os.walk(char_dir):
        lab = os.path.basename(root).strip().upper()
        if lab not in classes:
            continue
        for fn in files:
            if not fn.lower().endswith(EXT):
                continue
            if seen.get(lab, 0) >= max_per_class:
                continue
            im = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                images.append(im); labels.append(lab)
                seen[lab] = seen.get(lab, 0) + 1
    print(f"  DatasetCharacter: {len(images)} chars across {len(seen)} classes")
    return images, labels


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plates-dir", default="data/PlateTrainingDataset",
                    help="Folder of cropped plate images (filename = label)")
    ap.add_argument("--char-dir",   default="data/raw/DatasetCharacter",
                    help="DatasetCharacter folder for full class coverage")
    ap.add_argument("--max-per-class", type=int, default=400)
    ap.add_argument("--augment-factor", type=int, default=5)
    args = ap.parse_args()

    images, labels = [], []

    if args.plates_dir and os.path.isdir(args.plates_dir):
        print("Loading characters from cropped plates ...")
        pi, pl = load_from_cropped_plates(args.plates_dir, args.augment_factor)
        images += pi; labels += pl
    else:
        print(f"WARNING: --plates-dir '{args.plates_dir}' not found, skipping.")

    if args.char_dir and os.path.isdir(args.char_dir):
        print("Loading DatasetCharacter ...")
        ci, cl = load_char_dir(args.char_dir, args.max_per_class)
        images += ci; labels += cl
    else:
        print(f"WARNING: --char-dir '{args.char_dir}' not found, skipping.")

    labels = np.array(labels)
    if len(images) < 50:
        raise SystemExit("Too few characters. Check --plates-dir / --char-dir.")

    counts = {c: int(np.sum(labels == c)) for c in sorted(set(labels))}
    print(f"\nTotal: {len(images)} chars, {len(counts)} classes")
    print("per-class:", counts)

    idx = np.arange(len(images))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels,
                              random_state=RANDOM_STATE)
    print(f"\nTraining on {len(tr)} samples ...")
    ocr = OCRClassifier().fit([images[i] for i in tr], labels[tr])

    preds = [ocr.predict_char(images[i]) for i in te]
    truth = [labels[i] for i in te]
    acc   = character_accuracy(preds, list(truth))
    crit  = SUCCESS_CRITERIA["char_accuracy"]
    print(f"\nCharacter accuracy (held-out): {acc:.3f}  "
          f"(target >= {crit:.2f})  {'PASS' if acc >= crit else 'FAIL'}")
    ocr.save()
    print(f"Saved → {OCR_SVM_PATH}")


if __name__ == "__main__":
    main()
