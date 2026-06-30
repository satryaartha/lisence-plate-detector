"""
scripts/train_ocr_improved.py

Improved OCR trainer combining:
1. Real characters from DatasetCharacter (full class coverage)
2. Real characters cut from your own plate photos (domain match)
3. Augmentation: small rotation, gaussian noise, blur, contrast jitter
   -> makes the model robust to worn/tilted/varying-quality plate chars

Run:
  python scripts/train_ocr_improved.py \
      --char-dir data/raw/DatasetCharacter \
      --plates-dir data/raw \
      --augment-factor 4
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

EXT = (".png", ".jpg", ".jpeg", ".bmp")
RNG  = np.random.default_rng(42)


# --------------------------------------------------------------------------- #
# Augmentation
# --------------------------------------------------------------------------- #
def augment(img: np.ndarray, n: int) -> list[np.ndarray]:
    """Return n augmented variants of a grayscale glyph image."""
    out = []
    for _ in range(n):
        g = img.copy().astype(np.float32)
        # 1. small rotation (-8..+8 deg)
        angle = float(RNG.uniform(-8, 8))
        h, w = g.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        g = cv2.warpAffine(g, M, (w, h), borderValue=0)
        # 2. contrast/brightness jitter
        alpha = float(RNG.uniform(0.75, 1.25))
        beta  = float(RNG.uniform(-20, 20))
        g = np.clip(g * alpha + beta, 0, 255)
        # 3. gaussian noise
        noise = RNG.normal(0, float(RNG.uniform(0, 15)), g.shape)
        g = np.clip(g + noise, 0, 255)
        # 4. occasional blur
        if RNG.random() < 0.4:
            ks = int(RNG.choice([3, 5]))
            g = cv2.GaussianBlur(g, (ks, ks), 0)
        out.append(g.astype(np.uint8))
    return out


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
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


def _detect_plate(img_bgr):
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
        if hh == 0: continue
        ar, hr = ww / float(hh), hh / float(H)
        if 0.025 <= hr <= 0.16 and 0.1 <= ar <= 1.2:
            chars.append((x, y, ww, hh, float(cent[i][0]), float(cent[i][1])))
    if len(chars) < 3: return None
    chars.sort(key=lambda c: c[4])
    def run_from(seed):
        run, cur = [seed], seed
        for o in chars:
            if o[4] <= cur[4]: continue
            med = float(np.median([g[3] for g in run]))
            gap = o[0] - (cur[0] + cur[2])
            if (abs(o[5]-cur[5]) <= 0.6*cur[3] and 0.65*med <= o[3] <= 1.5*med
                    and -0.35*cur[3] <= gap <= 1.2*cur[3]):
                run.append(o); cur = o
        return run
    best, bsc = None, -1
    for seed in chars:
        run = run_from(seed)
        if len(run) < 3: continue
        xs0=min(g[0] for g in run); ys0=min(g[1] for g in run)
        xs1=max(g[0]+g[2] for g in run); ys1=max(g[1]+g[3] for g in run)
        gw,gh = xs1-xs0, ys1-ys0; ar = gw/float(gh+1e-6)
        avg_h = float(np.mean([g[3] for g in run]))
        if avg_h/H < 0.035 or gw < 0.10*W or not (1.6 <= ar <= 10.0): continue
        score = len(run) + (avg_h/H)*40 + min(ar/5,1)
        if score > bsc: best, bsc = (xs0,ys0,xs1,ys1,avg_h), score
    if best is None: return None
    xs0,ys0,xs1,ys1,avg_h = best
    px = int(0.06*(xs1-xs0)); x=max(0,xs0-px); w=min(W-x,(xs1-xs0)+2*px)
    y=max(0,ys0-int(0.35*avg_h)); h=min(H-y,int((ys1-ys0)+1.4*avg_h))
    return (x,y,w,h)


def _deskew(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(g,(5,5),0), 50, 150)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=g.shape[1]//4,maxLineGap=20)
    if lines is None: return crop_bgr
    angles = [np.degrees(np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0]))
              for l in lines if abs(np.degrees(np.arctan2(l[0][3]-l[0][1],l[0][2]-l[0][0])))<30]
    if not angles: return crop_bgr
    angle = float(np.median(angles))
    if abs(angle) < 0.5: return crop_bgr
    H,W = crop_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((W/2,H/2), angle, 1.0)
    return cv2.warpAffine(crop_bgr, M, (W,H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _segment_top_row(crop_bgr):
    crop = _deskew(crop_bgr)
    g = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY) if crop.ndim==3 else crop
    s = 280.0/max(1,g.shape[0]); g = cv2.resize(g,(max(1,int(g.shape[1]*s)),280))
    g = cv2.bilateralFilter(cv2.createCLAHE(3.0,(8,8)).apply(g),9,50,50)
    bw1 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,-8)
    bw2 = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,-8)
    bw = bw1 if float(bw1.mean())<float(bw2.mean()) else bw2
    bw = cv2.morphologyEx(bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    bw = cv2.medianBlur(bw,3); H,W = bw.shape
    hproj = np.convolve(bw.sum(axis=1).astype(float),np.ones(9)/9,mode='same')
    thr = float(hproj.max())*0.15
    bands, s0 = [], None
    for i in range(H):
        v = float(hproj[i])
        if v>thr and s0 is None: s0=i
        elif v<=thr and s0 is not None:
            if i-s0>H*0.10: bands.append((s0,i)); s0=None
    if s0: bands.append((s0,H))
    if not bands: return []
    r0,r1 = bands[0]; strip=bw[r0:r1,:]; SH,SW=strip.shape
    n,_l,st,ct = cv2.connectedComponentsWithStats(strip,8)
    blobs = []
    for i in range(1,n):
        x2,y2,ww,hh,a = int(st[i][0]),int(st[i][1]),int(st[i][2]),int(st[i][3]),int(st[i][4])
        cx,cy = float(ct[i][0]),float(ct[i][1])
        if hh==0: continue
        if 0.15<=hh/float(SH)<=0.98 and 0.05<=ww/float(hh)<=1.3 and a>=0.08*ww*hh and ww<0.25*SW:
            blobs.append([x2,y2,ww,hh,cx,cy,strip[y2:y2+hh,x2:x2+ww]])
    if not blobs: return []
    blobs.sort(key=lambda c:c[4]); best=[]
    for seed in blobs:
        run=[seed]; cur=seed
        for o in blobs:
            if o[4]<=cur[4]: continue
            med=float(np.median([g2[3] for g2 in run])); gap=o[0]-(cur[0]+cur[2])
            if abs(o[5]-cur[5])<=0.55*cur[3] and 0.55*med<=o[3]<=1.65*med and -0.4*cur[3]<=gap<=1.5*cur[3]:
                run.append(o); cur=o
        if len(run)>len(best): best=run
    return [c[6] for c in sorted(best,key=lambda c:c[0])]


def label_from_name(path):
    stem = re.sub(r"[^A-Z0-9]","",os.path.splitext(os.path.basename(path))[0].upper())
    m = re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}", stem)
    return m.group(0) if m else stem


def load_from_plates(plates_dir, limit):
    files = []
    for ext in ("jpg","jpeg","png","bmp"):
        files += glob.glob(os.path.join(plates_dir,f"**/*.{ext}"),recursive=True)
    files = sorted(f for f in files if "character" not in f.lower())[:limit]
    images, labels, used = [], [], 0
    for p in files:
        label = label_from_name(p)
        if not (5 <= len(label) <= 9): continue
        img = cv2.imread(p)
        if img is None: continue
        box = _detect_plate(img)
        if box is None: continue
        x,y,w,h = box
        glyphs = _segment_top_row(img[y:y+h,x:x+w])
        if len(glyphs) == len(label):
            for gimg,ch in zip(glyphs, label):
                if ch in OCR_CLASSES:
                    images.append(gimg); labels.append(ch)
            used += 1
    print(f"  Plate photos aligned: {used}/{len(files)} -> {len(images)} real chars")
    return images, labels


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char-dir",   default="data/raw/DatasetCharacter")
    ap.add_argument("--plates-dir", default="data/raw")
    ap.add_argument("--limit",      type=int, default=2000)
    ap.add_argument("--max-per-class", type=int, default=500)
    ap.add_argument("--augment-factor", type=int, default=4,
                    help="How many augmented copies per real plate character")
    args = ap.parse_args()

    images, labels = [], []

    if args.char_dir and os.path.isdir(args.char_dir):
        ci, cl = load_char_dir(args.char_dir, args.max_per_class)
        images += ci; labels += cl

    if args.plates_dir and os.path.isdir(args.plates_dir):
        print("Collecting characters from plate photos ...")
        pi, pl = load_from_plates(args.plates_dir, args.limit)
        # augment plate chars (they're domain-matched, worth multiplying)
        aug_i, aug_l = [], []
        for im, lb in zip(pi, pl):
            aug_i += augment(im, args.augment_factor)
            aug_l += [lb] * args.augment_factor
        print(f"  After augmentation: {len(aug_i)} extra chars")
        images += pi + aug_i; labels += pl + aug_l

    labels = np.array(labels)
    if len(images) < 50:
        raise SystemExit("Too few characters. Check --char-dir / --plates-dir.")
    counts = {c: int(np.sum(labels==c)) for c in sorted(set(labels))}
    print(f"\nTotal: {len(images)} chars, {len(counts)} classes")
    print("per-class:", counts)

    idx = np.arange(len(images))
    tr, te = train_test_split(idx, test_size=0.2, stratify=labels, random_state=RANDOM_STATE)
    print(f"\nTraining on {len(tr)} samples ...")
    ocr = OCRClassifier().fit([images[i] for i in tr], labels[tr])

    preds = [ocr.predict_char(images[i]) for i in te]
    truth = [labels[i] for i in te]
    acc   = character_accuracy(preds, list(truth))
    crit  = SUCCESS_CRITERIA["char_accuracy"]
    print(f"\nCharacter accuracy (held-out): {acc:.3f}  "
          f"(target >= {crit:.2f})  {'PASS' if acc >= crit else 'FAIL'}")
    ocr.save()
    print(f"Saved -> {OCR_SVM_PATH}")


if __name__ == "__main__":
    main()
