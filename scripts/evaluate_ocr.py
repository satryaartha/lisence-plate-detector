"""
scripts/evaluate_ocr.py  —  Comprehensive OCR Evaluation

Metrics yang dihasilkan:
  • Character Accuracy (CA)
  • Word Accuracy (WA) — seluruh plat terbaca benar
  • Precision, Recall, F1-score per kelas (macro / weighted)
  • Confusion matrix (visual ASCII + saved PNG)
  • Per-class breakdown (TP, FP, FN, precision, recall, F1)
  • Word Error Rate (WER)
  • Character Error Rate (CER)
  • Top-N most confused pairs

Run:
  python scripts/evaluate_ocr.py \
      --plates-dir data/PlateTrainingDataset \
      --char-dir   data/raw/DatasetCharacter   (opsional, untuk eval karakter bersih)
      --mode       plates                      (plates | chars | both)

Output:
  outputs/evaluation_report.txt   — laporan teks lengkap
  outputs/confusion_matrix.png    — heatmap confusion matrix
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime

import cv2
import joblib
import numpy as np
from skimage.feature import hog

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OCR_CLASSES, OCR_SVM_PATH

# --------------------------------------------------------------------------- #
# Feature extraction (must match training)
# --------------------------------------------------------------------------- #
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm="L2-Hys", transform_sqrt=True)


def normalize_char(g, out_size=40, margin_ratio=0.18):
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:
        b = cv2.bitwise_not(b)
    ys, xs = np.where(b > 0)
    if xs.size == 0:
        return cv2.resize(b, (out_size, out_size))
    crop = b[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    ch, cw = crop.shape
    inner = int(out_size * (1 - 2 * margin_ratio))
    scale = inner / max(ch, cw)
    nh, nw = max(1, int(ch * scale)), max(1, int(cw * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), np.uint8)
    oy, ox = (out_size - nh) // 2, (out_size - nw) // 2
    canvas[oy:oy + nh, ox:ox + nw] = crop
    return canvas


def char_hog(g):
    g = g if g.ndim == 2 else cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    r = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA)
    return hog(r, feature_vector=True, **HOG_PARAMS).astype(np.float32)


# --------------------------------------------------------------------------- #
# Segmentation (same as app)
# --------------------------------------------------------------------------- #
def segment_cropped_plate(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    H, W = g.shape
    s = 200.0 / H
    g = cv2.resize(g, (max(1, int(W * s)), 200))
    BH, BW = g.shape
    mx, my = int(BW * 0.03), int(BH * 0.05)
    g = g[my:BH - my, mx:BW - mx]
    BH, BW = g.shape
    g = cv2.bilateralFilter(cv2.createCLAHE(2.0, (8, 8)).apply(g), 5, 30, 30)

    def _bw_variants():
        _, bo = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ba = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, -8)
        ba = cv2.morphologyEx(ba, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        ba = cv2.medianBlur(ba, 3)
        out = []
        for bw in [bo, cv2.bitwise_not(bo), ba, cv2.bitwise_not(ba)]:
            out.append(cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)))
        return out

    def _extract(bw, band, min_hr=0.25):
        r0, r1 = band
        strip = bw[r0:r1, :]; SH, SW = strip.shape
        n, _l, st2, ct = cv2.connectedComponentsWithStats(strip, 8)
        blobs = []
        for i in range(1, n):
            x2,y2,ww,hh,a = int(st2[i][0]),int(st2[i][1]),int(st2[i][2]),int(st2[i][3]),int(st2[i][4])
            if hh == 0: continue
            hr, ar = hh / float(SH), ww / float(hh)
            if hr < min_hr or hr > 0.99: continue
            if ar < 0.08 or ar > 1.6: continue
            if a < 0.04 * ww * hh: continue
            blobs.append([x2,y2,ww,hh,float(ct[i][0]),float(ct[i][1]),
                          strip[y2:y2+hh, x2:x2+ww]])
        if not blobs: return []
        blobs.sort(key=lambda c: c[4]); best = []
        for seed in blobs:
            run, cur = [seed], seed
            for o in blobs:
                if o[4] <= cur[4]: continue
                med = float(np.median([g2[3] for g2 in run]))
                gap = o[0] - (cur[0] + cur[2])
                if (abs(o[5]-cur[5]) <= 0.4*cur[3] and 0.5*med <= o[3] <= 1.6*med
                        and -0.3*cur[3] <= gap <= 2.5*cur[3]):
                    run.append(o); cur = o
            if len(run) > len(best): best = run
        return [c[6] for c in sorted(best, key=lambda c: c[0])]

    best_num = []; best_num_sc = -999
    for bw in _bw_variants():
        hproj = np.convolve(bw.sum(axis=1).astype(float), np.ones(3) / 3, mode='same')
        mx_h = float(hproj.max())
        if mx_h == 0: continue
        thr1 = mx_h * 0.28
        bands, s0 = [], None
        for i in range(BH):
            v = float(hproj[i])
            if v > thr1 and s0 is None: s0 = i
            elif v <= thr1 and s0 is not None:
                if i - s0 > BH * 0.10: bands.append((s0, i))
                s0 = None
        if s0: bands.append((s0, BH))
        if not bands: continue
        nb = max(bands, key=lambda b: b[1] - b[0])
        num = _extract(bw, nb, min_hr=0.30)
        num_sc = len(num) - abs(len(num) - 7) * 2
        if num_sc > best_num_sc:
            best_num_sc = num_sc; best_num = num
    return best_num


def label_from_name(path):
    stem = re.sub(r"[^A-Z0-9]", "", os.path.splitext(os.path.basename(path))[0].upper())
    m = re.match(r"^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,4}", stem)
    return m.group(0) if m else ""


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: list[str], y_pred: list[str], classes: list[str]):
    n = len(y_true)
    assert n == len(y_pred), "length mismatch"

    # Per-class TP/FP/FN
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fn[t] += 1
            fp[p] += 1

    per_class = {}
    for c in classes:
        p_val = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r_val = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1    = 2 * p_val * r_val / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        support = tp[c] + fn[c]
        per_class[c] = dict(precision=p_val, recall=r_val, f1=f1,
                            tp=tp[c], fp=fp[c], fn=fn[c], support=support)

    # Aggregates
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n
    present = [c for c in classes if per_class[c]["support"] > 0]
    macro_p  = np.mean([per_class[c]["precision"] for c in present])
    macro_r  = np.mean([per_class[c]["recall"]    for c in present])
    macro_f1 = np.mean([per_class[c]["f1"]        for c in present])

    total_sup = sum(per_class[c]["support"] for c in present)
    weighted_p  = sum(per_class[c]["precision"] * per_class[c]["support"] for c in present) / total_sup
    weighted_r  = sum(per_class[c]["recall"]    * per_class[c]["support"] for c in present) / total_sup
    weighted_f1 = sum(per_class[c]["f1"]        * per_class[c]["support"] for c in present) / total_sup

    # Confusion matrix
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    # Top confused pairs
    confused = Counter()
    for t, p in zip(y_true, y_pred):
        if t != p:
            confused[(t, p)] += 1

    return dict(
        accuracy=accuracy, n=n,
        macro=dict(precision=macro_p, recall=macro_r, f1=macro_f1),
        weighted=dict(precision=weighted_p, recall=weighted_r, f1=weighted_f1),
        per_class=per_class, confusion_matrix=cm,
        top_confused=confused.most_common(15),
        classes=classes,
    )


def word_accuracy(words_true, words_pred):
    correct = sum(t == p for t, p in zip(words_true, words_pred))
    return correct / len(words_true) if words_true else 0.0


def character_error_rate(words_true, words_pred):
    """Levenshtein-based CER."""
    def edit(a, b):
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
                prev = temp
        return dp[n]

    total_edit = sum(edit(t, p) for t, p in zip(words_true, words_pred))
    total_len  = sum(len(t) for t in words_true)
    return total_edit / total_len if total_len > 0 else 0.0


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def print_confusion_matrix(cm, classes, max_show=20):
    present = [c for c in classes if cm[classes.index(c), :].sum() > 0
               or cm[:, classes.index(c)].sum() > 0][:max_show]
    idx = {c: classes.index(c) for c in present}
    sub = cm[np.ix_([idx[c] for c in present], [idx[c] for c in present])]

    header = "    " + " ".join(f"{c:>3}" for c in present)
    lines  = [header, "    " + "-" * (4 * len(present))]
    for i, c in enumerate(present):
        row = f"{c:>3} " + " ".join(f"{sub[i,j]:>3}" for j in range(len(present)))
        lines.append(row)
    return "\n".join(lines)


def save_confusion_matrix_png(cm, classes, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        present = [c for c in classes if cm[classes.index(c), :].sum() > 0]
        idx_map = [classes.index(c) for c in present]
        sub = cm[np.ix_(idx_map, idx_map)]

        fig, ax = plt.subplots(figsize=(max(8, len(present) * 0.55),
                                        max(7, len(present) * 0.5)))
        norm_sub = sub.astype(float)
        row_sums = norm_sub.sum(axis=1, keepdims=True)
        norm_sub = np.where(row_sums > 0, norm_sub / row_sums, 0)

        im = ax.imshow(norm_sub, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present, fontsize=8)
        ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.set_title("Confusion Matrix (row-normalised)", fontsize=13, fontweight="bold")

        for i in range(len(present)):
            for j in range(len(present)):
                if sub[i, j] > 0:
                    color = "white" if norm_sub[i, j] > 0.5 else "black"
                    ax.text(j, i, str(sub[i, j]), ha="center", va="center",
                            fontsize=7, color=color)

        plt.colorbar(im, ax=ax, label="Row-normalised recall")
        plt.tight_layout()
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        return True
    except ImportError:
        return False


def build_report(metrics, word_acc, cer, n_plates_total, n_plates_aligned,
                 mode, n_plate_chars=0):
    m = metrics
    lines = []
    sep = "=" * 65

    lines += [sep,
              "  OCR EVALUATION REPORT",
              f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
              f"  Mode      : {mode}",
              sep, ""]

    lines += ["── DATASET ─────────────────────────────────────────────────",
              f"  Plates total       : {n_plates_total}",
              f"  Plates aligned     : {n_plates_aligned}  "
              f"({n_plates_aligned/max(1,n_plates_total)*100:.1f}%  — segmentation rate)",
              f"  Characters (total) : {m['n']}",
              f"  Characters (plates): {n_plate_chars}",
              ""]

    lines += ["── CHARACTER-LEVEL METRICS (semua sumber) ──────────────────",
              f"  Accuracy           : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)",
              "",
              f"  Macro   Precision  : {m['macro']['precision']:.4f}",
              f"          Recall     : {m['macro']['recall']:.4f}",
              f"          F1-score   : {m['macro']['f1']:.4f}",
              "",
              f"  Weighted Precision : {m['weighted']['precision']:.4f}",
              f"           Recall    : {m['weighted']['recall']:.4f}",
              f"           F1-score  : {m['weighted']['f1']:.4f}",
              ""]

    if word_acc is not None:
        lines += ["── WORD / PLATE-LEVEL METRICS (dari plat teralign) ─────────",
                  f"  Word Accuracy      : {word_acc:.4f}  ({word_acc*100:.2f}%)",
                  f"  Word Error Rate    : {1-word_acc:.4f}  ({(1-word_acc)*100:.2f}%)",
                  f"  CER (Levenshtein)  : {cer:.4f}  ({cer*100:.2f}%)",
                  f"  Basis              : {n_plates_aligned} plat teralign",
                  "  Note: word accuracy = seluruh nomor plat terbaca benar persis",
                  ""]

    lines += ["── PER-CLASS BREAKDOWN ─────────────────────────────────────",
              f"  {'Class':>5}  {'Support':>7}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}"]
    lines.append("  " + "-" * 57)
    for c in m["classes"]:
        pc = m["per_class"][c]
        if pc["support"] == 0:
            continue
        lines.append(
            f"  {c:>5}  {pc['support']:>7}  {pc['precision']:>9.3f}  "
            f"{pc['recall']:>6.3f}  {pc['f1']:>6.3f}  "
            f"{pc['tp']:>4}  {pc['fp']:>4}  {pc['fn']:>4}"
        )
    lines.append("")

    lines += ["── TOP CONFUSED PAIRS ──────────────────────────────────────",
              f"  {'True':>5}  →  {'Pred':>5}  {'Count':>5}  Keterangan"]
    lines.append("  " + "-" * 45)
    notes = {
        ("1","I"): "garis tegak mirip",
        ("I","1"): "garis tegak mirip",
        ("2","Z"): "ujung diagonal mirip",
        ("Z","2"): "ujung diagonal mirip",
        ("C","G"): "lengkungan mirip",
        ("D","0"): "bulat mirip",
        ("O","0"): "bulat vs nol",
        ("8","B"): "kurva ganda mirip",
        ("B","8"): "kurva ganda mirip",
        ("7","1"): "garis lurus mirip",
        ("N","M"): "diagonal mirip",
    }
    for (t, p), cnt in m["top_confused"]:
        note = notes.get((t, p), "")
        lines.append(f"  {t:>5}  →  {p:>5}  {cnt:>5}  {note}")
    lines.append("")

    lines += ["── CONFUSION MATRIX (ASCII, aktif saja) ────────────────────",
              print_confusion_matrix(m["confusion_matrix"], m["classes"]),
              ""]

    lines.append(sep)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #
EXT = (".png", ".jpg", ".jpeg", ".bmp")


def load_plates(plates_dir, model):
    files = [f for f in os.listdir(plates_dir) if f.lower().endswith(EXT)]
    chars_true, chars_pred = [], []
    words_true, words_pred = [], []
    n_total = len(files); n_aligned = 0

    for fname in sorted(files):
        label = label_from_name(fname)
        if not (5 <= len(label) <= 9):
            continue
        img = cv2.imread(os.path.join(plates_dir, fname))
        if img is None:
            continue
        glyphs = segment_cropped_plate(img)
        if len(glyphs) != len(label):
            continue
        n_aligned += 1
        preds = [str(model.predict(char_hog(normalize_char(g)).reshape(1, -1))[0])
                 for g in glyphs]
        pred_word = "".join(preds)
        chars_true += list(label); chars_pred += preds
        words_true.append(label); words_pred.append(pred_word)

    return chars_true, chars_pred, words_true, words_pred, n_total, n_aligned


def load_char_dir(char_dir, model, max_per_class=300):
    classes = set(OCR_CLASSES)
    chars_true, chars_pred = [], []
    seen = defaultdict(int)
    for root, _d, files in os.walk(char_dir):
        lab = os.path.basename(root).strip().upper()
        if lab not in classes:
            continue
        for fn in files:
            if not fn.lower().endswith(EXT):
                continue
            if seen[lab] >= max_per_class:
                continue
            im = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            pred = str(model.predict(char_hog(normalize_char(im)).reshape(1, -1))[0])
            chars_true.append(lab); chars_pred.append(pred)
            seen[lab] += 1
    return chars_true, chars_pred


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate HOG+SVM OCR model on plate photos and/or character images."
    )
    ap.add_argument("--plates-dir", default="data/PlateTrainingDataset",
                    help="Folder of cropped plate images (filename = plate number)")
    ap.add_argument("--char-dir", default="",
                    help="DatasetCharacter folder (optional, for clean-char evaluation)")
    ap.add_argument("--mode", default="plates",
                    choices=["plates", "chars", "both"],
                    help="Evaluation mode")
    ap.add_argument("--model", default=OCR_SVM_PATH)
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading model from {args.model} ...")
    model = joblib.load(args.model)
    classes = sorted(set(OCR_CLASSES))

    chars_true, chars_pred = [], []
    words_true, words_pred = [], []
    n_total = n_aligned = 0

    if args.mode in ("plates", "both") and os.path.isdir(args.plates_dir):
        print(f"Evaluating on plate images in {args.plates_dir} ...")
        ct, cp, wt, wp, nt, na = load_plates(args.plates_dir, model)
        chars_true += ct; chars_pred += cp
        words_true += wt; words_pred += wp
        n_total += nt; n_aligned += na
        print(f"  Plates: {na}/{nt} aligned → {len(ct)} characters")

    if args.mode in ("chars", "both") and args.char_dir and os.path.isdir(args.char_dir):
        print(f"Evaluating on DatasetCharacter in {args.char_dir} ...")
        ct, cp = load_char_dir(args.char_dir, model)
        chars_true += ct; chars_pred += cp
        print(f"  Char images: {len(ct)} samples")

    if not chars_true:
        raise SystemExit("No evaluation data found. Check --plates-dir / --char-dir.")

    print(f"\nTotal characters to evaluate: {len(chars_true)}")
    print("Computing metrics ...")

    metrics  = compute_metrics(chars_true, chars_pred, classes)
    # Word-level metrics hanya dari plat (bukan char images)
    word_acc = word_accuracy(words_true, words_pred) if words_true else None
    cer      = character_error_rate(words_true, words_pred) if words_true else 0.0
    n_plate_chars = len([ct for ct, cp in zip(chars_true[:sum(len(l) for l in words_true)],
                                               chars_pred[:sum(len(l) for l in words_true)])
                         if True]) if words_true else 0
    # simpler: plate chars = sum of label lengths
    n_plate_chars = sum(len(w) for w in words_true)

    report = build_report(metrics, word_acc, cer, n_total, n_aligned,
                          args.mode, n_plate_chars)
    print("\n" + report)

    report_path = os.path.join(args.out_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved → {report_path}")

    cm_path = os.path.join(args.out_dir, "confusion_matrix.png")
    ok = save_confusion_matrix_png(metrics["confusion_matrix"], classes, cm_path)
    if ok:
        print(f"Confusion matrix saved → {cm_path}")
    else:
        print("(matplotlib not available — skipping confusion matrix PNG)")

    # Also save raw metrics as JSON for programmatic use
    json_metrics = {
        "accuracy": metrics["accuracy"],
        "word_accuracy": word_acc,
        "cer": cer,
        "macro": metrics["macro"],
        "weighted": metrics["weighted"],
        "n_chars": metrics["n"],
        "n_plates_total": n_total,
        "n_plates_aligned": n_aligned,
        "top_confused": [{"true": t, "pred": p, "count": c}
                         for (t, p), c in metrics["top_confused"]],
    }
    json_path = os.path.join(args.out_dir, "evaluation_metrics.json")
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Metrics JSON saved → {json_path}")


if __name__ == "__main__":
    main()
