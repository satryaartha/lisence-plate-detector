"""
scripts/evaluate_dataset_char.py  —  Evaluasi Dataset 1: DatasetCharacter

Mengevaluasi model HOG+SVM pada DatasetCharacter (karakter individual A-Z, 0-9).
Menghasilkan:
  • Confusion matrix (PNG + ASCII)
  • Accuracy, Precision, Recall, F1-score (macro & weighted)
  • Per-class breakdown
  • Classification report lengkap
  • HOG feature visualization (opsional)
  • Train/test split evaluation (80/20)

Run:
  python scripts/evaluate_dataset_char.py \
      --char-dir data/raw/DatasetCharacter \
      --out-dir  outputs/eval_dataset1
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OCR_CLASSES, OCR_SVM_PATH

EXT = (".png", ".jpg", ".jpeg", ".bmp")

# --------------------------------------------------------------------------- #
# Feature extraction (harus sama dengan training)
# --------------------------------------------------------------------------- #
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    transform_sqrt=True,
)


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
# Load DatasetCharacter
# --------------------------------------------------------------------------- #
def load_dataset(char_dir: str, max_per_class: int = 0):
    """
    Muat semua gambar dari DatasetCharacter.
    Struktur: char_dir/<char>/<file>.jpg  atau  char_dir/DatasetCharacter/<char>/...
    """
    classes = set(OCR_CLASSES)
    images, labels = [], []
    per_class: dict[str, int] = {}

    for root, _dirs, files in os.walk(char_dir):
        lab = os.path.basename(root).strip().upper()
        if lab not in classes:
            continue
        for fn in sorted(files):
            if not fn.lower().endswith(EXT):
                continue
            if max_per_class and per_class.get(lab, 0) >= max_per_class:
                continue
            im = cv2.imread(os.path.join(root, fn), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                images.append(im)
                labels.append(lab)
                per_class[lab] = per_class.get(lab, 0) + 1

    print(f"  Loaded {len(images)} images across {len(per_class)} classes")
    print(f"  Per-class counts: {dict(sorted(per_class.items()))}")
    return images, np.array(labels)


# --------------------------------------------------------------------------- #
# HOG feature visualization
# --------------------------------------------------------------------------- #
def save_hog_visualization(images, labels, out_dir, n_per_class=2):
    """Simpan contoh gambar + HOG visualization per kelas."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.feature import hog as sk_hog

        classes = sorted(set(labels))
        fig, axes = plt.subplots(
            len(classes), n_per_class * 2,
            figsize=(n_per_class * 4, len(classes) * 1.2)
        )
        if len(classes) == 1:
            axes = [axes]

        by_class: dict[str, list] = defaultdict(list)
        for img, lab in zip(images, labels):
            by_class[lab].append(img)

        for row, cls in enumerate(classes):
            samples = by_class[cls][:n_per_class]
            for col, img in enumerate(samples):
                norm = normalize_char(img)
                resized = cv2.resize(norm, (32, 32))
                _, hog_img = sk_hog(resized, visualize=True,
                                    feature_vector=True, **HOG_PARAMS)
                ax_img = axes[row][col * 2]
                ax_hog = axes[row][col * 2 + 1]
                ax_img.imshow(norm, cmap="gray"); ax_img.axis("off")
                ax_hog.imshow(hog_img, cmap="gray"); ax_hog.axis("off")
                if col == 0:
                    ax_img.set_title(f"'{cls}'", fontsize=7, pad=1)
                    ax_hog.set_title("HOG", fontsize=7, pad=1)

        plt.suptitle("Sample Images & HOG Features per Class",
                     fontsize=11, fontweight="bold", y=1.002)
        plt.tight_layout()
        path = os.path.join(out_dir, "hog_visualization.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  HOG visualization saved → {path}")
    except Exception as e:
        print(f"  HOG visualization skipped: {e}")


# --------------------------------------------------------------------------- #
# Confusion matrix PNG
# --------------------------------------------------------------------------- #
def save_confusion_matrix(cm, classes, out_dir, title="Confusion Matrix"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Row-normalise for readability
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0)

        fig, axes = plt.subplots(1, 2, figsize=(22, 10))

        for ax, data, subtitle, fmt in [
            (axes[0], cm_norm, "Row-Normalised (Recall per kelas)", ".2f"),
            (axes[1], cm.astype(float), "Raw Count", "d"),
        ]:
            im = ax.imshow(data, cmap="Blues",
                           vmin=0, vmax=(1 if fmt == ".2f" else cm.max()))
            ax.set_xticks(range(len(classes)))
            ax.set_xticklabels(classes, fontsize=8, rotation=45)
            ax.set_yticks(range(len(classes)))
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual", fontsize=11)
            ax.set_title(subtitle, fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    val = data[i, j]
                    if val > 0:
                        color = "white" if (cm_norm[i, j] > 0.5) else "black"
                        txt = (f"{val:.2f}" if fmt == ".2f" else str(int(val)))
                        ax.text(j, i, txt, ha="center", va="center",
                                fontsize=6, color=color)

        plt.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(out_dir, "confusion_matrix_dataset1.png")
        plt.savefig(path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved → {path}")
    except Exception as e:
        print(f"  Confusion matrix PNG skipped: {e}")


# --------------------------------------------------------------------------- #
# ASCII confusion matrix
# --------------------------------------------------------------------------- #
def ascii_confusion_matrix(cm, classes, max_show=20):
    present = [c for c in classes
               if cm[classes.index(c), :].sum() > 0][:max_show]
    idx = {c: classes.index(c) for c in present}
    sub = cm[np.ix_([idx[c] for c in present], [idx[c] for c in present])]
    header = "    " + " ".join(f"{c:>3}" for c in present)
    lines = [header, "    " + "-" * (4 * len(present))]
    for i, c in enumerate(present):
        row = f"{c:>3} " + " ".join(f"{sub[i,j]:>3}" for j in range(len(present)))
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Evaluasi model HOG+SVM pada DatasetCharacter (Dataset 1)"
    )
    ap.add_argument("--char-dir", default="data/raw/DatasetCharacter")
    ap.add_argument("--model", default=OCR_SVM_PATH)
    ap.add_argument("--out-dir", default="outputs/eval_dataset1")
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="Batasi jumlah sample per kelas (0 = semua)")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Proporsi data test (default 0.2 = 20%%)")
    ap.add_argument("--split-eval", action="store_true",
                    help="Evaluasi dengan train/test split (bukan semua data)")
    ap.add_argument("--hog-viz", action="store_true",
                    help="Simpan visualisasi HOG per kelas")
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  EVALUASI DATASET 1: DatasetCharacter")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Load data
    print(f"[1/4] Loading DatasetCharacter dari: {args.char_dir}")
    images, labels = load_dataset(args.char_dir, args.max_per_class)

    if len(images) == 0:
        raise SystemExit(
            f"Tidak ada gambar ditemukan di {args.char_dir}.\n"
            "Cek path folder DatasetCharacter."
        )

    # HOG feature extraction
    print(f"\n[2/4] Ekstraksi fitur HOG ({len(images)} gambar)...")
    X = np.array([char_hog(normalize_char(img)) for img in images])
    y = labels
    classes = sorted(set(OCR_CLASSES))
    print(f"  Feature shape: {X.shape}  (setiap karakter = {X.shape[1]} fitur HOG)")

    # Load model
    print(f"\n[3/4] Loading model SVM dari: {args.model}")
    model = joblib.load(args.model)
    print(f"  Model classes: {len(model.classes_)}")

    # Evaluate
    print(f"\n[4/4] Evaluasi...")
    if args.split_eval:
        print(f"  Mode: Train/Test Split ({int((1-args.test_size)*100)}/{int(args.test_size*100)})")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=args.test_size,
            stratify=y, random_state=args.random_state
        )
        y_pred = model.predict(X_te)
        y_true = y_te
        n_test = len(y_te)
        print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}")
    else:
        print(f"  Mode: Evaluasi seluruh dataset ({len(X)} sampel)")
        y_pred = model.predict(X)
        y_true = y
        n_test = len(y)

    # ── Metrics ─────────────────────────────────────────────
    acc = accuracy_score(y_true, y_pred)
    prec_macro  = precision_score(y_true, y_pred, average="macro",  zero_division=0)
    rec_macro   = recall_score(y_true, y_pred,    average="macro",  zero_division=0)
    f1_macro    = f1_score(y_true, y_pred,        average="macro",  zero_division=0)
    prec_weight = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_weight  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
    f1_weight   = f1_score(y_true, y_pred,        average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Top confused pairs
    from collections import Counter
    confused = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p)
    top_confused = confused.most_common(15)

    # ── Print report ────────────────────────────────────────
    sep = "=" * 60
    report_lines = [
        sep,
        "  LAPORAN EVALUASI — DATASET 1: DatasetCharacter",
        f"  Tanggal   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model     : {args.model}",
        f"  Char-dir  : {args.char_dir}",
        f"  Mode      : {'Train/Test Split' if args.split_eval else 'Full Dataset'}",
        sep, "",
        "── INFO DATASET ─────────────────────────────────────────",
        f"  Total sampel dievaluasi : {n_test}",
        f"  Jumlah kelas            : {len(set(y_true))}",
        f"  Fitur HOG per karakter  : {X.shape[1]} dimensi",
        f"  HOG config              : orientations=9, px/cell=(8,8), cells/block=(2,2)",
        "",
        "── METRIK KESELURUHAN ───────────────────────────────────",
        f"  Accuracy                : {acc:.4f}  ({acc*100:.2f}%)",
        "",
        f"  Macro   Precision       : {prec_macro:.4f}",
        f"          Recall          : {rec_macro:.4f}",
        f"          F1-score        : {f1_macro:.4f}",
        "",
        f"  Weighted Precision      : {prec_weight:.4f}",
        f"           Recall         : {rec_weight:.4f}",
        f"           F1-score       : {f1_weight:.4f}",
        "",
        "── CLASSIFICATION REPORT (sklearn) ─────────────────────",
        classification_report(y_true, y_pred, labels=classes,
                              target_names=classes, zero_division=0),
        "── TOP 15 PASANGAN SALAH ────────────────────────────────",
        f"  {'Aktual':>6}  →  {'Prediksi':>8}  {'Jumlah':>6}",
        "  " + "-" * 30,
    ]
    for (t, p), cnt in top_confused:
        report_lines.append(f"  {t:>6}  →  {p:>8}  {cnt:>6}")

    report_lines += [
        "",
        "── CONFUSION MATRIX (ASCII) ─────────────────────────────",
        ascii_confusion_matrix(cm, classes),
        "",
        sep,
    ]

    report = "\n".join(report_lines)
    print("\n" + report)

    # Save report
    report_path = os.path.join(args.out_dir, "evaluation_report_dataset1.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved → {report_path}")

    # Save confusion matrix PNG
    print("Saving confusion matrix PNG...")
    save_confusion_matrix(cm, classes, args.out_dir,
                          title="Confusion Matrix — Dataset 1 (DatasetCharacter)\nHOG + SVM")

    # Optional HOG visualization
    if args.hog_viz:
        print("Saving HOG visualization...")
        save_hog_visualization(images, labels, args.out_dir)

    # Save JSON summary
    import json
    summary = {
        "dataset": "DatasetCharacter",
        "n_samples": n_test,
        "n_classes": len(set(y_true)),
        "hog_features": int(X.shape[1]),
        "accuracy": float(acc),
        "macro": {
            "precision": float(prec_macro),
            "recall": float(rec_macro),
            "f1": float(f1_macro),
        },
        "weighted": {
            "precision": float(prec_weight),
            "recall": float(rec_weight),
            "f1": float(f1_weight),
        },
        "top_confused": [
            {"true": t, "pred": p, "count": c} for (t, p), c in top_confused
        ],
        "generated_at": datetime.now().isoformat(),
    }
    json_path = os.path.join(args.out_dir, "metrics_dataset1.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON metrics saved → {json_path}")

    print(f"\n{'='*60}")
    print(f"  ✓ Evaluasi selesai")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Macro F1  : {f1_macro*100:.2f}%")
    print(f"  Output    : {args.out_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
