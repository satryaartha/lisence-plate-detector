"""
scripts/test_on_dataset.py

Test the model on a folder of *already-cropped* plate images where the file name
is the ground-truth plate number (e.g. AA4795BE.jpg -> "AA4795BE"). This matches
the Kaggle "haarcascadeplatenumber" recognition images.

For each plate it segments characters, reads the number (and the expiry row),
then reports character accuracy, word accuracy and WER against the file names,
plus how many expiry dates parsed. A few annotated examples are saved.

Run:
  python scripts/test_on_dataset.py --dir data/raw --limit 300
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTPUT_DIR, SUCCESS_CRITERIA
from src import evaluation as ev
from src.ocr import OCRClassifier
from src.pipeline import LicensePlateExpiryPipeline


def _label_from_name(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    # keep only A-Z and 0-9, uppercase (drops separators like '_', '-', spaces)
    return re.sub(r"[^A-Z0-9]", "", stem.upper())


def _list_images(folder: str, recursive: bool) -> list[str]:
    files = []
    pat = "**/*" if recursive else "*"
    for ext in ("jpg", "jpeg", "png", "bmp"):
        files += glob.glob(os.path.join(folder, f"{pat}.{ext}"), recursive=recursive)
    # never treat the character-training folder as plates
    files = [f for f in files if "character" not in f.lower()]
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/raw", help="folder of cropped plate images")
    ap.add_argument("--limit", type=int, default=300, help="max images to test")
    ap.add_argument("--recursive", action="store_true",
                    help="also search sub-folders (e.g. data/raw/dataset)")
    ap.add_argument("--save-examples", type=int, default=8)
    args = ap.parse_args()

    ocr = OCRClassifier.load()
    pipe = LicensePlateExpiryPipeline(None, ocr)   # detection not needed

    images = _list_images(args.dir, args.recursive)[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {args.dir}")
    print(f"Testing on {len(images)} cropped plates from {args.dir}\n")

    preds, truths = [], []
    expiry_parsed = read_ok = 0
    saved = 0
    for path in images:
        truth = _label_from_name(path)
        if not truth:
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        res = pipe.run_on_plate(img)
        if not res.success:
            preds.append(""); truths.append(truth)
            continue
        read_ok += 1
        preds.append(res.plate_number); truths.append(truth)
        if res.expiry and res.expiry.is_expired is not None:
            expiry_parsed += 1
        if saved < args.save_examples:
            vis = img.copy()
            tag = res.plate_number
            if res.expiry and res.expiry.is_expired is not None:
                tag += "  " + ("EXPIRED" if res.expiry.is_expired else "VALID")
            cv2.putText(vis, tag, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"example_{saved}.png"), vis)
            saved += 1

    n = len(truths)
    metrics = {
        "char_accuracy": ev.character_accuracy(preds, truths),
        "word_accuracy": ev.word_accuracy(preds, truths),
        "wer": ev.word_error_rate(preds, truths),
    }
    print(ev.report(metrics, SUCCESS_CRITERIA))
    print(f"\nplates with characters segmented : {read_ok}/{n}")
    print(f"plates with a parseable expiry   : {expiry_parsed}/{n}")
    print(f"\nexamples (pred vs truth):")
    for p, t in list(zip(preds, truths))[:10]:
        mark = "OK " if p == t else "x  "
        print(f"  {mark} pred={p!r:14} truth={t!r}")
    print(f"\nannotated examples saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
