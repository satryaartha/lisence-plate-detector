"""
scripts/prepare_dataset.py

Turn the raw Kaggle dataset into the (positive plate crops / negative background
crops) needed to train the HOG + SVM plate classifier.

The Kaggle "haarcascadeplatenumber" dataset can ship in a few shapes, so this
script supports several input layouts (pick with --layout):

  haar    : OpenCV positives annotation file (info.dat / pos.txt):
            each line is  <image> <N> <x y w h> [<x y w h> ...]   (DEFAULT)
  voc     : images/ + annotations/ (Pascal VOC .xml with <bndbox>)
  yolo    : images/ + labels/ (YOLO .txt: class cx cy w h, normalised)
  folders : plates/ + background/ (already-cropped images, no annotation)

Not sure which one? Inspect the extracted dataset first:
  python scripts/prepare_dataset.py --raw data/raw --inspect

Output (under data/processed/):
  positives/*.png   cropped license plates
  negatives/*.png   random non-plate crops

Run:
  python scripts/prepare_dataset.py --layout haar --raw data/raw
"""
from __future__ import annotations

import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET

import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DIR, RAW_DATA_DIR, RANDOM_STATE

random.seed(RANDOM_STATE)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(folder: str) -> list[str]:
    files = []
    for ext in IMG_EXT:
        files += glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True)
    return sorted(files)


def _save(crop, out_dir: str, name: str) -> None:
    if crop is None or crop.size == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, name), crop)


def _random_negatives(img, boxes, n: int):
    """Sample n random crops that do not overlap any plate box."""
    h, w = img.shape[:2]
    negs = []
    tries = 0
    while len(negs) < n and tries < n * 20:
        tries += 1
        cw = random.randint(int(w * 0.1), int(w * 0.4))
        ch = random.randint(int(h * 0.05), int(h * 0.25))
        if cw < 10 or ch < 10:
            continue
        x = random.randint(0, w - cw)
        y = random.randint(0, h - ch)
        overlap = any(not (x + cw < bx or x > bx + bw or y + ch < by or y > by + bh)
                      for (bx, by, bw, bh) in boxes)
        if not overlap:
            negs.append(img[y:y + ch, x:x + cw])
    return negs


def _parse_voc(xml_path: str) -> list[tuple[int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = int(float(b.find("xmin").text))
        ymin = int(float(b.find("ymin").text))
        xmax = int(float(b.find("xmax").text))
        ymax = int(float(b.find("ymax").text))
        boxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
    return boxes


def _parse_yolo(txt_path: str, w: int, h: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            bw_px, bh_px = bw * w, bh * h
            x = int(cx * w - bw_px / 2)
            y = int(cy * h - bh_px / 2)
            boxes.append((max(0, x), max(0, y), int(bw_px), int(bh_px)))
    return boxes


def from_annotations(raw: str, layout: str, neg_per_img: int) -> tuple[int, int]:
    img_dir = os.path.join(raw, "images")
    images = _list_images(img_dir if os.path.isdir(img_dir) else raw)
    pos_dir = os.path.join(PROCESSED_DIR, "positives")
    neg_dir = os.path.join(PROCESSED_DIR, "negatives")
    n_pos = n_neg = 0
    for ip in images:
        img = cv2.imread(ip)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ip))[0]
        if layout == "voc":
            xml = os.path.join(raw, "annotations", stem + ".xml")
            boxes = _parse_voc(xml) if os.path.exists(xml) else []
        else:  # yolo
            txt = os.path.join(raw, "labels", stem + ".txt")
            boxes = _parse_yolo(txt, w, h)
        for j, (x, y, bw, bh) in enumerate(boxes):
            _save(img[y:y + bh, x:x + bw], pos_dir, f"{stem}_{j}.png")
            n_pos += 1
        for k, neg in enumerate(_random_negatives(img, boxes, neg_per_img)):
            _save(neg, neg_dir, f"{stem}_neg{k}.png")
            n_neg += 1
    return n_pos, n_neg


def _find_annotation_file(raw: str) -> str | None:
    """Look for an OpenCV-style annotation file (info.dat / pos.txt / *.dat)."""
    preferred = ["info.dat", "pos.txt", "positives.txt", "annotation.txt",
                 "annotations.txt", "info.txt"]
    for name in preferred:
        for hit in glob.glob(os.path.join(raw, "**", name), recursive=True):
            return hit
    # fall back to any .dat / .txt that looks like "path N x y w h ..."
    for pat in ("**/*.dat", "**/*.txt"):
        for hit in glob.glob(os.path.join(raw, pat), recursive=True):
            try:
                with open(hit) as f:
                    first = f.readline().split()
                if len(first) >= 6 and first[1].isdigit():
                    return hit
            except Exception:
                continue
    return None


def _resolve_image(token: str, ann_dir: str, raw: str) -> str | None:
    """Resolve an image path referenced inside the annotation file."""
    cands = [token,
             os.path.join(ann_dir, token),
             os.path.join(raw, token),
             os.path.join(raw, os.path.basename(token))]
    for c in cands:
        if os.path.exists(c):
            return c
    # last resort: search by basename anywhere under raw
    hits = glob.glob(os.path.join(raw, "**", os.path.basename(token)),
                     recursive=True)
    return hits[0] if hits else None


def from_haar(raw: str, ann_file: str | None, neg_per_img: int) -> tuple[int, int]:
    """Parse the OpenCV positives format:  <image> <N> <x y w h> [<x y w h> ...]"""
    ann_file = ann_file or _find_annotation_file(raw)
    if not ann_file or not os.path.exists(ann_file):
        raise SystemExit(
            "Could not find an OpenCV annotation file. Pass it with --ann "
            "(e.g. --ann data/raw/info.dat). Run with --inspect to see the "
            "dataset structure."
        )
    print(f"Using annotation file: {ann_file}")
    ann_dir = os.path.dirname(ann_file)
    pos_dir = os.path.join(PROCESSED_DIR, "positives")
    neg_dir = os.path.join(PROCESSED_DIR, "negatives")
    n_pos = n_neg = 0
    with open(ann_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            token, rest = parts[0], parts[1:]
            try:
                n = int(rest[0])
                nums = list(map(int, rest[1:1 + 4 * n]))
            except (ValueError, IndexError):
                continue
            img_path = _resolve_image(token, ann_dir, raw)
            if img_path is None:
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            stem = os.path.splitext(os.path.basename(img_path))[0]
            boxes = [tuple(nums[i:i + 4]) for i in range(0, len(nums), 4)]
            for j, (x, y, bw, bh) in enumerate(boxes):
                _save(img[y:y + bh, x:x + bw], pos_dir, f"{stem}_{j}.png")
                n_pos += 1
            for k, neg in enumerate(_random_negatives(img, boxes, neg_per_img)):
                _save(neg, neg_dir, f"{stem}_neg{k}.png")
                n_neg += 1
    return n_pos, n_neg


def inspect(raw: str) -> None:
    """Print a quick view of the dataset structure to pick the right layout."""
    print(f"Inspecting: {raw}\n")
    for root, dirs, files in os.walk(raw):
        depth = root.replace(raw, "").count(os.sep)
        if depth > 2:
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root) or raw}/  ({len(files)} files)")
        for fn in sorted(files)[:4]:
            print(f"{indent}  {fn}")
    ann = _find_annotation_file(raw)
    print(f"\nDetected annotation file: {ann or 'none'}")
    if ann:
        with open(ann) as f:
            print("First lines:")
            for _ in range(3):
                print("  " + f.readline().strip())


def from_folders(raw: str) -> tuple[int, int]:
    pos = _list_images(os.path.join(raw, "plates"))
    neg = _list_images(os.path.join(raw, "background"))
    pos_dir = os.path.join(PROCESSED_DIR, "positives")
    neg_dir = os.path.join(PROCESSED_DIR, "negatives")
    for i, p in enumerate(pos):
        _save(cv2.imread(p), pos_dir, f"plate_{i}.png")
    for i, n in enumerate(neg):
        _save(cv2.imread(n), neg_dir, f"bg_{i}.png")
    return len(pos), len(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW_DATA_DIR)
    ap.add_argument("--layout", choices=["haar", "voc", "yolo", "folders"],
                    default="haar")
    ap.add_argument("--ann", default=None,
                    help="path to the OpenCV annotation file (haar layout)")
    ap.add_argument("--neg-per-img", type=int, default=3)
    ap.add_argument("--inspect", action="store_true",
                    help="just print the dataset structure and exit")
    args = ap.parse_args()

    if args.inspect:
        inspect(args.raw)
        return

    if args.layout == "haar":
        n_pos, n_neg = from_haar(args.raw, args.ann, args.neg_per_img)
    elif args.layout == "folders":
        n_pos, n_neg = from_folders(args.raw)
    else:
        n_pos, n_neg = from_annotations(args.raw, args.layout, args.neg_per_img)
    print(f"Done. positives={n_pos}  negatives={n_neg}")
    print(f"Saved under {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
