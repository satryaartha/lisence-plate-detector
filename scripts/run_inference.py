"""
scripts/run_inference.py

Run the trained pipeline on a single image and print the result. Optionally save
an annotated visualisation.

Run:
  python scripts/run_inference.py path/to/car.jpg --save outputs/result.png
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import OUTPUT_DIR
from src.ocr import OCRClassifier
from src.pipeline import LicensePlateExpiryPipeline
from src.plate_classifier import PlateClassifier
from src.plate_detection import draw_boxes
from src.preprocessing import read_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--save", default=None, help="path to save annotated image")
    ap.add_argument("--cropped", action="store_true",
                    help="input is already a cropped plate (skip detection)")
    args = ap.parse_args()

    ocr = OCRClassifier.load()
    img = read_image(args.image)

    if args.cropped:
        # No detection/classifier needed; the whole image is the plate.
        pipe = LicensePlateExpiryPipeline(None, ocr)
        res = pipe.run_on_plate(img, box=(0, 0, img.shape[1], img.shape[0]))
    else:
        plate_clf = PlateClassifier.load()
        pipe = LicensePlateExpiryPipeline(plate_clf, ocr)
        res = pipe.run(img)

    print(f"success      : {res.success}")
    print(f"message      : {res.message}")
    if res.plate_box:
        print(f"plate box    : {res.plate_box}")
    if res.success:
        print(f"plate number : {res.plate_number}")
        e = res.expiry
        print(f"expiry text  : {e.raw_text!r}")
        if e.is_expired is None:
            print(f"expiry status: UNKNOWN ({e.reason})")
        else:
            print(f"expiry status: {'EXPIRED' if e.is_expired else 'VALID'} "
                  f"({e.month:02d}/{e.year}) - {e.reason}")
    print("timings (s)  :", {k: round(v, 4) for k, v in res.timings.items()})

    save_path = args.save
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "result.png")
    if res.plate_box:
        vis = draw_boxes(img, [res.plate_box])
        label = (res.plate_number +
                 ("  EXPIRED" if (res.expiry and res.expiry.is_expired) else
                  "  VALID" if (res.expiry and res.expiry.is_expired is False) else ""))
        x, y = res.plate_box[0], max(20, res.plate_box[1] - 10)
        cv2.putText(vis, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.imwrite(save_path, vis)
        print(f"saved        : {save_path}")


if __name__ == "__main__":
    main()
