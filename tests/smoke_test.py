"""Quick end-to-end smoke test on synthetic data (run from repo root)."""
import random

import numpy as np

from src.ocr import OCRClassifier, build_synthetic_dataset
from src.plate_classifier import PlateClassifier
from src.features import plate_hog
from src.pipeline import LicensePlateExpiryPipeline
from src.synthetic import make_plate, make_scene, random_plate_text
from src.preprocessing import preprocess
from src.plate_detection import find_candidates
from src import evaluation as ev

random.seed(0); np.random.seed(0)

print("1) Train OCR SVM on synthetic glyphs...")
imgs, labels = build_synthetic_dataset(samples_per_class=25)
ocr = OCRClassifier().fit(imgs, labels)
# self-check on a few fresh glyphs
from src.ocr import render_glyph, _find_fonts
fonts = _find_fonts()
correct = sum(ocr.predict_char(render_glyph(c, random.choice(fonts))) == c
              for c in "AB1234XY")
print(f"   OCR sanity on 8 chars: {correct}/8 correct")

print("2) Build plate-detection training set (positives + negatives)...")
pos_crops, neg_crops = [], []
for i in range(40):
    num, exp, _ = random_plate_text()
    plate = make_plate(num, exp, dark=random.random() < 0.5)
    scene, (x, y, w, h) = make_scene(plate, seed=i)
    gray = preprocess(scene)
    pos_crops.append(gray[y:y + h, x:x + w])
    # negatives: random non-plate crops from the same scene
    for _ in range(2):
        rw, rh = random.randint(40, 120), random.randint(20, 70)
        rx = random.randint(0, scene.shape[1] - rw - 1)
        ry = random.randint(0, scene.shape[0] - rh - 1)
        if abs(rx - x) > w or abs(ry - y) > h:
            neg_crops.append(gray[ry:ry + rh, rx:rx + rw])

X = np.vstack([plate_hog(c) for c in pos_crops + neg_crops])
yv = np.array([1] * len(pos_crops) + [0] * len(neg_crops))
plate_clf = PlateClassifier().fit(X, yv)
print(f"   trained on {len(pos_crops)} pos / {len(neg_crops)} neg")

print("3) Run full pipeline on fresh synthetic scenes...")
pipe = LicensePlateExpiryPipeline(plate_clf, ocr)
pred_boxes, gt_boxes, scores = [], [], []
ocr_preds, ocr_truth = [], []
detected = 0
N = 12
for i in range(100, 100 + N):
    num, exp, truth = random_plate_text()
    plate = make_plate(num, exp, dark=random.random() < 0.5)
    scene, gt = make_scene(plate, seed=i)
    res = pipe.run(scene)
    gt_boxes.append(gt)
    pred_boxes.append(res.plate_box)
    scores.append(1.0 if res.success else 0.0)
    if res.success:
        detected += 1
        ocr_preds.append(res.plate_number)
        ocr_truth.append(truth)

print(f"   plates detected: {detected}/{N}")
print(f"   mean IoU       : {ev.mean_iou(pred_boxes, gt_boxes):.3f}")
print(f"   mAP@0.5        : {ev.average_precision(pred_boxes, scores, gt_boxes):.3f}")
if ocr_preds:
    print(f"   char accuracy  : {ev.character_accuracy(ocr_preds, ocr_truth):.3f}")
    print(f"   word accuracy  : {ev.word_accuracy(ocr_preds, ocr_truth):.3f}")
    print(f"   WER            : {ev.word_error_rate(ocr_preds, ocr_truth):.3f}")
    print(f"   example: pred={ocr_preds[0]!r} truth={ocr_truth[0]!r}")

print("4) Expiry check demo...")
from src.expiry import check_expiry
from datetime import date
for txt in ["10.27", "01.24", "05.26"]:
    r = check_expiry(txt, today=date(2026, 6, 10))
    print(f"   {txt} -> expired={r.is_expired}  ({r.reason})")

print("\nSMOKE TEST COMPLETE")
