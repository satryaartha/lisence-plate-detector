"""
End-to-end pipeline orchestration.

Ties together every stage of the proposed method:

    image
      -> preprocess (grayscale + Gaussian)
      -> Canny + contour candidates
      -> HOG + SVM plate selection
      -> character segmentation
      -> HOG + SVM OCR  (split into number row + expiry row)
      -> expiry parse vs current date
      -> result
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.char_segmentation import CharBox, segment_characters, split_rows
from src.expiry import ExpiryResult, check_expiry
from src.ocr import OCRClassifier
from src.plate_classifier import PlateClassifier
from src.plate_detection import Candidate, find_candidates
from src.preprocessing import preprocess


@dataclass
class PipelineResult:
    success: bool
    plate_box: tuple[int, int, int, int] | None = None
    plate_number: str = ""
    expiry: ExpiryResult | None = None
    candidates: list[Candidate] = field(default_factory=list)
    chars: list[CharBox] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    message: str = ""


class LicensePlateExpiryPipeline:
    def __init__(self, plate_clf: PlateClassifier, ocr: OCRClassifier):
        self.plate_clf = plate_clf
        self.ocr = ocr

    def run_on_plate(self, plate_gray: np.ndarray,
                     box=None, candidates=None, timings=None) -> PipelineResult:
        """Read an already-localised plate: segment -> OCR -> expiry.

        Use this directly when the input image is *already a cropped plate*
        (no detection needed); `run()` calls it after detection.
        """
        timings = timings or {}
        if plate_gray.ndim == 3:
            plate_gray = preprocess(plate_gray)

        t0 = time.perf_counter()
        chars = segment_characters(plate_gray)
        timings["segmentation"] = time.perf_counter() - t0
        if not chars:
            return PipelineResult(False, plate_box=box, candidates=candidates or [],
                                  timings=timings,
                                  message="No characters segmented from the plate.")

        ph = plate_gray.shape[0]
        top_row, bottom_row = split_rows(chars, ph)

        t0 = time.perf_counter()
        plate_number = self.ocr.read(top_row) if top_row else self.ocr.read(chars)
        bottom_text = self.ocr.read_digits(bottom_row) if bottom_row else ""
        timings["ocr"] = time.perf_counter() - t0

        return PipelineResult(
            success=True, plate_box=box, plate_number=plate_number,
            expiry=check_expiry(bottom_text), candidates=candidates or [],
            chars=chars, timings=timings, message="OK",
        )

    def run(self, image: np.ndarray) -> PipelineResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        gray = preprocess(image)
        timings["preprocess"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        candidates = find_candidates(gray)
        timings["detection"] = time.perf_counter() - t0
        if not candidates:
            return PipelineResult(False, candidates=[], timings=timings,
                                  message="No plate-like contours found.")

        t0 = time.perf_counter()
        plate = self.plate_clf.select_plate(candidates)
        timings["classification"] = time.perf_counter() - t0
        if plate is None:
            return PipelineResult(False, candidates=candidates, timings=timings,
                                  message="No candidate classified as a plate.")

        return self.run_on_plate(plate.crop, box=plate.box,
                                 candidates=candidates, timings=timings)
