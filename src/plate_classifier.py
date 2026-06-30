"""
Stage 4 - SVM classifier (plate vs non-plate).

Given HOG features of a candidate region, decide whether it is a license plate.
We wrap an sklearn SVC in a small class that standardises features and persists
to disk. Among many plate candidates, the one classified as a plate with the
highest decision score is selected as the final plate.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    PLATE_SVM_C,
    PLATE_SVM_KERNEL,
    PLATE_SVM_PATH,
    RANDOM_STATE,
)
from src.features import plate_hog
from src.plate_detection import Candidate


class PlateClassifier:
    """HOG + SVM classifier for plate / non-plate regions."""

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                C=PLATE_SVM_C,
                kernel=PLATE_SVM_KERNEL,
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])
        self._fitted = False

    # ----------------------------------------------------------------- train
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlateClassifier":
        """Fit on HOG feature matrix X (N, D) and labels y (1 = plate)."""
        self.model.fit(X, y)
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def score_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive (plate) class."""
        return self.model.predict_proba(X)[:, 1]

    def select_plate(self, candidates: list[Candidate]) -> Candidate | None:
        """Pick the best plate among geometric candidates.

        Returns the candidate classified as a plate with the highest
        probability, or None if no candidate is classified as a plate.
        """
        if not candidates:
            return None
        X = np.vstack([plate_hog(c.crop) for c in candidates])
        proba = self.score_proba(X)
        preds = (proba >= 0.5).astype(int)
        best_idx, best_p = None, -1.0
        for i, (p, prob) in enumerate(zip(preds, proba)):
            if p == 1 and prob > best_p:
                best_idx, best_p = i, prob
        if best_idx is None:
            return None
        return candidates[best_idx]

    # ------------------------------------------------------------- persist
    def save(self, path: str = PLATE_SVM_PATH) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str = PLATE_SVM_PATH) -> "PlateClassifier":
        obj = cls()
        obj.model = joblib.load(path)
        obj._fitted = True
        return obj
