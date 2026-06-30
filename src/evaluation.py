"""
Evaluation metrics.

Implements every metric named in the proposal's Evaluation section:

  Classification : confusion matrix, accuracy, precision, recall, F1
  Detection      : IoU (mean), mAP@0.5 (single class)
  OCR            : character accuracy, word accuracy, Word Error Rate (WER)

All formulas follow the proposal definitions so reported numbers map 1:1 to the
success criteria in config.SUCCESS_CRITERIA.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
@dataclass
class ClassificationMetrics:
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def accuracy(self) -> float:
        d = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / d if d else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def classification_metrics(y_true, y_pred, positive=1) -> ClassificationMetrics:
    """Build a binary confusion matrix and derived metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    tn = int(np.sum((y_pred != positive) & (y_true != positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    return ClassificationMetrics(tp, tn, fp, fn)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def iou(box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int]) -> float:
    """Intersection over Union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def mean_iou(pred_boxes, gt_boxes) -> float:
    """Mean IoU over paired (pred, gt) boxes; skips entries where pred is None."""
    vals = [iou(p, g) for p, g in zip(pred_boxes, gt_boxes) if p is not None]
    return float(np.mean(vals)) if vals else 0.0


def average_precision(pred_boxes, scores, gt_boxes, iou_thr: float = 0.5) -> float:
    """Single-class AP (≈ mAP for one class) at a fixed IoU threshold.

    Each image is assumed to contain at most one ground-truth plate. A detection
    is a TP if its IoU with that image's GT box exceeds `iou_thr`.
    """
    order = np.argsort(-np.asarray(scores))
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    matched = set()
    for rank, idx in enumerate(order):
        gt = gt_boxes[idx]
        pred = pred_boxes[idx]
        if pred is None or gt is None:
            fp[rank] = 1
            continue
        if iou(pred, gt) >= iou_thr and idx not in matched:
            tp[rank] = 1
            matched.add(idx)
        else:
            fp[rank] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    n_gt = sum(1 for g in gt_boxes if g is not None)
    recall = cum_tp / (n_gt + 1e-9)
    precision = cum_tp / (cum_tp + cum_fp + 1e-9)
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = precision[recall >= t]
        ap += (np.max(p) if p.size else 0.0) / 11.0
    return float(ap)


# --------------------------------------------------------------------------- #
# OCR
# --------------------------------------------------------------------------- #
def _levenshtein(a: str, b: str) -> tuple[int, int, int]:
    """Return (substitutions, deletions, insertions) via edit-distance backtrace."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,        # deletion
                           dp[i][j - 1] + 1,        # insertion
                           dp[i - 1][j - 1] + cost) # sub/match
    # Backtrace to count operation types
    i, j, s, d, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            s += 1; i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1; i -= 1
        else:
            ins += 1; j -= 1
    return s, d, ins


def character_accuracy(preds: list[str], truths: list[str]) -> float:
    """Fraction of characters recognised correctly (position-wise via edit dist)."""
    total_chars, total_errors = 0, 0
    for p, t in zip(preds, truths):
        s, d, ins = _levenshtein(p, t)
        total_errors += s + d + ins
        total_chars += len(t)
    return 1.0 - (total_errors / total_chars) if total_chars else 0.0


def word_accuracy(preds: list[str], truths: list[str]) -> float:
    """Fraction of plates whose entire string is read correctly."""
    if not truths:
        return 0.0
    correct = sum(1 for p, t in zip(preds, truths) if p == t)
    return correct / len(truths)


def word_error_rate(preds: list[str], truths: list[str]) -> float:
    """WER = (S + D + I) / N, aggregated over all plates."""
    s_tot = d_tot = i_tot = n_tot = 0
    for p, t in zip(preds, truths):
        s, d, ins = _levenshtein(p, t)
        s_tot += s; d_tot += d; i_tot += ins; n_tot += len(t)
    return (s_tot + d_tot + i_tot) / n_tot if n_tot else 0.0


def report(metrics: dict[str, float], criteria: dict[str, float]) -> str:
    """Pretty-print a pass/fail table against success criteria."""
    lines = [f"{'Metric':<16}{'Value':>10}{'Target':>10}{'Status':>10}"]
    lines.append("-" * 46)
    for name, val in metrics.items():
        target = criteria.get(name)
        if target is None:
            status = ""
            tgt_s = "-"
        else:
            # WER is an upper bound (lower is better)
            ok = (val <= target) if name == "wer" else (val >= target)
            status = "PASS" if ok else "FAIL"
            tgt_s = f"{target:.2f}"
        lines.append(f"{name:<16}{val:>10.3f}{tgt_s:>10}{status:>10}")
    return "\n".join(lines)
