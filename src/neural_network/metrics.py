from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def accuracy_score(y_true: Array, y_pred_binary: NDArray[np.int64]) -> float:
    return float(np.mean(y_pred_binary == y_true.astype(np.int64)))


def precision_recall_f1(y_true: Array, y_pred_binary: NDArray[np.int64]) -> tuple[float, float, float]:
    y_true_i = y_true.astype(np.int64).reshape(-1)
    y_pred_i = y_pred_binary.astype(np.int64).reshape(-1)

    tp = int(np.sum((y_true_i == 1) & (y_pred_i == 1)))
    fp = int(np.sum((y_true_i == 0) & (y_pred_i == 1)))
    fn = int(np.sum((y_true_i == 1) & (y_pred_i == 0)))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return float(precision), float(recall), float(f1)
