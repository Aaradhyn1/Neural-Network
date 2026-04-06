from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def impute_missing_with_mean(x: Array) -> Array:
    """Fill NaN values with column-wise means."""
    x_filled = x.copy()
    col_means = np.nanmean(x_filled, axis=0)
    nan_rows, nan_cols = np.where(np.isnan(x_filled))
    x_filled[nan_rows, nan_cols] = col_means[nan_cols]
    return x_filled


def minmax_normalize(x: Array, eps: float = 1e-8) -> Array:
    """Scale each feature to [0, 1]."""
    x_min = np.min(x, axis=0, keepdims=True)
    x_max = np.max(x, axis=0, keepdims=True)
    return (x - x_min) / np.maximum(x_max - x_min, eps)


def train_val_test_split(
    x: Array,
    y: Array,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[tuple[Array, Array], tuple[Array, Array], tuple[Array, Array]]:
    """Split arrays into train/validation/test partitions."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be > 0 and sum to < 1")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    n = x.shape[0]
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    x_train, y_train = x_shuffled[:n_train], y_shuffled[:n_train]
    x_val, y_val = x_shuffled[n_train : n_train + n_val], y_shuffled[n_train : n_train + n_val]
    x_test, y_test = x_shuffled[n_train + n_val :], y_shuffled[n_train + n_val :]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)
