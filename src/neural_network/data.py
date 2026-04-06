from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def make_parity_data(bits: int = 4, repeats: int = 128, seed: int = 7) -> tuple[Array, Array]:
    """Generate a non-linear parity dataset.

    For 4-bit parity, each sample label is 1 when the number of 1s is odd.
    """
    rng = np.random.default_rng(seed)
    patterns = np.array(list(itertools.product([0.0, 1.0], repeat=bits)), dtype=np.float64)
    labels = (np.sum(patterns, axis=1, keepdims=True) % 2).astype(np.float64)

    x = np.repeat(patterns, repeats=repeats, axis=0)
    y = np.repeat(labels, repeats=repeats, axis=0)

    # Shuffle for SGD/Adam training.
    indices = rng.permutation(x.shape[0])
    return x[indices], y[indices]


def iterate_minibatches(
    x: Array,
    y: Array,
    batch_size: int = 32,
    seed: int = 123,
) -> list[tuple[Array, Array]]:
    """Shuffle data and return mini-batches."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(x.shape[0])
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    batches: list[tuple[Array, Array]] = []
    for start in range(0, x.shape[0], batch_size):
        end = start + batch_size
        batches.append((x_shuffled[start:end], y_shuffled[start:end]))
    return batches
