from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_classification_data(
    samples: int = 1000,
    features: int = 4,
    batch_size: int = 32,
) -> DataLoader:
    """Create a synthetic binary classification dataset."""
    x = torch.randn(samples, features)
    weights = torch.tensor([1.2, -0.7, 0.9, -1.1][:features], dtype=torch.float32)
    logits = x @ weights + 0.2 * torch.randn(samples)
    y = (logits > 0).float().unsqueeze(1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
