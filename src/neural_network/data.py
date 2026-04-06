from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def make_synthetic_classification_data(
    samples: int = 1000,
    features: int = 4,
    batch_size: int = 32,
    noise_level: float = 0.05,
    imbalance_ratio: float = 0.5,
    seed: int | None = 42,
    device: str | torch.device = "cpu",
) -> DataLoader:
    """
    Creates a production-ready synthetic binary classification DataLoader.
    
    Args:
        samples: Total number of data points.
        features: Dimensionality of the input (X).
        batch_size: Samples per iteration.
        noise_level: Probability of flipping labels (simulates real-world noise).
        imbalance_ratio: Fraction of positive (1) samples (0.5 is balanced).
        seed: Random state for reproducibility.
        device: Target device ('cpu' or 'cuda').
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 1. Generate core features on device
    x = torch.randn(samples, features, device=device)
    
    # 2. Dynamic weight generation for any number of features
    weights = torch.linspace(1.5, -1.5, steps=features, device=device)
    bias = 0.5
    
    # 3. Generate non-linear logits (XW + noise)
    logits = (x @ weights) + bias
    
    # 4. Handle Imbalance: Adjust threshold to meet target ratio
    # We use the percentile of the logits to ensure strict ratio control
    threshold = torch.quantile(logits, 1.0 - imbalance_ratio)
    y = (logits > threshold).float().unsqueeze(1)
    
    # 5. Inject Label Noise: Randomly flip a percentage of labels
    if noise_level > 0:
        flip_mask = torch.rand(samples, 1, device=device) < noise_level
        y = torch.where(flip_mask, 1.0 - y, y)
    
    # 6. Optimized DataLoader setup
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        # Pin memory is faster if moving data from CPU to GPU during training
        pin_memory=(device == "cpu" and torch.cuda.is_available())
    )
