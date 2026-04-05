from __future__ import annotations

import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """A simple MLP for binary classification."""

    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
