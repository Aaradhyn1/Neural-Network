from __future__ import annotations

import torch
from torch import nn, Tensor


class ResidualBlock(nn.Module):
    """A pre-norm residual block for stable deep learning."""
    def __init__(self, size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.GELU(),  # Modern alternative to ReLU
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class FeedForwardNetwork(nn.Module):
    """
    An advanced FFN with residual paths, layer normalization, 
    and smart weight initialization.
    """
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        num_layers: int = 3,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        # 1. Input projection
        self.stem = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU()
        )
        
        # 2. Stack of Residual Blocks
        self.backbone = nn.Sequential(
            *[ResidualBlock(hidden_size, dropout) for _ in range(num_layers)]
        )
        
        # 3. Output head (Linear layer)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

        # 4. Custom Weight Initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            # Kaiming initialization is superior for GELU/ReLU
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.backbone(x)
        return self.head(x)
