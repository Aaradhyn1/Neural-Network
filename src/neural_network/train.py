from __future__ import annotations

import torch
from torch import nn

from neural_network.data import make_synthetic_classification_data
from neural_network.model import FeedForwardNetwork


def train_model(
    input_size: int = 4,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cpu",
) -> FeedForwardNetwork:
    loader = make_synthetic_classification_data(features=input_size)
    model = FeedForwardNetwork(input_size=input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model


def accuracy(model: FeedForwardNetwork, device: str = "cpu") -> float:
    loader = make_synthetic_classification_data(samples=300, features=4, batch_size=64)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            probs = torch.sigmoid(model(x_batch))
            preds = (probs > 0.5).float()
            correct += (preds == y_batch).sum().item()
            total += y_batch.numel()
    return correct / total if total else 0.0
