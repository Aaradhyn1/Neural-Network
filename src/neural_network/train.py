from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neural_network.data import iterate_minibatches, make_parity_data
from neural_network.model import NeuralNetwork
from neural_network.model_library import get_model_spec

Array = NDArray[np.float64]


def binary_cross_entropy(y_true: Array, y_pred: Array) -> float:
    """Binary cross-entropy loss for probabilities in [0, 1]."""
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(np.mean(loss))


def _cosine_lr(base_lr: float, epoch: int, total_epochs: int) -> float:
    """Cosine decay schedule for smoother convergence."""
    progress = (epoch - 1) / max(total_epochs - 1, 1)
    return 0.5 * base_lr * (1.0 + np.cos(np.pi * progress))


def train_model(
    input_size: int = 4,
    hidden_sizes: tuple[int, ...] | None = None,
    model_preset: str = "parity",
    epochs: int = 2000,
    lr: float = 0.01,
    batch_size: int = 32,
    optimizer: str = "adam",
    weight_decay: float = 1e-4,
    grad_clip: float | None = 1.0,
    use_cosine_lr: bool = True,
    seed: int = 42,
    log_every: int = 100,
) -> NeuralNetwork:
    """Train a modular NumPy DNN on non-linear parity data."""
    if hidden_sizes is None:
        hidden_sizes = get_model_spec(model_preset).hidden_sizes

    x, y = make_parity_data(bits=input_size, repeats=128, seed=seed)
    model = NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes, seed=seed)

    step = 0
    for epoch in range(1, epochs + 1):
        batches = iterate_minibatches(x, y, batch_size=batch_size, seed=seed + epoch)
        epoch_loss = 0.0
        current_lr = _cosine_lr(lr, epoch, epochs) if use_cosine_lr else lr

        for x_batch, y_batch in batches:
            step += 1
            preds = model.forward_propagation(x_batch)

            # BCE + sigmoid simplifies gradient wrt logits to (pred - y),
            # because dL/dA and dA/dZ terms cancel analytically.
            grad_loss = preds - y_batch
            model.backpropagation(grad_loss)
            model.update_parameters(
                learning_rate=current_lr,
                optimizer=optimizer,
                step=step,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
            )

            epoch_loss += binary_cross_entropy(y_batch, preds)

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            mean_loss = epoch_loss / len(batches)
            print(f"Epoch {epoch:04d}/{epochs} - lr: {current_lr:.6f} - loss: {mean_loss:.6f}")

    return model


def accuracy(model: NeuralNetwork, bits: int = 4) -> float:
    x_eval, y_eval = make_parity_data(bits=bits, repeats=16, seed=999)
    preds = model.predict(x_eval)
    return float(np.mean(preds == y_eval.astype(np.int64)))


def run_demo() -> None:
    """Train on 4-bit parity and print final accuracy."""
    model = train_model(model_preset="llm_wide")
    final_acc = accuracy(model, bits=4)
    print(f"Final 4-bit parity accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    run_demo()
