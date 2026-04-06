from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neural_network.data import iterate_minibatches, make_parity_data
from neural_network.metrics import accuracy_score
from neural_network.model import NeuralNetwork
from neural_network.model_library import get_model_spec
from neural_network.preprocessing import minmax_normalize, train_val_test_split

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
    early_stopping_patience: int = 200,
    seed: int = 42,
    log_every: int = 100,
) -> NeuralNetwork:
    """Train a modular NumPy DNN on non-linear parity data with validation monitoring."""
    if hidden_sizes is None:
        hidden_sizes = get_model_spec(model_preset).hidden_sizes

    x, y = make_parity_data(bits=input_size, repeats=128, seed=seed)
    x = minmax_normalize(x)
    (x_train, y_train), (x_val, y_val), _ = train_val_test_split(x, y, seed=seed)

    model = NeuralNetwork(input_size=input_size, hidden_sizes=hidden_sizes, seed=seed)

    best_state = model.state_dict()
    best_val_loss = float("inf")
    epochs_since_improvement = 0

    step = 0
    for epoch in range(1, epochs + 1):
        batches = iterate_minibatches(x_train, y_train, batch_size=batch_size, seed=seed + epoch)
        epoch_loss = 0.0
        current_lr = _cosine_lr(lr, epoch, epochs) if use_cosine_lr else lr

        for x_batch, y_batch in batches:
            step += 1
            preds = model.forward_propagation(x_batch)
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

        val_preds = model.forward_propagation(x_val)
        val_loss = binary_cross_entropy(y_val, val_preds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            train_acc = accuracy_score(y_train, model.predict(x_train))
            val_acc = accuracy_score(y_val, model.predict(x_val))
            mean_loss = epoch_loss / len(batches)
            print(
                f"Epoch {epoch:04d}/{epochs} - lr: {current_lr:.6f} - "
                f"train_loss: {mean_loss:.6f} - val_loss: {val_loss:.6f} - "
                f"train_acc: {train_acc:.3f} - val_acc: {val_acc:.3f}"
            )

        if epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    # Restore best validation checkpoint.
    return NeuralNetwork.from_state_dict(input_size=input_size, state=best_state)


def accuracy(model: NeuralNetwork, bits: int = 4) -> float:
    x_eval, y_eval = make_parity_data(bits=bits, repeats=16, seed=999)
    x_eval = minmax_normalize(x_eval)
    preds = model.predict(x_eval)
    return accuracy_score(y_eval, preds)


def run_demo() -> None:
    """Train on 4-bit parity and print final accuracy."""
    model = train_model(model_preset="llm_wide")
    final_acc = accuracy(model, bits=4)
    print(f"Final 4-bit parity accuracy: {final_acc:.3f}")


if __name__ == "__main__":
    run_demo()
