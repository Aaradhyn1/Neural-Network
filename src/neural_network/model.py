from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LayerConfig:
    """Configuration for a dense layer in the network."""

    in_features: int
    out_features: int
    activation: str


class Layer:
    """Fully connected layer with cached tensors used for backpropagation.

    Forward pass (per batch):
        Z = X @ W + b
        A = activation(Z)

    During backpropagation we reuse cached X and Z to compute:
        dW = (X.T @ dZ) / m
        db = sum(dZ) / m
        dX = dZ @ W.T

    where:
      - X has shape (m, in_features)
      - W has shape (in_features, out_features)
      - dZ has shape (m, out_features)
      - X.T @ dZ produces (in_features, out_features)
    """

    def __init__(self, config: LayerConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.weights = self._init_weights(
            config.in_features,
            config.out_features,
            config.activation,
            rng,
        )
        self.biases = np.zeros((1, config.out_features), dtype=np.float64)

        self._last_input: Array | None = None
        self._last_linear: Array | None = None

        # Gradient buffers (filled during backward pass).
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

        # Adam state.
        self._mw = np.zeros_like(self.weights)
        self._vw = np.zeros_like(self.weights)
        self._mb = np.zeros_like(self.biases)
        self._vb = np.zeros_like(self.biases)

    @staticmethod
    def _init_weights(
        in_features: int,
        out_features: int,
        activation: str,
        rng: np.random.Generator,
    ) -> Array:
        # He initialization is ideal for ReLU hidden layers.
        if activation in {"relu", "leaky_relu"}:
            std = np.sqrt(2.0 / in_features)
        else:
            # Xavier-like scaling is stable for sigmoid/tanh style layers.
            std = np.sqrt(1.0 / in_features)
        return rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)

    def forward(self, x: Array) -> Array:
        self._last_input = x
        self._last_linear = x @ self.weights + self.biases
        return _apply_activation(self._last_linear, self.config.activation)

    def backward(self, grad_output: Array) -> Array:
        """Compute local gradients and return gradient wrt input.

        If grad_output = dL/dA, then for element-wise activations:
            dZ = dL/dA * dA/dZ
        and chain rule gives dL/dX = dZ @ W.T.
        """
        if self._last_input is None or self._last_linear is None:
            raise RuntimeError("backward called before forward")

        batch_size = self._last_input.shape[0]
        grad_activation = _activation_gradient(self._last_linear, self.config.activation)
        grad_linear = grad_output * grad_activation

        self.grad_weights = (self._last_input.T @ grad_linear) / batch_size
        self.grad_biases = np.sum(grad_linear, axis=0, keepdims=True) / batch_size
        grad_input = grad_linear @ self.weights.T
        return grad_input

    def apply_gradients(
        self,
        learning_rate: float,
        optimizer: str,
        step: int,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
    ) -> None:
        if grad_clip is not None:
            self.grad_weights = np.clip(self.grad_weights, -grad_clip, grad_clip)
            self.grad_biases = np.clip(self.grad_biases, -grad_clip, grad_clip)

        if weight_decay > 0.0:
            # L2 regularization contribution to gradient.
            self.grad_weights = self.grad_weights + weight_decay * self.weights

        if optimizer == "sgd":
            self.weights -= learning_rate * self.grad_weights
            self.biases -= learning_rate * self.grad_biases
            return

        if optimizer != "adam":
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Adam updates with bias correction.
        self._mw = beta1 * self._mw + (1.0 - beta1) * self.grad_weights
        self._vw = beta2 * self._vw + (1.0 - beta2) * (self.grad_weights**2)
        self._mb = beta1 * self._mb + (1.0 - beta1) * self.grad_biases
        self._vb = beta2 * self._vb + (1.0 - beta2) * (self.grad_biases**2)

        mw_hat = self._mw / (1.0 - beta1**step)
        vw_hat = self._vw / (1.0 - beta2**step)
        mb_hat = self._mb / (1.0 - beta1**step)
        vb_hat = self._vb / (1.0 - beta2**step)

        self.weights -= learning_rate * mw_hat / (np.sqrt(vw_hat) + eps)
        self.biases -= learning_rate * mb_hat / (np.sqrt(vb_hat) + eps)


class NeuralNetwork:
    """Modular feed-forward network supporting any hidden-layer layout."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        output_size: int = 1,
        output_activation: str = "sigmoid",
        seed: int = 42,
    ) -> None:
        if output_size != 1:
            raise ValueError(
                "This implementation is for binary classification with one output neuron."
            )

        layer_sizes = (input_size, *hidden_sizes, output_size)
        rng = np.random.default_rng(seed)

        self.layers: list[Layer] = []
        for i in range(len(layer_sizes) - 1):
            activation = "relu" if i < len(layer_sizes) - 2 else output_activation
            self.layers.append(
                Layer(
                    LayerConfig(
                        in_features=layer_sizes[i],
                        out_features=layer_sizes[i + 1],
                        activation=activation,
                    ),
                    rng=rng,
                )
            )

    def forward_propagation(self, x: Array) -> Array:
        activations = x
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backpropagation(self, grad_loss: Array) -> None:
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_parameters(
        self,
        learning_rate: float,
        optimizer: str,
        step: int,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
    ) -> None:
        for layer in self.layers:
            layer.apply_gradients(
                learning_rate,
                optimizer,
                step,
                beta1,
                beta2,
                eps,
                weight_decay=weight_decay,
                grad_clip=grad_clip,
            )

    def predict_proba(self, x: Array) -> Array:
        return self.forward_propagation(x)

    def predict(self, x: Array) -> NDArray[np.int64]:
        return (self.predict_proba(x) >= 0.5).astype(np.int64)

    # Backward-compatible aliases.
    forward = forward_propagation
    backward = backpropagation

    def state_dict(self) -> dict[str, object]:
        return {
            "layers": [
                {
                    "weights": layer.weights.copy(),
                    "biases": layer.biases.copy(),
                    "activation": layer.config.activation,
                }
                for layer in self.layers
            ]
        }

    @classmethod
    def from_state_dict(cls, input_size: int, state: dict[str, object]) -> NeuralNetwork:
        layer_payload = state["layers"]
        if not isinstance(layer_payload, list) or not layer_payload:
            raise ValueError("state_dict has invalid layer payload")

        hidden_sizes = tuple(item["weights"].shape[1] for item in layer_payload[:-1])
        output_size = layer_payload[-1]["weights"].shape[1]
        output_activation = str(layer_payload[-1]["activation"])

        model = cls(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            output_activation=output_activation,
        )
        for layer, layer_state in zip(model.layers, layer_payload, strict=True):
            layer.weights = layer_state["weights"].copy()
            layer.biases = layer_state["biases"].copy()
        return model


# Backward-compatible type alias for existing imports.
FeedForwardNetwork = NeuralNetwork


def _relu(x: Array) -> Array:
    return np.maximum(0.0, x)


def _sigmoid(x: Array) -> Array:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _apply_activation(x: Array, activation: str) -> Array:
    if activation == "relu":
        return _relu(x)
    if activation == "leaky_relu":
        return np.where(x > 0.0, x, 0.01 * x)
    if activation == "tanh":
        return np.tanh(x)
    if activation == "sigmoid":
        return _sigmoid(x)
    if activation == "linear":
        return x
    raise ValueError(f"Unsupported activation: {activation}")


def _activation_gradient(x: Array, activation: str) -> Array:
    if activation == "relu":
        return (x > 0.0).astype(np.float64)
    if activation == "leaky_relu":
        return np.where(x > 0.0, 1.0, 0.01).astype(np.float64)
    if activation == "tanh":
        t = np.tanh(x)
        return 1.0 - t * t
    if activation == "sigmoid":
        s = _sigmoid(x)
        return s * (1.0 - s)
    if activation == "linear":
        return np.ones_like(x)
    raise ValueError(f"Unsupported activation: {activation}")
