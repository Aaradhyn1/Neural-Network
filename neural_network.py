"""Simple feedforward neural network implemented with NumPy.

This module provides a minimal neural network class with ReLU hidden layers
and a sigmoid output layer, plus a small XOR training demo.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable, List, Tuple

import numpy as np


@dataclasses.dataclass
class LayerCache:
    input_data: np.ndarray
    weighted_sum: np.ndarray
    activation: np.ndarray


class NeuralNetwork:
    """A simple fully connected neural network for binary classification."""

    def __init__(
        self,
        layer_sizes: Iterable[int],
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.layer_sizes = list(layer_sizes)
        if len(self.layer_sizes) < 2:
            raise ValueError("layer_sizes must contain at least input and output sizes.")
        self.learning_rate = learning_rate
        self.random_state = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        self.weights.clear()
        self.biases.clear()
        for input_size, output_size in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weight = self.random_state.normal(0.0, 1.0, size=(input_size, output_size))
            bias = np.zeros((1, output_size))
            self.weights.append(weight / np.sqrt(input_size))
            self.biases.append(bias)

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, values)

    @staticmethod
    def _relu_derivative(values: np.ndarray) -> np.ndarray:
        return (values > 0).astype(values.dtype)

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    @staticmethod
    def _sigmoid_derivative(values: np.ndarray) -> np.ndarray:
        sigmoid_values = NeuralNetwork._sigmoid(values)
        return sigmoid_values * (1.0 - sigmoid_values)

    def forward(self, inputs: np.ndarray) -> Tuple[np.ndarray, List[LayerCache]]:
        """Run a forward pass and return output and caches for backprop."""
        activation = inputs
        caches: List[LayerCache] = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weighted_sum = activation @ weight + bias
            if index < len(self.weights) - 1:
                activation = self._relu(weighted_sum)
            else:
                activation = self._sigmoid(weighted_sum)
            caches.append(LayerCache(inputs, weighted_sum, activation))
            inputs = activation
        return activation, caches

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return float(loss)

    def train(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 1000,
        report_every: int = 100,
    ) -> List[float]:
        """Train the network with batch gradient descent."""
        losses: List[float] = []
        for epoch in range(1, epochs + 1):
            predictions, caches = self.forward(inputs)
            loss = self._binary_cross_entropy(targets, predictions)
            losses.append(loss)

            gradients_w, gradients_b = self._backward(inputs, targets, caches)
            self._apply_gradients(gradients_w, gradients_b)

            if report_every and epoch % report_every == 0:
                print(f"Epoch {epoch}: loss={loss:.4f}")
        return losses

    def _backward(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        caches: List[LayerCache],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        gradients_w: List[np.ndarray] = []
        gradients_b: List[np.ndarray] = []

        activation = caches[-1].activation
        error = (activation - targets) / targets.shape[0]

        for layer_index in reversed(range(len(self.weights))):
            cache = caches[layer_index]
            if layer_index == len(self.weights) - 1:
                delta = error * self._sigmoid_derivative(cache.weighted_sum)
            else:
                delta = error * self._relu_derivative(cache.weighted_sum)

            previous_activation = inputs if layer_index == 0 else caches[layer_index - 1].activation
            grad_w = previous_activation.T @ delta
            grad_b = np.sum(delta, axis=0, keepdims=True)

            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)

            error = delta @ self.weights[layer_index].T

        return gradients_w, gradients_b

    def _apply_gradients(
        self,
        gradients_w: List[np.ndarray],
        gradients_b: List[np.ndarray],
    ) -> None:
        for index, (grad_w, grad_b) in enumerate(zip(gradients_w, gradients_b)):
            self.weights[index] -= self.learning_rate * grad_w
            self.biases[index] -= self.learning_rate * grad_b

    def predict(self, inputs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return class predictions for the inputs."""
        probabilities, _ = self.forward(inputs)
        return (probabilities >= threshold).astype(int)


def _xor_demo() -> None:
    inputs = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    targets = np.array([[0.0], [1.0], [1.0], [0.0]])

    network = NeuralNetwork(layer_sizes=[2, 4, 1], learning_rate=0.5, seed=42)
    network.train(inputs, targets, epochs=2000, report_every=500)

    predictions = network.predict(inputs)
    print("Predictions:")
    for input_row, prediction in zip(inputs, predictions):
        print(f"{input_row} -> {prediction[0]}")


if __name__ == "__main__":
    _xor_demo()
