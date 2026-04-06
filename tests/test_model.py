import numpy as np

from neural_network.model import NeuralNetwork


def test_model_output_shape() -> None:
    model = NeuralNetwork(input_size=4, hidden_sizes=(8, 8))
    x = np.random.default_rng(0).normal(size=(8, 4))
    y = model.forward_propagation(x)
    assert y.shape == (8, 1)


def test_tanh_output_activation_supported() -> None:
    model = NeuralNetwork(input_size=4, hidden_sizes=(8,), output_activation="tanh")
    x = np.random.default_rng(1).normal(size=(3, 4))
    y = model.forward_propagation(x)
    assert y.shape == (3, 1)
    assert np.all(y <= 1.0) and np.all(y >= -1.0)
