import torch

from neural_network.model import FeedForwardNetwork


def test_model_output_shape() -> None:
    model = FeedForwardNetwork(input_size=4)
    x = torch.randn(8, 4)
    y = model(x)
    assert y.shape == (8, 1)
