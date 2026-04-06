from __future__ import annotations

import pytest
import torch
from torch import nn

from neural_network.model import FeedForwardNetwork

@pytest.mark.parametrize("batch_size", [1, 8, 32])
@pytest.mark.parametrize("input_dim", [4, 16, 128])
@pytest.mark.parametrize("hidden_dim", [32, 64])
def test_model_forward_pass(batch_size: int, input_dim: int, hidden_dim: int) -> None:
    """Tests architectural integrity across various input scales."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeedForwardNetwork(input_size=input_dim, hidden_size=hidden_dim).to(device)
    model.eval()

    # 1. Shape Validation
    x = torch.randn(batch_size, input_dim, device=device)
    with torch.inference_mode():
        logits = model(x)
    
    assert logits.shape == (batch_size, 1), f"Expected (B, 1), got {logits.shape}"

    # 2. Numerical Stability Check
    assert not torch.isnan(logits).any(), "Model produced NaN outputs"
    assert not torch.isinf(logits).any(), "Model produced Inf outputs"


def test_model_gradient_flow() -> None:
    """Verifies that the model is actually learnable (gradients reach the input)."""
    input_dim = 4
    model = FeedForwardNetwork(input_size=input_dim)
    model.train()
    
    x = torch.randn(2, input_dim, requires_grad=True)
    logits = model(x)
    loss = logits.mean()
    loss.backward()

    # Verify every weight layer has non-zero gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"
            assert torch.count_nonzero(param.grad) > 0, f"Vanishing gradient in {name}"


def test_reproducibility() -> None:
    """Ensures deterministic output when manual seeds are set."""
    torch.manual_seed(42)
    model = FeedForwardNetwork(input_size=4)
    x = torch.randn(1, 4)
    
    out1 = model(x)
    
    # Re-init same architecture with same seed
    torch.manual_seed(42)
    model_clone = FeedForwardNetwork(input_size=4)
    out2 = model_clone(x)

    assert torch.allclose(out1, out2, atol=1e-6), "Model outputs are not reproducible"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_parity() -> None:
    """Ensures CPU and GPU outputs are identical for the same weights."""
    model_cpu = FeedForwardNetwork(input_size=4)
    model_gpu = FeedForwardNetwork(input_size=4).cuda()
    model_gpu.load_state_dict(model_cpu.state_dict())
    
    x_cpu = torch.randn(1, 4)
    x_gpu = x_cpu.cuda()

    with torch.no_grad():
        out_cpu = model_cpu(x_cpu)
        out_gpu = model_gpu(x_gpu)

    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5)
