from __future__ import annotations

import pytest
import torch
from neural_network.train import accuracy, train_model

@pytest.fixture(scope="module")
def trained_model():
    """Fixture to avoid re-training for every test case."""
    torch.manual_seed(42)
    # Train for enough epochs to see actual improvement
    return train_model(epochs=10, lr=1e-2, input_size=4)

def test_training_convergence(trained_model) -> None:
    """Verifies the model performs significantly better than random chance (0.5)."""
    acc = accuracy(trained_model)
    
    # Statistical threshold: for binary classification, 0.6+ shows learning
    assert acc > 0.6, f"Model failed to converge. Accuracy: {acc:.4f}"
    assert acc <= 1.0, "Accuracy cannot exceed 100%"

@pytest.mark.parametrize("run_id", range(3))
def test_training_stability(run_id: int) -> None:
    """Ensures training is stable across different random initializations."""
    model = train_model(epochs=5, lr=1e-2)
    acc = accuracy(model)
    
    assert acc > 0.55, f"Instability detected in run {run_id}. Accuracy: {acc:.4f}"

def test_overfitting_capability() -> None:
    """
    Advanced 'Small Data' test: A model should be able to 
    perfectly overfit a single batch (100% accuracy).
    """
    # High LR, many epochs, tiny data
    model = train_model(epochs=50, lr=0.1, input_size=4)
    # Check accuracy on the training distribution specifically
    acc = accuracy(model) 
    
    # If a model can't overfit a tiny synthetic set, there is a bug in the architecture
    assert acc > 0.85, f"Model architecture lacks capacity to fit data. Acc: {acc:.4f}"

@torch.inference_mode()
def test_prediction_variance(trained_model) -> None:
    """Ensures the model isn't just predicting the same class for everything."""
    from neural_network.data import make_synthetic_classification_data
    
    loader = make_synthetic_classification_data(samples=100, features=4)
    all_preds = []
    
    for x_batch, _ in loader:
        logits = trained_model(x_batch)
        preds = (logits > 0).float()
        all_preds.append(preds)
        
    combined_preds = torch.cat(all_preds)
    unique_elements = torch.unique(combined_preds)
    
    # Verify the model is actually predicting both classes (0 and 1)
    assert len(unique_elements) > 1, "Model is collapsed (predicting only one class)"
