from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

# Assuming architecture from previous steps
from neural_network.model import FeedForwardNetwork

class InferenceEngine:
    """High-performance wrapper for model deployment."""
    
    def __init__(self, model_path: Path, device: str | None = None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        )
        self.model, self.metadata = self._load_resource(model_path)
        self.input_size = self.metadata["input_size"]

    def _load_resource(self, path: Path) -> tuple[FeedForwardNetwork, dict[str, Any]]:
        """Loads checkpoint with metadata and safety checks."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        
        model = FeedForwardNetwork(
            input_size=payload["input_size"],
            hidden_size=payload.get("hidden_size", 64),
            num_layers=payload.get("num_layers", 2)
        ).to(self.device)
        
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model, payload

    @torch.inference_mode()
    def predict(self, features: list[float] | list[list[float]]) -> dict[str, Any]:
        """Performs batch-aware inference with latency profiling."""
        start_time = time.perf_counter()
        
        # Normalize input to 2D tensor [Batch, Features]
        tensor_input = torch.atleast_2d(torch.as_tensor(features, dtype=torch.float32, device=self.device))
        
        if tensor_input.shape[1] != self.input_size:
            raise ValueError(f"Feature mismatch: Expected {self.input_size}, got {tensor_input.shape[1]}")

        logits = self.model(tensor_input)
        probs = torch.sigmoid(logits).squeeze(-1)
        
        # Handle single vs batch output
        results = {
            "probabilities": probs.tolist() if probs.dim() > 0 else [probs.item()],
            "classes": (probs > 0.5).int().tolist() if probs.dim() > 0 else [int(probs > 0.5)],
            "latency_ms": (time.perf_counter() - start_time) * 1000,
            "device": str(self.device)
        }
        return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Production Inference CLI")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--features", type=float, nargs="+", help="Input features")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    engine = InferenceEngine(args.model_path)

    try:
        output = engine.predict(args.features)
        
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(f"--- Inference Results ({output['device']}) ---")
            for i, (p, c) in enumerate(zip(output["probabilities"], output["classes"])):
                print(f"Sample {i}: Class {c} (Conf: {p:.4f})")
            print(f"Latency: {output['latency_ms']:.2f}ms")

    except Exception as e:
        print(f"Inference Error: {e}")

if __name__ == "__main__":
    main()
