from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

from neural_network.model import FeedForwardNetwork

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Inference Engine")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        required=True,
        help="Space-separated feature values (e.g., 1.2 -0.5 0.8 2.1)",
    )
    return parser.parse_args()

def load_inference_model(path: Path, device: torch.device) -> FeedForwardNetwork:
    """Loads model and metadata with safety checks."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # weights_only=True is a security best practice for loading pickles
    payload = torch.load(path, map_location=device, weights_only=True)
    
    model = FeedForwardNetwork(
        input_size=payload["input_size"],
        hidden_size=payload.get("hidden_size", 64),
        num_layers=payload.get("num_layers", 2)
    ).to(device)
    
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

@torch.inference_mode()
def main() -> None:
    args = parse_args()
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 2. Prepare Model
        model = load_inference_model(args.model_path, device)
        input_size = model.stem[0].in_features # Introspect architecture for validation

        # 3. Validate and Tensorize Input
        if len(args.features) != input_size:
            raise ValueError(f"Input mismatch: model needs {input_size} features, got {len(args.features)}")

        # Create batch of 1 on the correct device
        x = torch.tensor([args.features], dtype=torch.float32, device=device)

        # 4. Perform Inference
        logits = model(x)
        prob = torch.sigmoid(logits).item()
        pred_class = int(prob > 0.5)

        # 5. Professional Output
        print("\n--- Inference Result ---")
        print(f"Confidence: {prob:.2%}")
        print(f"Category:   {'Positive (1)' if pred_class else 'Negative (0)'}")
        print("------------------------\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

