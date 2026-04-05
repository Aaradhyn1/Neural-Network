from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_network.model import FeedForwardNetwork


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict using trained model")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=float,
        nargs="+",
        required=True,
        help="Feature values in the same order used during training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.model_path, map_location="cpu")
    input_size = payload["input_size"]

    if len(args.features) != input_size:
        raise ValueError(f"Expected {input_size} features, got {len(args.features)}")

    model = FeedForwardNetwork(input_size=input_size)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    x = torch.tensor([args.features], dtype=torch.float32)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    print(f"Predicted probability: {prob:.4f}")
    print(f"Predicted class: {int(prob > 0.5)}")


if __name__ == "__main__":
    main()
