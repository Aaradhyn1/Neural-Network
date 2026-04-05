from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_network.model import FeedForwardNetwork
from neural_network.train import accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.model_path, map_location="cpu")
    model = FeedForwardNetwork(input_size=payload["input_size"])
    model.load_state_dict(payload["state_dict"])

    score = accuracy(model)
    print(f"Accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
