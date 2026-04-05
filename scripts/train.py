from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_network.train import accuracy, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feed-forward neural network")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_model(input_size=args.input_size, epochs=args.epochs, lr=args.lr)
    score = accuracy(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_size": args.input_size,
            "state_dict": model.state_dict(),
        },
        args.output,
    )
    print(f"Model saved to {args.output}")
    print(f"Synthetic validation accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
