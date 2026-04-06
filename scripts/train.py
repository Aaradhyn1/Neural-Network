from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Support running scripts directly without package installation.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from neural_network.train import accuracy, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NumPy neural network on parity data")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--input-size", type=int, default=4)
    parser.add_argument("--model", choices=["tiny", "parity", "deep", "llm_wide"], default="llm_wide")
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-cosine-lr", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/model.pkl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_model(
        input_size=args.input_size,
        model_preset=args.model,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_cosine_lr=not args.no_cosine_lr,
        early_stopping_patience=args.early_stopping_patience,
    )
    score = accuracy(model, bits=args.input_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        pickle.dump({"input_size": args.input_size, "state": model.state_dict()}, f)

    print(f"Model saved to {args.output}")
    print(f"Parity validation accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
