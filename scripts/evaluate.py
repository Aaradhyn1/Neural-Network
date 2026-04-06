from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from neural_network.model import FeedForwardNetwork
from neural_network.train import accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.model_path.open("rb") as f:
        payload = pickle.load(f)

    input_size = int(payload["input_size"])
    model = FeedForwardNetwork.from_state_dict(input_size=input_size, state=payload["state"])
    score = accuracy(model, bits=input_size)
    print(f"Accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
