from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

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
    with args.model_path.open("rb") as f:
        payload = pickle.load(f)

    input_size = int(payload["input_size"])
    if len(args.features) != input_size:
        raise ValueError(f"Expected {input_size} features, got {len(args.features)}")

    model = FeedForwardNetwork.from_state_dict(input_size=input_size, state=payload["state"])
    x = np.array([args.features], dtype=np.float64)
    prob = float(model.predict_proba(x)[0, 0])

    print(f"Predicted probability: {prob:.4f}")
    print(f"Predicted class: {int(prob >= 0.5)}")


if __name__ == "__main__":
    main()
