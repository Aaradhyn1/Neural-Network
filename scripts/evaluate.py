from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from neural_network.data import make_parity_data
from neural_network.metrics import accuracy_score, precision_recall_f1
from neural_network.model import FeedForwardNetwork
from neural_network.preprocessing import minmax_normalize


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

    x_eval, y_eval = make_parity_data(bits=input_size, repeats=16, seed=999)
    x_eval = minmax_normalize(x_eval)
    preds = model.predict(x_eval)

    acc = accuracy_score(y_eval, preds)
    precision, recall, f1 = precision_recall_f1(y_eval, preds)
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")


if __name__ == "__main__":
    main()
