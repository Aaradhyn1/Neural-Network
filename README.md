# Neural-Network (Python + NumPy)

A production-style, from-scratch Deep Neural Network implementation using only **NumPy**.

## Highlights

- **Modular architecture** with reusable `Layer` and `NeuralNetwork` classes.
- **Model library presets** (`tiny`, `parity`, `deep`, `llm_wide`) for broader/wider architectures.
- **He initialization** for ReLU/Leaky-ReLU hidden layers.
- **ReLU + Sigmoid** default stack for binary classification.
- **Forward propagation + backpropagation** with explicit matrix calculus.
- **Optimizers:** SGD and Adam.
- **Training stability improvements:** optional gradient clipping, L2 weight decay, cosine LR decay.
- **Loss:** Binary Cross-Entropy (BCE).
- **Training logs** every 100 epochs.
- **Demo task:** 4-bit parity (a non-linear classification problem).

## Project Structure

```text
.
├── src/neural_network/model.py          # Layer + NeuralNetwork implementation
├── src/neural_network/model_library.py  # Named model presets
├── src/neural_network/data.py           # 4-bit parity dataset + batching
├── src/neural_network/train.py          # BCE, training loop, demo
├── scripts/train.py                     # CLI training entrypoint
├── scripts/evaluate.py                  # CLI evaluation
├── scripts/predict.py                   # CLI prediction
└── tests/                               # Unit tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
python scripts/train.py --model llm_wide --epochs 1200 --optimizer adam --output checkpoints/model.pkl
python scripts/evaluate.py --model-path checkpoints/model.pkl
python scripts/predict.py --model-path checkpoints/model.pkl --features 1 0 1 0
```

## Notes on Backpropagation

For a dense layer with input `X`, weights `W`, biases `b`, and pre-activation `Z = XW + b`:

- `dW = X^T · dZ / m`
- `db = sum(dZ) / m`
- `dX = dZ · W^T`

These transposes ensure shape alignment while applying the chain rule layer-by-layer.
