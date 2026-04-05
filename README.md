# Neural-Network (Python)

A clean starter repository for training, evaluating, and running a feed-forward neural network with PyTorch.

## Project Structure

```text
.
├── .github/workflows/ci.yml      # Lint + test in GitHub Actions
├── src/neural_network/            # Reusable Python package
├── scripts/                       # CLI scripts (train, evaluate, predict)
├── tests/                         # Unit tests
├── data/                          # Data directory (keep your datasets here)
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Dev dependencies
├── pyproject.toml                 # Tooling and packaging config
└── README.md
```

## Quick Start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

2. Train a demo model on synthetic data:

```bash
python scripts/train.py --epochs 40 --output checkpoints/model.pt
```

3. Evaluate the model:

```bash
python scripts/evaluate.py --model-path checkpoints/model.pt
```

4. Run prediction:

```bash
python scripts/predict.py --model-path checkpoints/model.pt --features 0.1 0.2 -0.3 0.4
```

## Using Your Own Files/Data

- Put your dataset files into `data/`.
- Extend `src/neural_network/data.py` to load your specific format.
- Keep your custom models in `src/neural_network/model.py` or add additional modules.
- Keep scripts in `scripts/` for reproducible training and inference workflows.

## GitHub Setup

This repository includes:
- a Python-focused `.gitignore`
- formatting/lint/test configuration in `pyproject.toml`
- a CI workflow at `.github/workflows/ci.yml`

Push this branch to GitHub and Actions will run automatically on push/PR.
