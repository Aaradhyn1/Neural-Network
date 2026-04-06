# Neural-Network (Python + NumPy)

A production-style, from-scratch Deep Neural Network implementation using only **NumPy**.

## Highlights

- **Modular architecture** with reusable `Layer` and `NeuralNetwork` classes.
- **Model library presets** (`tiny`, `parity`, `deep`, `llm_wide`) for broader/wider architectures.
- **He initialization** for ReLU/Leaky-ReLU hidden layers.
- **ReLU + Sigmoid** default stack for binary classification.
- **Forward propagation + backpropagation** with explicit matrix calculus.
- **Optimizers:** SGD and Adam.
- **Training stability improvements:** gradient clipping, L2 weight decay, cosine LR decay, and early stopping.
- **Evaluation metrics:** accuracy, precision, recall, and F1.
- **Data prep utilities:** missing-value imputation, min-max normalization, and train/val/test splitting.

## 8-Step Workflow (Implemented)

1. **Define problem**: binary classification target (parity/churn-style yes/no).  
2. **Collect & preprocess data**: use `preprocessing.py` for imputation + scaling.  
3. **Prepare train/val/test splits**: `train_val_test_split` utility.  
4. **Initialize parameters**: He/Xavier-style random initialization in model layers.  
5. **Forward propagation**: dense linear transforms + activation functions.  
6. **Cost function**: binary cross-entropy.  
7. **Training**: backprop + Adam/SGD + regularization + LR scheduling.  
8. **Evaluation**: accuracy/precision/recall/F1 on held-out data.  

## Project Structure

```text
.
├── src/neural_network/model.py          # Layer + NeuralNetwork implementation
├── src/neural_network/model_library.py  # Named model presets
├── src/neural_network/preprocessing.py  # Impute/normalize/split utilities
├── src/neural_network/metrics.py        # Accuracy/precision/recall/F1
├── src/neural_network/data.py           # 4-bit parity dataset + batching
├── src/neural_network/train.py          # BCE, training loop, early stopping
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
