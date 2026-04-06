from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn

# Assuming these are in your local package
from neural_network.model import FeedForwardNetwork
from neural_network.train import evaluate_model # Updated name for clarity

# Configure professional logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced Model Evaluation Utility")
    parser.add_argument(
        "--model-path", 
        type=Path, 
        required=True, 
        help="Path to the saved .pt or .pth checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cuda", "cpu", "mps"], 
        default=None,
        help="Target device (auto-detected if not specified)"
    )
    return parser.parse_args()

def load_resource(path: Path, device: torch.device) -> nn.Module:
    """Safely loads a model checkpoint and handles hardware mapping."""
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
        
    logger.info(f"Loading checkpoint from {path} to {device}...")
    payload = torch.load(path, map_location=device, weights_only=True)
    
    # Reconstruct model from saved metadata
    model = FeedForwardNetwork(
        input_size=payload["input_size"],
        hidden_size=payload.get("hidden_size", 64),
        num_layers=payload.get("num_layers", 2)
    ).to(device)
    
    model.load_state_dict(payload["state_dict"])
    model.eval() # Set to evaluation mode (disables dropout, etc.)
    return model

@torch.inference_mode() # Faster and more memory-efficient than no_grad()
def main() -> None:
    args = parse_args()
    
    # 1. Hardware Detection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )

    try:
        # 2. Model Prep
        model = load_resource(args.model_path, device)
        
        # 3. High-Performance Evaluation
        logger.info("Starting evaluation...")
        score = evaluate_model(model, device=device)
        
        # 4. Result Formatting
        print("\n" + "="*30)
        print(f"DEVICE:   {device.type.upper()}")
        print(f"ACCURACY: {score:.4%}") # Formats as 98.23%
        print("="*30 + "\n")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
