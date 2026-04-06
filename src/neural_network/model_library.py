from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Named architecture preset for quick model selection."""

    hidden_sizes: tuple[int, ...]
    description: str


MODEL_LIBRARY: dict[str, ModelSpec] = {
    "tiny": ModelSpec(hidden_sizes=(16, 8), description="Small baseline for fast iteration."),
    "parity": ModelSpec(hidden_sizes=(64, 64, 32), description="Wider default for 4-bit parity."),
    "deep": ModelSpec(
        hidden_sizes=(128, 128, 64, 32),
        description="Deep + wide stack for harder non-linear toy datasets.",
    ),
    "llm_wide": ModelSpec(
        hidden_sizes=(256, 256, 128, 64),
        description="Extra-wide MLP preset with improved optimization defaults.",
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    """Return a model preset by name."""
    try:
        return MODEL_LIBRARY[name]
    except KeyError as exc:
        options = ", ".join(sorted(MODEL_LIBRARY.keys()))
        raise ValueError(f"Unknown model preset '{name}'. Available presets: {options}") from exc
