"""Checkpoint state helpers for long MuScripter Flash training runs."""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

_STATE_FORMAT = 1


def save_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_state: dict[str, torch.Tensor],
    best_step: int,
    best_f1: float,
    best_threshold: float,
    best_metrics: dict[str, float],
    best_per_dataset: dict[str, dict[str, float]],
    baseline_threshold: float,
    baseline_metrics: dict[str, float],
    baseline_per_dataset: dict[str, dict[str, float]],
    last_loss: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    config: dict[str, Any],
) -> Path:
    """Atomically save everything required to resume a training loop."""
    if step < 0:
        raise ValueError("step must be non-negative")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")

    payload: dict[str, Any] = {
        "format": _STATE_FORMAT,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_state": copy.deepcopy(best_state),
        "best_step": int(best_step),
        "best_f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "best_metrics": copy.deepcopy(best_metrics),
        "best_per_dataset": copy.deepcopy(best_per_dataset),
        "baseline_threshold": float(baseline_threshold),
        "baseline_metrics": copy.deepcopy(baseline_metrics),
        "baseline_per_dataset": copy.deepcopy(baseline_per_dataset),
        "last_loss": float(last_loss),
        "config": copy.deepcopy(config),
        "rng": {
            "python": rng.getstate(),
            "numpy_generator": copy.deepcopy(np_rng.bit_generator.state),
            "numpy_global": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rng: random.Random,
    np_rng: np.random.Generator,
    expected_config: dict[str, Any],
) -> dict[str, Any]:
    """Load a saved loop state and restore model, optimizer, and RNG streams."""
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if int(payload.get("format", -1)) != _STATE_FORMAT:
        raise ValueError(
            f"unsupported Flash training-state format: {payload.get('format')!r}"
        )

    actual_config = payload.get("config")
    if actual_config != expected_config:
        raise ValueError(
            "training-state configuration mismatch: "
            f"saved={actual_config!r}, requested={expected_config!r}"
        )

    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])

    rng_state = payload["rng"]
    rng.setstate(rng_state["python"])
    np_rng.bit_generator.state = copy.deepcopy(rng_state["numpy_generator"])
    np.random.set_state(rng_state["numpy_global"])
    torch.set_rng_state(rng_state["torch"])
    return payload
