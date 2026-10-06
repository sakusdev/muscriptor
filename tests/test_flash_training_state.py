import random
from pathlib import Path

import numpy as np
import torch

from muscriptor.flash_neural import FlashNeuralNet
from muscriptor.flash_training_state import load_training_state, save_training_state


def _optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1e-4)


def test_training_state_roundtrip_restores_model_optimizer_and_rng(tmp_path: Path):
    torch.manual_seed(123)
    np.random.seed(456)
    rng = random.Random(789)
    np_rng = np.random.default_rng(321)

    model = FlashNeuralNet()
    optimizer = _optimizer(model)
    waveform = torch.randn(2, 2048)
    loss = model(waveform).square().mean()
    loss.backward()
    optimizer.step()

    config = {
        "datasets": ["MusicNet", "URMP"],
        "batch_size": 16,
        "learning_rate": 1e-4,
        "real_fraction": 0.8,
        "eval_every": 50,
    }
    state_path = tmp_path / "state.pt"
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    save_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        step=75,
        best_state=best_state,
        best_step=50,
        best_f1=0.7,
        best_threshold=0.6,
        best_metrics={"precision": 0.65, "recall": 0.76, "f1": 0.7},
        best_per_dataset={"MusicNet": {"f1": 0.68}},
        baseline_threshold=0.5,
        baseline_metrics={"precision": 0.5, "recall": 0.6, "f1": 0.55},
        baseline_per_dataset={"MusicNet": {"f1": 0.54}},
        last_loss=0.25,
        rng=rng,
        np_rng=np_rng,
        config=config,
    )

    expected_python = rng.random()
    expected_generator = float(np_rng.random())
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(()))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    rng.random()
    np_rng.random()
    np.random.random()
    torch.rand(())

    restored = load_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        rng=rng,
        np_rng=np_rng,
        expected_config=config,
    )

    assert restored["step"] == 75
    assert restored["best_step"] == 50
    assert restored["best_f1"] == 0.7
    assert restored["last_loss"] == 0.25
    for key, value in model.state_dict().items():
        assert torch.equal(value, best_state[key])

    assert rng.random() == expected_python
    assert float(np_rng.random()) == expected_generator
    assert float(np.random.random()) == expected_numpy
    assert float(torch.rand(())) == expected_torch


def test_training_state_rejects_config_mismatch(tmp_path: Path):
    model = FlashNeuralNet()
    optimizer = _optimizer(model)
    rng = random.Random(1)
    np_rng = np.random.default_rng(2)
    state_path = tmp_path / "state.pt"

    save_training_state(
        state_path,
        model=model,
        optimizer=optimizer,
        step=0,
        best_state=model.state_dict(),
        best_step=0,
        best_f1=0.0,
        best_threshold=0.5,
        best_metrics={"precision": 0.0, "recall": 0.0, "f1": 0.0},
        best_per_dataset={},
        baseline_threshold=0.5,
        baseline_metrics={"precision": 0.0, "recall": 0.0, "f1": 0.0},
        baseline_per_dataset={},
        last_loss=0.0,
        rng=rng,
        np_rng=np_rng,
        config={"datasets": ["MusicNet"], "batch_size": 16},
    )

    try:
        load_training_state(
            state_path,
            model=model,
            optimizer=optimizer,
            rng=rng,
            np_rng=np_rng,
            expected_config={"datasets": ["MusicNet"], "batch_size": 32},
        )
    except ValueError as exc:
        assert "configuration mismatch" in str(exc)
    else:
        raise AssertionError("expected training-state configuration mismatch")
