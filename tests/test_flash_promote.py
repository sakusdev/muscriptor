from pathlib import Path

import torch

from muscriptor.flash_neural import FlashNeuralNet
from muscriptor.flash_promote import evaluate_candidate


def _checkpoint(
    path: Path, *, candidate_f1: float, baseline_f1: float, after: float
) -> None:
    model = FlashNeuralNet(window_samples=2048)
    metadata = {
        "window_samples": 2048,
        "f1": candidate_f1,
        "baseline_f1": baseline_f1,
        "baseline_per_dataset": {"MusicNet": {"f1": 0.60}},
        "per_dataset": {"MusicNet": {"f1": after}},
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)


def test_promotion_gate_passes_improved_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / "candidate.pt"
    _checkpoint(checkpoint, candidate_f1=0.72, baseline_f1=0.68, after=0.61)

    report = evaluate_candidate(
        checkpoint,
        min_global_gain=0.01,
        max_dataset_regression=0.02,
        max_p95_ms=10_000.0,
        benchmark_iterations=2,
    )

    assert report["passed"] is True
    assert report["global_gain"] > 0.03
    assert report["failures"] == []


def test_promotion_gate_rejects_dataset_regression(tmp_path: Path):
    checkpoint = tmp_path / "candidate.pt"
    _checkpoint(checkpoint, candidate_f1=0.75, baseline_f1=0.68, after=0.55)

    report = evaluate_candidate(
        checkpoint,
        min_global_gain=0.01,
        max_dataset_regression=0.02,
        max_p95_ms=10_000.0,
        benchmark_iterations=2,
    )

    assert report["passed"] is False
    assert any("MusicNet" in failure for failure in report["failures"])


def test_promotion_gate_rejects_missing_global_gain(tmp_path: Path):
    checkpoint = tmp_path / "candidate.pt"
    _checkpoint(checkpoint, candidate_f1=0.685, baseline_f1=0.68, after=0.61)

    report = evaluate_candidate(
        checkpoint,
        min_global_gain=0.01,
        max_dataset_regression=0.02,
        max_p95_ms=10_000.0,
        benchmark_iterations=2,
    )

    assert report["passed"] is False
    assert any("global F1 gain" in failure for failure in report["failures"])
