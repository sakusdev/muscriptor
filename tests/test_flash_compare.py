from pathlib import Path

import torch

from muscriptor import flash_compare
from muscriptor.flash_multidataset import TimedPiece
from muscriptor.flash_neural import FlashNeuralNet


def _checkpoint(path: Path, *, bias: float = 0.0) -> None:
    model = FlashNeuralNet()
    with torch.no_grad():
        model.classifier[-1].bias.fill_(bias)
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {
                "sample_rate": 16000,
                "window_samples": 2048,
                "min_midi": 21,
                "max_midi": 108,
                "backend": "test",
            },
        },
        path,
    )


def _piece(name: str) -> TimedPiece:
    labels = torch.zeros((8, 88), dtype=torch.float32)
    labels[:, 48] = 1.0
    audio = torch.zeros(4096, dtype=torch.float32)
    return TimedPiece(
        name=name,
        dataset="Fixture",
        audio=audio,
        labels=labels,
        frame_origin_seconds=0.128,
        frame_hop_seconds=0.010,
    )


def test_compare_report_uses_same_validation_sets(monkeypatch, tmp_path: Path):
    baseline = tmp_path / "baseline.pt"
    candidate = tmp_path / "candidate.pt"
    report_path = tmp_path / "report.json"
    _checkpoint(baseline)
    _checkpoint(candidate, bias=0.2)

    validation = {"Fixture": [_piece("heldout")]}
    monkeypatch.setattr(
        flash_compare,
        "_split_datasets",
        lambda **kwargs: ({"Fixture": [_piece("train")]}, validation),
    )
    monkeypatch.setattr(
        flash_compare,
        "_benchmark_cpu",
        lambda model, windows: {
            "median_ms_per_window": 1.0,
            "min_ms_per_window": 0.9,
            "max_ms_per_window": 1.1,
            "batch_windows": int(windows.shape[0]),
            "repeats": 12,
        },
    )

    report = flash_compare.compare_checkpoints(
        candidate,
        baseline=baseline,
        bach10=tmp_path,
        output=report_path,
        max_frames_per_dataset=8,
        latency_windows=2,
    )

    assert report["evaluation"]["datasets"] == ["Fixture"]
    assert report["baseline"]["path"] == str(baseline.resolve())
    assert report["candidate"]["path"] == str(candidate.resolve())
    assert "Fixture" in report["per_dataset"]
    assert report_path.is_file()


def test_checkpoint_geometry_mismatch_is_rejected(tmp_path: Path):
    path = tmp_path / "bad.pt"
    model = FlashNeuralNet()
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {
                "sample_rate": 44100,
                "window_samples": 2048,
                "min_midi": 21,
                "max_midi": 108,
            },
        },
        path,
    )

    try:
        flash_compare._load_checkpoint(path)
    except ValueError as exc:
        assert "geometry mismatch" in str(exc)
    else:
        raise AssertionError("expected checkpoint geometry mismatch")


def test_representative_windows_returns_requested_count():
    windows = flash_compare._representative_windows(
        {"Fixture": [_piece("a"), _piece("b")]}, count=5
    )

    assert windows.shape == (5, 2048)
