"""Compare a candidate MuScripter Flash checkpoint against a baseline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from muscriptor.flash_multidataset import (
    _WINDOW_SAMPLES,
    _causal_window,
    _evaluate,
    _split_datasets,
)
from muscriptor.flash_neural import FlashNeuralNet

_EXPECTED_GEOMETRY = (16_000, 2_048, 21, 108)


def default_baseline_checkpoint() -> Path:
    return Path(__file__).resolve().parent / "checkpoints" / "flash-neural-bach10.pt"


def _load_checkpoint(path: str | Path) -> tuple[FlashNeuralNet, dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    geometry = (
        int(metadata.get("sample_rate", 16_000)),
        int(metadata.get("window_samples", _WINDOW_SAMPLES)),
        int(metadata.get("min_midi", 21)),
        int(metadata.get("max_midi", 108)),
    )
    if geometry != _EXPECTED_GEOMETRY:
        raise ValueError(f"checkpoint geometry mismatch for {path}: {geometry}")

    model = FlashNeuralNet(window_samples=_WINDOW_SAMPLES)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, metadata


def _representative_windows(
    datasets: dict[str, list[Any]],
    *,
    count: int = 32,
) -> torch.Tensor:
    examples: list[torch.Tensor] = []
    pieces = [piece for name in sorted(datasets) for piece in datasets[name]]
    if not pieces:
        raise ValueError("no validation pieces available for latency benchmark")

    piece_index = 0
    while len(examples) < count:
        piece = pieces[piece_index % len(pieces)]
        frame_count = int(piece.labels.shape[0])
        if frame_count:
            fraction = ((len(examples) * 37) % 101) / 100.0
            frame_index = min(frame_count - 1, int(fraction * frame_count))
            examples.append(_causal_window(piece, frame_index))
        piece_index += 1
        if piece_index > len(pieces) * max(2, count):
            break

    if not examples:
        raise ValueError("validation datasets contain no frames")
    return torch.stack(examples)


def _benchmark_cpu(
    model: FlashNeuralNet,
    windows: torch.Tensor,
    *,
    warmup: int = 4,
    repeats: int = 12,
) -> dict[str, float]:
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    with torch.inference_mode():
        for _ in range(warmup):
            model(windows)

        timings_ms: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(windows)
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            timings_ms.append(elapsed_ms / windows.shape[0])

    return {
        "median_ms_per_window": float(statistics.median(timings_ms)),
        "min_ms_per_window": float(min(timings_ms)),
        "max_ms_per_window": float(max(timings_ms)),
        "batch_windows": int(windows.shape[0]),
        "repeats": repeats,
    }


def _metric_delta(
    candidate: dict[str, float], baseline: dict[str, float]
) -> dict[str, float]:
    return {
        key: float(candidate[key] - baseline[key])
        for key in ("precision", "recall", "f1")
    }


def compare_checkpoints(
    candidate: str | Path,
    *,
    baseline: str | Path | None = None,
    bach10: str | Path | None = None,
    urmp: str | Path | None = None,
    musicnet: str | Path | None = None,
    output: str | Path | None = None,
    max_frames_per_dataset: int = 2_048,
    latency_windows: int = 32,
) -> dict[str, Any]:
    """Evaluate candidate and baseline on identical held-out dataset frames."""
    if max_frames_per_dataset <= 0 or latency_windows <= 0:
        raise ValueError("frame and latency sample counts must be positive")

    candidate_path = Path(candidate).expanduser().resolve()
    baseline_path = (
        Path(baseline or default_baseline_checkpoint()).expanduser().resolve()
    )
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)

    _, validation_sets = _split_datasets(
        bach10=bach10,
        urmp=urmp,
        musicnet=musicnet,
    )
    candidate_model, candidate_metadata = _load_checkpoint(candidate_path)
    baseline_model, baseline_metadata = _load_checkpoint(baseline_path)

    baseline_threshold, baseline_metrics, baseline_per_dataset = _evaluate(
        baseline_model,
        validation_sets,
        max_frames_per_dataset=max_frames_per_dataset,
    )
    candidate_threshold, candidate_metrics, candidate_per_dataset = _evaluate(
        candidate_model,
        validation_sets,
        max_frames_per_dataset=max_frames_per_dataset,
    )

    windows = _representative_windows(validation_sets, count=latency_windows)
    baseline_latency = _benchmark_cpu(baseline_model, windows)
    candidate_latency = _benchmark_cpu(candidate_model, windows)

    per_dataset: dict[str, Any] = {}
    for name in sorted(validation_sets):
        baseline_dataset = baseline_per_dataset[name]
        candidate_dataset = candidate_per_dataset[name]
        per_dataset[name] = {
            "baseline": baseline_dataset,
            "candidate": candidate_dataset,
            "delta": _metric_delta(candidate_dataset, baseline_dataset),
        }

    global_delta = _metric_delta(candidate_metrics, baseline_metrics)
    latency_ratio = candidate_latency["median_ms_per_window"] / max(
        baseline_latency["median_ms_per_window"], 1e-9
    )
    regressions = [
        name for name, metrics in per_dataset.items() if metrics["delta"]["f1"] < 0.0
    ]

    report: dict[str, Any] = {
        "format": 1,
        "candidate": {
            "path": str(candidate_path),
            "metadata": candidate_metadata,
            "decision_threshold": candidate_threshold,
            "metrics": candidate_metrics,
            "latency": candidate_latency,
        },
        "baseline": {
            "path": str(baseline_path),
            "metadata": baseline_metadata,
            "decision_threshold": baseline_threshold,
            "metrics": baseline_metrics,
            "latency": baseline_latency,
        },
        "delta": {
            **global_delta,
            "latency_ratio": float(latency_ratio),
            "latency_ms_per_window": float(
                candidate_latency["median_ms_per_window"]
                - baseline_latency["median_ms_per_window"]
            ),
        },
        "per_dataset": per_dataset,
        "dataset_f1_regressions": regressions,
        "evaluation": {
            "max_frames_per_dataset": max_frames_per_dataset,
            "latency_windows": latency_windows,
            "datasets": sorted(validation_sets),
            "note": (
                "This compares checkpoints on the trainer's configured holdout sets. "
                "If those holdouts influenced checkpoint selection, this is a regression "
                "report rather than an independent final test."
            ),
        },
    }

    if output is not None:
        output_path = Path(output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp = output_path.with_suffix(output_path.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp.replace(output_path)

    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a MuScripter Flash candidate checkpoint to a baseline"
    )
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--bach10")
    parser.add_argument("--urmp")
    parser.add_argument("--musicnet")
    parser.add_argument("--output")
    parser.add_argument("--max-frames-per-dataset", type=int, default=2_048)
    parser.add_argument("--latency-windows", type=int, default=32)
    args = parser.parse_args()

    compare_checkpoints(
        args.candidate,
        baseline=args.baseline,
        bach10=args.bach10,
        urmp=args.urmp,
        musicnet=args.musicnet,
        output=args.output,
        max_frames_per_dataset=args.max_frames_per_dataset,
        latency_windows=args.latency_windows,
    )


if __name__ == "__main__":
    main()
