"""Quality gate for promoting MuScripter Flash multidataset checkpoints."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from muscriptor.flash_neural import FlashNeuralNet


def _benchmark_p95_ms(model: FlashNeuralNet, *, iterations: int = 100) -> float:
    if iterations <= 0:
        raise ValueError("benchmark_iterations must be positive")
    model.eval()
    sample = torch.zeros(1, 2048, dtype=torch.float32)
    with torch.inference_mode():
        for _ in range(5):
            model(sample)
        times: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    index = min(len(times) - 1, max(0, round(0.95 * (len(times) - 1))))
    return float(times[index])


def evaluate_candidate(
    checkpoint: str | Path,
    *,
    min_global_gain: float = 0.01,
    max_dataset_regression: float = 0.02,
    max_p95_ms: float = 15.0,
    benchmark_iterations: int = 100,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    model = FlashNeuralNet(window_samples=int(metadata.get("window_samples", 2048)))
    model.load_state_dict(payload["model"])

    candidate_f1 = float(metadata.get("f1", 0.0))
    baseline_f1 = float(metadata.get("baseline_f1", 0.0))
    global_gain = candidate_f1 - baseline_f1

    current_per_dataset = metadata.get("per_dataset", {}) or {}
    baseline_per_dataset = metadata.get("baseline_per_dataset", {}) or {}
    regressions: dict[str, float] = {}
    dataset_failures: list[str] = []
    for name, baseline in sorted(baseline_per_dataset.items()):
        if name not in current_per_dataset:
            dataset_failures.append(f"missing candidate metrics for {name}")
            continue
        before = float(baseline.get("f1", 0.0))
        after = float(current_per_dataset[name].get("f1", 0.0))
        regression = before - after
        regressions[name] = regression
        if regression > max_dataset_regression:
            dataset_failures.append(
                f"{name} F1 regressed by {regression:.4f} (> {max_dataset_regression:.4f})"
            )

    p95_ms = _benchmark_p95_ms(model, iterations=benchmark_iterations)
    failures: list[str] = []
    if global_gain < min_global_gain:
        failures.append(
            f"global F1 gain {global_gain:.4f} is below required {min_global_gain:.4f}"
        )
    failures.extend(dataset_failures)
    if p95_ms > max_p95_ms:
        failures.append(
            f"CPU p95 inference {p95_ms:.3f} ms exceeds {max_p95_ms:.3f} ms"
        )

    return {
        "checkpoint": str(checkpoint),
        "candidate_f1": candidate_f1,
        "baseline_f1": baseline_f1,
        "global_gain": global_gain,
        "per_dataset_regression": regressions,
        "cpu_p95_ms": p95_ms,
        "criteria": {
            "min_global_gain": min_global_gain,
            "max_dataset_regression": max_dataset_regression,
            "max_p95_ms": max_p95_ms,
        },
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate/promote a Flash checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--report")
    parser.add_argument("--install-to")
    parser.add_argument("--min-global-gain", type=float, default=0.01)
    parser.add_argument("--max-dataset-regression", type=float, default=0.02)
    parser.add_argument("--max-p95-ms", type=float, default=15.0)
    parser.add_argument("--benchmark-iterations", type=int, default=100)
    args = parser.parse_args()

    report = evaluate_candidate(
        args.checkpoint,
        min_global_gain=args.min_global_gain,
        max_dataset_regression=args.max_dataset_regression,
        max_p95_ms=args.max_p95_ms,
        benchmark_iterations=args.benchmark_iterations,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)
    if args.install_to:
        destination = Path(args.install_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.checkpoint, destination)
        print(f"promoted checkpoint: {destination}")


if __name__ == "__main__":
    main()
