"""Managed train -> compare -> promote workflow for MuScripter Flash."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from muscriptor.flash_compare import compare_checkpoints
from muscriptor.flash_promote import evaluate_candidate
from muscriptor.flash_train import run_training


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _comparison_gate(
    report: dict[str, Any],
    *,
    min_global_gain: float,
    max_dataset_regression: float,
    max_latency_ratio: float,
) -> dict[str, Any]:
    failures: list[str] = []
    global_gain = float(report["delta"]["f1"])
    if global_gain < min_global_gain:
        failures.append(
            f"re-evaluated global F1 gain {global_gain:.4f} is below "
            f"required {min_global_gain:.4f}"
        )

    for name, metrics in sorted(report["per_dataset"].items()):
        delta = float(metrics["delta"]["f1"])
        if delta < -max_dataset_regression:
            failures.append(
                f"re-evaluated {name} F1 regressed by {-delta:.4f} "
                f"(> {max_dataset_regression:.4f})"
            )

    latency_ratio = float(report["delta"]["latency_ratio"])
    if latency_ratio > max_latency_ratio:
        failures.append(
            f"candidate CPU median latency ratio {latency_ratio:.3f} exceeds "
            f"{max_latency_ratio:.3f}"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "criteria": {
            "min_global_gain": min_global_gain,
            "max_dataset_regression": max_dataset_regression,
            "max_latency_ratio": max_latency_ratio,
        },
    }


def run_pipeline(
    *,
    workdir: str | Path,
    bach10: str | Path | None = None,
    urmp: str | Path | None = None,
    musicnet: str | Path | None = None,
    base_checkpoint: str | Path | None = None,
    steps: int = 3_000,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    real_fraction: float = 0.80,
    eval_every: int = 50,
    save_every: int = 50,
    seed: int = 20261007,
    fresh: bool = False,
    max_frames_per_dataset: int = 2_048,
    latency_windows: int = 32,
    compare_min_global_gain: float = 0.0,
    compare_max_dataset_regression: float = 0.02,
    compare_max_latency_ratio: float = 1.25,
    promote_min_global_gain: float = 0.01,
    promote_max_dataset_regression: float = 0.02,
    promote_max_p95_ms: float = 15.0,
    promote_benchmark_iterations: int = 100,
    promote_to: str | Path | None = None,
) -> dict[str, Any]:
    """Run training, fresh checkpoint comparison, and promotion gates."""
    root = Path(workdir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    comparison_path = root / "comparison.json"
    promotion_path = root / "promotion.json"
    pipeline_path = root / "pipeline.json"

    training = run_training(
        workdir=root,
        bach10=bach10,
        urmp=urmp,
        musicnet=musicnet,
        base_checkpoint=base_checkpoint,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        real_fraction=real_fraction,
        eval_every=eval_every,
        save_every=save_every,
        seed=seed,
        fresh=fresh,
    )

    candidate = root / "model.pt"
    comparison = compare_checkpoints(
        candidate,
        baseline=base_checkpoint,
        bach10=bach10,
        urmp=urmp,
        musicnet=musicnet,
        output=comparison_path,
        max_frames_per_dataset=max_frames_per_dataset,
        latency_windows=latency_windows,
    )
    comparison_gate = _comparison_gate(
        comparison,
        min_global_gain=compare_min_global_gain,
        max_dataset_regression=compare_max_dataset_regression,
        max_latency_ratio=compare_max_latency_ratio,
    )

    promotion = evaluate_candidate(
        candidate,
        min_global_gain=promote_min_global_gain,
        max_dataset_regression=promote_max_dataset_regression,
        max_p95_ms=promote_max_p95_ms,
        benchmark_iterations=promote_benchmark_iterations,
    )
    _atomic_json(promotion_path, promotion)

    gates_passed = bool(comparison_gate["passed"] and promotion["passed"])
    installed_to: str | None = None
    if promote_to is not None and gates_passed:
        destination = Path(promote_to).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
        installed_to = str(destination)

    result: dict[str, Any] = {
        "format": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workdir": str(root),
        "candidate": str(candidate),
        "training": training,
        "comparison_report": str(comparison_path),
        "comparison_gate": comparison_gate,
        "promotion_report": str(promotion_path),
        "promotion_gate": promotion,
        "passed": gates_passed,
        "promote_requested": promote_to is not None,
        "installed_to": installed_to,
    }
    _atomic_json(pipeline_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MuScripter Flash train -> compare -> promote pipeline"
    )
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--bach10")
    parser.add_argument("--urmp")
    parser.add_argument("--musicnet")
    parser.add_argument("--base")
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--real-fraction", type=float, default=0.80)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--max-frames-per-dataset", type=int, default=2_048)
    parser.add_argument("--latency-windows", type=int, default=32)
    parser.add_argument("--compare-min-global-gain", type=float, default=0.0)
    parser.add_argument("--compare-max-dataset-regression", type=float, default=0.02)
    parser.add_argument("--compare-max-latency-ratio", type=float, default=1.25)
    parser.add_argument("--promote-min-global-gain", type=float, default=0.01)
    parser.add_argument("--promote-max-dataset-regression", type=float, default=0.02)
    parser.add_argument("--promote-max-p95-ms", type=float, default=15.0)
    parser.add_argument("--promote-benchmark-iterations", type=int, default=100)
    parser.add_argument("--promote-to")
    args = parser.parse_args()

    result = run_pipeline(
        workdir=args.workdir,
        bach10=args.bach10,
        urmp=args.urmp,
        musicnet=args.musicnet,
        base_checkpoint=args.base,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        real_fraction=args.real_fraction,
        eval_every=args.eval_every,
        save_every=args.save_every,
        seed=args.seed,
        fresh=args.fresh,
        max_frames_per_dataset=args.max_frames_per_dataset,
        latency_windows=args.latency_windows,
        compare_min_global_gain=args.compare_min_global_gain,
        compare_max_dataset_regression=args.compare_max_dataset_regression,
        compare_max_latency_ratio=args.compare_max_latency_ratio,
        promote_min_global_gain=args.promote_min_global_gain,
        promote_max_dataset_regression=args.promote_max_dataset_regression,
        promote_max_p95_ms=args.promote_max_p95_ms,
        promote_benchmark_iterations=args.promote_benchmark_iterations,
        promote_to=args.promote_to,
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
