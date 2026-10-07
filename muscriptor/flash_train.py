"""Convenience runner for long MuScripter Flash real-data training jobs."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from muscriptor.flash_multidataset import finetune_multidataset


@dataclass(frozen=True)
class TrainingRunPaths:
    root: Path
    model: Path
    metrics: Path
    state: Path
    manifest: Path


def default_base_checkpoint() -> Path:
    """Return the bundled Bach10-adapted Flash checkpoint."""
    return Path(__file__).resolve().parent / "checkpoints" / "flash-neural-bach10.pt"


def resolve_run_paths(workdir: str | Path) -> TrainingRunPaths:
    root = Path(workdir).expanduser().resolve()
    return TrainingRunPaths(
        root=root,
        model=root / "model.pt",
        metrics=root / "metrics.json",
        state=root / "training-state.pt",
        manifest=root / "run.json",
    )


def _dataset_path(value: str | Path | None, name: str) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")
    return path


def _backup_existing_state(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_name(f"{path.stem}.{stamp}.bak{path.suffix}")
    shutil.move(path, backup)
    return backup


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def run_training(
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
) -> dict[str, Any]:
    """Run or resume one managed Flash multi-dataset training job.

    The work directory owns all durable outputs. If ``training-state.pt`` exists,
    the run resumes automatically unless ``fresh=True``. A fresh run preserves
    the previous state as a timestamped backup instead of silently deleting it.
    """
    datasets = {
        "bach10": _dataset_path(bach10, "Bach10"),
        "urmp": _dataset_path(urmp, "URMP"),
        "musicnet": _dataset_path(musicnet, "MusicNet"),
    }
    if not any(datasets.values()):
        raise ValueError("provide at least one of bach10, urmp, or musicnet")

    paths = resolve_run_paths(workdir)
    paths.root.mkdir(parents=True, exist_ok=True)

    base = Path(base_checkpoint or default_base_checkpoint()).expanduser().resolve()
    if not base.is_file():
        raise FileNotFoundError(f"base checkpoint not found: {base}")

    backup: Path | None = None
    if fresh:
        backup = _backup_existing_state(paths.state)
    resume_state = paths.state if paths.state.is_file() else None

    manifest: dict[str, Any] = {
        "format": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "workdir": str(paths.root),
        "base_checkpoint": str(base),
        "datasets": {
            name: str(path) if path is not None else None
            for name, path in datasets.items()
        },
        "training": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "real_fraction": real_fraction,
            "eval_every": eval_every,
            "save_every": save_every,
            "seed": seed,
        },
        "resume_state": str(resume_state) if resume_state else None,
        "fresh": fresh,
        "backed_up_state": str(backup) if backup else None,
        "outputs": {
            "model": str(paths.model),
            "metrics": str(paths.metrics),
            "state": str(paths.state),
        },
        "status": "running",
    }
    _write_manifest(paths.manifest, manifest)

    print(f"Flash training workdir: {paths.root}", flush=True)
    print(f"Base checkpoint: {base}", flush=True)
    if resume_state is not None:
        print(f"Auto-resume: {resume_state}", flush=True)
    elif backup is not None:
        print(f"Fresh run; previous state backed up to: {backup}", flush=True)
    else:
        print("Fresh run; no previous state found", flush=True)

    try:
        result = finetune_multidataset(
            base,
            paths.model,
            bach10=datasets["bach10"],
            urmp=datasets["urmp"],
            musicnet=datasets["musicnet"],
            metrics_path=paths.metrics,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            real_fraction=real_fraction,
            eval_every=eval_every,
            seed=seed,
            resume_state=resume_state,
            state_output=paths.state,
            save_every=save_every,
        )
    except BaseException as exc:
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        manifest["status"] = "interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_manifest(paths.manifest, manifest)
        raise

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "completed"
    manifest["result"] = result
    _write_manifest(paths.manifest, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume a managed MuScripter Flash training job"
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
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start over and preserve any existing training state as a backup",
    )
    args = parser.parse_args()

    run_training(
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
    )


if __name__ == "__main__":
    main()
