"""Multi-dataset real-audio fine-tuning for MuScripter Flash.

This module extends the Bach10-only real-data path with URMP.  URMP provides
aligned note annotations for real multi-instrument performances.  Training
remains causal: each target is predicted from only the audio that has arrived
up to that target time.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from muscriptor.flash_neural import FlashNeuralNet, _best_threshold, synthetic_batch
from muscriptor.flash_realdata import load_bach10
from muscriptor.utils.audio import load_audio

_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 2_048
_MIN_MIDI = 21
_MAX_MIDI = 108
_URMP_HOP_SECONDS = 0.010
_URMP_FRAME_ORIGIN_SECONDS = 0.010
_URMP_TEST_IDS = {1, 2, 12, 13, 24, 25, 31, 38, 39}


@dataclass
class TimedPiece:
    """Audio plus 88-key frame labels with an explicit frame clock."""

    name: str
    dataset: str
    audio: torch.Tensor
    labels: torch.Tensor
    frame_origin_seconds: float
    frame_hop_seconds: float


def _bach10_to_timed() -> tuple[float, float]:
    return 0.023, 0.010


def load_bach10_timed(root: str | Path) -> list[TimedPiece]:
    """Load Bach10 and expose its label clock explicitly."""
    origin, hop = _bach10_to_timed()
    return [
        TimedPiece(
            name=piece.name,
            dataset="Bach10",
            audio=piece.audio,
            labels=piece.labels,
            frame_origin_seconds=origin,
            frame_hop_seconds=hop,
        )
        for piece in load_bach10(root)
    ]


def _frequency_to_midi(frequency_hz: float) -> int | None:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        return None
    midi = round(69.0 + 12.0 * math.log2(frequency_hz / 440.0))
    if not _MIN_MIDI <= midi <= _MAX_MIDI:
        return None
    return int(midi)


def _read_urmp_notes(path: str | Path) -> np.ndarray:
    """Read one URMP Notes_*.txt file as onset/frequency/duration rows."""
    path = Path(path)
    if path.stat().st_size == 0:
        return np.empty((0, 3), dtype=np.float32)
    rows = np.loadtxt(path, dtype=np.float32)
    rows = np.atleast_2d(rows)
    if rows.shape[1] < 3:
        raise ValueError(f"expected onset/frequency/duration columns in {path}")
    return np.ascontiguousarray(rows[:, :3], dtype=np.float32)


def _labels_from_urmp_notes(
    note_files: list[str | Path],
    *,
    audio_samples: int,
) -> torch.Tensor:
    """Rasterize aligned URMP note annotations onto a 10 ms 88-key grid."""
    if audio_samples <= 0:
        raise ValueError("audio_samples must be positive")
    frame_count = max(1, math.ceil(audio_samples / _SAMPLE_RATE / _URMP_HOP_SECONDS))
    labels = np.zeros((frame_count, 88), dtype=np.float32)

    for note_file in note_files:
        for onset, frequency, duration in _read_urmp_notes(note_file):
            midi = _frequency_to_midi(float(frequency))
            duration = float(duration)
            onset = float(onset)
            if midi is None or not math.isfinite(onset) or duration <= 0.0:
                continue

            offset = onset + duration
            # Label index i represents time (i + 1) * 10 ms.  Pick the first
            # frame at/after onset and stop before the note offset.
            start = max(0, math.ceil(onset / _URMP_HOP_SECONDS) - 1)
            stop = max(start + 1, math.ceil(offset / _URMP_HOP_SECONDS) - 1)
            start = min(start, frame_count)
            stop = min(stop, frame_count)
            if start < stop:
                labels[start:stop, midi - _MIN_MIDI] = 1.0

    return torch.from_numpy(labels)


def _urmp_piece_id(directory: Path) -> int | None:
    prefix = directory.name.split("_", 1)[0]
    try:
        return int(prefix)
    except ValueError:
        return None


def load_urmp(root: str | Path) -> list[TimedPiece]:
    """Load URMP mixture audio and aligned Notes_*.txt annotations."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    pieces: list[TimedPiece] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        mixes = sorted(directory.glob("AuMix*.wav"))
        note_files = sorted(directory.glob("Notes_*.txt"))
        if not mixes or not note_files:
            continue

        audio = load_audio(mixes[0], target_sr=_SAMPLE_RATE).squeeze(0).float()
        labels = _labels_from_urmp_notes(note_files, audio_samples=audio.numel())
        pieces.append(
            TimedPiece(
                name=directory.name,
                dataset="URMP",
                audio=audio,
                labels=labels,
                frame_origin_seconds=_URMP_FRAME_ORIGIN_SECONDS,
                frame_hop_seconds=_URMP_HOP_SECONDS,
            )
        )

    if len(pieces) < 2:
        raise ValueError(f"expected at least two URMP pieces under {root}, found {len(pieces)}")
    return pieces


def _causal_window(piece: TimedPiece, frame_index: int) -> torch.Tensor:
    """Return the 128 ms audio window ending at this piece's target time."""
    if not 0 <= frame_index < piece.labels.shape[0]:
        raise IndexError(frame_index)

    target_seconds = piece.frame_origin_seconds + frame_index * piece.frame_hop_seconds
    end = round(target_seconds * _SAMPLE_RATE)
    start = end - _WINDOW_SAMPLES

    if start >= 0:
        frame = piece.audio[start:end]
    else:
        available_end = max(0, end)
        frame = torch.cat(
            [
                torch.zeros(-start, dtype=piece.audio.dtype),
                piece.audio[:available_end],
            ]
        )

    if frame.numel() < _WINDOW_SAMPLES:
        frame = torch.nn.functional.pad(frame, (0, _WINDOW_SAMPLES - frame.numel()))
    elif frame.numel() > _WINDOW_SAMPLES:
        frame = frame[-_WINDOW_SAMPLES:]
    return frame.contiguous()


def _split_datasets(
    *,
    bach10: str | Path | None,
    urmp: str | Path | None,
) -> tuple[dict[str, list[TimedPiece]], dict[str, list[TimedPiece]]]:
    train: dict[str, list[TimedPiece]] = {}
    validation: dict[str, list[TimedPiece]] = {}

    if bach10 is not None:
        pieces = load_bach10_timed(bach10)
        if len(pieces) < 3:
            raise ValueError("Bach10 needs at least three pieces")
        train["Bach10"] = pieces[:-2]
        validation["Bach10"] = pieces[-2:]

    if urmp is not None:
        pieces = load_urmp(urmp)
        urmp_train = [
            piece
            for piece in pieces
            if (_urmp_piece_id(Path(piece.name)) or -1) not in _URMP_TEST_IDS
        ]
        urmp_validation = [
            piece
            for piece in pieces
            if (_urmp_piece_id(Path(piece.name)) or -1) in _URMP_TEST_IDS
        ]
        if not urmp_train or not urmp_validation:
            raise ValueError(
                "URMP split requires both train pieces and MT3-compatible test IDs "
                f"{sorted(_URMP_TEST_IDS)}"
            )
        train["URMP"] = urmp_train
        validation["URMP"] = urmp_validation

    if not train:
        raise ValueError("provide at least one of --bach10 or --urmp")
    return train, validation


def _real_batch(
    datasets: dict[str, list[TimedPiece]],
    batch_size: int,
    *,
    rng: random.Random,
    np_rng: np.random.Generator,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample datasets uniformly, then sample pieces/frames within each dataset."""
    names = sorted(datasets)
    audio = torch.empty((batch_size, _WINDOW_SAMPLES), dtype=torch.float32)
    labels = torch.empty((batch_size, 88), dtype=torch.float32)

    for index in range(batch_size):
        dataset_name = rng.choice(names)
        piece = rng.choice(datasets[dataset_name])
        frame_index = rng.randrange(piece.labels.shape[0])
        frame = _causal_window(piece, frame_index).clone()
        if augment:
            frame.mul_(rng.uniform(0.65, 1.20))
            noise_std = rng.uniform(0.0, 0.004)
            if noise_std:
                noise = torch.from_numpy(
                    np_rng.normal(0.0, noise_std, size=_WINDOW_SAMPLES).astype(np.float32)
                )
                frame.add_(noise)
            if rng.random() < 0.5:
                frame.neg_()
            frame.clamp_(-1.0, 1.0)
        audio[index] = frame
        labels[index] = piece.labels[frame_index]
    return audio, labels


def _evaluate(
    model: FlashNeuralNet,
    datasets: dict[str, list[TimedPiece]],
    *,
    max_frames_per_dataset: int = 2_048,
    batch_size: int = 128,
) -> tuple[float, dict[str, float], dict[str, dict[str, float]]]:
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    per_dataset: dict[str, dict[str, float]] = {}

    model.eval()
    with torch.inference_mode():
        for dataset_name, pieces in sorted(datasets.items()):
            examples: list[tuple[TimedPiece, int]] = []
            total_frames = sum(piece.labels.shape[0] for piece in pieces)
            stride = max(1, total_frames // max_frames_per_dataset)
            counter = 0
            for piece in pieces:
                for frame_index in range(piece.labels.shape[0]):
                    if counter % stride == 0:
                        examples.append((piece, frame_index))
                    counter += 1
            examples = examples[:max_frames_per_dataset]

            logits_batches: list[torch.Tensor] = []
            target_batches: list[torch.Tensor] = []
            for offset in range(0, len(examples), batch_size):
                chunk = examples[offset : offset + batch_size]
                waveforms = torch.stack(
                    [_causal_window(piece, frame_index) for piece, frame_index in chunk]
                )
                targets = torch.stack(
                    [piece.labels[frame_index] for piece, frame_index in chunk]
                )
                logits_batches.append(model(waveforms).cpu())
                target_batches.append(targets.cpu())

            logits = torch.cat(logits_batches)
            targets = torch.cat(target_batches)
            threshold, metrics = _best_threshold(logits, targets)
            per_dataset[dataset_name] = {"threshold": threshold, **metrics}
            all_logits.append(logits)
            all_targets.append(targets)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    threshold, metrics = _best_threshold(logits, targets)
    return threshold, metrics, per_dataset


def finetune_multidataset(
    base_checkpoint: str | Path,
    output: str | Path,
    *,
    bach10: str | Path | None = None,
    urmp: str | Path | None = None,
    metrics_path: str | Path | None = None,
    steps: int = 600,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    real_fraction: float = 0.80,
    eval_every: int = 50,
    seed: int = 20261007,
) -> dict[str, Any]:
    """Fine-tune one Flash checkpoint across one or more real datasets."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if not 0.0 < real_fraction <= 1.0:
        raise ValueError("real_fraction must be in (0, 1]")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    train_sets, validation_sets = _split_datasets(bach10=bach10, urmp=urmp)
    payload = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    geometry = (
        int(metadata.get("sample_rate", _SAMPLE_RATE)),
        int(metadata.get("window_samples", _WINDOW_SAMPLES)),
        int(metadata.get("min_midi", _MIN_MIDI)),
        int(metadata.get("max_midi", _MAX_MIDI)),
    )
    if geometry != (_SAMPLE_RATE, _WINDOW_SAMPLES, _MIN_MIDI, _MAX_MIDI):
        raise ValueError(f"base checkpoint geometry mismatch: {geometry}")

    model = FlashNeuralNet(window_samples=_WINDOW_SAMPLES)
    model.load_state_dict(payload["model"])
    baseline_threshold, baseline_metrics, baseline_per_dataset = _evaluate(
        model, validation_sets
    )
    print(
        "baseline holdout: "
        f"f1={baseline_metrics['f1']:.4f} "
        f"p={baseline_metrics['precision']:.4f} "
        f"r={baseline_metrics['recall']:.4f} "
        f"threshold={baseline_threshold:.3f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=2e-4
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((88,), 20.0))
    real_count = max(1, round(batch_size * real_fraction))
    synth_count = batch_size - real_count

    best_f1 = baseline_metrics["f1"]
    best_threshold = baseline_threshold
    best_metrics = dict(baseline_metrics)
    best_per_dataset = copy.deepcopy(baseline_per_dataset)
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    last_loss = 0.0

    model.train()
    for step in range(1, steps + 1):
        real_audio, real_labels = _real_batch(
            train_sets,
            real_count,
            rng=rng,
            np_rng=np_rng,
            augment=True,
        )
        if synth_count:
            synth_audio, synth_labels = synthetic_batch(
                synth_count,
                sample_rate=_SAMPLE_RATE,
                window_samples=_WINDOW_SAMPLES,
                rng=rng,
            )
            waveforms = torch.cat([real_audio, synth_audio])
            labels = torch.cat([real_labels, synth_labels])
        else:
            waveforms, labels = real_audio, real_labels

        permutation = torch.randperm(waveforms.shape[0])
        logits = model(waveforms[permutation])
        loss = criterion(logits, labels[permutation])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.item())

        if step == 1 or step % eval_every == 0 or step == steps:
            threshold, metrics, per_dataset = _evaluate(model, validation_sets)
            print(
                f"step={step:04d}/{steps} loss={last_loss:.5f} "
                f"f1={metrics['f1']:.4f} p={metrics['precision']:.4f} "
                f"r={metrics['recall']:.4f} threshold={threshold:.3f}",
                flush=True,
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_metrics = dict(metrics)
                best_per_dataset = copy.deepcopy(per_dataset)
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            model.train()

    model.load_state_dict(best_state)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "format": 4,
        "backend": "log-spectrum-mlp-multidataset-finetuned",
        "sample_rate": _SAMPLE_RATE,
        "window_samples": _WINDOW_SAMPLES,
        "window_ms": 128.0,
        "min_midi": _MIN_MIDI,
        "max_midi": _MAX_MIDI,
        "source_checkpoint": str(base_checkpoint),
        "datasets": sorted(train_sets),
        "train_piece_counts": {name: len(pieces) for name, pieces in train_sets.items()},
        "validation_piece_counts": {
            name: len(pieces) for name, pieces in validation_sets.items()
        },
        "steps": steps,
        "best_step": best_step,
        "batch_size": batch_size,
        "real_fraction": real_fraction,
        "seed": seed,
        "train_loss": last_loss,
        "baseline_decision_threshold": baseline_threshold,
        "baseline_precision": baseline_metrics["precision"],
        "baseline_recall": baseline_metrics["recall"],
        "baseline_f1": baseline_metrics["f1"],
        "baseline_per_dataset": baseline_per_dataset,
        "decision_threshold": best_threshold,
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "per_dataset": best_per_dataset,
    }
    torch.save({"model": model.state_dict(), "metadata": result}, output)

    if metrics_path is not None:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune MuScripter Flash across real audio datasets"
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bach10")
    parser.add_argument("--urmp")
    parser.add_argument("--metrics")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--real-fraction", type=float, default=0.80)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()

    finetune_multidataset(
        args.base,
        args.output,
        bach10=args.bach10,
        urmp=args.urmp,
        metrics_path=args.metrics,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        real_fraction=args.real_fraction,
        eval_every=args.eval_every,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
