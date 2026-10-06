"""Real-recording fine-tuning for the MuScripter Flash neural backend.

The synthetic checkpoint is useful for bootstrapping pitch geometry, but it does
not contain real instrument timbre.  This module fine-tunes that checkpoint on
Bach10: ten short, real-recorded four-part Bach chorales with frame-level ground
truth pitches.  Training remains causal: every target is predicted from only the
128 ms of audio immediately preceding the target frame.

Bach10 itself is not vendored in this repository.  Pass a local checkout/path to
``python -m muscriptor.flash_realdata finetune``.  The GitHub Actions workflow in
this repository downloads a public mirror at training time.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from muscriptor.flash_neural import (
    FlashNeuralNet,
    _best_threshold,
    synthetic_batch,
)
from muscriptor.utils.audio import load_audio

_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 2_048
_MIN_MIDI = 21
_MAX_MIDI = 108
_BACH10_FRAME_CENTER_SECONDS = 0.023
_BACH10_HOP_SECONDS = 0.010


@dataclass
class Bach10Piece:
    """One Bach10 mixture plus frame-level 88-key labels."""

    name: str
    audio: torch.Tensor  # mono [samples] at 16 kHz
    labels: torch.Tensor  # [frames, 88], active-state labels


def _extract_gtf0_matrix(mat: dict[str, Any]) -> np.ndarray:
    """Find the Bach10 4 x frame ground-truth pitch matrix in a MAT payload."""
    candidates: list[np.ndarray] = []
    for key, value in mat.items():
        if key.startswith("__") or not isinstance(value, np.ndarray):
            continue
        if value.ndim != 2 or not np.issubdtype(value.dtype, np.number):
            continue
        array = np.asarray(value)
        if array.shape[0] == 4:
            candidates.append(array)
        elif array.shape[1] == 4:
            candidates.append(array.T)
    if not candidates:
        raise ValueError("could not find a 4 x frame GTF0 matrix in MAT file")
    # MATLAB files can contain helper arrays.  The frame matrix is by far the
    # widest 4-row numeric array.
    return max(candidates, key=lambda array: array.shape[1]).astype(np.float32)


def _labels_from_gtf0(gtf0: np.ndarray) -> torch.Tensor:
    """Convert Bach10 per-part MIDI pitches to an 88-key multi-hot matrix."""
    array = np.asarray(gtf0, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected 2D GTF0 matrix, got {array.shape}")
    if array.shape[0] != 4 and array.shape[1] == 4:
        array = array.T
    if array.shape[0] != 4:
        raise ValueError(f"expected four Bach10 parts, got {array.shape}")

    frames = array.shape[1]
    labels = np.zeros((frames, _MAX_MIDI - _MIN_MIDI + 1), dtype=np.float32)
    rounded = np.rint(array).astype(np.int16)
    for part in range(4):
        valid = (rounded[part] >= _MIN_MIDI) & (rounded[part] <= _MAX_MIDI)
        frame_indices = np.flatnonzero(valid)
        note_indices = rounded[part, valid] - _MIN_MIDI
        labels[frame_indices, note_indices] = 1.0
    return torch.from_numpy(labels)


def load_bach10(root: str | Path) -> list[Bach10Piece]:
    """Load Bach10 mixtures and frame-level ground truth from ``root``."""
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError(
            "Bach10 fine-tuning requires scipy. Run with `uv run --with scipy ...`."
        ) from exc

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    pieces: list[Bach10Piece] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        mixture = directory / f"{directory.name}.wav"
        gtf0_files = sorted(directory.glob("*-GTF0s.mat"))
        if not mixture.exists() or not gtf0_files:
            continue

        waveform = load_audio(mixture, target_sr=_SAMPLE_RATE).squeeze(0).float()
        matrix = _extract_gtf0_matrix(loadmat(gtf0_files[0]))
        labels = _labels_from_gtf0(matrix)
        if labels.shape[0] < 2:
            continue
        pieces.append(Bach10Piece(directory.name, waveform, labels))

    if len(pieces) < 3:
        raise ValueError(
            f"expected at least three Bach10 pieces under {root}, found {len(pieces)}"
        )
    return pieces


def _causal_window(piece: Bach10Piece, frame_index: int) -> torch.Tensor:
    """Return the 128 ms window ending at a Bach10 target frame center."""
    if not 0 <= frame_index < piece.labels.shape[0]:
        raise IndexError(frame_index)

    target_seconds = _BACH10_FRAME_CENTER_SECONDS + frame_index * _BACH10_HOP_SECONDS
    end = round(target_seconds * _SAMPLE_RATE)
    start = end - _WINDOW_SAMPLES

    if start >= 0:
        frame = piece.audio[start:end]
    else:
        frame = torch.cat(
            [
                torch.zeros(-start, dtype=piece.audio.dtype),
                piece.audio[:end],
            ]
        )

    if frame.numel() < _WINDOW_SAMPLES:
        frame = torch.nn.functional.pad(frame, (0, _WINDOW_SAMPLES - frame.numel()))
    elif frame.numel() > _WINDOW_SAMPLES:
        frame = frame[-_WINDOW_SAMPLES:]
    return frame.contiguous()


def _real_batch(
    pieces: list[Bach10Piece],
    batch_size: int,
    *,
    rng: random.Random,
    np_rng: np.random.Generator,
    augment: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio = torch.empty((batch_size, _WINDOW_SAMPLES), dtype=torch.float32)
    labels = torch.empty((batch_size, 88), dtype=torch.float32)

    for index in range(batch_size):
        piece = rng.choice(pieces)
        frame_index = rng.randrange(piece.labels.shape[0])
        frame = _causal_window(piece, frame_index).clone()
        if augment:
            gain = rng.uniform(0.65, 1.20)
            frame.mul_(gain)
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


def _evaluate_real(
    model: FlashNeuralNet,
    pieces: list[Bach10Piece],
    *,
    max_frames: int = 2_048,
    batch_size: int = 128,
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    """Evaluate on deterministic evenly spaced frames from holdout pieces."""
    examples: list[tuple[Bach10Piece, int]] = []
    total_frames = sum(piece.labels.shape[0] for piece in pieces)
    stride = max(1, total_frames // max_frames)
    counter = 0
    for piece in pieces:
        for frame_index in range(piece.labels.shape[0]):
            if counter % stride == 0:
                examples.append((piece, frame_index))
            counter += 1
    examples = examples[:max_frames]

    logits_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            chunk = examples[offset : offset + batch_size]
            waveforms = torch.stack([_causal_window(piece, index) for piece, index in chunk])
            targets = torch.stack([piece.labels[index] for piece, index in chunk])
            logits_batches.append(model(waveforms).cpu())
            target_batches.append(targets.cpu())

    logits = torch.cat(logits_batches)
    targets = torch.cat(target_batches)
    threshold, metrics = _best_threshold(logits, targets)
    return threshold, metrics, logits, targets


def finetune_bach10(
    dataset: str | Path,
    base_checkpoint: str | Path,
    output: str | Path,
    *,
    metrics_path: str | Path | None = None,
    steps: int = 400,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    seed: int = 20261007,
    validation_pieces: int = 2,
    real_fraction: float = 0.75,
    eval_every: int = 50,
) -> dict[str, Any]:
    """Fine-tune the synthetic Flash model on real Bach10 recordings."""
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive")
    if not 0.0 < real_fraction <= 1.0:
        raise ValueError("real_fraction must be in (0, 1]")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    pieces = load_bach10(dataset)
    if not 1 <= validation_pieces < len(pieces):
        raise ValueError("validation_pieces must leave at least one training piece")
    train_pieces = pieces[:-validation_pieces]
    val_pieces = pieces[-validation_pieces:]

    payload = torch.load(Path(base_checkpoint), map_location="cpu", weights_only=False)
    base_metadata = dict(payload.get("metadata", {}))
    expected = (
        int(base_metadata.get("sample_rate", _SAMPLE_RATE)),
        int(base_metadata.get("window_samples", _WINDOW_SAMPLES)),
        int(base_metadata.get("min_midi", _MIN_MIDI)),
        int(base_metadata.get("max_midi", _MAX_MIDI)),
    )
    if expected != (_SAMPLE_RATE, _WINDOW_SAMPLES, _MIN_MIDI, _MAX_MIDI):
        raise ValueError(f"base checkpoint geometry mismatch: {expected}")

    model = FlashNeuralNet(window_samples=_WINDOW_SAMPLES)
    model.load_state_dict(payload["model"])

    baseline_threshold, baseline_metrics, _, _ = _evaluate_real(model, val_pieces)
    print(
        "baseline real holdout: "
        f"f1={baseline_metrics['f1']:.4f} "
        f"precision={baseline_metrics['precision']:.4f} "
        f"recall={baseline_metrics['recall']:.4f} "
        f"threshold={baseline_threshold:.3f}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((88,), 20.0))

    real_count = max(1, round(batch_size * real_fraction))
    synth_count = batch_size - real_count
    best_f1 = baseline_metrics["f1"]
    best_threshold = baseline_threshold
    best_metrics = dict(baseline_metrics)
    best_state = copy.deepcopy(model.state_dict())
    best_step = 0
    last_loss = 0.0

    model.train()
    for step in range(1, steps + 1):
        real_audio, real_labels = _real_batch(
            train_pieces,
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
            waveforms = torch.cat([real_audio, synth_audio], dim=0)
            labels = torch.cat([real_labels, synth_labels], dim=0)
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
            threshold, metrics, _, _ = _evaluate_real(model, val_pieces)
            print(
                f"step={step:04d}/{steps} loss={last_loss:.5f} "
                f"real_f1={metrics['f1']:.4f} p={metrics['precision']:.4f} "
                f"r={metrics['recall']:.4f} threshold={threshold:.3f}",
                flush=True,
            )
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_metrics = dict(metrics)
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
            model.train()

    model.load_state_dict(best_state)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "format": 3,
        "backend": "log-spectrum-mlp-bach10-finetuned",
        "sample_rate": _SAMPLE_RATE,
        "window_samples": _WINDOW_SAMPLES,
        "window_ms": 128.0,
        "min_midi": _MIN_MIDI,
        "max_midi": _MAX_MIDI,
        "source_checkpoint": str(base_checkpoint),
        "dataset": "Bach10 v1.1",
        "train_pieces": [piece.name for piece in train_pieces],
        "validation_pieces": [piece.name for piece in val_pieces],
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
        "decision_threshold": best_threshold,
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)

    if metrics_path is not None:
        path = Path(metrics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune MuScripter Flash on real audio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    finetune = subparsers.add_parser("finetune", help="fine-tune on Bach10 real recordings")
    finetune.add_argument("--dataset", required=True)
    finetune.add_argument("--base", required=True)
    finetune.add_argument("--output", required=True)
    finetune.add_argument("--metrics")
    finetune.add_argument("--steps", type=int, default=400)
    finetune.add_argument("--batch-size", type=int, default=32)
    finetune.add_argument("--learning-rate", type=float, default=2e-4)
    finetune.add_argument("--seed", type=int, default=20261007)
    finetune.add_argument("--validation-pieces", type=int, default=2)
    finetune.add_argument("--real-fraction", type=float, default=0.75)
    finetune.add_argument("--eval-every", type=int, default=50)

    args = parser.parse_args()
    finetune_bach10(
        args.dataset,
        args.base,
        args.output,
        metrics_path=args.metrics,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        validation_pieces=args.validation_pieces,
        real_fraction=args.real_fraction,
        eval_every=args.eval_every,
    )


if __name__ == "__main__":
    main()
