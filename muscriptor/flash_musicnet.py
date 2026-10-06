"""MusicNet dataset adapter for MuScripter Flash.

MusicNet stores PCM WAV recordings in ``train_data`` / ``test_data`` and
aligned CSV note labels in ``train_labels`` / ``test_labels``.  CSV note times
are sample indices in the original 44.1 kHz recording.  This adapter converts
those intervals to the same 10 ms, 88-key frame representation used by the
Flash multi-dataset trainer while keeping the original train/test split.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import torch

from muscriptor.flash_multidataset import TimedPiece
from muscriptor.utils.audio import load_audio

_SAMPLE_RATE = 16_000
_MUSICNET_SOURCE_SAMPLE_RATE = 44_100
_MUSICNET_HOP_SECONDS = 0.010
_MUSICNET_FRAME_ORIGIN_SECONDS = 0.010
_MIN_MIDI = 21
_MAX_MIDI = 108


def _find_musicnet_split_dirs(root: str | Path, split: str) -> tuple[Path, Path]:
    """Resolve MusicNet's ``<split>_data`` and ``<split>_labels`` directories."""
    if split not in {"train", "test"}:
        raise ValueError("MusicNet split must be 'train' or 'test'")

    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    data_name = f"{split}_data"
    label_name = f"{split}_labels"
    candidates: list[Path] = [root / data_name]
    candidates.extend(sorted(root.rglob(data_name)))

    seen: set[Path] = set()
    for data_dir in candidates:
        try:
            key = data_dir.resolve()
        except OSError:
            key = data_dir
        if key in seen:
            continue
        seen.add(key)
        label_dir = data_dir.parent / label_name
        if data_dir.is_dir() and label_dir.is_dir():
            return data_dir, label_dir

    raise FileNotFoundError(
        f"could not find {data_name}/ and {label_name}/ under {root}"
    )


def _read_musicnet_csv(path: str | Path) -> list[tuple[int, int, int]]:
    """Read MusicNet note intervals as source-sample start/end and MIDI note."""
    rows: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"start_time", "end_time", "note"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"MusicNet CSV {path} must contain {sorted(required)} columns"
            )
        for row in reader:
            start = int(row["start_time"])
            end = int(row["end_time"])
            note = int(row["note"])
            rows.append((start, end, note))
    return rows


def _labels_from_musicnet_csv(
    path: str | Path,
    *,
    audio_samples: int,
) -> torch.Tensor:
    """Rasterize MusicNet source-sample intervals to a 10 ms 88-key grid."""
    if audio_samples <= 0:
        raise ValueError("audio_samples must be positive")

    frame_count = max(
        1,
        math.ceil(audio_samples / _SAMPLE_RATE / _MUSICNET_HOP_SECONDS),
    )
    labels = np.zeros((frame_count, 88), dtype=np.float32)

    for start_sample, end_sample, note in _read_musicnet_csv(path):
        if end_sample <= start_sample or not _MIN_MIDI <= note <= _MAX_MIDI:
            continue
        onset = start_sample / _MUSICNET_SOURCE_SAMPLE_RATE
        offset = end_sample / _MUSICNET_SOURCE_SAMPLE_RATE

        # Label index i represents time (i + 1) * 10 ms.  MusicNet intervals
        # are half-open, so activate frames at/after onset and before offset.
        start = max(0, math.ceil(onset / _MUSICNET_HOP_SECONDS) - 1)
        stop = max(start + 1, math.ceil(offset / _MUSICNET_HOP_SECONDS) - 1)
        start = min(start, frame_count)
        stop = min(stop, frame_count)
        if start < stop:
            labels[start:stop, note - _MIN_MIDI] = 1.0

    return torch.from_numpy(labels)


def load_musicnet_split(root: str | Path, split: str) -> list[TimedPiece]:
    """Load one official MusicNet split as Flash ``TimedPiece`` objects."""
    data_dir, label_dir = _find_musicnet_split_dirs(root, split)
    pieces: list[TimedPiece] = []

    for wav_path in sorted(data_dir.glob("*.wav")):
        csv_path = label_dir / f"{wav_path.stem}.csv"
        if not csv_path.is_file():
            continue
        audio = load_audio(wav_path, target_sr=_SAMPLE_RATE).squeeze(0).float()
        labels = _labels_from_musicnet_csv(csv_path, audio_samples=audio.numel())
        pieces.append(
            TimedPiece(
                name=wav_path.stem,
                dataset="MusicNet",
                audio=audio,
                labels=labels,
                frame_origin_seconds=_MUSICNET_FRAME_ORIGIN_SECONDS,
                frame_hop_seconds=_MUSICNET_HOP_SECONDS,
            )
        )

    if not pieces:
        raise ValueError(
            f"no paired MusicNet WAV/CSV recordings found in {data_dir} and {label_dir}"
        )
    return pieces


def load_musicnet(root: str | Path) -> tuple[list[TimedPiece], list[TimedPiece]]:
    """Load MusicNet's official train and test splits."""
    return load_musicnet_split(root, "train"), load_musicnet_split(root, "test")
