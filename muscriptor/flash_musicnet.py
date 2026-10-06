"""MusicNet dataset adapter for MuScripter Flash.

MusicNet stores WAV recordings in ``train_data`` / ``test_data`` and aligned
CSV note labels in ``train_labels`` / ``test_labels``. CSV note times are sample
indices in the original 44.1 kHz recording. This adapter converts those
intervals to the same 10 ms, 88-key frame representation used by the Flash
multi-dataset trainer while keeping the original train/test split.

MusicNet is large, so neither audio nor frame labels are expanded wholesale in
RAM. Each ``TimedPiece`` carries a lazy audio-window loader, and labels are kept
as merged per-pitch frame intervals that materialize one 88-key vector only when
a training/evaluation frame is requested.
"""

from __future__ import annotations

import csv
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from muscriptor.flash_multidataset import TimedPiece
from muscriptor.utils.audio import resample

_SAMPLE_RATE = 16_000
_WINDOW_SAMPLES = 2_048
_MUSICNET_SOURCE_SAMPLE_RATE = 44_100
_MUSICNET_HOP_SECONDS = 0.010
_MUSICNET_FRAME_ORIGIN_SECONDS = 0.010
_MIN_MIDI = 21
_MAX_MIDI = 108
_NOTE_COUNT = _MAX_MIDI - _MIN_MIDI + 1


class SparseMusicNetLabels:
    """Memory-efficient random access to MusicNet's 10 ms 88-key labels."""

    def __init__(
        self,
        frame_count: int,
        intervals: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> None:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        self.frame_count = int(frame_count)
        self.intervals = intervals

    @property
    def shape(self) -> tuple[int, int]:
        return self.frame_count, _NOTE_COUNT

    def __len__(self) -> int:
        return self.frame_count

    def _frame(self, frame_index: int) -> torch.Tensor:
        if frame_index < 0:
            frame_index += self.frame_count
        if not 0 <= frame_index < self.frame_count:
            raise IndexError(frame_index)

        vector = torch.zeros(_NOTE_COUNT, dtype=torch.float32)
        for pitch_index, (starts, stops) in self.intervals.items():
            position = int(np.searchsorted(starts, frame_index, side="right")) - 1
            if position >= 0 and int(stops[position]) > frame_index:
                vector[pitch_index] = 1.0
        return vector

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if len(index) != 2:
                raise IndexError(index)
            frame_index, pitch_index = index
            return self._frame(int(frame_index))[pitch_index]
        if isinstance(index, slice):
            indices = range(*index.indices(self.frame_count))
            frames = [self._frame(frame_index) for frame_index in indices]
            if not frames:
                return torch.empty((0, _NOTE_COUNT), dtype=torch.float32)
            return torch.stack(frames)
        return self._frame(int(index))


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


def _musicnet_audio_info(path: str | Path) -> tuple[float, int]:
    """Return duration seconds and actual audio sample rate without decoding audio."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("MusicNet lazy loading requires soundfile") from exc

    info = sf.info(str(path))
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"invalid MusicNet audio metadata: {path}")
    return info.frames / float(info.samplerate), int(info.samplerate)


def _load_musicnet_window(path: str | Path, target_seconds: float) -> torch.Tensor:
    """Seek and decode only the causal 128 ms window ending at ``target_seconds``."""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("MusicNet lazy loading requires soundfile") from exc

    path = Path(path)
    with sf.SoundFile(str(path), "r") as handle:
        source_rate = int(handle.samplerate)
        source_window = max(1, round(_WINDOW_SAMPLES * source_rate / _SAMPLE_RATE))
        target_end = round(target_seconds * source_rate)
        desired_start = target_end - source_window
        pad_left = max(0, -desired_start)
        read_start = max(0, desired_start)
        read_end = min(max(0, target_end), len(handle))
        read_count = max(0, read_end - read_start)

        if read_start < len(handle):
            handle.seek(read_start)
            data = handle.read(read_count, dtype="float32", always_2d=True)
        else:
            data = np.empty((0, handle.channels), dtype=np.float32)

    if data.size:
        mono = data.mean(axis=1, dtype=np.float32)
    else:
        mono = np.empty(0, dtype=np.float32)
    source = torch.from_numpy(np.ascontiguousarray(mono)).float()

    if pad_left:
        source = F.pad(source, (pad_left, 0))
    if source.numel() < source_window:
        source = F.pad(source, (0, source_window - source.numel()))
    elif source.numel() > source_window:
        source = source[-source_window:]

    target = resample(source.unsqueeze(0), source_rate, _SAMPLE_RATE).squeeze(0)
    if target.numel() < _WINDOW_SAMPLES:
        target = F.pad(target, (_WINDOW_SAMPLES - target.numel(), 0))
    elif target.numel() > _WINDOW_SAMPLES:
        target = target[-_WINDOW_SAMPLES:]
    return target.contiguous()


def _frame_count(
    *,
    audio_samples: int | None,
    duration_seconds: float | None,
) -> int:
    if duration_seconds is None:
        if audio_samples is None or audio_samples <= 0:
            raise ValueError("provide positive audio_samples or duration_seconds")
        duration_seconds = audio_samples / _SAMPLE_RATE
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    return max(1, math.ceil(duration_seconds / _MUSICNET_HOP_SECONDS))


def _sample_interval_to_frames(
    start_sample: int,
    end_sample: int,
    frame_count: int,
) -> tuple[int, int] | None:
    if end_sample <= start_sample:
        return None
    onset = start_sample / _MUSICNET_SOURCE_SAMPLE_RATE
    offset = end_sample / _MUSICNET_SOURCE_SAMPLE_RATE
    start = max(0, math.ceil(onset / _MUSICNET_HOP_SECONDS) - 1)
    stop = max(0, math.ceil(offset / _MUSICNET_HOP_SECONDS) - 1)
    start = min(start, frame_count)
    stop = min(stop, frame_count)
    if start >= stop:
        return None
    return start, stop


def _merge_intervals(intervals: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping/touching frame intervals for one pitch."""
    if not intervals:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty

    intervals.sort()
    merged: list[list[int]] = []
    for start, stop in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    starts = np.asarray([interval[0] for interval in merged], dtype=np.int32)
    stops = np.asarray([interval[1] for interval in merged], dtype=np.int32)
    return starts, stops


def _sparse_labels_from_musicnet_csv(
    path: str | Path,
    *,
    audio_samples: int | None = None,
    duration_seconds: float | None = None,
) -> SparseMusicNetLabels:
    frame_count = _frame_count(
        audio_samples=audio_samples,
        duration_seconds=duration_seconds,
    )
    grouped: dict[int, list[tuple[int, int]]] = {}
    for start_sample, end_sample, note in _read_musicnet_csv(path):
        if not _MIN_MIDI <= note <= _MAX_MIDI:
            continue
        interval = _sample_interval_to_frames(start_sample, end_sample, frame_count)
        if interval is None:
            continue
        grouped.setdefault(note - _MIN_MIDI, []).append(interval)

    merged = {
        pitch_index: _merge_intervals(intervals)
        for pitch_index, intervals in grouped.items()
    }
    return SparseMusicNetLabels(frame_count, merged)


def _labels_from_musicnet_csv(
    path: str | Path,
    *,
    audio_samples: int | None = None,
    duration_seconds: float | None = None,
) -> torch.Tensor:
    """Materialize dense labels for tests/small utilities; training stays sparse."""
    sparse = _sparse_labels_from_musicnet_csv(
        path,
        audio_samples=audio_samples,
        duration_seconds=duration_seconds,
    )
    return sparse[:]


def load_musicnet_split(root: str | Path, split: str) -> list[TimedPiece]:
    """Load one official MusicNet split with lazy audio and sparse labels."""
    data_dir, label_dir = _find_musicnet_split_dirs(root, split)
    pieces: list[TimedPiece] = []

    for wav_path in sorted(data_dir.glob("*.wav")):
        csv_path = label_dir / f"{wav_path.stem}.csv"
        if not csv_path.is_file():
            continue
        duration_seconds, _ = _musicnet_audio_info(wav_path)
        labels = _sparse_labels_from_musicnet_csv(
            csv_path,
            duration_seconds=duration_seconds,
        )
        pieces.append(
            TimedPiece(
                name=wav_path.stem,
                dataset="MusicNet",
                audio=None,
                labels=labels,
                frame_origin_seconds=_MUSICNET_FRAME_ORIGIN_SECONDS,
                frame_hop_seconds=_MUSICNET_HOP_SECONDS,
                window_loader=partial(_load_musicnet_window, wav_path),
            )
        )

    if not pieces:
        raise ValueError(
            f"no paired MusicNet WAV/CSV recordings found in {data_dir} and {label_dir}"
        )
    return pieces


def load_musicnet(root: str | Path) -> tuple[list[TimedPiece], list[TimedPiece]]:
    """Load MusicNet's official train and test splits without eager expansion."""
    return load_musicnet_split(root, "train"), load_musicnet_split(root, "test")
