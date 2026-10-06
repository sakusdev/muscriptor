import csv
import wave
from pathlib import Path

import numpy as np
import torch

from muscriptor.flash_multidataset import _causal_window
from muscriptor.flash_musicnet import (
    _find_musicnet_split_dirs,
    _labels_from_musicnet_csv,
    _read_musicnet_csv,
    load_musicnet,
)


def _write_wav(path: Path, *, seconds: float = 0.08, sample_rate: int = 44_100) -> None:
    samples = int(seconds * sample_rate)
    data = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def _write_labels(path: Path, *, note: int = 69) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start_time",
                "end_time",
                "instrument",
                "note",
                "start_beat",
                "end_beat",
                "note_value",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "start_time": 441,
                "end_time": 1323,
                "instrument": 41,
                "note": note,
                "start_beat": 0.0,
                "end_beat": 1.0,
                "note_value": "Quarter",
            }
        )


def _write_split(root: Path, split: str, recording_id: str, note: int) -> None:
    data_dir = root / f"{split}_data"
    label_dir = root / f"{split}_labels"
    data_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    _write_wav(data_dir / f"{recording_id}.wav")
    _write_labels(label_dir / f"{recording_id}.csv", note=note)


def test_read_musicnet_csv_preserves_sample_indices(tmp_path: Path):
    labels = tmp_path / "1727.csv"
    _write_labels(labels, note=72)

    rows = _read_musicnet_csv(labels)

    assert rows == [(441, 1323, 72)]


def test_musicnet_rasterization_uses_44100hz_sample_clock(tmp_path: Path):
    labels = tmp_path / "1727.csv"
    _write_labels(labels)

    frames = _labels_from_musicnet_csv(labels, audio_samples=1600)

    # 441 samples = 10 ms; 1323 samples = 30 ms. The interval is [10, 30) ms.
    assert frames.shape == (10, 88)
    assert frames[0, 69 - 21] == 1
    assert frames[1, 69 - 21] == 1
    assert frames[2, 69 - 21] == 0


def test_musicnet_subframe_note_does_not_invent_future_activity(tmp_path: Path):
    labels = tmp_path / "1727.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_time", "end_time", "note"])
        writer.writeheader()
        writer.writerow({"start_time": 44, "end_time": 220, "note": 69})

    frames = _labels_from_musicnet_csv(labels, audio_samples=1600)

    # Roughly [1, 5) ms contains no 10 ms target frame, so no false label.
    assert frames[:, 69 - 21].sum().item() == 0


def test_find_musicnet_split_dirs_accepts_nested_archive_root(tmp_path: Path):
    nested = tmp_path / "musicnet"
    _write_split(nested, "train", "1727", 69)

    data_dir, label_dir = _find_musicnet_split_dirs(tmp_path, "train")

    assert data_dir == nested / "train_data"
    assert label_dir == nested / "train_labels"


def test_load_musicnet_preserves_split_and_decodes_windows_lazily(tmp_path: Path):
    _write_split(tmp_path, "train", "1727", 69)
    _write_split(tmp_path, "test", "2303", 72)

    train, test = load_musicnet(tmp_path)

    assert [piece.name for piece in train] == ["1727"]
    assert [piece.name for piece in test] == ["2303"]
    assert train[0].dataset == "MusicNet"
    assert test[0].dataset == "MusicNet"
    assert train[0].audio is None
    assert train[0].window_loader is not None
    assert train[0].labels.shape == (8, 88)
    assert train[0].labels[0, 69 - 21] == 1
    assert test[0].labels[0, 72 - 21] == 1

    window = _causal_window(train[0], 4)
    assert window.shape == (2048,)
    assert torch.count_nonzero(window).item() == 0
