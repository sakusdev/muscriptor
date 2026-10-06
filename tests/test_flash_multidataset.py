import math
import wave
from pathlib import Path

import numpy as np
import torch

from muscriptor.flash_multidataset import (
    TimedPiece,
    _causal_window,
    _frequency_to_midi,
    _labels_from_urmp_notes,
    _split_datasets,
    load_urmp,
)


def _write_wav(path: Path, *, seconds: float = 0.08, sample_rate: int = 16_000) -> None:
    samples = int(seconds * sample_rate)
    data = np.zeros(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())


def _write_urmp_piece(root: Path, name: str, note: int = 69) -> Path:
    directory = root / name
    directory.mkdir()
    _write_wav(directory / f"AuMix_{name}.wav")
    frequency = 440.0 * 2.0 ** ((note - 69) / 12.0)
    (directory / f"Notes_1_vn_{name}.txt").write_text(
        f"0.000000\t{frequency:.6f}\t0.030000\n",
        encoding="utf-8",
    )
    return directory


def test_frequency_to_midi():
    assert _frequency_to_midi(440.0) == 69
    assert _frequency_to_midi(261.625565) == 60
    assert _frequency_to_midi(0.0) is None
    assert _frequency_to_midi(math.nan) is None


def test_urmp_note_rasterization_uses_arrived_frame_times(tmp_path: Path):
    notes = tmp_path / "Notes_1_vn_01_Test.txt"
    notes.write_text("0.015\t440.0\t0.020\n", encoding="utf-8")

    labels = _labels_from_urmp_notes([notes], audio_samples=1600)

    assert labels.shape == (10, 88)
    # Frame index 1 is t=20 ms and frame index 2 is t=30 ms.
    assert labels[0, 69 - 21] == 0
    assert labels[1, 69 - 21] == 1
    assert labels[2, 69 - 21] == 1
    assert labels[3, 69 - 21] == 0


def test_urmp_causal_window_never_reads_future_samples():
    audio = torch.arange(5000, dtype=torch.float32)
    labels = torch.zeros((30, 88), dtype=torch.float32)
    piece = TimedPiece(
        name="03_Test",
        dataset="URMP",
        audio=audio,
        labels=labels,
        frame_origin_seconds=0.010,
        frame_hop_seconds=0.010,
    )

    # index 19 => target time 200 ms => sample 3200 at 16 kHz.
    window = _causal_window(piece, 19)

    assert window.shape == (2048,)
    assert window[-1].item() == 3199
    assert window[0].item() == 3200 - 2048


def test_load_urmp_and_mt3_compatible_split(tmp_path: Path):
    _write_urmp_piece(tmp_path, "01_Test_vn")
    _write_urmp_piece(tmp_path, "03_Train_vn", note=72)

    pieces = load_urmp(tmp_path)
    assert len(pieces) == 2
    assert {piece.name for piece in pieces} == {"01_Test_vn", "03_Train_vn"}
    assert all(piece.dataset == "URMP" for piece in pieces)
    assert pieces[0].labels.shape[1] == 88

    train, validation = _split_datasets(bach10=None, urmp=tmp_path)
    assert [piece.name for piece in train["URMP"]] == ["03_Train_vn"]
    assert [piece.name for piece in validation["URMP"]] == ["01_Test_vn"]
