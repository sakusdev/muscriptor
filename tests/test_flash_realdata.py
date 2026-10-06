import numpy as np
import torch

from muscriptor.flash_realdata import (
    Bach10Piece,
    _causal_window,
    _extract_gtf0_matrix,
    _labels_from_gtf0,
)


def test_extract_gtf0_matrix_accepts_transposed_payload():
    expected = np.zeros((4, 123), dtype=np.float64)
    payload = {
        "__header__": "ignored",
        "small": np.zeros((2, 2)),
        "pitch": expected.T,
    }
    extracted = _extract_gtf0_matrix(payload)
    assert extracted.shape == (4, 123)
    assert extracted.dtype == np.float32


def test_labels_from_gtf0_builds_88_key_multihot():
    gtf0 = np.array(
        [
            [60.0, 0.0, 72.2],
            [64.0, 67.0, 0.0],
            [67.0, 67.0, 76.0],
            [48.0, 43.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = _labels_from_gtf0(gtf0)
    assert labels.shape == (3, 88)
    assert labels[0, 60 - 21] == 1
    assert labels[0, 64 - 21] == 1
    assert labels[0, 67 - 21] == 1
    assert labels[0, 48 - 21] == 1
    assert labels[1, 67 - 21] == 1
    assert labels[1, 43 - 21] == 1
    assert labels[2, 72 - 21] == 1
    assert labels[2, 76 - 21] == 1


def test_causal_window_never_reads_future_samples():
    # At frame 20, Bach10 target center is 223 ms => sample 3568 at 16 kHz.
    audio = torch.arange(6000, dtype=torch.float32)
    labels = torch.zeros((100, 88), dtype=torch.float32)
    piece = Bach10Piece("test", audio, labels)

    window = _causal_window(piece, 20)

    assert window.shape == (2048,)
    assert window[-1].item() == 3567
    assert window[0].item() == 3568 - 2048


def test_causal_window_left_pads_early_frames():
    audio = torch.arange(1000, dtype=torch.float32)
    labels = torch.zeros((10, 88), dtype=torch.float32)
    piece = Bach10Piece("test", audio, labels)

    window = _causal_window(piece, 0)

    assert window.shape == (2048,)
    # Frame zero is centered at 23 ms -> 368 samples have arrived.
    assert torch.count_nonzero(window[: 2048 - 368]) == 0
    assert torch.equal(window[-368:], audio[:368])
