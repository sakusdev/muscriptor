import random
from pathlib import Path

import numpy as np
import torch

from muscriptor.flash import FlashConfig
from muscriptor.flash_neural import (
    FlashNeuralNet,
    NeuralPitchDetector,
    synthetic_batch,
)


def test_flash_neural_model_shape():
    model = FlashNeuralNet()
    waveform = torch.zeros(2, 2048)
    logits = model(waveform)
    assert logits.shape == (2, 88)


def test_synthetic_batch_shapes_and_labels():
    audio, labels = synthetic_batch(4, rng=random.Random(1234))
    assert audio.shape == (4, 2048)
    assert labels.shape == (4, 88)
    assert audio.dtype == torch.float32
    assert labels.dtype == torch.float32
    assert torch.all((labels == 0) | (labels == 1))


def test_detector_loads_checkpoint_and_silence_is_empty(tmp_path: Path):
    config = FlashConfig()
    model = FlashNeuralNet()
    checkpoint = tmp_path / "flash.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {
                "sample_rate": config.sample_rate,
                "window_samples": config.window_samples,
                "min_midi": config.min_midi,
                "max_midi": config.max_midi,
            },
        },
        checkpoint,
    )

    detector = NeuralPitchDetector(config, checkpoint)
    frame = np.zeros(config.window_samples, dtype=np.float32)
    assert detector.detect(frame) == {}


def test_detector_rejects_window_mismatch(tmp_path: Path):
    config = FlashConfig()
    model = FlashNeuralNet()
    checkpoint = tmp_path / "flash.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "metadata": {
                "sample_rate": config.sample_rate,
                "window_samples": 1024,
                "min_midi": config.min_midi,
                "max_midi": config.max_midi,
            },
        },
        checkpoint,
    )

    try:
        NeuralPitchDetector(config, checkpoint)
    except ValueError as exc:
        assert "window mismatch" in str(exc)
    else:
        raise AssertionError("expected checkpoint window mismatch")
