"""Tiny trainable neural backend for MuScripter Flash.

This module deliberately targets the realtime Flash contract rather than the
five-second autoregressive MuScriptor model. The first checkpoint is trained on
synthetic polyphonic audio so GitHub Actions can produce a usable model without
shipping a large external dataset into CI.

The network consumes only the current rolling Flash window; it never needs
future audio. That makes it suitable for the existing 128 ms window / 32 ms hop
streaming path. Synthetic training is only a bootstrap: real recorded stems and
MIDI-aligned data should replace or fine-tune it for production quality.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from muscriptor.flash import FlashConfig, PitchEstimate


class FlashNeuralNet(nn.Module):
    """Small causal-ish raw-audio Conv1D + GRU multi-pitch classifier.

    The model predicts the active state of MIDI notes 21..108 from one rolling
    audio window. Convolutions use no right-padding and the final prediction is
    taken from the last recurrent state, so inference never requires samples
    beyond the current Flash window.
    """

    def __init__(self, note_count: int = 88) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=4),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=4),
            nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2),
            nn.GELU(),
        )
        self.gru = nn.GRU(48, 64, batch_first=True)
        self.head = nn.Linear(64, note_count)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"expected [batch, samples], got {tuple(waveform.shape)}")
        features = self.encoder(waveform.unsqueeze(1)).transpose(1, 2)
        _, hidden = self.gru(features)
        return self.head(hidden[-1])


class NeuralPitchDetector:
    """Flash detector wrapper around a trained :class:`FlashNeuralNet`."""

    def __init__(
        self,
        config: FlashConfig,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        metadata = payload.get("metadata", {})

        expected_sr = int(metadata.get("sample_rate", config.sample_rate))
        expected_samples = int(metadata.get("window_samples", config.window_samples))
        min_midi = int(metadata.get("min_midi", 21))
        max_midi = int(metadata.get("max_midi", 108))
        if expected_sr != config.sample_rate:
            raise ValueError(
                f"checkpoint expects {expected_sr} Hz, Flash uses {config.sample_rate} Hz"
            )
        if expected_samples != config.window_samples:
            raise ValueError(
                "checkpoint window mismatch: "
                f"expects {expected_samples} samples, Flash uses {config.window_samples}"
            )
        if (min_midi, max_midi) != (config.min_midi, config.max_midi):
            raise ValueError(
                "checkpoint MIDI range mismatch: "
                f"expects {min_midi}..{max_midi}, Flash uses "
                f"{config.min_midi}..{config.max_midi}"
            )

        self.model = FlashNeuralNet(max_midi - min_midi + 1).to(self.device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.min_midi = min_midi

    def detect(self, frame: np.ndarray) -> dict[int, PitchEstimate]:
        array = np.asarray(frame, dtype=np.float32).reshape(-1)
        if array.size != self.config.window_samples:
            raise ValueError(
                f"expected {self.config.window_samples} samples, got {array.size}"
            )
        rms = float(np.sqrt(np.mean(array * array) + 1e-12))
        if rms < self.config.silence_rms:
            return {}

        peak = max(float(np.max(np.abs(array))), 1e-6)
        normalized = np.clip(array / peak, -1.0, 1.0)
        tensor = torch.from_numpy(normalized).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.sigmoid(self.model(tensor))[0].cpu().numpy()

        max_probability = float(np.max(probabilities))
        threshold = max(
            self.config.min_confidence,
            max_probability * self.config.relative_threshold,
        )
        indices = np.flatnonzero(probabilities >= threshold)
        if indices.size == 0:
            return {}

        ranked = indices[np.argsort(probabilities[indices])[::-1]][
            : self.config.max_polyphony
        ]
        return {
            self.min_midi + int(index): PitchEstimate(
                midi_note=self.min_midi + int(index),
                confidence=float(probabilities[index]),
                strength=float(probabilities[index]),
            )
            for index in ranked
        }


def _synth_one(
    *,
    sample_rate: int,
    samples: int,
    min_midi: int,
    max_midi: int,
    max_polyphony: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    label = np.zeros(max_midi - min_midi + 1, dtype=np.float32)
    audio = np.zeros(samples, dtype=np.float32)
    t = np.arange(samples, dtype=np.float32) / float(sample_rate)

    if rng.random() < 0.12:
        note_count = 0
    else:
        note_count = rng.randint(1, min(4, max_polyphony))

    notes = rng.sample(range(min_midi, max_midi + 1), k=note_count)
    if note_count:
        norm = 1.0 / math.sqrt(note_count)
        for note in notes:
            label[note - min_midi] = 1.0
            cents = rng.uniform(-7.0, 7.0)
            freq = 440.0 * 2.0 ** ((note - 69 + cents / 100.0) / 12.0)
            amplitude = rng.uniform(0.20, 0.55) * norm
            phase = rng.uniform(0.0, math.tau)
            partial = np.zeros(samples, dtype=np.float32)
            for harmonic, harmonic_gain in ((1, 1.0), (2, 0.28), (3, 0.12)):
                harmonic_freq = freq * harmonic
                if harmonic_freq >= sample_rate / 2:
                    continue
                partial += harmonic_gain * np.sin(
                    math.tau * harmonic_freq * t + phase * harmonic
                ).astype(np.float32)

            attack_samples = max(1, int(samples * rng.uniform(0.01, 0.15)))
            envelope = np.ones(samples, dtype=np.float32)
            envelope[:attack_samples] = np.linspace(
                0.0, 1.0, attack_samples, dtype=np.float32
            )
            envelope *= np.linspace(
                1.0, rng.uniform(0.65, 1.0), samples, dtype=np.float32
            )
            audio += amplitude * partial * envelope

    noise_sigma = rng.uniform(0.002, 0.02)
    audio += np.random.normal(0.0, noise_sigma, size=samples).astype(np.float32)
    audio *= rng.uniform(0.55, 1.0)
    peak = max(float(np.max(np.abs(audio))), 1.0)
    return np.clip(audio / peak, -1.0, 1.0), label


def synthetic_batch(
    batch_size: int,
    *,
    sample_rate: int = 16_000,
    window_samples: int = 2048,
    min_midi: int = 21,
    max_midi: int = 108,
    max_polyphony: int = 4,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    audio = np.empty((batch_size, window_samples), dtype=np.float32)
    labels = np.empty((batch_size, max_midi - min_midi + 1), dtype=np.float32)
    for index in range(batch_size):
        audio[index], labels[index] = _synth_one(
            sample_rate=sample_rate,
            samples=window_samples,
            min_midi=min_midi,
            max_midi=max_midi,
            max_polyphony=max_polyphony,
            rng=rng,
        )
    return torch.from_numpy(audio), torch.from_numpy(labels)


def _metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.45,
) -> dict[str, float]:
    predictions = torch.sigmoid(logits) >= threshold
    truth = targets >= 0.5
    tp = float((predictions & truth).sum().item())
    fp = float((predictions & ~truth).sum().item())
    fn = float((~predictions & truth).sum().item())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def train_synthetic(
    output: str | Path,
    *,
    metrics_path: str | Path | None = None,
    steps: int = 300,
    batch_size: int = 32,
    learning_rate: float = 2e-3,
    seed: int = 20261006,
    sample_rate: int = 16_000,
    window_samples: int = 2048,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    model = FlashNeuralNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    pos_weight = torch.full((88,), 12.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    last_loss = 0.0
    for step in range(1, steps + 1):
        waveforms, labels = synthetic_batch(
            batch_size,
            sample_rate=sample_rate,
            window_samples=window_samples,
            rng=rng,
        )
        logits = model(waveforms)
        loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last_loss = float(loss.item())
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"step={step:04d}/{steps} loss={last_loss:.5f}", flush=True)

    model.eval()
    validation_rng = random.Random(seed + 1)
    val_audio, val_labels = synthetic_batch(
        256,
        sample_rate=sample_rate,
        window_samples=window_samples,
        rng=validation_rng,
    )
    with torch.inference_mode():
        val_logits = model(val_audio)
        val_loss = float(criterion(val_logits, val_labels).item())
    metric_values = _metrics(val_logits, val_labels)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "format": 1,
        "backend": "conv1d-gru-synthetic-bootstrap",
        "sample_rate": sample_rate,
        "window_samples": window_samples,
        "window_ms": window_samples * 1000.0 / sample_rate,
        "min_midi": 21,
        "max_midi": 108,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "train_loss": last_loss,
        "validation_loss": val_loss,
        **metric_values,
    }
    torch.save({"model": model.state_dict(), "metadata": metadata}, output)

    if metrics_path is not None:
        metrics_file = Path(metrics_path)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def _smoke(checkpoint: str | Path) -> None:
    config = FlashConfig()
    detector = NeuralPitchDetector(config, checkpoint)
    frame = np.zeros(config.window_samples, dtype=np.float32)
    estimates = detector.detect(frame)
    if estimates:
        raise RuntimeError("silence smoke test unexpectedly emitted notes")
    print("checkpoint smoke test: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/test MuScripter Flash neural backend")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the synthetic bootstrap model")
    train.add_argument("--output", required=True)
    train.add_argument("--metrics")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--seed", type=int, default=20261006)

    smoke = subparsers.add_parser("smoke", help="load a checkpoint and run silence inference")
    smoke.add_argument("checkpoint")

    args = parser.parse_args()
    if args.command == "train":
        train_synthetic(
            args.output,
            metrics_path=args.metrics,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
    else:
        _smoke(args.checkpoint)


if __name__ == "__main__":
    main()
