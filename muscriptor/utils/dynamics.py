"""Audio-derived MIDI velocity estimation.

MuScriptor's note tokens carry only binary on/off state, so the legacy MIDI writer
uses one fixed playback velocity for every onset. This module recovers useful
macro-dynamics from the source recording without changing the transcription model.

The estimator intentionally stays conservative: it measures a short post-onset
energy envelope from the mix, then normalizes it per decoded instrument. Notes in
the same chord therefore share an accent, while piano/bass/drums each keep their
own dynamic range. It is not source separation and should not be interpreted as
ground-truth performer velocity.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict

import torch
import torch.nn.functional as F

from muscriptor.tokenizer.notes import Note

_ENVELOPE_WINDOW_S = 0.080
_ENVELOPE_HOP_S = 0.005
_MIN_ACTIVE_DB = -70.0
_PITCHED_VELOCITY_RANGE = (68, 124)
_DRUM_VELOCITY_RANGE = (54, 127)
_MIN_DYNAMIC_SPAN_DB = 6.0


def _energy_envelope(wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Return a short-time attack-sensitive amplitude envelope on CPU."""

    x = wav.detach().float().cpu()
    if x.ndim == 2:
        x = x.mean(dim=0)
    elif x.ndim != 1:
        raise ValueError(f"wav must have shape [T] or [C, T], got {tuple(x.shape)}")

    window = max(1, round(_ENVELOPE_WINDOW_S * sample_rate))
    hop = max(1, round(_ENVELOPE_HOP_S * sample_rate))
    if x.numel() < window:
        x = F.pad(x, (0, window - x.numel()))

    framed = x[None, None, :]
    rms = torch.sqrt(
        F.avg_pool1d(framed.square(), kernel_size=window, stride=hop).clamp_min(1e-12)
    ).flatten()
    peak = F.max_pool1d(framed.abs(), kernel_size=window, stride=hop).flatten()

    # RMS carries the musical level; a small peak term makes attacks and drum
    # accents survive dense mixes a little better.
    return 0.80 * rms + 0.20 * peak


def _robust_bounds(levels_db: torch.Tensor) -> tuple[float, float]:
    """Robust dB bounds with a minimum span around the median."""

    if levels_db.numel() == 1:
        center = float(levels_db[0])
        half = _MIN_DYNAMIC_SPAN_DB / 2
        return center - half, center + half

    low = float(torch.quantile(levels_db, 0.10))
    high = float(torch.quantile(levels_db, 0.90))
    if high - low >= _MIN_DYNAMIC_SPAN_DB:
        return low, high

    center = float(torch.median(levels_db))
    half = _MIN_DYNAMIC_SPAN_DB / 2
    return center - half, center + half


def _velocity(level_db: float, low_db: float, high_db: float, is_drum: bool) -> int:
    lo, hi = _DRUM_VELOCITY_RANGE if is_drum else _PITCHED_VELOCITY_RANGE
    phase = (level_db - low_db) / max(high_db - low_db, 1e-6)
    phase = max(0.0, min(1.0, phase))

    # Slight upward curve: ordinary notes remain near the familiar ~100 range,
    # while genuinely quiet/loud attacks still separate clearly.
    phase = phase**0.75
    return int(round(lo + (hi - lo) * phase))


def apply_audio_velocities(
    notes: list[Note], wav: torch.Tensor, sample_rate: int
) -> list[Note]:
    """Return notes with audio-derived midi_velocity values.

    Dynamics are normalized independently for each decoded MIDI program (drums
    included). Chords do not overweight the normalization: each 5 ms onset frame
    contributes once per program.

    On silent input the notes are returned unchanged, preserving the legacy fixed
    velocity fallback in the MIDI writer.
    """

    if not notes:
        return notes

    envelope = _energy_envelope(wav, sample_rate)
    if envelope.numel() == 0:
        return notes

    levels_db = 20.0 * torch.log10(envelope.clamp_min(1e-7))
    if float(torch.quantile(levels_db, 0.95)) < _MIN_ACTIVE_DB:
        return notes

    hop = max(1, round(_ENVELOPE_HOP_S * sample_rate))
    max_frame = len(levels_db) - 1

    frame_for_note: list[int] = []
    grouped_frames: dict[tuple[int, bool], set[int]] = defaultdict(set)
    for note in notes:
        frame = round(note.onset * sample_rate / hop)
        frame = max(0, min(max_frame, frame))
        frame_for_note.append(frame)
        grouped_frames[(note.program, note.is_drum)].add(frame)

    bounds: dict[tuple[int, bool], tuple[float, float]] = {}
    for group, frames in grouped_frames.items():
        values = levels_db[torch.tensor(sorted(frames), dtype=torch.long)]
        bounds[group] = _robust_bounds(values)

    out: list[Note] = []
    for note, frame in zip(notes, frame_for_note, strict=True):
        low_db, high_db = bounds[(note.program, note.is_drum)]
        out.append(
            dataclasses.replace(
                note,
                midi_velocity=_velocity(
                    float(levels_db[frame]), low_db, high_db, note.is_drum
                ),
            )
        )
    return out
