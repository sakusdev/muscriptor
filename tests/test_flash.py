import math

import numpy as np
import pytest

from muscriptor.flash import FlashConfig, FlashEngine


def _tone(
    frequencies: list[float],
    duration: float,
    sample_rate: int = 16_000,
    amplitude: float = 0.35,
) -> np.ndarray:
    t = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
    signal = np.zeros_like(t)
    for frequency in frequencies:
        signal += np.sin(2.0 * math.pi * frequency * t).astype(np.float32)
    signal *= amplitude / max(len(frequencies), 1)
    return signal


def _feed(engine: FlashEngine, audio: np.ndarray) -> list:
    events = []
    block = engine.config.hop_samples
    for start in range(0, len(audio), block):
        events.extend(engine.process_block(audio[start : start + block]))
    return events


def test_default_note_on_wait_fits_250ms_budget():
    config = FlashConfig()
    assert config.nominal_note_on_latency_ms == pytest.approx(128.0)
    assert config.nominal_note_on_latency_ms < config.latency_budget_ms


def test_rejects_invalid_latency_geometry():
    with pytest.raises(ValueError, match="hop_ms"):
        FlashConfig(window_ms=64.0, hop_ms=80.0)


def test_a4_note_on_and_note_off():
    engine = FlashEngine(FlashConfig(min_confidence=0.40, release_frames=2))

    on_events = _feed(engine, _tone([440.0], 0.45))
    a4_on = [
        event for event in on_events if event.type == "note_on" and event.note == 69
    ]
    assert a4_on
    assert a4_on[0].latency_ms < 250.0
    assert 1 <= a4_on[0].velocity <= 127
    assert 69 in engine.active_notes

    off_events = _feed(engine, np.zeros(round(0.25 * 16_000), dtype=np.float32))
    assert any(event.type == "note_off" and event.note == 69 for event in off_events)
    assert 69 not in engine.active_notes


def test_a_major_triad_is_polyphonic():
    config = FlashConfig(min_confidence=0.40, max_polyphony=8)
    engine = FlashEngine(config)
    events = _feed(engine, _tone([440.0, 554.365, 659.255], 0.45))

    notes = {event.note for event in events if event.type == "note_on"}
    assert {69, 73, 76}.issubset(notes)


def test_all_notes_off_flushes_state():
    engine = FlashEngine(FlashConfig(min_confidence=0.40))
    _feed(engine, _tone([261.626], 0.35))
    assert 60 in engine.active_notes

    flushed = engine.all_notes_off()
    assert any(event.type == "note_off" and event.note == 60 for event in flushed)
    assert not engine.active_notes
