import json
from pathlib import Path

import mido
import numpy as np

from muscriptor.flash_colab import (
    _audio_to_flash_pcm,
    _web_midi_payload,
    _write_midi,
    process_stream,
    start_flash,
    stop_flash,
)


def _a4_chunk(duration: float = 0.20, sample_rate: int = 16_000):
    t = np.arange(round(duration * sample_rate), dtype=np.float32) / sample_rate
    audio = (0.35 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
    return sample_rate, audio


def test_audio_bridge_normalizes_int16_and_resamples():
    sample_rate = 8_000
    t = np.arange(800, dtype=np.float32) / sample_rate
    mono = (0.25 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
    stereo = np.stack([mono, mono], axis=1)

    pcm, seconds = _audio_to_flash_pcm((sample_rate, stereo), 16_000)

    assert pcm.dtype == np.float32
    assert pcm.ndim == 1
    assert 1500 <= len(pcm) <= 1700
    assert seconds == 0.1
    assert np.max(np.abs(pcm)) <= 1.05


def test_streaming_session_exports_midi_and_web_midi(tmp_path: Path, monkeypatch):
    from muscriptor import flash_colab

    monkeypatch.setattr(flash_colab, "_WORKDIR", tmp_path)
    state, _, _, start_payload = start_flash(128, 32, 0.40, 2, 8)
    assert json.loads(start_payload)["messages"] == []

    realtime_messages = []
    for _ in range(4):
        state, _, payload = process_stream(_a4_chunk(), state)
        realtime_messages.extend(json.loads(payload)["messages"])

    state, midi_path, status, stop_payload = stop_flash(state)
    realtime_messages.extend(json.loads(stop_payload)["messages"])

    assert midi_path is not None
    assert "MIDI saved" in status
    assert any(message[0] == 0x90 and message[1] == 69 for message in realtime_messages)
    assert any(message[0] == 0x80 and message[1] == 69 for message in realtime_messages)

    midi = mido.MidiFile(midi_path)
    notes = [
        message.note
        for track in midi.tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    ]
    assert 69 in notes


def test_web_midi_payload_serializes_note_messages():
    from muscriptor.flash import FlashMidiEvent
    from muscriptor.flash_colab import _new_state

    state = _new_state()
    payload = json.loads(
        _web_midi_payload(
            state,
            [
                FlashMidiEvent("note_on", 60, 91, 0.1, 0.9, 128.0),
                FlashMidiEvent("note_off", 60, 0, 0.5, 0.8, 0.0),
            ],
        )
    )

    assert payload["seq"] == 1
    assert payload["messages"] == [[0x90, 60, 91], [0x80, 60, 0]]


def test_write_midi_preserves_note_on_and_off(tmp_path: Path):
    from muscriptor.flash import FlashMidiEvent

    path = tmp_path / "events.mid"
    events = [
        FlashMidiEvent("note_on", 60, 90, 0.10, 0.9, 130.0),
        FlashMidiEvent("note_off", 60, 0, 0.60, 0.9, 0.0),
    ]
    _write_midi(events, path)

    midi = mido.MidiFile(path)
    channel_messages = [
        message
        for track in midi.tracks
        for message in track
        if message.type in {"note_on", "note_off"}
    ]
    assert [message.type for message in channel_messages] == ["note_on", "note_off"]
    assert channel_messages[0].note == 60
    assert channel_messages[1].note == 60
