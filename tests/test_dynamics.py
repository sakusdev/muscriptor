"""Tests for audio-derived MIDI dynamics."""

import torch

from muscriptor.tokenizer.notes import Note
from muscriptor.utils.dynamics import apply_audio_velocities
from muscriptor.utils.midi import notes_to_midi


def _note(onset: float, pitch: int, program: int = 0, is_drum: bool = False) -> Note:
    return Note(
        is_drum=is_drum,
        program=program,
        onset=onset,
        offset=onset + 0.2,
        pitch=pitch,
    )


def _audio_with_blocks(sample_rate: int = 16000) -> torch.Tensor:
    wav = torch.zeros(1, sample_rate * 2)
    # Quiet and loud attacks with identical duration so the estimator sees level,
    # not just event length.
    wav[:, int(0.20 * sample_rate) : int(0.30 * sample_rate)] = 0.05
    wav[:, int(1.00 * sample_rate) : int(1.10 * sample_rate)] = 0.50
    return wav


def test_louder_attack_gets_higher_velocity():
    notes = [_note(0.20, 60), _note(1.00, 64)]
    out = apply_audio_velocities(notes, _audio_with_blocks(), 16000)

    assert out[0].midi_velocity is not None
    assert out[1].midi_velocity is not None
    assert out[1].midi_velocity > out[0].midi_velocity


def test_chord_notes_share_the_same_accent():
    notes = [_note(1.00, 60), _note(1.00, 64), _note(1.00, 67)]
    out = apply_audio_velocities(notes, _audio_with_blocks(), 16000)

    assert len({n.midi_velocity for n in out}) == 1


def test_silence_preserves_legacy_fixed_velocity_fallback():
    notes = [_note(0.20, 60)]
    out = apply_audio_velocities(notes, torch.zeros(1, 32000), 16000)

    assert out[0].midi_velocity is None


def test_programs_are_normalized_independently():
    # Both instruments see the same source mix, but each program builds its own
    # dynamic bounds instead of one loud instrument crushing the other's range.
    notes = [
        _note(0.20, 60, program=0),
        _note(1.00, 64, program=0),
        _note(0.20, 40, program=33),
        _note(1.00, 43, program=33),
    ]
    out = apply_audio_velocities(notes, _audio_with_blocks(), 16000)

    piano = [n.midi_velocity for n in out if n.program == 0]
    bass = [n.midi_velocity for n in out if n.program == 33]
    assert piano[1] > piano[0]
    assert bass[1] > bass[0]


def test_midi_writer_uses_per_note_velocity_when_present():
    notes = [
        Note(False, 0, 0.0, 0.2, 60, midi_velocity=42),
        Note(False, 0, 0.5, 0.7, 64, midi_velocity=117),
    ]
    midi = notes_to_midi(notes)

    velocities = [
        msg.velocity
        for track in midi.tracks
        for msg in track
        if msg.type == "note_on" and msg.velocity > 0
    ]
    assert velocities == [42, 117]
