"""Tests for dependency-light helpers in the Gradio Colab app."""

from muscriptor.colab_app import _safe_stem, _tempo_mode


def test_safe_stem_strips_paths_and_unsafe_characters():
    assert _safe_stem("../../Gardenia?.wav") == "Gardenia_"


def test_safe_stem_never_returns_empty():
    assert _safe_stem("?.wav") == "_"


def test_tempo_modes_match_transcription_api():
    assert _tempo_mode("Best effort") == "best-effort"
    assert _tempo_mode("Strict") is True
    assert _tempo_mode("Off") is False
