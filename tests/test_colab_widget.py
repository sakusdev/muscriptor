"""Tests for the dependency-light parts of the Colab widget."""

from types import SimpleNamespace

from muscriptor.colab_widget import _first_upload, _safe_filename, _tempo_mode


def test_safe_filename_strips_paths_and_unsafe_characters():
    assert _safe_filename("../../Gardenia?.wav") == "Gardenia_.wav"


def test_first_upload_supports_ipywidgets_7_dict_shape():
    value = {
        "song.wav": {
            "metadata": {"name": "song.wav"},
            "content": memoryview(b"abc"),
        }
    }
    assert _first_upload(value) == ("song.wav", b"abc")


def test_first_upload_supports_ipywidgets_8_sequence_shape():
    value = (SimpleNamespace(name="song.flac", content=memoryview(b"xyz")),)
    assert _first_upload(value) == ("song.flac", b"xyz")


def test_first_upload_handles_empty_value():
    assert _first_upload(()) is None
    assert _first_upload({}) is None


def test_tempo_modes_match_transcription_api():
    assert _tempo_mode("best-effort") == "best-effort"
    assert _tempo_mode("strict") is True
    assert _tempo_mode("off") is False
