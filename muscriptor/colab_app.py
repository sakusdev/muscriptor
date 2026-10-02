"""Gradio-based Google Colab UI for MuScriptor.

Gradio is substantially more reliable than ipywidgets inside Colab for file
upload/download workflows, while still rendering inline as a notebook widget.
"""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path

_MODEL = None
_MODEL_KEY: tuple[str, str | None, str | None] | None = None

_WORKDIR = Path("/content/muscriptor") if Path("/content").exists() else Path.cwd()
_WORKDIR.mkdir(parents=True, exist_ok=True)


def _safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" .")
    return stem or "transcription"


def _tempo_mode(value: str):
    return {"Best effort": "best-effort", "Strict": True, "Off": False}[value]


def _maybe_hf_login(token: str) -> str:
    """Authenticate from explicit token, Colab Secret, or existing environment."""

    token = (token or "").strip()
    if not token:
        token = os.environ.get("HF_TOKEN", "").strip()

    if not token:
        try:
            from google.colab import userdata

            token = (userdata.get("HF_TOKEN") or "").strip()
        except Exception:  # noqa: BLE001 - Colab secret provider uses custom errors
            token = ""

    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        return "Hugging Face authentication: OK"

    return "Hugging Face token: not provided (existing login/cache may still work)"


def _load_model(model_name: str, device: str, dtype: str):
    """Keep exactly one checkpoint resident and reuse it between songs."""

    global _MODEL, _MODEL_KEY

    from muscriptor.transcription_model import TranscriptionModel

    device_arg = None if device == "Auto" else device.lower()
    dtype_arg = None if dtype == "Auto" else dtype
    key = (model_name, device_arg, dtype_arg)

    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL, False

    _MODEL = None
    _MODEL_KEY = None
    gc.collect()

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _MODEL = TranscriptionModel.load_model(
        weights_path=model_name,
        device=device_arg,
        dtype=dtype_arg,
    )
    _MODEL_KEY = key
    return _MODEL, True


def transcribe_file(
    audio_path: str | None,
    model_name: str,
    device: str,
    dtype: str,
    tempo: str,
    quantize: bool,
    dynamic_velocity: bool,
    instruments: str,
    hf_token: str,
) -> tuple[str | None, str]:
    """Transcribe one uploaded audio file and return (MIDI path, status text)."""

    if not audio_path:
        return None, "Upload an audio file first."

    audio = Path(audio_path)
    if not audio.exists():
        return None, f"Input file not found: {audio}"

    lines: list[str] = []
    try:
        lines.append(_maybe_hf_login(hf_token))
        lines.append(f"Model: {model_name}")

        model, loaded = _load_model(model_name, device, dtype)
        lines.append("Model loaded." if loaded else "Reusing cached model.")

        instrument_names = None
        if instruments.strip():
            from muscriptor.tokenizer.mt3 import resolve_instrument_names

            tokens = [part.strip() for part in instruments.split(",") if part.strip()]
            instrument_names = resolve_instrument_names(tokens)
            lines.append("Instruments: " + ", ".join(instrument_names))

        midi_bytes, grid = model.transcribe_and_postprocess(
            audio,
            instruments=instrument_names,
            detect_tempo=_tempo_mode(tempo),
            quantize=quantize,
            dynamic_velocity=dynamic_velocity,
        )

        out = _WORKDIR / f"{_safe_stem(audio)}.mid"
        out.write_bytes(midi_bytes)

        if grid is None:
            lines.append("Tempo: 120 BPM placeholder")
        else:
            meter = (
                f", {grid.beats_per_bar}/4"
                if grid.beats_per_bar is not None
                else ", meter not fixed"
            )
            lines.append(f"Tempo: {grid.bpm:.3f} BPM{meter}")

        lines.append(f"Saved: {out.name}")
        lines.append(f"Size: {len(midi_bytes) / 1024:.1f} KiB")
        return str(out), "\n".join(lines)

    except Exception as exc:  # noqa: BLE001 - UI surfaces model/download errors verbatim
        import traceback

        lines.append("")
        lines.append(f"ERROR: {exc}")
        lines.append(traceback.format_exc())
        return None, "\n".join(lines)


def build_app():
    """Build and return the Gradio Blocks app."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required. In Colab run: %pip install -q 'gradio>=5,<7'"
        ) from exc

    with gr.Blocks(title="MuScriptor · Colab") as demo:
        gr.Markdown(
            """
# MuScriptor · Colab

Upload audio, transcribe it with MuScriptor, and download the resulting MIDI.
**Large (1.4B)** is selected by default. The loaded checkpoint is cached for the
next song in the same Colab session.
"""
        )

        with gr.Row():
            audio = gr.Audio(
                label="Audio",
                type="filepath",
                sources=["upload"],
            )
            midi = gr.File(label="MIDI output", interactive=False)

        with gr.Row():
            model_name = gr.Dropdown(
                choices=["large", "medium", "small"],
                value="large",
                label="Model",
            )
            tempo = gr.Dropdown(
                choices=["Best effort", "Strict", "Off"],
                value="Best effort",
                label="Tempo detection",
            )

        with gr.Accordion("Advanced", open=False):
            with gr.Row():
                device = gr.Dropdown(
                    choices=["Auto", "cuda", "cpu"],
                    value="Auto",
                    label="Device",
                )
                dtype = gr.Dropdown(
                    choices=["Auto", "float32", "float16", "bfloat16"],
                    value="Auto",
                    label="dtype",
                )
            instruments = gr.Textbox(
                label="Instrument constraints",
                placeholder="e.g. piano,bass,drums  (blank = auto)",
            )
            hf_token = gr.Textbox(
                label="Hugging Face token (optional)",
                type="password",
                placeholder="Uses Colab Secret HF_TOKEN automatically when available",
            )
            quantize = gr.Checkbox(
                value=False,
                label="Quantize to detected subdivision",
            )
            dynamic_velocity = gr.Checkbox(
                value=True,
                label="Dynamic velocity from source audio",
            )

        run = gr.Button("Transcribe", variant="primary")
        status = gr.Textbox(
            label="Status / log",
            lines=10,
            interactive=False,
        )

        run.click(
            fn=transcribe_file,
            inputs=[
                audio,
                model_name,
                device,
                dtype,
                tempo,
                quantize,
                dynamic_velocity,
                instruments,
                hf_token,
            ],
            outputs=[midi, status],
            show_progress="full",
        )

        gr.Markdown(
            """
**Before first use:** accept the MuScriptor model conditions on Hugging Face.
You can store a token as a Colab Secret named \`HF_TOKEN\`; then you do not need
to paste it into this UI.
"""
        )

    return demo


def launch(**kwargs):
    """Launch the Colab app inline."""

    demo = build_app()
    launch_kwargs = {
        "inline": True,
        "show_error": True,
        "quiet": False,
    }
    launch_kwargs.update(kwargs)
    return demo.launch(**launch_kwargs)
