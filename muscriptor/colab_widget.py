"""Google Colab widget for interactive MuScriptor transcription.

This module intentionally has no ipywidgets/google-colab dependency at import time.
Call :func:`launch` inside a Colab notebook after installing `ipywidgets`.
"""

from __future__ import annotations

import gc
import html
import re
from pathlib import Path
from typing import Any

_MODEL = None
_MODEL_KEY: tuple[str, str | None, str | None] | None = None
_LAST_OUTPUT: Path | None = None


def _safe_filename(name: str) -> str:
    """Return a filesystem-safe upload name while preserving the extension."""

    name = Path(name or "input.wav").name
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip(" .")
    return cleaned or "input.wav"


def _first_upload(value: Any) -> tuple[str, bytes] | None:
    """Read one FileUpload value from ipywidgets 7 or 8."""

    if not value:
        return None

    if isinstance(value, dict):
        # ipywidgets 7: {filename: {name, content, ...}}
        item = next(iter(value.values()))
    else:
        # ipywidgets 8: tuple/list of upload records
        item = value[0]

    if isinstance(item, dict):
        name = item.get("name") or item.get("metadata", {}).get("name") or "input.wav"
        data = item.get("content")
    else:
        name = getattr(item, "name", "input.wav")
        data = getattr(item, "content", None)

    if data is None:
        return None
    return _safe_filename(str(name)), bytes(data)


def _tempo_mode(value: str):
    return {"best-effort": "best-effort", "strict": True, "off": False}[value]


def _load_model(model_name: str, device: str, dtype: str):
    """Load at most one model into memory and reuse it across transcriptions."""

    global _MODEL, _MODEL_KEY

    from muscriptor.transcription_model import TranscriptionModel

    device_arg = None if device == "auto" else device
    dtype_arg = None if dtype == "auto" else dtype
    key = (model_name, device_arg, dtype_arg)

    if _MODEL is not None and _MODEL_KEY == key:
        return _MODEL, False

    # Colab GPU RAM is precious. Keep one checkpoint resident at a time.
    _MODEL = None
    _MODEL_KEY = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    _MODEL = TranscriptionModel.load_model(
        weights_path=model_name,
        device=device_arg,
        dtype=dtype_arg,
    )
    _MODEL_KEY = key
    return _MODEL, True


def launch():
    """Display the MuScriptor Colab UI and return its top-level widget."""

    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError as exc:
        raise RuntimeError(
            "ipywidgets is required. In Colab run: %pip install -q ipywidgets"
        ) from exc

    try:
        from google.colab import output as colab_output

        colab_output.enable_custom_widget_manager()
    except ImportError:
        pass

    title = widgets.HTML(
        """
        <div style="padding:14px 16px;border:1px solid #ddd;border-radius:12px">
          <div style="font-size:22px;font-weight:700">MuScriptor · Colab</div>
          <div style="margin-top:4px;color:#666">
            Upload audio → transcribe with Small / Medium / Large → download MIDI.
            The model stays loaded for the next song.
          </div>
        </div>
        """
    )

    upload = widgets.FileUpload(
        accept=".wav,.mp3,.flac,.ogg,.oga,.m4a,.aac",
        multiple=False,
        description="Audio",
    )
    model = widgets.Dropdown(
        options=[
            ("Large · 1.4B · best quality", "large"),
            ("Medium · 307M", "medium"),
            ("Small · 103M", "small"),
        ],
        value="large",
        description="Model",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )
    device = widgets.Dropdown(
        options=[("Auto", "auto"), ("CUDA", "cuda"), ("CPU", "cpu")],
        value="auto",
        description="Device",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    dtype = widgets.Dropdown(
        options=[
            ("Auto", "auto"),
            ("float32", "float32"),
            ("float16", "float16"),
            ("bfloat16", "bfloat16"),
        ],
        value="auto",
        description="dtype",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="260px"),
    )
    tempo = widgets.Dropdown(
        options=[
            ("Best effort · recommended", "best-effort"),
            ("Strict fixed tempo", "strict"),
            ("Off · 120 BPM placeholder", "off"),
        ],
        value="best-effort",
        description="Tempo",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="420px"),
    )
    quantize = widgets.Checkbox(
        value=False,
        description="Quantize to detected subdivision",
        indent=False,
    )
    dynamic_velocity = widgets.Checkbox(
        value=True,
        description="Dynamic velocity from source audio",
        indent=False,
    )
    instruments = widgets.Text(
        value="",
        placeholder="e.g. piano,bass,drums  (blank = auto)",
        description="Instruments",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="520px"),
    )
    hf_token = widgets.Password(
        value="",
        placeholder="optional if HF_TOKEN secret/login is already configured",
        description="HF token",
        style={"description_width": "90px"},
        layout=widgets.Layout(width="520px"),
    )

    transcribe = widgets.Button(
        description="Transcribe",
        button_style="primary",
        icon="music",
        layout=widgets.Layout(width="160px", height="40px"),
    )
    download = widgets.Button(
        description="Download MIDI",
        icon="download",
        disabled=True,
        layout=widgets.Layout(width="160px", height="40px"),
    )
    status = widgets.HTML("<span style='color:#666'>Ready.</span>")
    log = widgets.Output(
        layout={
            "border": "1px solid #eee",
            "padding": "8px",
            "max_height": "280px",
            "overflow_y": "auto",
        }
    )

    def set_busy(busy: bool) -> None:
        transcribe.disabled = busy
        upload.disabled = busy
        model.disabled = busy
        device.disabled = busy
        dtype.disabled = busy
        tempo.disabled = busy
        quantize.disabled = busy
        dynamic_velocity.disabled = busy
        instruments.disabled = busy
        hf_token.disabled = busy

    def on_transcribe(_button) -> None:
        global _LAST_OUTPUT

        uploaded = _first_upload(upload.value)
        if uploaded is None:
            status.value = "<b style='color:#c62828'>Choose one audio file first.</b>"
            return

        name, data = uploaded
        workspace = Path("/content/muscriptor") if Path("/content").exists() else Path.cwd()
        workspace.mkdir(parents=True, exist_ok=True)
        input_path = workspace / name
        input_path.write_bytes(data)
        output_path = workspace / f"{input_path.stem}.mid"

        set_busy(True)
        download.disabled = True
        _LAST_OUTPUT = None
        log.clear_output()
        status.value = "<b>Loading model…</b>"

        try:
            with log:
                print(f"Input: {input_path.name}")
                print(f"Model: {model.value}")

                token = hf_token.value.strip()
                if not token:
                    try:
                        from google.colab import userdata

                        token = userdata.get("HF_TOKEN") or ""
                    except Exception:
                        token = ""
                if token:
                    from huggingface_hub import login

                    login(token=token, add_to_git_credential=False)
                    hf_token.value = ""
                    print("Hugging Face authentication: OK")

                loaded_model, newly_loaded = _load_model(
                    model.value, device.value, dtype.value
                )
                print("Model loaded." if newly_loaded else "Reusing cached model.")

                instrument_names = None
                if instruments.value.strip():
                    from muscriptor.tokenizer.mt3 import resolve_instrument_names

                    tokens = [
                        token.strip()
                        for token in instruments.value.split(",")
                        if token.strip()
                    ]
                    instrument_names = resolve_instrument_names(tokens)
                    print("Instruments:", ", ".join(instrument_names))

                status.value = "<b>Transcribing…</b>"
                midi_bytes, grid = loaded_model.transcribe_and_postprocess(
                    input_path,
                    instruments=instrument_names,
                    detect_tempo=_tempo_mode(tempo.value),
                    quantize=quantize.value,
                    dynamic_velocity=dynamic_velocity.value,
                )
                output_path.write_bytes(midi_bytes)
                _LAST_OUTPUT = output_path

                if grid is None:
                    tempo_text = "120 BPM placeholder"
                else:
                    meter = (
                        f", {grid.beats_per_bar}/4"
                        if grid.beats_per_bar is not None
                        else ""
                    )
                    tempo_text = f"{grid.bpm:.3f} BPM{meter}"

                print(f"Tempo: {tempo_text}")
                print(f"Saved: {output_path}")
                print(f"Size: {len(midi_bytes) / 1024:.1f} KiB")

            status.value = (
                "<b style='color:#2e7d32'>Done.</b> "
                + html.escape(output_path.name)
            )
            download.disabled = False
        except Exception as exc:
            status.value = (
                "<b style='color:#c62828'>Failed:</b> " + html.escape(str(exc))
            )
            with log:
                import traceback

                traceback.print_exc()
        finally:
            set_busy(False)

    def on_download(_button) -> None:
        if _LAST_OUTPUT is None or not _LAST_OUTPUT.exists():
            return
        try:
            from google.colab import files

            files.download(str(_LAST_OUTPUT))
        except ImportError:
            with log:
                print(f"MIDI is saved at: {_LAST_OUTPUT}")

    transcribe.on_click(on_transcribe)
    download.on_click(on_download)

    advanced = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.HBox([device, dtype]),
                    instruments,
                    hf_token,
                    widgets.HTML(
                        "<small>Accept the MuScriptor model license on Hugging Face "
                        "before the first run. In Colab you can also save HF_TOKEN "
                        "under Secrets.</small>"
                    ),
                    quantize,
                    dynamic_velocity,
                ],
                layout=widgets.Layout(gap="6px"),
            )
        ],
        selected_index=None,
    )
    advanced.set_title(0, "Advanced")

    panel = widgets.VBox(
        [
            title,
            upload,
            model,
            tempo,
            advanced,
            widgets.HBox([transcribe, download]),
            status,
            log,
        ],
        layout=widgets.Layout(width="100%", gap="10px"),
    )
    display(panel)
    return panel
