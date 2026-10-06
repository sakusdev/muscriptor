"""Google Colab / Gradio frontend for MuScripter Flash.

The Colab runtime cannot open the user's local microphone with PortAudio.
Instead, the browser captures microphone audio and Gradio streams short chunks
to the remote runtime. Flash then processes those chunks with the same
``FlashEngine`` used by the native CLI.

The engine's own latency accounting excludes browser/network transport. The UI
shows a rough server-side stream lag separately so Colab does not pretend to
provide a hard end-to-end 250 ms guarantee.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from muscriptor.flash import FlashConfig, FlashEngine, FlashMidiEvent
from muscriptor.utils.audio import resample

_WORKDIR = (
    Path("/content/muscriptor-flash") if Path("/content").exists() else Path.cwd()
)
_WORKDIR.mkdir(parents=True, exist_ok=True)

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


@dataclass
class FlashColabState:
    engine: FlashEngine
    events: list[FlashMidiEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    chunks: int = 0
    input_seconds: float = 0.0
    last_observed_lag_ms: float = 0.0


def _note_name(note: int) -> str:
    return f"{_NOTE_NAMES[note % 12]}{note // 12 - 1}"


def _new_state(
    window_ms: float = 128.0,
    hop_ms: float = 32.0,
    confidence: float = 0.46,
    release_frames: int = 3,
    max_polyphony: int = 12,
) -> FlashColabState:
    config = FlashConfig(
        window_ms=float(window_ms),
        hop_ms=float(hop_ms),
        min_confidence=float(confidence),
        release_frames=int(release_frames),
        max_polyphony=int(max_polyphony),
    )
    return FlashColabState(engine=FlashEngine(config))


def _audio_to_flash_pcm(
    audio: tuple[int, np.ndarray] | None,
    target_sample_rate: int,
) -> tuple[np.ndarray, float]:
    if audio is None:
        return np.empty(0, dtype=np.float32), 0.0

    sample_rate, values = audio
    array = np.asarray(values)
    if array.size == 0:
        return np.empty(0, dtype=np.float32), 0.0

    if np.issubdtype(array.dtype, np.integer):
        scale = float(max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32, copy=False)

    if array.ndim == 2:
        array = array.mean(axis=1, dtype=np.float32)
    elif array.ndim != 1:
        raise ValueError(f"Unexpected microphone audio shape: {array.shape}")

    input_seconds = len(array) / float(sample_rate)
    if sample_rate != target_sample_rate:
        waveform = torch.from_numpy(np.ascontiguousarray(array))[None, :]
        array = (
            resample(waveform, int(sample_rate), int(target_sample_rate))
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
    return np.ascontiguousarray(array), input_seconds


def _event_line(event: FlashMidiEvent) -> str:
    name = _note_name(event.note)
    if event.type == "note_on":
        return (
            f"{event.stream_time:7.3f}s  ON   {name:4s}  "
            f"vel={event.velocity:3d}  conf={event.confidence:.2f}  "
            f"engine={event.latency_ms:.1f}ms"
        )
    return f"{event.stream_time:7.3f}s  OFF  {name:4s}"


def _status(state: FlashColabState, extra: str | None = None) -> str:
    engine = state.engine
    stats = engine.stats
    active = ", ".join(_note_name(note) for note in sorted(engine.active_notes)) or "—"
    recent = (
        "\n".join(_event_line(event) for event in state.events[-18:])
        or "(waiting for notes)"
    )
    prefix = f"{extra}\n\n" if extra else ""
    return (
        f"{prefix}Active notes: {active}\n"
        f"Chunks: {state.chunks} | audio: {state.input_seconds:.2f}s | "
        f"analyses: {stats.analyses}\n"
        f"Max Flash DSP: {stats.max_compute_ms:.1f}ms | "
        f"Flash budget misses: {stats.budget_misses} | "
        f"rough transport/queue lag: {state.last_observed_lag_ms:.1f}ms\n\n"
        f"Recent MIDI events\n{recent}"
    )


def start_flash(
    window_ms: float,
    hop_ms: float,
    confidence: float,
    release_frames: int,
    max_polyphony: int,
) -> tuple[FlashColabState, str, None]:
    """Reset Flash state when browser microphone recording starts."""
    state = _new_state(
        window_ms=window_ms,
        hop_ms=hop_ms,
        confidence=confidence,
        release_frames=release_frames,
        max_polyphony=max_polyphony,
    )
    return (
        state,
        _status(state, "Flash started. Play an instrument into the microphone."),
        None,
    )


def process_stream(
    audio: tuple[int, np.ndarray] | None,
    state: FlashColabState | None,
) -> tuple[FlashColabState, str]:
    """Process one browser microphone chunk."""
    if state is None:
        state = _new_state()

    pcm, input_seconds = _audio_to_flash_pcm(audio, state.engine.config.sample_rate)
    if pcm.size:
        state.events.extend(state.engine.process_block(pcm))
        state.chunks += 1
        state.input_seconds += input_seconds

        elapsed = time.perf_counter() - state.started_at
        state.last_observed_lag_ms = max(0.0, (elapsed - state.input_seconds) * 1000.0)

    return state, _status(state)


def _write_midi(events: list[FlashMidiEvent], output: Path) -> None:
    import mido

    ticks_per_beat = 480
    tempo = mido.bpm2tempo(120)
    midi = mido.MidiFile(type=0, ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="MuScripter Flash", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))

    ordered = sorted(
        events,
        key=lambda event: (event.stream_time, 0 if event.type == "note_off" else 1),
    )
    last_time = 0.0
    for event in ordered:
        delta_seconds = max(0.0, event.stream_time - last_time)
        delta_ticks = round(mido.second2tick(delta_seconds, ticks_per_beat, tempo))
        if event.type == "note_on":
            message = mido.Message(
                "note_on",
                note=event.note,
                velocity=event.velocity,
                channel=0,
                time=delta_ticks,
            )
        else:
            message = mido.Message(
                "note_off",
                note=event.note,
                velocity=0,
                channel=0,
                time=delta_ticks,
            )
        track.append(message)
        last_time = event.stream_time

    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(output)


def stop_flash(
    state: FlashColabState | None,
) -> tuple[FlashColabState, str | None, str]:
    """Flush active notes and export the live event stream to a MIDI file."""
    if state is None:
        state = _new_state()
        return state, None, _status(state, "No Flash session was active.")

    state.events.extend(state.engine.all_notes_off())
    if not state.events:
        return state, None, _status(state, "Stopped. No MIDI notes were detected.")

    output = _WORKDIR / f"muscriptor-flash-{int(time.time())}.mid"
    _write_midi(state.events, output)
    return state, str(output), _status(state, f"Stopped. MIDI saved as {output.name}")


def process_uploaded_file(
    audio: tuple[int, np.ndarray] | None,
    window_ms: float,
    hop_ms: float,
    confidence: float,
    release_frames: int,
    max_polyphony: int,
) -> tuple[str | None, str]:
    """Run the streaming engine over an uploaded clip for deterministic testing."""
    state = _new_state(
        window_ms=window_ms,
        hop_ms=hop_ms,
        confidence=confidence,
        release_frames=release_frames,
        max_polyphony=max_polyphony,
    )
    pcm, _ = _audio_to_flash_pcm(audio, state.engine.config.sample_rate)
    if pcm.size == 0:
        return None, "Upload an audio clip first."

    block = state.engine.config.hop_samples
    started = time.perf_counter()
    for offset in range(0, len(pcm), block):
        state.events.extend(state.engine.process_block(pcm[offset : offset + block]))
    state.events.extend(state.engine.all_notes_off())
    elapsed = time.perf_counter() - started

    output = _WORKDIR / f"muscriptor-flash-upload-{int(time.time())}.mid"
    _write_midi(state.events, output)
    realtime_factor = (len(pcm) / state.engine.config.sample_rate) / max(elapsed, 1e-9)
    summary = _status(
        state,
        f"Processed {len(pcm) / state.engine.config.sample_rate:.2f}s of audio in "
        f"{elapsed:.3f}s ({realtime_factor:.1f}x realtime).",
    )
    return str(output), summary


def build_app():
    """Build the browser-microphone Flash UI used in Colab."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is required. In Colab run: %pip install -q 'gradio>=6,<7'"
        ) from exc

    with gr.Blocks(title="MuScripter Flash · Colab") as demo:
        gr.Markdown(
            """
# MuScripter Flash · Colab

Live browser microphone → streamed audio chunks → **FlashEngine** → MIDI events.

The native Flash target is **≤250 ms NoteOn latency**. In Colab, browser/network
transport is additional and variable, so the UI reports Flash DSP latency and a
rough transport/queue lag separately. For hard realtime use, run
`muscriptor-flash live` locally.
"""
        )

        with gr.Accordion("Flash tuning", open=False):
            with gr.Row():
                window_ms = gr.Slider(64, 224, value=128, step=16, label="Window (ms)")
                hop_ms = gr.Slider(16, 64, value=32, step=8, label="Hop (ms)")
            with gr.Row():
                confidence = gr.Slider(
                    0.20, 0.90, value=0.46, step=0.01, label="Confidence floor"
                )
                release_frames = gr.Slider(
                    1, 8, value=3, step=1, label="Release frames"
                )
                max_polyphony = gr.Slider(
                    1, 24, value=12, step=1, label="Max polyphony"
                )

        with gr.Tab("Live microphone"):
            microphone = gr.Audio(
                label="Microphone",
                sources=["microphone"],
                type="numpy",
                streaming=True,
            )
            live_state = gr.State(value=None)
            live_midi = gr.File(label="Recorded Flash MIDI", interactive=False)
            live_status = gr.Textbox(
                label="Flash event stream / latency",
                lines=24,
                interactive=False,
            )

            microphone.start_recording(
                fn=start_flash,
                inputs=[
                    window_ms,
                    hop_ms,
                    confidence,
                    release_frames,
                    max_polyphony,
                ],
                outputs=[live_state, live_status, live_midi],
                queue=False,
            )
            microphone.stream(
                fn=process_stream,
                inputs=[microphone, live_state],
                outputs=[live_state, live_status],
                stream_every=0.1,
                time_limit=600,
                concurrency_limit=1,
                show_progress="hidden",
            )
            microphone.stop_recording(
                fn=stop_flash,
                inputs=[live_state],
                outputs=[live_state, live_midi, live_status],
            )

        with gr.Tab("Uploaded audio benchmark"):
            uploaded = gr.Audio(
                label="Audio clip",
                sources=["upload"],
                type="numpy",
            )
            uploaded_midi = gr.File(label="Flash MIDI", interactive=False)
            uploaded_status = gr.Textbox(
                label="Benchmark / events",
                lines=24,
                interactive=False,
            )
            run_upload = gr.Button("Run through FlashEngine", variant="primary")
            run_upload.click(
                fn=process_uploaded_file,
                inputs=[
                    uploaded,
                    window_ms,
                    hop_ms,
                    confidence,
                    release_frames,
                    max_polyphony,
                ],
                outputs=[uploaded_midi, uploaded_status],
            )

        gr.Markdown(
            """
**Tip:** Chrome/Edge usually gives the most reliable microphone permission in
Colab. If the inline widget cannot access the microphone, open the Gradio frame
in a new tab and grant microphone permission there.
"""
        )

    return demo


def launch(**kwargs):
    demo = build_app()
    launch_kwargs = {
        "inline": True,
        "show_error": True,
        "quiet": False,
    }
    launch_kwargs.update(kwargs)
    return demo.launch(**launch_kwargs)
