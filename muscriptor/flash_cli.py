"""Command-line interface for MuScripter Flash."""

from __future__ import annotations

from typing import Annotated

import typer

from muscriptor.flash import (
    FlashConfig,
    FlashEngine,
    LiveFlashSession,
    MidoMidiSink,
    list_audio_devices,
    list_midi_outputs,
)

app = typer.Typer(
    add_completion=False,
    help="MuScripter Flash — low-latency live audio-to-MIDI",
)


@app.command()
def live(
    midi_port: Annotated[
        str | None,
        typer.Option(
            "--midi-port",
            "-m",
            help="System MIDI output port. A unique substring is accepted.",
        ),
    ] = None,
    input_device: Annotated[
        str | None,
        typer.Option(
            "--input-device",
            "-i",
            help="sounddevice input index or device name.",
        ),
    ] = None,
    virtual_midi: Annotated[
        bool,
        typer.Option(
            "--virtual-midi",
            help="Create a virtual MIDI output named 'MuScripter Flash'.",
        ),
    ] = False,
    sample_rate: Annotated[
        int,
        typer.Option("--sample-rate", help="Audio sample rate used by Flash."),
    ] = 16_000,
    window_ms: Annotated[
        float,
        typer.Option(
            "--window-ms",
            help="Rolling analysis window in milliseconds.",
        ),
    ] = 128.0,
    hop_ms: Annotated[
        float,
        typer.Option(
            "--hop-ms",
            help="Time between analyses in milliseconds.",
        ),
    ] = 32.0,
    latency_budget_ms: Annotated[
        float,
        typer.Option(
            "--latency-budget-ms",
            help="Target NoteOn latency budget used for runtime diagnostics.",
        ),
    ] = 250.0,
    confidence: Annotated[
        float,
        typer.Option(
            "--confidence",
            help="Absolute pitch confidence floor (0..1).",
        ),
    ] = 0.46,
    max_polyphony: Annotated[
        int,
        typer.Option(
            "--max-polyphony",
            help="Maximum simultaneous MIDI notes emitted by the MVP detector.",
        ),
    ] = 12,
    attack_frames: Annotated[
        int,
        typer.Option(
            "--attack-frames",
            help="Consecutive positive frames required before NoteOn.",
        ),
    ] = 1,
    release_frames: Annotated[
        int,
        typer.Option(
            "--release-frames",
            help="Missing frames required before NoteOff.",
        ),
    ] = 3,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Print emitted MIDI events and latency."),
    ] = False,
) -> None:
    """Stream microphone audio to a MIDI output in real time."""
    try:
        config = FlashConfig(
            sample_rate=sample_rate,
            window_ms=window_ms,
            hop_ms=hop_ms,
            latency_budget_ms=latency_budget_ms,
            min_confidence=confidence,
            max_polyphony=max_polyphony,
            attack_frames=attack_frames,
            release_frames=release_frames,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2)

    nominal = config.nominal_note_on_latency_ms
    if nominal >= config.latency_budget_ms:
        typer.echo(
            "Warning: the configured analysis/attack wait already uses "
            f"{nominal:.1f} ms of the {config.latency_budget_ms:.1f} ms budget "
            "before compute/device overhead.",
            err=True,
        )

    try:
        sink = MidoMidiSink(midi_port, virtual=virtual_midi)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    # Numeric strings are the most convenient form for sounddevice device IDs.
    resolved_device: int | str | None = input_device
    if input_device is not None:
        try:
            resolved_device = int(input_device)
        except ValueError:
            pass

    engine = FlashEngine(config)
    session = LiveFlashSession(
        engine=engine,
        sink=sink,
        input_device=resolved_device,
    )

    typer.echo("MuScripter Flash")
    typer.echo(f"  MIDI: {sink.name}")
    typer.echo(
        f"  analysis: {config.window_ms:.0f} ms window / {config.hop_ms:.0f} ms hop"
    )
    typer.echo(
        f"  nominal NoteOn latency: {config.nominal_note_on_latency_ms:.1f} ms "
        f"(budget {config.latency_budget_ms:.1f} ms)"
    )
    typer.echo("Press Ctrl+C to stop.")

    try:
        session.run(verbose=verbose)
    except KeyboardInterrupt:
        typer.echo("Stopped.")
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    finally:
        stats = engine.stats
        typer.echo(
            f"Flash stats: analyses={stats.analyses}, note_on={stats.note_ons}, "
            f"note_off={stats.note_offs}, max_compute={stats.max_compute_ms:.1f}ms, "
            f"budget_misses={stats.budget_misses}",
            err=True,
        )


@app.command("list-audio")
def list_audio() -> None:
    """List available microphone/input devices."""
    try:
        devices = list_audio_devices()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if not devices:
        typer.echo("No input devices found.")
    for device in devices:
        typer.echo(device)


@app.command("list-midi")
def list_midi() -> None:
    """List available MIDI output ports."""
    try:
        ports = list_midi_outputs()
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)
    if not ports:
        typer.echo("No MIDI output ports found.")
    for port in ports:
        typer.echo(port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
