# MuScripter Flash

MuScripter Flash is the low-latency live transcription path for MuScriptor.
It is designed around a **250 ms NoteOn latency budget** rather than the
five-second chunking used by the normal high-accuracy transformer.

## Status

This first implementation is an MVP for the realtime pipeline:

- microphone input through PortAudio / `sounddevice`
- 128 ms rolling analysis window by default
- 32 ms analysis hop
- polyphonic spectral pitch estimation
- immediate NoteOn with configurable attack frames
- debounced NoteOff with configurable release frames
- velocity estimation
- system MIDI output through `mido` + `python-rtmidi`
- latency instrumentation and budget-miss counting
- panic-style All Notes Off when the session exits
- Google Colab browser-microphone streaming UI
- uploaded-audio Flash benchmark mode for Colab

The spectral detector is intentionally lightweight and deterministic. It is
not intended to match the normal MuScriptor transformer's transcription
quality. It establishes the streaming API, timing behavior, MIDI state
management, and device path needed for a future causal neural Flash model.

## Install

```bash
uv sync --extra flash
```

or from a built package:

```bash
pip install 'muscriptor[flash]'
```

## Find devices

```bash
muscriptor-flash list-audio
muscriptor-flash list-midi
```

## Start live transcription

```bash
muscriptor-flash live --midi-port "loopMIDI Port"
```

The MIDI port can be an exact name or a unique substring. If the system has
exactly one MIDI output, `--midi-port` may be omitted.

On platforms that support virtual RtMidi ports:

```bash
muscriptor-flash live --virtual-midi
```

For event/latency diagnostics:

```bash
muscriptor-flash live --midi-port "loopMIDI Port" --verbose
```

## Google Colab

Open `notebooks/MuScriptor_Flash_Colab.ipynb`. The notebook installs the
`work/muscriptor-flash` branch plus Gradio and launches a browser UI with two
modes:

- **Live microphone**: the browser records microphone audio and sends roughly
  100 ms chunks to the Colab runtime. FlashEngine processes those chunks and
  shows MIDI events, active notes, Flash DSP time, budget misses, and a rough
  transport/queue-lag estimate. Stopping the recording exports a `.mid` file.
- **Uploaded audio benchmark**: feeds an uploaded clip through the exact same
  streaming engine in hop-sized blocks without browser/network timing noise.
  This is useful for tuning confidence/polyphony and measuring engine speed.

The Colab runtime is remote, so the native **250 ms target is not an end-to-end
latency guarantee** there. Browser capture, network transport, Gradio queueing,
and notebook scheduling are additional. The native CLI remains the path for a
local DAW / virtual-MIDI workflow with predictable latency.

The current Flash spectral MVP is CPU-friendly, so a GPU runtime is not
required for the Colab notebook.

## Default latency geometry

The default configuration is:

- window: 128 ms
- hop: 32 ms
- attack: 1 frame
- release: 3 frames
- target budget: 250 ms

The nominal NoteOn wait is therefore 128 ms before DSP, audio-driver, MIDI,
and scheduler overhead are added. Each emitted event records the measured DSP
component and the engine counts any analysis whose estimated latency exceeds
the configured budget.

## Architecture

```text
Audio device
    |
    v
32 ms input blocks
    |
    v
128 ms rolling ring buffer
    |
    v
Flash pitch detector
    |
    v
attack/release note state machine
    |
    +----> MIDI NoteOn / NoteOff ----> RtMidi port
    |
    +----> latency / confidence stats
```

In Colab, the capture side is different but the engine is the same:

```text
Browser microphone
    |
    v
Gradio stream (~100 ms chunks)
    |
    v
Colab runtime
    |
    v
FlashEngine
    |
    +----> live event/status display
    |
    +----> MIDI file on stop
```

The normal MuScriptor model remains unchanged. A recording can therefore be
processed later by the high-accuracy path to create a corrected final MIDI
without forcing the realtime path to wait for future audio.

## Next model step

The intended second phase is to replace the MVP spectral detector with a
causal neural backend that keeps the same `FlashEngine` streaming contract.
That model should output per-pitch onset/frame probabilities incrementally so
it can preserve the sub-250-ms behavior while improving instrument separation,
polyphonic accuracy, and robustness on real mixes.
