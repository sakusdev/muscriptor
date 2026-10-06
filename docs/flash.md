# MuScripter Flash

MuScripter Flash is the low-latency live transcription path for MuScriptor.
It is designed around a **250 ms NoteOn latency budget** rather than the
five-second chunking used by the normal high-accuracy transformer.

## Status

The realtime pipeline now includes:

- microphone input through PortAudio / `sounddevice`
- 128 ms rolling analysis window by default
- 32 ms analysis hop
- **88-key neural multi-pitch detection** for the native CLI
- real-recording fine-tuned and synthetic neural checkpoints
- deterministic spectral fallback
- immediate NoteOn with configurable attack frames
- debounced NoteOff with configurable release frames
- velocity estimation
- system MIDI output through `mido` + `python-rtmidi`
- latency instrumentation and budget-miss counting
- panic-style All Notes Off when the session exits
- Google Colab browser-microphone streaming UI
- browser Web MIDI output from Colab to a local MIDI port
- 32 ms Colab transport cadence
- browser-side active-note tracking and automatic panic on disconnect/page close
- uploaded-audio Flash benchmark mode for Colab

The native CLI prefers bundled backends in this order:

1. `muscriptor/checkpoints/flash-neural-bach10.pt`
2. `muscriptor/checkpoints/flash-neural-synthetic.pt`
3. the spectral detector

Both neural models consume only the current 128 ms audio window; they do not
look into future audio. The spectral backend remains useful as a deterministic
CPU-friendly fallback and for debugging.

## Neural model training

The first neural bootstrap is trained on procedurally generated tones and
polyphonic mixtures with exact 88-key labels. It establishes broad pitch
coverage without requiring an external training corpus.

The real-recording checkpoint then fine-tunes that model on Bach10. The current
split uses eight pieces for training and two pieces (`09-Jesus` and
`10-NunBitten`) as a held-out evaluation set. Training mixes 75% Bach10 windows
with 25% synthetic windows so the model does not completely discard the wider
synthetic pitch prior.

On the current held-out Bach10 split:

| Checkpoint | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| synthetic bootstrap | 0.497 | 0.620 | 0.552 |
| Bach10 fine-tuned | **0.617** | **0.808** | **0.700** |

This is a narrow frame-level validation on two Bach10 quartet recordings, not a
production benchmark across arbitrary genres, instruments, microphones or
mixes. Broader real-data training and evaluation are still needed.

The real-data trainer is `muscriptor/flash_realdata.py`. The
`train-flash-bach10` GitHub Actions workflow is manual (`workflow_dispatch`) so
retraining does not consume Actions minutes on every source-code push. It:

1. downloads Bach10,
2. fine-tunes from the synthetic checkpoint,
3. requires an improvement on the held-out pieces,
4. smoke-tests the resulting checkpoint,
5. uploads it as an Actions artifact, and
6. commits the accepted checkpoint and metrics back to the training branch.

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

The best bundled neural checkpoint is selected automatically by the native
CLI. A specific checkpoint can be forced with:

```bash
muscriptor-flash live \
  --midi-port "loopMIDI Port" \
  --neural-checkpoint path/to/flash-model.pt
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

Open `notebooks/MuScriptor_Flash_Colab.ipynb`. The notebook installs the Flash
frontend plus Gradio and launches a browser UI with live microphone and uploaded
benchmark modes.

### Live microphone + realtime local MIDI

The browser records microphone audio and sends chunks at roughly **32 ms**
cadence to the Colab runtime. FlashEngine processes those chunks and returns
NoteOn/NoteOff messages to the browser. The browser then forwards them through
the Web MIDI API to a **local** MIDI output such as loopMIDI, a hardware MIDI
interface, or a DAW-visible virtual MIDI port.

Typical Windows path:

```text
Microphone
   |
   v
Chrome / Edge
   |
   v
Colab / FlashEngine
   |
   v
Web MIDI bridge in browser
   |
   v
loopMIDI
   |
   v
DAW / software instrument
```

Before recording:

1. Start loopMIDI (or connect a hardware MIDI output).
2. Enter a unique part of the output name, for example `loopMIDI`.
3. Press **Connect realtime MIDI**.
4. Allow MIDI-device access in the browser prompt.
5. Start microphone recording.

The browser bridge tracks active notes locally. The UI includes a **Panic / All
Notes Off** button, and a panic is also sent automatically if the selected MIDI
output disconnects or the page closes. Stopping microphone capture flushes any
active Flash notes and sends their NoteOff messages before exporting the
recorded `.mid` file.

Idle audio chunks reuse the previous MIDI payload, so they do not trigger
redundant Web MIDI sends.

Web MIDI requires a supported browser, a secure context, and explicit user
permission. Desktop Chrome/Edge is the recommended path. If Colab's inline
frame blocks MIDI permission, open the Gradio frame in its own tab and connect
MIDI there.

### Uploaded audio benchmark

This feeds an uploaded clip through the exact same streaming engine in hop-sized
blocks without browser/network timing noise. It is useful for tuning confidence
and polyphony and measuring engine speed.

The Colab runtime is remote, so the native **250 ms target is not an end-to-end
latency guarantee** there. Browser capture, network transport, Gradio queueing,
and notebook scheduling are additional. The realtime Web MIDI bridge removes
the need for the Colab VM itself to see the user's local MIDI devices, but it
does not remove the network round trip.

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

Native:

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
Flash neural detector
    |       \
    |        +-- spectral fallback
    v
attack/release note state machine
    |
    +----> MIDI NoteOn / NoteOff ----> RtMidi port
    |
    +----> latency / confidence stats
```

Colab realtime output:

```text
Browser microphone
    |
    v
Gradio stream (~32 ms cadence)
    |
    v
Colab runtime
    |
    v
FlashEngine
    |
    v
NoteOn / NoteOff JSON
    |
    v
Browser Web MIDI API
    |
    v
Local MIDI output (loopMIDI / hardware / DAW)
```

The normal MuScriptor model remains unchanged. A recording can therefore be
processed later by the high-accuracy path to create a corrected final MIDI
without forcing the realtime path to wait for future audio.

## Next model steps

The next quality gains should come from broader real recordings rather than
more synthetic-only training. Good targets are datasets with aligned isolated
instrument tracks and note annotations, followed by cross-dataset evaluation.
The Flash API and latency contract do not need to change as those checkpoints
improve.
