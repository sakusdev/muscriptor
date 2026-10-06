<p align="center">
  <img src="web/logo_muscriptor_final.png" alt="MuScriptor logo" width="300">
</p>

# MuScriptor

MuScriptor is a multi-instrument music transcription model developed by [Kyutai](https://kyutai.org) and [Mirelo](https://www.mirelo.ai).
It turns a recording into MIDI and into sheet music.
It's the most accurate open-source transcription model.
You can use the model [here](https://muscriptor.kyutai.org) or self-host it using this repository.


[Use it](https://muscriptor.kyutai.org) | [Paper](https://arxiv.org/abs/2607.08168v1) | [HuggingFace](https://huggingface.co/MuScriptor)

<!-- TODO: record the demo GIF (web UI piano roll), save it as assets/demo.gif,
     then uncomment:
<p align="center">
  <img src="assets/demo.gif" alt="MuScriptor web UI: live piano roll while transcribing" width="700">
</p>
-->

## HuggingFace login (required)

To use MuScriptor locally, you first need to log into [HuggingFace](https://huggingface.co/MuScriptor)
and accept the CC BY-NC 4.0 license.

1. Accept the model license on the model page for the [small](https://huggingface.co/MuScriptor/muscriptor-small),
   [medium](https://huggingface.co/MuScriptor/muscriptor-medium) or [large](https://huggingface.co/MuScriptor/muscriptor-large) model
   (access is granted automatically).
2. Authenticate on your machine:

   ```bash
   uvx hf auth login
   ```

   or set a token (create one at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)):

   ```bash
   export HF_TOKEN=hf_...
   ```

The weights are then automatically downloaded on first use and cached locally.

> **MuScripter Flash is different:** Flash ships its own low-latency neural
> checkpoints and does not use the Hugging Face transformer weights, so Hugging
> Face login is not required for Flash itself.

## Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sakusdev/muscriptor/blob/main/notebooks/MuScriptor_Colab.ipynb)

The Colab notebook provides an inline **Gradio** UI for:

- audio upload
- Small / Medium / **Large (1.4B)** model selection
- best-effort / strict / disabled tempo detection
- optional quantization
- dynamic MIDI velocity
- instrument constraints
- one-click MIDI download

Large is selected by default, and the loaded model is cached in the Colab session
so transcribing a second song does not reload the checkpoint. Gradio is used instead
of ipywidgets because Colab file upload/download interactions are more reliable that
way. Before the first run, accept the model license on Hugging Face and either add
`HF_TOKEN` to Colab **Secrets**, paste a token into the Advanced section, or
authenticate in the notebook environment another way.

## MuScripter Flash — realtime audio to MIDI

**MuScripter Flash** is the low-latency transcription path. It is designed for
live microphone input and realtime MIDI output instead of waiting for the normal
high-accuracy transformer to process a full recording.

Current defaults:

- 16 kHz mono processing
- 128 ms rolling analysis window
- 32 ms analysis hop
- 250 ms NoteOn latency budget
- immediate NoteOn path
- debounced NoteOff path
- **88-key neural multi-pitch detection**
- dynamic MIDI velocity
- All Notes Off / panic handling

Flash automatically selects the best bundled realtime backend in this order:

1. `flash-neural-bach10.pt` — neural model fine-tuned on real Bach10 recordings
2. `flash-neural-synthetic.pt` — synthetic neural bootstrap model
3. the deterministic spectral detector as a final fallback

The neural model sees only the current 128 ms window and never future audio, so
it preserves the realtime streaming contract. The Bach10 fine-tuned checkpoint
starts from the synthetic model and was trained on 8 real-recorded Bach10 pieces
with 2 pieces held out. On that held-out split, frame-level 88-key F1 improved
from **0.552 to 0.700** (precision **0.617**, recall **0.808**). These numbers are
a narrow Bach10 validation result, not a claim of production accuracy across all
instruments, microphones, genres or mixes. The normal MuScriptor transformer is
still the high-accuracy choice for offline transcription.

### Local Flash setup

Clone the repository and install the Flash device dependencies:

```bash
git clone https://github.com/sakusdev/muscriptor.git
cd muscriptor
uv sync --extra flash
```

List available audio inputs and MIDI outputs:

```bash
uv run muscriptor-flash list-audio
uv run muscriptor-flash list-midi
```

Start realtime transcription to a MIDI output:

```bash
uv run muscriptor-flash live --midi-port "loopMIDI Port"
```

The best bundled neural checkpoint is loaded automatically. To force a specific
checkpoint, use:

```bash
uv run muscriptor-flash live \
  --midi-port "loopMIDI Port" \
  --neural-checkpoint path/to/flash-model.pt
```

The MIDI port can be an exact name or a unique substring. If the machine has
exactly one MIDI output, `--midi-port` can be omitted.

For event and latency diagnostics:

```bash
uv run muscriptor-flash live --midi-port "loopMIDI Port" --verbose
```

On platforms where RtMidi supports virtual output ports, Flash can create one:

```bash
uv run muscriptor-flash live --virtual-midi
```

The local CLI is the recommended mode for the **lowest and most predictable
latency**, because microphone capture, Flash processing and MIDI output all stay
on the same machine.

### Windows + loopMIDI + DAW

A practical Windows setup is:

```text
Microphone
   |
   v
MuScripter Flash
   |
   v
loopMIDI
   |
   v
FL Studio / Ableton Live / another DAW or software instrument
```

1. Install and start loopMIDI.
2. Create a virtual port such as `MuScripter Flash`.
3. Run:

   ```bash
   uv run muscriptor-flash live --midi-port "MuScripter Flash"
   ```

4. Select that loopMIDI port as a MIDI input in your DAW or software instrument.
5. Play into the selected microphone.

If a note ever remains stuck, stop Flash or use the DAW's panic / All Notes Off
function. Flash also flushes active notes when the native session exits.

### Flash on Google Colab with realtime local MIDI

Open [`notebooks/MuScriptor_Flash_Colab.ipynb`](notebooks/MuScriptor_Flash_Colab.ipynb).
The notebook installs the Flash frontend and launches a Gradio UI.

The Colab realtime path is:

```text
Microphone
   |
   v
Chrome / Edge
   |
   | browser audio stream (~32 ms cadence)
   v
Google Colab
   |
   v
FlashEngine
   |
   | NoteOn / NoteOff
   v
Web MIDI API in the browser
   |
   v
loopMIDI / hardware MIDI output
   |
   v
DAW / synth
```

On Windows with loopMIDI:

1. Start loopMIDI and create a port, for example `MuScripter Flash`.
2. Open the Flash Colab notebook and run its cells.
3. In the **Live microphone** tab, enter `MuScripter Flash` (or another unique
   part of the MIDI output name).
4. Press **Connect realtime MIDI**.
5. Allow MIDI-device access when the browser asks.
6. Start microphone recording in the Flash UI.
7. Route the loopMIDI port into your DAW or software instrument.

The browser forwards each detected NoteOn/NoteOff to the selected local MIDI
output. The UI also provides:

- live active-note and event display
- Flash DSP latency and rough transport/queue-lag diagnostics
- **Panic / All Notes Off**
- active-note tracking in the browser
- automatic panic on MIDI disconnect or page close
- NoteOff flush when recording stops
- `.mid` export of the captured Flash session
- an uploaded-audio benchmark mode for testing the same FlashEngine without
  browser/network timing noise

Desktop **Chrome or Edge** is recommended because Web MIDI is not available in
every browser. Web MIDI requires a secure context and explicit user permission.
If the inline Colab frame blocks MIDI permission, open the Gradio UI in its own
tab and connect MIDI there.

Colab is remote, so its **250 ms target is not an end-to-end latency guarantee**.
Browser capture, network transport, Gradio streaming and notebook scheduling are
added on top of Flash processing. The transport cadence is approximately 32 ms,
but actual round-trip latency depends on the network and Colab runtime.

### Re-training the Flash neural model

The repository contains both synthetic bootstrap training and real-recording
fine-tuning code. The real-data path is in `muscriptor/flash_realdata.py` and the
manual GitHub Actions workflow is `.github/workflows/train-flash-bach10.yml`.
That workflow downloads Bach10, fine-tunes the synthetic checkpoint, requires an
improvement on the held-out pieces, smoke-tests the checkpoint, uploads an
artifact, and commits the accepted model back to the development branch.

For more implementation details, see [`docs/flash.md`](docs/flash.md).

## Try it locally

After Hugging Face authentication, you can use MuScriptor with `uvx` without having to clone this repo.

Some platforms need an extra `uvx` flag, on every `uvx muscriptor` command:

| Platform | Command |
|---|---|
| Linux, macOS with Apple Silicon | `uvx muscriptor serve` |
| Windows (to use the GPU) | `uvx --torch-backend=cu128 muscriptor serve` |
| macOS with Intel | `uvx --python 3.12 muscriptor serve` |

On Windows the default PyTorch backend is `cpu`, so the GPU needs
`--torch-backend=cu128`. On Intel Macs, PyTorch stopped shipping x86_64 wheels
after torch 2.2.2, which supports Python ≤ 3.12, so the Python version has to
be pinned (if you install with pip/uv instead, use Python 3.10–3.12).

## Web UI

You can host the web UI locally with:

```bash
uvx muscriptor serve
```

This gives you the same UI as hosted on https://muscriptor.kyutai.org/, just with a different look.

The sheet music download needs **MuseScore 4 or newer** installed separately (see
[Sheet music](#sheet-music) below). Without it, everything except that download still works.

## Command-line interface (CLI)

```bash
uvx muscriptor transcribe path/to/audio_file.wav
```

See `--help` for all the options.

### Expressive MIDI post-processing

This fork improves the MIDI produced after transcription without changing the
MuScriptor model itself:

- **Dynamic velocity is on by default.** MuScriptor tokens only contain note
  on/off state, so upstream writes every onset at velocity 100. The post-processor
  now estimates macro-dynamics from short source-audio onset windows and normalizes
  them per decoded instrument. Use `--fixed-velocity` for the original behavior.
- **Best-effort tempo keeps useful BPM on live performances.** If beat tracking is
  coherent but the performance drifts too much for a strict fixed grid,
  `--detect-tempo best-effort` writes the fitted average BPM while withholding meter
  and subdivision data. This avoids a needless 120 BPM placeholder without
  quantizing expressive timing to a false grid.
- **Raw MIDI can be quantized explicitly** with `--quantize`. It remains off by
  default for listening/editing; `--format sheets` still quantizes automatically.

Example:

```bash
uvx muscriptor transcribe song.wav --model large --quantize
```

### Sheet music

Using the CLI with `--format sheets` engraves the transcription as readable notation instead of
writing a single MIDI file.

```bash
muscriptor transcribe audio.wav --format sheets --output score/
```

The output structure looks like this:

```
score/
├── score.mid                       the transcription, as quantized MIDI
├── score.musicxml                  the engraved score, as MusicXML
├── full_score.pdf                  every instrument on one system
├── 01_electric_guitar.pdf          one PDF per instrument …
├── 01_electric_guitar_tab.pdf      … and a tablature PDF for fretted ones
├── 02_electric_bass.pdf
├── 02_electric_bass_tab.pdf
└── 03_drum_kit.pdf
```

This needs **MuseScore 4 or newer** installed separately. Downloads for every
platform are at [musescore.org/en/download](https://musescore.org/en/download).
Set `$MUSCRIPTOR_MUSESCORE` if it lives somewhere unusual.

It works best if there is a steady tempo (i.e. playing with a metronome), because
that allows us to quantize the notes (snap them to a grid) for a cleaner transcription.
Rubato recordings will work significantly worse.

## Using from Python

MuScriptor is also on PyPI, so you can install it with with uv (recommended) or with pip:

```bash
uv add muscriptor
```

```bash
pip install muscriptor
```

Ask your coding agent to show you around the codebase.

## Models

Three variants are published under the [MuScriptor](https://huggingface.co/MuScriptor)
HuggingFace organization. Everywhere a model is selected (`load_model()`, the
CLI's `--model`, `serve --model`) you can pass the bare size keyword and the
weights are downloaded and cached automatically. The architecture is a transformer decoder only. Here are the detailed model sizes:

| Variant | Parameters | Layers | Dim | HuggingFace repo |
|---|---|---|---|---|
| `small` | 103M | 14 | 768 | [muscriptor-small](https://huggingface.co/MuScriptor/muscriptor-small) |
| `medium` (default) | 307M | 24 | 1024 | [muscriptor-medium](https://huggingface.co/MuScriptor/muscriptor-medium) |
| `large` | 1.4B | 48 | 1536 | [muscriptor-large](https://huggingface.co/MuScriptor/muscriptor-large) |

`small` is the practical choice on CPU-only machines, `medium` is the default
speed/accuracy trade-off, and `large` is the most accurate but really wants a
GPU. On Apple Silicon the model runs on Metal (MPS) automatically.

## Developing

To set up for development, get [uv](https://docs.astral.sh/uv/getting-started/installation/),
clone this repo and run:
```bash
uv sync
```

For the web UI, you also need [pnpm](https://pnpm.io/installation) and Node
(can be installed [via pnpm](https://pnpm.io/cli/runtime)).
Then run:

```bash
cd web
pnpm install
pnpm run build
```

If you're not editing the frontend, you only need to do this once.
If you are, run `pnpm dev` instead for a hot-reloading dev server.
Start the backend alongside it with it using `uv run muscriptor serve --port 8222`
and then open the frontend on http://localhost:5173/.

### Run

After this setup, you can run Muscriptor from your local repository using
`uv` (note - not `uvx` like before):

```bash
uv run muscriptor serve
# or 
uv run muscriptor transcribe path/to/audio_file.wav
```

Again, see `--help` for more options.

## License

The code in this repository is released under the [MIT license](LICENSE).

The model weights, published on
[HuggingFace](https://huggingface.co/MuScriptor), are released under the
[CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use).

The MuseScore General SoundFont downloaded for playback is
distributed under its own (MIT) license.

## Citation

```bibtex
@misc{rouard2026muscriptoropenmodelmultiinstrument,
      title={MuScriptor: An Open Model for Multi-Instrument Music Transcription}, 
      author={Simon Rouard and Michael Krause and Axel Roebel and Carl-Johann Simon-Gabriel and Alexandre Défossez},
      year={2026},
      eprint={2607.08168},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2607.08168}, 
}
```
