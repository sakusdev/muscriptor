# MuScripter Flash real-audio training

Flash uses a causal 128 ms audio window and predicts the active state of MIDI
notes 21..108. The training code is split into two stages:

1. `flash_neural.py` creates the small synthetic bootstrap checkpoint.
2. real recordings fine-tune that checkpoint without changing the realtime
   `FlashEngine` contract.

## Current checkpoints

- `flash-neural-synthetic.pt` — synthetic bootstrap.
- `flash-neural-bach10.pt` — fine-tuned on Bach10 real recordings and currently
  the best bundled checkpoint.
- `flash-neural-multidataset.pt` — reserved output name for a broader checkpoint
  trained with `flash_multidataset.py`; it is not bundled until a reproducible
  cross-dataset run has been completed and accepted.

Do not treat a checkpoint as better just because it was trained on more data.
Keep held-out benchmarks and only promote it after the target metrics improve.

## Bach10

`muscriptor.flash_realdata` is the small reproducible real-audio stage. The
accepted checkpoint trains on eight Bach10 pieces and holds out the final two.
The training workflow remains useful as a fast regression check because the
whole dataset is small enough for a GitHub-hosted Actions runner.

## URMP

URMP contains 44 real chamber-music performances from duets through quintets.
Each piece provides a mixture WAV, isolated instrument recordings, score files,
and aligned `Notes_*.txt` annotations. A note row contains onset time,
frequency, and duration. `muscriptor.flash_multidataset` converts these rows to
an 88-key 10 ms activity grid and always builds each model input from samples at
or before the target time.

URMP is much larger than Bach10 (the full package is about 12.5 GB), so it is
not downloaded automatically by normal CI. Keep the dataset outside the Git
repository and pass its extracted root explicitly.

The URMP validation split follows the MT3/YourMT3-compatible test IDs:

```text
1, 2, 12, 13, 24, 25, 31, 38, 39
```

The remaining available pieces are training data.

## MusicNet

MusicNet contains 330 real classical recordings and more than one million
aligned note labels. The official archive contains these directories:

```text
train_data/
train_labels/
test_data/
test_labels/
```

`muscriptor.flash_musicnet` pairs WAV and CSV files by recording ID and preserves
that official train/test split. MusicNet's CSV `start_time` and `end_time` are
sample indices on the original 44.1 kHz recording, not seconds. The adapter
converts those indices to the same 10 ms 88-key target clock used by Flash.

MusicNet is intentionally **lazy** in the Flash trainer:

- WAV files are not decoded in full. Each requested training frame seeks only
  the causal 128 ms source window ending at that frame, then resamples that
  slice to 16 kHz.
- labels are not expanded to a full `frames x 88` matrix. Per-pitch note
  intervals are merged and stored sparsely; an 88-key vector is materialized
  only for the requested frame.

This keeps working memory from scaling with the entire decoded MusicNet corpus.
The compressed MusicNet archive is about 11.1 GB and expands substantially, so
normal CI never downloads it. Keep it in external/local storage and pass its
extracted root with `--musicnet`.

## Train on one dataset

Start from the current Bach10 checkpoint so the model keeps the real-audio
adaptation already learned there.

URMP only:

```bash
uv run python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --urmp /data/URMP \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt \
  --metrics muscriptor/checkpoints/flash-neural-multidataset-metrics.json
```

MusicNet only:

```bash
uv run python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --musicnet /data/musicnet \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt \
  --metrics muscriptor/checkpoints/flash-neural-multidataset-metrics.json
```

## Train on Bach10 + URMP + MusicNet

Install SciPy for the Bach10 MAT annotations, then pass all three roots:

```bash
uv run --with scipy python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --bach10 /data/Bach10_v1.1 \
  --urmp /data/URMP \
  --musicnet /data/musicnet \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt \
  --metrics muscriptor/checkpoints/flash-neural-multidataset-metrics.json
```

Any subset of `--bach10`, `--urmp`, and `--musicnet` is valid.

Real batches choose the source dataset uniformly before choosing a piece and a
frame. This prevents MusicNet's much larger recording count from automatically
swamping Bach10 or URMP. By default 80% of each batch is real audio and 20%
remains synthetic so the model keeps exposure to a broad MIDI pitch range.

## Resume long training runs

URMP and especially MusicNet can make a useful training run much longer than a
Bach10 regression run. Use `--state-output` to save an interruption-safe state
periodically:

```bash
uv run python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --musicnet /data/musicnet \
  --steps 3000 \
  --state-output /data/flash-training-state.pt \
  --save-every 50 \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt
```

The state file contains the current model, optimizer, best model so far,
baseline/best metrics, current step, and Python/NumPy/Torch random-number states.
It is written to a temporary file and atomically replaces the previous state so
an interrupted write is less likely to destroy the last usable checkpoint.

Resume by passing the same training configuration and a larger or equal final
`--steps` target:

```bash
uv run python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --musicnet /data/musicnet \
  --steps 3000 \
  --resume-state /data/flash-training-state.pt \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt
```

If `--state-output` is omitted while resuming, the file passed to
`--resume-state` is updated in place. A resume state is rejected if the dataset
piece lists, batch size, learning rate, real-data fraction, evaluation cadence,
seed, or base-backend identity differ from the saved configuration. The initial
holdout baseline is also restored from the state instead of being recomputed.

`--steps` means the **final step number**, not “additional steps.” For example,
a state saved at step 1200 with `--steps 3000` continues from step 1201 through
3000.

## Promotion rule

A multi-dataset checkpoint should not replace `flash-neural-bach10.pt` in the
runtime search order until it has been evaluated on data that was not used for
training or threshold selection. At minimum record:

- global frame precision / recall / F1;
- per-dataset precision / recall / F1;
- chosen decision threshold;
- realtime inference speed on CPU;
- regressions on very low/high notes and dense polyphony.

`flash_multidataset.py` writes the global and per-dataset holdout metrics into
the checkpoint metadata and optional metrics JSON.

## Next model step

The data path now supports Bach10, URMP, and MusicNet without changing the
realtime inference contract. Long runs are resumable and autosaved. The next
quality step is to run a reproducible full multi-dataset training job outside
normal CI, compare it against the Bach10-only checkpoint on every holdout set,
and only then promote the resulting `flash-neural-multidataset.pt` into the
runtime checkpoint search order.
