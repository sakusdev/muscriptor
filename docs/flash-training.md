# MuScripter Flash real-audio training

Flash uses a causal 128 ms audio window and predicts the active state of MIDI
notes 21..108.  The training code is split into two stages:

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
Keep a held-out benchmark and only promote it after the target metrics improve.

## Bach10

`muscriptor.flash_realdata` is the small reproducible real-audio stage.  The
accepted checkpoint trains on eight Bach10 pieces and holds out the final two.
The training workflow remains useful as a fast regression check because the
whole dataset is small enough for a GitHub-hosted Actions runner.

## URMP

URMP contains 44 real chamber-music performances from duets through quintets.
Each piece provides a mixture WAV, isolated instrument recordings, score files,
and aligned `Notes_*.txt` annotations.  A note row contains onset time,
frequency, and duration.  `muscriptor.flash_multidataset` converts these rows to
an 88-key 10 ms activity grid and always builds each model input from samples at
or before the target time.

URMP is much larger than Bach10 (the full package is about 12.5 GB), so it is
not downloaded automatically by normal CI.  Keep the dataset outside the Git
repository and pass its extracted root explicitly.

The URMP validation split follows the MT3/YourMT3-compatible test IDs:

```text
1, 2, 12, 13, 24, 25, 31, 38, 39
```

The remaining available pieces are training data.

## Train on URMP only

Start from the current Bach10 checkpoint so the model keeps the real-audio
adaptation already learned there:

```bash
uv run python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --urmp /data/URMP \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt \
  --metrics muscriptor/checkpoints/flash-neural-multidataset-metrics.json
```

## Train on Bach10 + URMP

Install SciPy for the Bach10 MAT annotations, then pass both roots:

```bash
uv run --with scipy python -m muscriptor.flash_multidataset \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --bach10 /data/Bach10_v1.1 \
  --urmp /data/URMP \
  --output muscriptor/checkpoints/flash-neural-multidataset.pt \
  --metrics muscriptor/checkpoints/flash-neural-multidataset-metrics.json
```

Real batches choose the source dataset uniformly before choosing a piece and a
frame.  This prevents the dataset with the most pieces from automatically
swamping smaller sources.  By default 80% of each batch is real audio and 20%
remains synthetic so the model keeps exposure to a broad MIDI pitch range.

## Promotion rule

A multi-dataset checkpoint should not replace `flash-neural-bach10.pt` in the
runtime search order until it has been evaluated on data that was not used for
training or threshold selection.  At minimum record:

- global frame precision / recall / F1;
- per-dataset precision / recall / F1;
- chosen decision threshold;
- realtime inference speed on CPU;
- regressions on very low/high notes and dense polyphony.

`flash_multidataset.py` writes the global and per-dataset holdout metrics into
the checkpoint metadata and optional metrics JSON.

## Next datasets

MusicNet is the next useful source after URMP because it contains hundreds of
real classical recordings with note-level labels and a much wider collection of
performances.  It is also roughly 11 GB, so support should follow the same
pattern: local/external dataset storage, deterministic split, selective loading,
and no unconditional download in normal CI.
