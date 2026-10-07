# MuScripter Flash managed training runner

`muscriptor-flash-train` is the recommended entry point for long Bach10 / URMP / MusicNet training jobs.

It creates one durable work directory containing:

```text
model.pt
metrics.json
training-state.pt
run.json
```

If `training-state.pt` already exists, the runner resumes it automatically. The saved state includes the current model, optimizer, best model, evaluation state, and RNG state, so a Colab or local job can continue instead of restarting from step 0.

## MusicNet example

```bash
muscriptor-flash-train \
  --workdir /data/muscriptor-runs/musicnet-01 \
  --musicnet /data/musicnet \
  --steps 3000
```

Running the same command again after an interruption resumes from the existing state file.

## Bach10 + URMP + MusicNet

```bash
muscriptor-flash-train \
  --workdir /data/muscriptor-runs/full-01 \
  --bach10 /data/Bach10_v1.1 \
  --urmp /data/URMP \
  --musicnet /data/musicnet \
  --steps 5000 \
  --batch-size 32 \
  --eval-every 50 \
  --save-every 50
```

The bundled `flash-neural-bach10.pt` is used as the default base checkpoint. Use `--base` to override it.

## Intentionally restarting

Use `--fresh` only when you want a new run from step 0:

```bash
muscriptor-flash-train \
  --workdir /data/muscriptor-runs/full-01 \
  --musicnet /data/musicnet \
  --steps 3000 \
  --fresh
```

The previous `training-state.pt` is moved to a timestamped `.bak.pt` file instead of being deleted.

## Google Colab

Open `notebooks/MuScriptor_Flash_Training_Colab.ipynb`, mount Google Drive, and put the run work directory on Drive. The notebook calls the same managed runner, so reconnecting to Colab and rerunning the training cell resumes automatically from Drive.

Large datasets are not downloaded automatically. Extract Bach10, URMP, and/or MusicNet separately and point the notebook or CLI at their roots.
