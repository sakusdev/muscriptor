# MuScripter Flash training in Google Colab

Use `notebooks/MuScriptor_Flash_Training_Colab.ipynb` for long Flash fine-tuning runs with Google Drive-backed autosave state.

## What the notebook does

- installs MuScripter and the Flash training CLI;
- mounts Google Drive;
- uses the bundled `flash-neural-bach10.pt` as the starting checkpoint;
- accepts any subset of Bach10, URMP, and MusicNet dataset roots;
- writes the multidataset checkpoint, metrics JSON, and resumable training state to Drive;
- automatically adds `--resume-state` when the Drive state file already exists.

The training command is also available directly as:

```bash
muscriptor-flash-train --help
```

For example:

```bash
muscriptor-flash-train \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --musicnet /data/musicnet \
  --steps 3000 \
  --state-output /data/flash-training-state.pt \
  --save-every 50 \
  --output /data/flash-neural-multidataset.pt \
  --metrics /data/flash-neural-multidataset-metrics.json
```

If the run stops at step 1200, rerun with the same training configuration and:

```bash
muscriptor-flash-train \
  --base muscriptor/checkpoints/flash-neural-bach10.pt \
  --musicnet /data/musicnet \
  --steps 3000 \
  --resume-state /data/flash-training-state.pt \
  --output /data/flash-neural-multidataset.pt
```

`--steps` is the final target step, so this resumes at 1201 and continues through 3000.

## Colab storage recommendation

Keep the resumable state and output checkpoint on Google Drive. Large datasets can also live on Drive, but MusicNet training reads many short random windows, so local Colab storage is faster if enough disk is available. A practical setup is to keep the source archive on Drive, extract it under `/content`, and keep only `flash-training-state.pt`, metrics, and final checkpoints on Drive.

The MusicNet adapter remains lazy: it reads only the causal 128 ms audio window needed for the sampled frame and keeps labels as sparse note intervals rather than decoding the whole corpus into RAM.

## Promotion rule

A completed multidataset checkpoint is not automatically made the default runtime model. Compare global and per-dataset held-out F1, precision, recall, threshold, and CPU inference speed against the bundled Bach10 checkpoint first. Only promote a checkpoint when the broader evaluation improves without a major regression on another holdout set.
