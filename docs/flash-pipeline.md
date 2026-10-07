# MuScripter Flash train -> compare -> promote pipeline

`muscriptor-flash-pipeline` connects the three Flash model lifecycle stages into one guarded workflow:

```text
managed training / resume
        |
        v
candidate model.pt
        |
        v
fresh baseline-vs-candidate comparison
        |
        v
comparison gate
        |
        v
metadata + CPU p95 promotion gate
        |
        v
optional installation only when both gates pass
```

## Example

```bash
muscriptor-flash-pipeline \
  --workdir /data/muscriptor-runs/full-01 \
  --bach10 /data/Bach10_v1.1 \
  --urmp /data/URMP \
  --musicnet /data/musicnet \
  --steps 5000
```

The work directory gains these files in addition to the managed trainer outputs:

```text
comparison.json
promotion.json
pipeline.json
```

Training keeps the existing automatic resume behavior. Running the same command again continues from `training-state.pt`.

## Guarded installation

To copy the candidate only after both gates pass:

```bash
muscriptor-flash-pipeline \
  --workdir /data/muscriptor-runs/full-01 \
  --musicnet /data/musicnet \
  --steps 5000 \
  --promote-to /opt/muscriptor/flash-neural-multidataset.pt
```

No file is copied when either gate fails.

The comparison gate defaults to:

- re-evaluated global F1 must not decrease;
- no dataset may regress by more than 0.02 F1;
- candidate median CPU latency must stay within 1.25x of baseline.

The existing promotion gate then requires by default:

- checkpoint-metadata global F1 gain >= 0.01;
- per-dataset regression <= 0.02 F1;
- CPU p95 inference <= 15 ms.

All thresholds are configurable from the CLI.

## Important evaluation limitation

`comparison.json` reruns candidate and baseline on the trainer's configured holdout sets. If those holdouts affected model selection during training, the result is a regression check rather than a genuinely independent final test. Runtime promotion should still be confirmed on untouched real audio, especially dense polyphony, very low/high pitches, and non-classical material.
