# MuScripter Flash checkpoint comparison

Use `muscriptor-flash-compare` after a real-data training run to compare its best `model.pt` against the currently bundled Bach10 checkpoint on the same configured holdout datasets.

## Example

```bash
muscriptor-flash-compare \
  --candidate /data/muscriptor-runs/full-01/model.pt \
  --bach10 /data/Bach10_v1.1 \
  --urmp /data/URMP \
  --musicnet /data/musicnet \
  --output /data/muscriptor-runs/full-01/comparison.json
```

The default baseline is the bundled `flash-neural-bach10.pt`. Use `--baseline` to compare against another checkpoint.

The report includes:

- global precision / recall / F1 for baseline and candidate;
- per-dataset precision / recall / F1 and deltas;
- a list of datasets where candidate F1 regressed;
- the independently selected decision threshold for each checkpoint;
- CPU median/min/max milliseconds per 128 ms model window;
- candidate/baseline CPU latency ratio;
- checkpoint metadata and evaluation settings.

## Comparison vs promotion gate

`muscriptor-flash-compare` and `muscriptor-flash-promote` are complementary:

1. `muscriptor-flash-compare` reruns both candidate and baseline on identical configured holdout frames and produces a detailed regression report.
2. `muscriptor-flash-promote` applies the repository's automatic quality gate to the candidate metadata and CPU p95 latency before installation.

A practical flow is `train -> compare -> promote`.

## Interpreting the report

The comparison intentionally uses exactly the same dataset adapters and holdout splits as the Flash multi-dataset trainer. This makes it useful for regression checking, but there is an important limitation: if those holdout sets influenced checkpoint selection during training, this is **not an independent final test**.

Do not promote a candidate solely because global F1 improved. Check per-dataset regressions, CPU latency, dense polyphony and pitch-edge behavior, then confirm quality on genuinely independent audio before changing the runtime default checkpoint.

For a faster smoke test, reduce the sampled validation frames:

```bash
muscriptor-flash-compare \
  --candidate ./model.pt \
  --musicnet /data/musicnet \
  --max-frames-per-dataset 256 \
  --latency-windows 16
```
