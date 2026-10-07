# MuScripter Flash checkpoint promotion

A newly trained multidataset checkpoint should not become the default realtime model only because its global score increased. Use the promotion gate after training.

```bash
muscriptor-flash-promote /data/flash-neural-multidataset-candidate.pt \
  --report /data/flash-promotion-report.json \
  --install-to muscriptor/checkpoints/flash-neural-multidataset.pt
```

The command exits non-zero and does not install the checkpoint if any gate fails.

Default criteria:

- global held-out F1 must improve by at least `0.01` relative to the baseline checkpoint evaluated on the same holdouts;
- no per-dataset held-out F1 may regress by more than `0.02`;
- CPU p95 inference for one 128 ms Flash window must be at most `15 ms` on the machine running the gate.

The thresholds can be overridden with `--min-global-gain`, `--max-dataset-regression`, and `--max-p95-ms`. The generated JSON report records the candidate/baseline F1 values, per-dataset regressions, CPU p95 timing, criteria, failures, and final pass/fail result.

A passed checkpoint installed as `muscriptor/checkpoints/flash-neural-multidataset.pt` is preferred by the realtime CLI. Runtime fallback order is:

1. promoted multidataset checkpoint;
2. Bach10 real-audio checkpoint;
3. synthetic neural checkpoint;
4. spectral detector.

This naming convention is intentional: training output can be kept under a candidate filename, and only a checkpoint that passes the gate should be copied to the canonical runtime filename.
