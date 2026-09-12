# Results

Only **aggregates** are committed: one CSV per (sweep, task) with, per
configuration, the mean and *sample* standard deviation (n−1) over completed
seeds, the seed count, and the raw values. Per-run result CSVs and checkpoints
are not committed.

Every table and figure in the README and the paper is rendered from these files
by `sparc-report`, so a table and a figure cannot disagree, and CI checks that
the README block matches a fresh render.

```bash
sparc-report check  --aggregates experiments/results/aggregate      # λ=0 identity, double-count
sparc-report table  --aggregates experiments/results/aggregate      # markdown (or --format latex)
sparc-report readme --aggregates experiments/results/aggregate      # splice into README.md
sparc-report figure --aggregates experiments/results/aggregate -o docs/figures/lambda.png      # or --format pgfplots
```

## Provenance

The archived study was run with the previous implementation (bit-identical to
this one on a fixed batch; see the phase-5 verification). The per-run CSVs it
produced were converted to this repository's naming with a one-time script and
aggregated by the released `sparc-report aggregate`, so the committed files were
produced by the shipped tool.

Six run_ids existed in two places in the source archive with slightly different
numbers: the DenseCL baseline and `densecl_lambda_0p5` (two separate runs of one
configuration), and `sparc_lambda_0p5` seeds 2–5 detection (the same checkpoint
fine-tuned twice, once per sweep). The committed sweep archive was taken as
canonical and the second copies dropped, since they share seed numbers with the
canonical runs. All differences were within seed noise (≤0.4 mIoU points, ≤0.3
AP points), and the second fine-tunes incidentally measure fine-tuning-only
nondeterminism at ≈0.1–0.3 AP points.

## Checks these files pass

- `sparc_lambda_0`, `densecl_lambda_0` and `moco` — three routes to the same
  objective — agree within 2·SE on both tasks.
- Every `config_id` names exactly one run (`densecl_lambda_0p5` is the DenseCL
  baseline and is reported once).
