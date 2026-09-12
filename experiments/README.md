# Experiments

Each sweep has a definition in [`configs/sweeps/`](../configs/sweeps) and an
expanded manifest here. The manifest is the single source of truth: both the
pretraining job and the evaluation job read the same row, so the swept value a
checkpoint was trained with and the one recorded against its result cannot
disagree.

Regenerate after editing a sweep definition:

```bash
sparc-sweep expand configs/sweeps/sparc_lambda.yaml -o experiments/sparc_lambda/manifest.csv
```

A test asserts the committed manifests match what the definitions expand to, so
a forgotten regeneration fails CI rather than going unnoticed.

## The sweeps

| Sweep | What varies | Runs |
|---|---|---|
| [`sparc_lambda`](sparc_lambda/) | `method.lambda_region` over 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1 | 7 × 5 seeds |
| [`densecl_lambda`](densecl_lambda/) | `method.dense_lambda` over the same seven values | 7 × 5 seeds |
| [`baselines`](baselines/) | MoCo v2, DenseCL, random init, supervised ImageNet | 4 × 5 seeds |

The two λ sweeps are the paper's headline ablations and are deliberately
parallel: both interpolate from the same MoCo-global objective at λ = 0, so
`sparc_lambda_0`, `densecl_lambda_0` and the MoCo v2 baseline are three routes to
the same model and must agree within seed noise.

**`densecl_lambda_0p5` is the DenseCL baseline.** `dense_lambda` defaults to 0.5,
so it appears in both the sweep and the baselines with the same `config_id`
because it is literally the same run. Report it once.

## Run identity

A `config_id` names a configuration; a `run_id` names one seed of it. Seed 1
keeps the bare `config_id`; seeds 2+ get `_seed<N>`.

That asymmetry is deliberate: it is what let an existing single-seed study be
reused as replicate #1 when seeds 2–5 were added, instead of renaming and
re-running everything.

## Running a sweep

One seed per submission, which keeps job arrays small and lets seeds be added
later without renumbering:

```bash
for seed in 1 2 3 4 5; do
    bash slurm/submit_sweep.sh --sweep sparc_lambda --seed "$seed"
done
```

Progress is derived from files on disk, never from a column in the manifest:

```bash
sparc-sweep status experiments/sparc_lambda/manifest.csv \
    --results-root "$SPARC_OUTPUT_ROOT/results" \
    --checkpoint-root "$SPARC_OUTPUT_ROOT/checkpoints"
```
