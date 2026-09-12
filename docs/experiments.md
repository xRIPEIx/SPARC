# Experiments

See [`experiments/README.md`](../experiments/README.md) for the sweeps and
[`experiments/results/README.md`](../experiments/results/README.md) for the
committed aggregates and their provenance. In brief:

1. A sweep is one YAML in `configs/sweeps/`.
2. `sparc-sweep expand` turns it into a manifest: one row per (configuration,
   seed). Pretraining and evaluation jobs both read the same row.
3. `sparc-sweep status` reports completion from files on disk, never from a
   column.
4. `sparc-report aggregate` turns per-run CSVs into one row per configuration
   (mean, sample s.d., n, values); `table`, `figure`, `readme` and `check`
   render from that.

Run identity: seed 1 keeps the bare `config_id`; seeds 2+ append `_seed<N>`.
