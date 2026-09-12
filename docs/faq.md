# FAQ

**Do MoCo v2 and DenseCL need superpixel masks?** No. Only SPARC does, and it
checks for them before spending any compute.

**Can I run without SLURM?** Yes; `slurm/` is optional. Call the CLIs directly.

**Multi-GPU?** Not implemented, on purpose. The published numbers depend on a
single-process batch of 64 and a 4096-entry memory bank; a data-parallel run
would change the negatives both losses see.

**Why is the global temperature 0.5 and not 0.1?** Because the study never set
it and inherited lightly's default. Every number was produced at 0.5, so the
config says 0.5 explicitly. Change it and you are running a different
experiment.

**Why are difficult VOC boxes included?** Same reason: an argparse default in
the original code overrode the dataset's default. Documented in
[datasets.md](datasets.md); flip `data.ignore_difficult` for the standard
protocol.

**Why is `densecl_lambda_0p5` the DenseCL baseline?** DenseCL's mix defaults to
0.5, so the baseline is a point on the DenseCL λ curve. It is one run and is
reported once.

**How different is a re-run allowed to be?** Seed-to-seed s.d. is ≈0.4 mIoU
points and ≈0.2 AP points over five seeds, so differences under about twice
that are not resolvable. See [reproduce.md](reproduce.md).

**My mask directory is missing / incomplete.** `sparc-masks` is resumable —
re-run it and it fills the gaps. `slurm/stage_dataset.sh` refuses to start a
job with a mismatched count.

**How do I add a backbone / superpixel method / objective?**
[extending.md](extending.md). Each is one registry entry.
