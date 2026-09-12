# Running on a cluster

See [`slurm/README.md`](../slurm/README.md). The short version: copy
`slurm/cluster_env.sh.example` to `slurm/cluster_env.sh`, edit it, and submit
sweeps with `slurm/submit_sweep.sh`. Nothing else in the repository hardcodes an
account, partition, GPU type or module name.

Three facts learned the expensive way, all baked into `slurm/`:

- **Stage the corpus onto node-local storage.** Reading 118k small files per
  epoch off a shared parallel filesystem measured 602 ms/batch, against
  57 ms/batch from node-local NVMe.
- **Keep mask sets as a single tar.** Filesystem quotas usually count files;
  a mask set is 118,287 of them.
- **Verify the mask count before training.** A precompute once reported success
  while silently leaving 287 images uncovered; it surfaced hours into training.

Login nodes commonly kill CPU-heavy processes. Run the test suite through the
scheduler (`srun --cpus-per-task=8 --mem=16G pytest`), not on the login node.
