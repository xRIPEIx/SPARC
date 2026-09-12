"""Run identity.

A `config_id` names a configuration; a `run_id` names one seed of it. Seed 1
keeps the bare config_id and seeds 2+ get a `_seed<N>` suffix.

That asymmetry is deliberate and load-bearing: it is what let an existing
single-seed study be reused as replicate #1 when seeds 2-5 were added, rather
than renaming and re-running everything. Manifests, checkpoint directories and
result filenames all depend on it agreeing exactly with the shell implementation
it replaces.
"""

from __future__ import annotations


def resolve_run_id(config_id: str, seed: int) -> str:
    if seed < 1:
        raise ValueError(f"seed must be >= 1, got {seed}")
    return config_id if seed == 1 else f"{config_id}_seed{seed}"


def parse_run_id(run_id: str) -> tuple[str, int]:
    """Inverse of resolve_run_id: recover (config_id, seed)."""
    marker = "_seed"
    index = run_id.rfind(marker)
    if index == -1:
        return run_id, 1
    suffix = run_id[index + len(marker) :]
    if not suffix.isdigit():
        return run_id, 1
    return run_id[:index], int(suffix)
