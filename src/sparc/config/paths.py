"""Dataset path resolution.

Previously a path could come from four places: pretraining flags, evaluation
flags with stale ./datasets defaults, cluster environment variables, and
hardcoded literals in tooling. This is the one place now.

`configs/paths.yaml` is committed with env-var-backed defaults, so a cluster
user drives it with exports and a laptop user edits one file. An optional,
gitignored `configs/paths.local.yaml` is merged on top -- that is the one-line
change that makes a fresh clone work on someone else's machine.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

PATHS_FILENAME = "paths.yaml"
LOCAL_PATHS_FILENAME = "paths.local.yaml"


def configs_dir() -> Path:
    """Locate the repo's configs/ directory.

    Walks up from this file, which works for an editable install from a clone.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs"
        if (candidate / PATHS_FILENAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not locate configs/{PATHS_FILENAME}. Run from a clone of the "
        f"repository, or pass an explicit --paths file."
    )


def load_paths(explicit: str | Path | None = None) -> DictConfig:
    """Load paths.yaml, then merge paths.local.yaml over it if present."""
    if explicit is not None:
        base = OmegaConf.load(Path(explicit))
        return OmegaConf.create({"paths": base.get("paths", base)})

    directory = configs_dir()
    merged = OmegaConf.load(directory / PATHS_FILENAME)
    local = directory / LOCAL_PATHS_FILENAME
    if local.is_file():
        merged = OmegaConf.merge(merged, OmegaConf.load(local))
    return merged
