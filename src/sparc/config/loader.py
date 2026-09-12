"""Config loading: base composition, path interpolation, CLI overrides.

Deliberately not Hydra. The execution model here is SLURM job arrays with an
index-to-config mapping, which is what Hydra's multirun layer would duplicate or
fight, and Hydra's working-directory behaviour would scatter output directories
through scratch. A YAML file plus the dataclass it validates against is also a
five-second explanation for a reviewer reading the repo once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from sparc.config.paths import load_paths
from sparc.config.schema import PretrainConfig

#: Key listing configs to compose underneath this one, resolved relative to it.
BASES_KEY = "defaults"


def _compose(path: Path, seen: set[Path] | None = None) -> DictConfig:
    """Load a YAML file, recursively merging anything in its `defaults:` list.

    Later entries win over earlier ones, and the file's own keys win over all of
    them -- the usual override order, so a specific config can always adjust
    what it inherits.
    """
    path = Path(path).resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Circular config inheritance at {path}")
    seen = seen | {path}

    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")

    node = OmegaConf.load(path)
    bases = node.pop(BASES_KEY, []) if isinstance(node, DictConfig) else []

    merged = OmegaConf.create({})
    for base in bases:
        merged = OmegaConf.merge(merged, _compose(path.parent / str(base), seen))
    return OmegaConf.merge(merged, node)


def apply_overrides(config: DictConfig, overrides: list[str]) -> DictConfig:
    """Apply `key.path=value` overrides.

    Unknown keys are rejected. argparse only ever validated flags it knew about,
    so a mistyped setting silently did nothing; here it stops the run.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override {item!r} is not of the form key.path=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        # throw_on_resolution_failure=False matters: a key whose value is an
        # unresolved interpolation (data.images = ${paths.coco_images}) still
        # *exists*, and checking for it must not try to resolve it.
        existing = OmegaConf.select(
            config, key, default=_MISSING, throw_on_resolution_failure=False
        )
        if existing is _MISSING:
            raise KeyError(
                f"Unknown config key {key!r}. It does not exist in the schema; "
                f"check the spelling against configs/ or sparc.config.schema."
            )
        OmegaConf.update(config, key, _parse_value(raw), merge=False)
    return config


def _parse_value(raw: str):
    """Interpret an override value as a YAML scalar.

    Without this, `data.masks=null` sets the four-character string "null"
    rather than None -- and because the schema types that field as `str | None`,
    the string is perfectly valid and the mistake surfaces much later as a
    confusing missing-directory error. Numbers and booleans would otherwise be
    coerced by the typed schema anyway; null, lists and dicts would not.

    Anything YAML cannot parse is kept verbatim, so paths and interpolation
    strings pass through untouched.
    """
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw
    return raw if parsed is None and raw.strip() not in {"null", "~", ""} else parsed


_MISSING = object()


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    paths_file: str | Path | None = None,
    schema: type = PretrainConfig,
) -> DictConfig:
    """Load, compose, validate and resolve a config.

    Order: compose `defaults:`, merge onto the typed schema (which is what makes
    unknown keys and wrong types an error), apply CLI overrides, then resolve
    `${paths.*}` and `${oc.env:...}` interpolations.
    """
    composed = _compose(Path(path))
    config = OmegaConf.merge(OmegaConf.structured(schema), composed)

    # Paths are merged in as a sibling node purely so ${paths.x} resolves, then
    # dropped again: they describe this machine, not this experiment, and should
    # not be baked into a checkpoint as though they were.
    #
    # Merged BEFORE overrides so that an override may itself reference a path
    # (--set data.masks='${paths.superpixel_root}/slic_n250'), and so that
    # overriding a key whose default is an interpolation works at all.
    with_paths = OmegaConf.merge(load_paths(paths_file), config)
    with_paths = apply_overrides(with_paths, overrides or [])
    OmegaConf.resolve(with_paths)
    resolved = OmegaConf.create({k: v for k, v in with_paths.items() if k != "paths"})
    return resolved  # type: ignore[return-value]


def to_dict(config: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]


def save_config(config: DictConfig, path: str | Path) -> Path:
    """Write the fully-resolved config next to a run's outputs.

    This is the provenance record the previous shell-script workflow provided
    only by accident, via whatever happened to be in the job script.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(config, resolve=True))
    return path
