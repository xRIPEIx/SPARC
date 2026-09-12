"""Checkpoint save, resume, and backbone transfer.

Transfer lives here rather than in the evaluation code because it is the hinge
between pretraining and downstream: it is the one place where a mistake yields a
randomly-initialised backbone and a completely plausible-looking metric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

LAST_NAME = "last.pth"

#: Wrapper prefixes that may sit in front of real backbone parameter names.
_PREFIXES = ("module.", "encoder.", "backbone.", "net.")
#: Keys that are never part of a transferable backbone.
_DROP_PATTERNS = (
    re.compile(r"^fc\."),
    re.compile(r"projection_head"),
    re.compile(r"region_projector"),
    re.compile(r"criterion"),
)


def checkpoint_name(method: str, dataset: str, arch: str, epoch: int) -> str:
    return f"{method}_{dataset}_{arch}_ep{epoch:04d}.pth"


def save_checkpoint(
    path: Path,
    *,
    method: nn.Module,
    method_name: str,
    config: dict[str, Any],
    epoch: int,
    avg_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> Path:
    """Write a checkpoint holding both the transferable encoder and full
    resume state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "method": method_name,
        "avg_loss": avg_loss,
        # The transferable weights, saved separately from the full module so
        # downstream evaluation never has to know this method's internal layout.
        "encoder": method.encoder_state_dict(),
        "model": method.state_dict(),
        # The fully-resolved config, so a checkpoint explains how it was made.
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    criteria = getattr(method, "criteria_state", lambda: {})()
    if criteria:
        payload["criteria"] = {name: mod.state_dict() for name, mod in criteria.items()}

    torch.save(payload, path)
    return path


def find_resume_checkpoint(output_dir: Path, explicit: str | None = None) -> Path | None:
    """Locate a checkpoint to resume from.

    `last.pth` wins; otherwise the highest-numbered periodic checkpoint. Sorting
    is by the parsed epoch number, not lexicographic: `ep0100` must beat `ep0090`
    and string ordering happens to agree only because of the zero padding.
    """
    if explicit and explicit != "auto":
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {path}")
        return path

    output_dir = Path(output_dir)
    last = output_dir / LAST_NAME
    if last.is_file():
        return last

    candidates = []
    for path in output_dir.glob("*_ep*.pth"):
        match = re.search(r"_ep(\d+)\.pth$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    return max(candidates)[1]


def load_resume_checkpoint(
    path: Path,
    *,
    method: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
) -> int:
    """Restore training state. Returns the epoch to start from."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    method.load_state_dict(payload["model"])

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    # Memory banks are part of the objective's state. Dropping them restarts the
    # queue empty, which changes the negatives for the epochs after a resume.
    criteria = getattr(method, "criteria_state", lambda: {})()
    for name, module in criteria.items():
        saved = payload.get("criteria", {}).get(name)
        if saved is not None:
            module.load_state_dict(saved)

    return int(payload["epoch"]) + 1


def strip_encoder_state(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reduce a saved state dict to plain backbone parameter names.

    Prefixes are stripped one at a time in a loop so that nested wrappers
    (`module.encoder.conv1.weight`) resolve fully.
    """
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        name = key
        changed = True
        while changed:
            changed = False
            for prefix in _PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        if any(pattern.search(name) for pattern in _DROP_PATTERNS):
            continue
        out[name] = value
    return out


@dataclass
class LoadReport:
    matched: int
    total: int
    missing: list[str]
    unexpected: list[str]

    @property
    def fraction(self) -> float:
        return self.matched / self.total if self.total else 0.0


def load_pretrained_encoder(
    module: nn.Module,
    checkpoint_path: str | Path,
    *,
    min_match: float = 0.95,
    allow_partial: bool = False,
) -> LoadReport:
    """Load SSL-pretrained weights into a downstream backbone.

    Raises unless most parameters actually matched.

    The previous implementation called load_state_dict(strict=False) and only
    *printed* the missing-key count. A mistyped architecture therefore produced a
    randomly-initialised backbone, trained happily, and reported a plausible
    metric with no error anywhere. For a repository backing a published number
    that failure has to be loud.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("encoder", "state_dict", "model"):
        if isinstance(payload, dict) and key in payload:
            raw = payload[key]
            break
    else:
        raw = payload

    state = strip_encoder_state(raw)
    target_keys = set(module.state_dict())
    matched = sorted(target_keys & set(state))
    report = LoadReport(
        matched=len(matched),
        total=len(target_keys),
        missing=sorted(target_keys - set(state)),
        unexpected=sorted(set(state) - target_keys),
    )

    if report.fraction < min_match and not allow_partial:
        raise RuntimeError(
            f"Refusing to load {checkpoint_path}: only {report.matched}/{report.total} "
            f"({report.fraction:.1%}) of the target backbone's parameters were found in the "
            f"checkpoint, below the {min_match:.0%} threshold.\n"
            f"The usual cause is an architecture mismatch -- check that the downstream "
            f"backbone matches the one this checkpoint was pretrained with.\n"
            f"First missing keys: {report.missing[:5]}\n"
            f"Pass allow_partial=True (CLI: --allow-partial-load) to proceed anyway."
        )

    module.load_state_dict({k: v for k, v in state.items() if k in target_keys}, strict=False)
    return report
