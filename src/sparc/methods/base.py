"""The SSL method contract.

Previously the per-method logic was spread across three places: the model class,
its compute_loss, and an if/elif/else inside the batch loop that did momentum
updates and loss assembly differently for each method. Adding a method meant
touching all three, and the trainer had to know which methods needed masks.

Here a method owns all of it. The trainer calls update_momentum then
training_step and knows nothing else.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import nn

from sparc.models.backbones import BackboneSpec


@dataclass
class SSLBatch:
    """Two augmented views, plus their aligned region masks when the method
    needs them.

    Built by the collate adapter, so the trainer never inspects a config flag to
    work out how to unpack a batch.
    """

    view_q: torch.Tensor
    view_k: torch.Tensor
    mask_q: torch.Tensor | None = None
    mask_k: torch.Tensor | None = None

    def to(self, device: torch.device, non_blocking: bool = True) -> SSLBatch:
        move = lambda t: None if t is None else t.to(device, non_blocking=non_blocking)  # noqa: E731
        return SSLBatch(move(self.view_q), move(self.view_k), move(self.mask_q), move(self.mask_k))


@dataclass
class StepOutput:
    """What a training step reports. `loss` is the only thing backpropagated;
    everything in `metrics` is detached and for logging only."""

    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class SSLMethod(nn.Module, abc.ABC):
    """Base class for a self-supervised pretraining objective.

    Subclasses implement three things. Nothing in the trainer, the CLI or the
    config schema needs to change to add one -- see docs/extending.md.
    """

    #: Whether batches must carry region masks. Replaces the ad-hoc runtime
    #: check that used to live in the argparse handling, and lets the config
    #: layer reject an unusable combination before any compute is spent.
    requires_region_masks: ClassVar[bool] = False

    def __init__(self, spec: BackboneSpec):
        super().__init__()
        self.backbone_name = spec.name
        self.feature_dim = spec.feature_dim

    @abc.abstractmethod
    def training_step(self, batch: SSLBatch) -> StepOutput:
        """Run one optimisation step's forward pass and return the loss."""

    @abc.abstractmethod
    def update_momentum(self, momentum: float) -> None:
        """Apply the EMA update to every momentum copy this method holds."""

    @abc.abstractmethod
    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        """The transferable backbone weights.

        Must be plain torchvision-style keys (conv1.weight, layer1.0....) with
        no wrapper prefixes, because this is exactly what downstream evaluation
        loads into a freshly built backbone.
        """

    def criteria_state(self) -> dict[str, nn.Module]:
        """Stateful loss modules to checkpoint, e.g. NT-Xent memory banks.

        Without these a resumed run restarts with an empty queue, which changes
        the effective negative distribution for the first epochs after a resume.
        """
        return {}
