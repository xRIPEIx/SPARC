"""The backbone contract.

A BackboneSpec carries everything the rest of SPARC needs to know about a
feature extractor, so that pretraining, FCN segmentation and FPN detection are
all written once rather than once per architecture.

Adding an architecture means producing one of these. It does not mean editing
the trainer, the downstream builders, or any argparse `choices=` list.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from torch import nn

#: Stage names, coarsest last. s1 is the highest-resolution stage used by the
#: feature pyramid; s4 is the final stage that global pooling and FCN consume.
STAGE_NAMES: tuple[str, ...] = ("s1", "s2", "s3", "s4")


@dataclass(frozen=True)
class BackboneSpec:
    """Everything SPARC needs to use an architecture.

    Attributes:
        name: Registry key this was built from, e.g. "resnet18".
        net: The trunk itself, classification head already removed. This is the
            module whose ``state_dict`` is saved as the SSL encoder and later
            transferred into a downstream model, so its key names matter.
        feature_dim: Channels of the final stage. Projection heads size
            themselves from this.
        stage_channels: Channels per stage, coarsest last. The feature pyramid
            is built from this, which is what removes the hardcoded
            ``in_channels_list=[64, 128, 256, 512]``.
        stage_strides: Total downsampling factor per stage, e.g. (4, 8, 16, 32).
        make_extractor: Given a subset of ``stage_names``, returns a module
            mapping an image to ``{stage_name: tensor}``. Parameters are shared
            with ``net`` rather than copied, so momentum updates applied to
            ``net`` are seen by the extractor.
    """

    name: str
    net: nn.Module
    feature_dim: int
    stage_channels: tuple[int, ...]
    stage_strides: tuple[int, ...]
    make_extractor: Callable[[Sequence[str]], nn.Module]
    stage_names: tuple[str, ...] = field(default=STAGE_NAMES)

    def __post_init__(self) -> None:
        n = len(self.stage_names)
        if len(self.stage_channels) != n:
            raise ValueError(
                f"{self.name}: stage_channels has {len(self.stage_channels)} entries "
                f"but there are {n} stages {self.stage_names}"
            )
        if len(self.stage_strides) != n:
            raise ValueError(
                f"{self.name}: stage_strides has {len(self.stage_strides)} entries "
                f"but there are {n} stages {self.stage_names}"
            )
        if self.feature_dim != self.stage_channels[-1]:
            raise ValueError(
                f"{self.name}: feature_dim {self.feature_dim} does not match the final "
                f"stage channel count {self.stage_channels[-1]}"
            )

    def final_extractor(self) -> nn.Module:
        """Extractor for the last stage only -- what SSL pretraining uses."""
        return self.make_extractor((self.stage_names[-1],))


class RenameOutputs(nn.Module):
    """Wraps an extractor to rename its output keys.

    torchvision's FPN and FCN want their own key names ("0".."3", "out"); the
    registry speaks in stage names. This adapter keeps that translation in one
    place instead of leaking it into every builder.
    """

    def __init__(self, inner: nn.Module, mapping: dict[str, str]):
        super().__init__()
        self.inner = inner
        self.mapping = dict(mapping)

    def forward(self, x):
        out = self.inner(x)
        return {self.mapping.get(k, k): v for k, v in out.items()}
