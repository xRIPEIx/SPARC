"""torchvision ResNet backbones.

One factory covers resnet18/34/50/101/152: the channel counts differ only by
BasicBlock vs Bottleneck expansion, which is read off the model rather than
hardcoded.
"""

from __future__ import annotations

import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from sparc.models.backbones.base import STAGE_NAMES, BackboneSpec
from sparc.models.backbones.registry import register_backbone

#: torchvision layer name -> SPARC stage name.
_LAYER_TO_STAGE = {"layer1": "s1", "layer2": "s2", "layer3": "s3", "layer4": "s4"}
_STAGE_TO_LAYER = {v: k for k, v in _LAYER_TO_STAGE.items()}
_STAGE_STRIDES = (4, 8, 16, 32)

_WEIGHTS = {
    "resnet18": "ResNet18_Weights",
    "resnet34": "ResNet34_Weights",
    "resnet50": "ResNet50_Weights",
    "resnet101": "ResNet101_Weights",
    "resnet152": "ResNet152_Weights",
}


@register_backbone("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
def torchvision_resnet(*, name: str, imagenet_init: bool = False, **_) -> BackboneSpec:
    weights = None
    if imagenet_init:
        enum = getattr(torchvision.models, _WEIGHTS[name])
        # IMAGENET1K_V2 exists only for some; DEFAULT resolves to the best available.
        weights = enum.DEFAULT
    net = getattr(torchvision.models, name)(weights=weights)

    # The classification head is dropped, not kept and ignored: it would
    # otherwise be saved into every SSL checkpoint and then stripped again on
    # transfer.
    net.fc = nn.Identity()

    expansion = net.layer1[0].expansion
    stage_channels = tuple(c * expansion for c in (64, 128, 256, 512))

    def make_extractor(stages):
        unknown = set(stages) - set(STAGE_NAMES)
        if unknown:
            raise ValueError(f"Unknown stages {sorted(unknown)}; expected {STAGE_NAMES}")
        # Ordered coarsest-last so IntermediateLayerGetter truncates the trunk
        # after the deepest requested stage.
        ordered = [s for s in STAGE_NAMES if s in stages]
        return IntermediateLayerGetter(net, {_STAGE_TO_LAYER[s]: s for s in ordered})

    return BackboneSpec(
        name=name,
        net=net,
        feature_dim=stage_channels[-1],
        stage_channels=stage_channels,
        stage_strides=_STAGE_STRIDES,
        make_extractor=make_extractor,
    )
