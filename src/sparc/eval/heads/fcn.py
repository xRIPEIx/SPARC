"""Generic FCN segmentation head.

Replaces the previous per-architecture builders (`fcn_resnet18`, plus a
torchvision call for resnet50), each of which hardcoded its own channel count.
This reads the channel count from the BackboneSpec, so it works for any
registered architecture including timm models.
"""

from __future__ import annotations

from torch import nn
from torchvision.models.segmentation.fcn import FCN, FCNHead

from sparc.models.backbones import BackboneSpec, RenameOutputs


def build_fcn(spec: BackboneSpec, num_classes: int, *, aux: bool = False) -> nn.Module:
    """FCN over the final backbone stage.

    Args:
        spec: Any registered backbone.
        num_classes: Output channels, 21 for VOC including background.
        aux: Add an auxiliary head on the penultimate stage. Off by default,
            matching the configuration the published numbers were produced with.
    """
    stages = (spec.stage_names[-1],) if not aux else spec.stage_names[-2:]
    mapping = {spec.stage_names[-1]: "out"}
    if aux:
        mapping[spec.stage_names[-2]] = "aux"

    backbone = RenameOutputs(spec.make_extractor(stages), mapping)
    classifier = FCNHead(spec.feature_dim, num_classes)
    aux_classifier = FCNHead(spec.stage_channels[-2], num_classes) if aux else None
    return FCN(backbone=backbone, classifier=classifier, aux_classifier=aux_classifier)
