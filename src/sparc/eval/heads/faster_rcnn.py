"""Generic Faster R-CNN detection head with a feature pyramid.

Replaces the previous `fasterrcnn_resnet18_fpn`, which hardcoded
`in_channels_list=[64, 128, 256, 512]` and so only worked for the two
BasicBlock ResNets.

`FeaturePyramidNetwork` is composed directly rather than going through
torchvision's `BackboneWithFPN`, because that helper builds its own
`IntermediateLayerGetter` from the model's named children -- re-imposing exactly
the torchvision-ResNet assumption this is meant to remove.

Anchor sizes and ROI-pooler geometry are copied verbatim from the previous
implementation so that resnet18 detection stays numerically unchanged.
"""

from __future__ import annotations

from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)

from sparc.models.backbones import BackboneSpec, RenameOutputs

#: FPN level names expected by MultiScaleRoIAlign.
_FPN_LEVELS = ("0", "1", "2", "3")


class _BodyWithFPN(nn.Module):
    def __init__(self, body: nn.Module, fpn: FeaturePyramidNetwork, out_channels: int):
        super().__init__()
        self.body = body
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x):
        return self.fpn(self.body(x))


def build_faster_rcnn(
    spec: BackboneSpec,
    num_classes: int,
    *,
    fpn_out_channels: int = 256,
    norm_layer: type[nn.Module] | None = None,
) -> nn.Module:
    """Faster R-CNN over a feature pyramid built from any registered backbone.

    Args:
        norm_layer: Normalisation for the FPN. Passed explicitly rather than
            inferred, because torchvision's own resnet50 builder silently
            selects FrozenBatchNorm2d when pretrained backbone weights are
            requested and plain BatchNorm2d otherwise -- which makes a
            supervised-ImageNet arm and an SSL arm not strictly like for like.
            Here the choice is the caller's and identical across init modes.
    """
    body = RenameOutputs(
        spec.make_extractor(spec.stage_names),
        dict(zip(spec.stage_names, _FPN_LEVELS, strict=True)),
    )
    fpn = FeaturePyramidNetwork(
        in_channels_list=list(spec.stage_channels),
        out_channels=fpn_out_channels,
        extra_blocks=LastLevelMaxPool(),
        norm_layer=norm_layer,
    )
    backbone = _BodyWithFPN(body, fpn, fpn_out_channels)

    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=list(_FPN_LEVELS), output_size=7, sampling_ratio=2
    )
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
