"""timm backbones, via the `timm:` name prefix.

timm cannot go through IntermediateLayerGetter -- most of its models have no
flat top-level stage children to truncate at -- so this uses timm's own
multi-scale API, which also reports channel counts and reductions directly.
That is what lets `backbone.name: timm:convnext_tiny` work with no code change
anywhere in SPARC.
"""

from __future__ import annotations

from torch import nn

from sparc.models.backbones.base import STAGE_NAMES, BackboneSpec
from sparc.models.backbones.registry import TIMM_PREFIX, register_backbone


class _SelectStages(nn.Module):
    """Adapts timm's list output to the {stage_name: tensor} contract."""

    def __init__(self, features: nn.Module, stages, all_stages):
        super().__init__()
        self.features = features
        self.stages = tuple(stages)
        self.all_stages = tuple(all_stages)

    def forward(self, x):
        outs = self.features(x)
        named = dict(zip(self.all_stages, outs, strict=False))
        return {s: named[s] for s in self.all_stages if s in self.stages}


@register_backbone(TIMM_PREFIX)
def timm_backbone(*, name: str, imagenet_init: bool = False, **_) -> BackboneSpec:
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on the install extra
        raise ImportError(f"Backbone {name!r} needs timm: pip install 'sparc-ssl[timm]'") from exc

    model_name = name[len(TIMM_PREFIX) :]
    # Negative indices: the last four stages, however many the model has.
    # Stage counts differ across families -- timm's resnet18 exposes five
    # (index 0 is the stride-2 stem) while convnext_tiny exposes four -- so a
    # fixed (1, 2, 3, 4) silently works for one family and IndexErrors on the
    # other. Counting from the end gives strides (4, 8, 16, 32) for both.
    features = timm.create_model(
        model_name,
        pretrained=imagenet_init,
        features_only=True,
        out_indices=(-4, -3, -2, -1),
    )
    stage_channels = tuple(features.feature_info.channels())
    stage_strides = tuple(features.feature_info.reduction())

    def make_extractor(stages):
        return _SelectStages(features, stages, STAGE_NAMES)

    return BackboneSpec(
        name=name,
        net=features,
        feature_dim=stage_channels[-1],
        stage_channels=stage_channels,
        stage_strides=stage_strides,
        make_extractor=make_extractor,
    )
