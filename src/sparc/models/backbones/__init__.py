"""Backbone registry and built-in architectures."""

# Importing the built-ins fires their registration decorators.
from sparc.models.backbones import torchvision_resnet  # noqa: E402,F401  (side effect)
from sparc.models.backbones.base import STAGE_NAMES, BackboneSpec, RenameOutputs
from sparc.models.backbones.registry import build_backbone, list_backbones, register_backbone

__all__ = [
    "STAGE_NAMES",
    "BackboneSpec",
    "RenameOutputs",
    "build_backbone",
    "list_backbones",
    "register_backbone",
]
