"""Backbone registry.

Replaces the previous arrangement, where adding an architecture meant editing
four places across two files -- the SSL trunk factory, the detection builder,
the segmentation builder, and two argparse `choices=` lists -- and where a typo
in `--arch` was caught only by an enumeration that had to be kept in sync by
hand.
"""

from __future__ import annotations

from collections.abc import Callable

from sparc.models.backbones.base import BackboneSpec

#: Factories take (imagenet_init) and return a BackboneSpec.
BackboneFactory = Callable[..., BackboneSpec]

_BACKBONES: dict[str, BackboneFactory] = {}

#: Prefix routed to the timm factory, e.g. "timm:convnext_tiny".
TIMM_PREFIX = "timm:"


def register_backbone(*names: str) -> Callable[[BackboneFactory], BackboneFactory]:
    """Register a factory under one or more names.

    Stacking the decorator is fine; `resnet18` and `resnet50` share one factory.
    """

    def decorator(factory: BackboneFactory) -> BackboneFactory:
        for name in names:
            if name in _BACKBONES:
                raise ValueError(f"Backbone {name!r} is already registered")
            _BACKBONES[name] = factory
        return factory

    return decorator


def list_backbones() -> list[str]:
    """Registered names, sorted. Any timm model also works via the timm: prefix."""
    return sorted(_BACKBONES)


def build_backbone(name: str, *, imagenet_init: bool = False, **kwargs) -> BackboneSpec:
    """Construct a backbone by registry name.

    Raises:
        KeyError: if the name is unknown. The message lists what is available,
            which is the part a `choices=` list used to provide.
    """
    if name.startswith(TIMM_PREFIX):
        from sparc.models.backbones import timm_backbone  # noqa: F401  (registers on import)

        factory = _BACKBONES.get(TIMM_PREFIX)
        if factory is None:  # pragma: no cover - only if timm import failed silently
            raise KeyError(f"{name!r} requires the timm extra: pip install 'sparc-ssl[timm]'")
        return factory(name=name, imagenet_init=imagenet_init, **kwargs)

    try:
        factory = _BACKBONES[name]
    except KeyError:
        raise KeyError(
            f"Unknown backbone {name!r}. Registered: {list_backbones()}. "
            f"Any timm model also works as '{TIMM_PREFIX}<model_name>' "
            f"(needs the timm extra). To add your own, see docs/extending.md."
        ) from None
    return factory(name=name, imagenet_init=imagenet_init, **kwargs)
