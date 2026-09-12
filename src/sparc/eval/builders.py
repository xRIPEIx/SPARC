"""Downstream model construction and weight initialisation."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from sparc.data.voc.classes import NUM_VOC_CLASSES
from sparc.engine.checkpoint import LoadReport, load_pretrained_encoder
from sparc.eval.heads.faster_rcnn import build_faster_rcnn
from sparc.eval.heads.fcn import build_fcn
from sparc.models.backbones import build_backbone

#: How the backbone is initialised before fine-tuning.
INIT_CHOICES = ("random", "supervised_imagenet", "ssl")


def build_downstream_model(
    task: str,
    *,
    backbone_name: str,
    init: str,
    ssl_ckpt: str | Path | None = None,
    num_classes: int = NUM_VOC_CLASSES,
    allow_partial_load: bool = False,
) -> tuple[nn.Module, LoadReport | None]:
    """Build a segmentation or detection model with the requested initialisation.

    One code path for every architecture and both tasks: the head builders size
    themselves from the backbone's reported channels.
    """
    if init not in INIT_CHOICES:
        raise ValueError(f"Unknown init {init!r}; expected one of {INIT_CHOICES}")
    if init == "ssl" and not ssl_ckpt:
        raise ValueError("init='ssl' requires ssl_ckpt")

    spec = build_backbone(backbone_name, imagenet_init=(init == "supervised_imagenet"))

    if task == "segmentation":
        model = build_fcn(spec, num_classes)
    elif task == "detection":
        model = build_faster_rcnn(spec, num_classes)
    else:
        raise ValueError(f"Unknown task {task!r}; expected 'segmentation' or 'detection'")

    report = None
    if init == "ssl":
        # Loaded into spec.net, whose parameters the extractor shares, so the
        # weights reach the assembled model without needing to know where the
        # head builder put things.
        report = load_pretrained_encoder(spec.net, ssl_ckpt, allow_partial=allow_partial_load)
        print(
            f"Loaded SSL encoder: {report.matched}/{report.total} parameters "
            f"({report.fraction:.1%}) from {ssl_ckpt}"
        )
    return model, report
