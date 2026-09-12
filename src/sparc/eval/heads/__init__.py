"""Task heads, built generically from a BackboneSpec."""

from sparc.eval.heads.faster_rcnn import build_faster_rcnn
from sparc.eval.heads.fcn import build_fcn

__all__ = ["build_faster_rcnn", "build_fcn"]
