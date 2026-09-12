"""Self-supervised pretraining objectives."""

# Importing the built-ins fires their registration decorators.
from sparc.methods import densecl, moco, sparc  # noqa: E402,F401  (side effect)
from sparc.methods.base import SSLBatch, SSLMethod, StepOutput
from sparc.methods.registry import get_method, list_methods, register_method

__all__ = [
    "SSLBatch",
    "SSLMethod",
    "StepOutput",
    "get_method",
    "list_methods",
    "register_method",
]
