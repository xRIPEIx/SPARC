"""SSL method registry."""

from __future__ import annotations

from collections.abc import Callable

from sparc.methods.base import SSLMethod

_METHODS: dict[str, type[SSLMethod]] = {}


def register_method(name: str) -> Callable[[type[SSLMethod]], type[SSLMethod]]:
    def decorator(cls: type[SSLMethod]) -> type[SSLMethod]:
        if name in _METHODS:
            raise ValueError(f"Method {name!r} is already registered")
        _METHODS[name] = cls
        return cls

    return decorator


def list_methods() -> list[str]:
    return sorted(_METHODS)


def get_method(name: str) -> type[SSLMethod]:
    try:
        return _METHODS[name]
    except KeyError:
        raise KeyError(
            f"Unknown method {name!r}. Registered: {list_methods()}. "
            f"To add your own, see docs/extending.md."
        ) from None
