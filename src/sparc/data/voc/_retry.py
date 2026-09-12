"""Retry wrapper for transient filesystem read failures.

torchvision's VOC datasets re-read from disk on every access with no caching.
Under heavy concurrent load on a shared parallel filesystem, individual reads
occasionally fail in ways that are not reproducible:

  * An 11-hour hyper-parameter search died 9 trials in with
    "ParseError: no element found: line 1, column 0" on an annotation XML --
    the signature of a zero-byte read. Re-parsing all 17,125 annotation files
    with the same parser afterwards found zero corrupted files.
  * A fine-tuning job died on epoch 2 with PIL.UnidentifiedImageError on a JPEG
    whose header and footer were intact, and which opened cleanly on the next
    attempt.

Neither was a bad file, and in both cases hours of compute were discarded by a
read that would have succeeded half a second later.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_read(load: Callable[[], T], *, what: str, attempts: int = 3) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return load()
        except Exception as error:  # noqa: BLE001 - deliberately broad; see module docstring
            last_error = error
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to load {what} after {attempts} attempts") from last_error
