"""Timestamped console logging."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str = "", *args: Any, **kwargs: Any) -> None:
    """Print a message prefixed with local wall-clock time."""
    if args:
        msg = msg % args if "%" in msg else " ".join((msg, *(str(a) for a in args)))
    print(f"[{ts()}] {msg}", **kwargs)
