"""Small shared validators for run-spec parsing."""

from __future__ import annotations

from typing import Any

from .errors import ConfigError


def coerce_bool(value: Any, where: str) -> bool:
    """Return a bool, accepting case-insensitive JSON-ish string booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ConfigError(f"{where}: expected bool, got {type(value).__name__}.")
