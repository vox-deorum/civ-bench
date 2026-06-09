"""Adjust-stage errors."""

from __future__ import annotations


class AdjustError(RuntimeError):
    """Raised when an adjust stage cannot run (bad config, missing inputs/baseline)."""
