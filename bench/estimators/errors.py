"""Estimator-stage errors."""

from __future__ import annotations


class EstimatorError(RuntimeError):
    """Raised when an estimator stage cannot run (bad config, missing inputs)."""
