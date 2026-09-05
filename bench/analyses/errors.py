"""Analysis-stage errors (stage 4)."""

from __future__ import annotations


class AnalysisError(RuntimeError):
    """Raised when an analysis stage cannot run or produce a result.

    Mirrors :class:`bench.adjust.errors.AdjustError` / ``EstimatorError``: the
    CLI catches it and exits 2 with a precise message (fail loud, never silent).
    """
