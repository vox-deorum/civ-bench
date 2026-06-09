"""Analyses layer — the pluggable analysis modules (stage 4).

Each analysis is a self-contained unit behind the :class:`Analysis` interface; the
JSON ``analyses[].module`` string selects it from :data:`ANALYSIS_REGISTRY`. A
module consumes its declared inputs via an :class:`AnalysisContext` and returns an
:class:`AnalysisResult` (tables + figures + summary); the :func:`run_analysis`
runner persists the result under ``<root>/analyses/<id>/``.

Pulls matplotlib/statsmodels (and, for ``ratings.*``, ``Rscript``) on import of
:mod:`registry`/:mod:`runner`, so import those lazily from the CLI run path — never
from the import-light config/pipeline layers.
"""

from __future__ import annotations

from .base import Analysis, AnalysisContext, AnalysisResult
from .errors import AnalysisError
from .registry import ANALYSIS_REGISTRY, get_analysis, list_analyses
from .runner import AnalysisRunResult, run_analysis

__all__ = [
    "ANALYSIS_REGISTRY",
    "Analysis",
    "AnalysisContext",
    "AnalysisError",
    "AnalysisResult",
    "AnalysisRunResult",
    "get_analysis",
    "list_analyses",
    "run_analysis",
]
