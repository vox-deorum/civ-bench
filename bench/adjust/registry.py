"""Adjust-module registry: name → adjust builder (currently only ``strength``).

Mirrors ``bench/estimators/registry.py``: a JSON ``adjust[].module`` string selects
a derived-table producer. The strength builder returns a :class:`StrengthArtifacts`
(panel + audit trails); the runner persists them.
"""

from __future__ import annotations

from typing import Callable, Dict

from .strength import build_strength_panel

# module id → builder. The builder signature is
# (predictions_path, panel_path, games_path, params, catalog, estimator_id).
ADJUST_REGISTRY: Dict[str, Callable] = {
    "strength": build_strength_panel,
}


def get_adjuster(name: str) -> Callable:
    """Return the adjust builder registered under ``name``."""
    if name not in ADJUST_REGISTRY:
        available = ", ".join(sorted(ADJUST_REGISTRY))
        raise ValueError(f"Unknown adjust module: '{name}'. Available: {available}")
    return ADJUST_REGISTRY[name]


def list_adjusters() -> list[str]:
    return sorted(ADJUST_REGISTRY)
