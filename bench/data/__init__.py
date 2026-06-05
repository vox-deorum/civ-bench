"""Data layer: canonical CSV loading + filtering (ported from
``shared/data_loading.py``). Pulls pandas only; catalog-driven."""

from __future__ import annotations

from .loading import (
    filter_non_llm_games,
    load_panel_data,
    load_turn_data,
)

__all__ = ["filter_non_llm_games", "load_panel_data", "load_turn_data"]
