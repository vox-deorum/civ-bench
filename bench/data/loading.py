"""Canonical CSV filtering (ported from ``shared/data_loading.py``).

The pipeline reads canonical CSVs through the stage loaders (``AnalysisContext``
etc.), so only the reusable filter helpers survive here:

* :func:`apply_filter_spec` applies benchmark filter semantics (preset name / inline
  object / list, with the global↔stage intersection) to an already-loaded frame,
  catalog-driven for ``only_llm`` / default exclusions.
* :func:`drop_problem_games` drops malformed-DB games named in ``import_issues.csv``.

The source's full ``load_turn_data`` / ``load_panel_data`` loaders (and their
``player_type`` fallback-mapping plumbing) were unused and have been removed;
``player_type`` is composed at extract now (benchmark.md §3.3).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..catalog import Catalog
from ..config.filters import intersect_filter_specs, resolve_filter_spec


def apply_filter_spec(
    df: pd.DataFrame,
    catalog: Optional[Catalog] = None,
    filter_spec=None,
    presets: Optional[dict] = None,
    stage_filter=None,
) -> pd.DataFrame:
    """Apply benchmark filter semantics to an already loaded dataframe.

    ``filter_spec`` is normally the resolved global ``data.filter``. When
    ``stage_filter`` is supplied, it is intersected with the global filter so a
    stage narrows the global selection instead of replacing it.
    """
    presets = presets or {}
    resolved = resolve_filter_spec(filter_spec, presets, "filter")
    if stage_filter is not None:
        resolved = intersect_filter_specs(
            resolved,
            resolve_filter_spec(stage_filter, presets, "stage.filter"),
        )

    out = df
    cond_col = "condition" if "condition" in out.columns else "experiment"

    experiments = resolved.get("experiments")
    if experiments is not None:
        out = out[out[cond_col].isin(_as_list(experiments))]

    excluded = resolved.get("exclude_experiments") or []
    if excluded:
        out = out[~out[cond_col].isin(_as_list(excluded))]

    if resolved.get("only_llm"):
        if catalog is None:
            raise ValueError(
                "filter only_llm=true requires a catalog to identify the non-LLM "
                "baseline seats/experiments, but none was provided."
            )
        if "player_type" in out.columns:
            baselines = {catalog.vanilla_label, catalog.null_label}
            out = out[~out["player_type"].isin(baselines)]
        out = out[~out[cond_col].isin(catalog.non_llm_experiments())]

    players = resolved.get("players")
    if players is not None and "player_type" in out.columns:
        out = out[out["player_type"].isin(_as_list(players))]

    if "turn_range" in resolved and "turn" in out.columns:
        lo, hi = resolved["turn_range"]
        if lo is not None:
            out = out[out["turn"] >= lo]
        if hi is not None:
            out = out[out["turn"] <= hi]

    min_games = resolved.get("min_games")
    if min_games is not None and "player_type" in out.columns:
        count_col = "game_id" if "game_id" in out.columns else None
        if count_col:
            counts = out.groupby("player_type")[count_col].nunique()
        else:
            counts = out["player_type"].value_counts()
        keep = counts[counts >= min_games].index
        out = out[out["player_type"].isin(keep)]

    return out


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def drop_problem_games(df: pd.DataFrame, problem_ids) -> pd.DataFrame:
    """Drop rows whose ``game_id`` is in ``problem_ids`` (malformed-DB games).

    A no-op when ``problem_ids`` is empty/None or the frame has no ``game_id``
    column, so callers can apply it unconditionally. Shared by every analysis input
    (``AnalysisContext._drop_problem_games``) and the adjust stage, so a flagged game
    is excluded consistently rather than only in ``load_table``.
    """
    if problem_ids is not None and len(problem_ids) and "game_id" in df.columns:
        return df[~df["game_id"].isin(problem_ids)]
    return df
