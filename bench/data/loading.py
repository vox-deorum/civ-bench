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
    condition_incomplete=None,
) -> pd.DataFrame:
    """Apply benchmark filter semantics to an already loaded dataframe.

    ``filter_spec`` is normally the resolved global ``data.filter``. When
    ``stage_filter`` is supplied, it is intersected with the global filter so a
    stage narrows the global selection instead of replacing it.

    ``condition_incomplete`` is the set of experiment/condition names a caller
    has already judged incomplete (computed once from the ``games`` table against
    the global ``min_condition_completeness``); it lets tables that carry no
    ``seed``/``seating_rotation`` grid (tokens, turns, panel, predictions) still
    drop incomplete conditions. A table that does carry the grid re-derives the
    set itself and the two are merged.
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

    mcc = resolved.get("min_condition_completeness")
    if mcc is not None and cond_col in out.columns:
        incomplete = set(condition_incomplete or ())
        if {"seed", "seating_rotation"} <= set(out.columns):
            incomplete |= incomplete_experiments(out, mcc, cond_col)
        if incomplete:
            out = out[~out[cond_col].astype(str).isin(incomplete)]

    return out


def _as_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def condition_completeness(
    df: pd.DataFrame, cond_col: Optional[str] = None
) -> dict[str, float]:
    """Per-condition fraction of controlled ``(seed, seating_rotation)`` slots it
    occupies, evaluated against the union of controlled slots as the reference grid
    (the whole controlled design).

    An experiment missing any reference slot scores below 1.0; an absent controlled
    grid yields ``{}``. Rows with the ``-1`` uncontrolled sentinel for either field
    never count as occupied slots. ``cond_col`` defaults to ``"condition"`` when the
    frame carries it, else ``"experiment"``.
    """
    cond = cond_col or _cond_col(df)
    needed = {cond, "seed", "seating_rotation"}
    if not needed <= set(df.columns):
        return {}
    controlled = df[
        (pd.to_numeric(df["seed"], errors="coerce").fillna(-1) != -1)
        & (pd.to_numeric(df["seating_rotation"], errors="coerce").fillna(-1) != -1)
    ]
    if controlled.empty:
        return {}
    reference = {
        (int(seed), int(rot))
        for seed, rot in controlled[["seed", "seating_rotation"]].itertuples(
            index=False, name=None
        )
    }
    if not reference:
        return {}
    out: dict[str, float] = {}
    for exp, grp in controlled.groupby(cond, sort=True):
        present = {
            (int(s), int(r))
            for s, r in grp[["seed", "seating_rotation"]].itertuples(index=False, name=None)
        }
        out[str(exp)] = len(reference & present) / len(reference)
    return out


def incomplete_experiments(
    df: pd.DataFrame, threshold: float, cond_col: Optional[str] = None
) -> set[str]:
    """Condition names whose slot completeness is below ``threshold`` (in ``(0, 1]``)."""
    return {
        exp for exp, frac in condition_completeness(df, cond_col).items() if frac < threshold
    }


def incomplete_experiments_from_games(
    games_path,
    resolved_filter: Optional[dict],
    problem_ids=None,
) -> Optional[set[str]]:
    """Incomplete-condition names computed once from the canonical ``games`` table.

    Returns ``None`` when the global filter sets no ``min_condition_completeness``
    (so callers skip a needless file read). ``problem_ids`` are malformed-DB
    ``game_id``s to exclude before the grid is computed (matching the loaders).
    """
    mcc = (resolved_filter or {}).get("min_condition_completeness")
    if mcc is None:
        return None
    games = drop_problem_games(pd.read_csv(games_path), problem_ids)
    return incomplete_experiments(games, mcc)


def _cond_col(df: pd.DataFrame) -> str:
    return "condition" if "condition" in df.columns else "experiment"


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
