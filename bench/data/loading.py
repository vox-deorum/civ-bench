"""Canonical CSV loading + filtering (ported from ``shared/data_loading.py``).

Two changes from the source, per plans/plan.md's migration map:

1. **`player_type` now comes from extract** — the loaders prefer a ``player_type``
   column already present in the CSV (composed by the extract stage from per-player
   metadata, benchmark.md §3.3). The static ``(condition, player_id)`` seat merge is
   used only as a fallback, and only when a :class:`Catalog` is supplied.
2. **Experiment groups are catalog-driven** — ``vanilla``/``null`` membership and
   default exclusions come from the supplied catalog, not module globals.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from ..catalog import Catalog


def filter_non_llm_games(df: pd.DataFrame, catalog: Catalog, keep_null_ai: bool = False) -> pd.DataFrame:
    col = "condition" if "condition" in df.columns else "experiment"
    exclude = catalog.vanilla_experiments() if keep_null_ai else catalog.non_llm_experiments()
    return df[~df[col].isin(exclude)]


def _load_csv_with_condition_mapping(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "experiment" in df.columns:
        df["condition"] = df["experiment"]
    return df


def _apply_player_type_mapping(
    df: pd.DataFrame,
    catalog: Optional[Catalog] = None,
    skip_mapping: bool = False,
) -> pd.DataFrame:
    # Orthodox path: player_type already composed at extract — keep it as-is.
    if "player_type" in df.columns and not skip_mapping:
        return df

    df = df.copy()
    if skip_mapping or catalog is None or "condition" not in df.columns:
        df["player_type"] = "Player " + df["player_id"].astype(str)
        return df

    # Fallback path: static (condition, player_id) seat map from the catalog.
    df["player_type"] = [
        catalog.fallback_player_type(cond, pid)
        for cond, pid in zip(df["condition"], df["player_id"])
    ]
    return df


def _apply_data_filters(
    df: pd.DataFrame,
    catalog: Optional[Catalog] = None,
    player_id=None,
    version_filter=None,
    condition_exclude=None,
    condition_include=None,
    turn_filter=None,
    min_turn=None,
    max_turn=None,
    exclude_non_llm=False,
) -> pd.DataFrame:
    cond_col = "condition" if "condition" in df.columns else "experiment"

    if version_filter is not None and "version" in df.columns:
        df = df[df["version"] == version_filter]

    if catalog is not None:
        excluded = catalog.default_excluded_experiments()
        if excluded:
            df = df[~df[cond_col].isin(excluded)]
        if exclude_non_llm:
            df = df[~df[cond_col].isin(catalog.non_llm_experiments())]

    if condition_exclude is not None:
        if isinstance(condition_exclude, (list, tuple)):
            df = df[~df[cond_col].isin(condition_exclude)]
        else:
            df = df[df[cond_col] != condition_exclude]

    if condition_include is not None:
        if isinstance(condition_include, (list, tuple)):
            df = df[df[cond_col].isin(condition_include)]
        else:
            df = df[df[cond_col] == condition_include]

    if player_id is not None:
        df = df[df["player_id"] == player_id]

    if "turn" in df.columns:
        if turn_filter is not None:
            df = df[df["turn"] == turn_filter]
        if min_turn is not None:
            df = df[df["turn"] >= min_turn]
        if max_turn is not None:
            df = df[df["turn"] <= max_turn]

    return df


def load_turn_data(
    csv_path="turn_data.csv",
    catalog: Optional[Catalog] = None,
    player_id=None,
    version_filter=None,
    condition_exclude=None,
    turn_filter=None,
    min_turn=None,
    max_turn=None,
    exclude_non_llm=False,
    skip_mapping=False,
) -> pd.DataFrame:
    df = _load_csv_with_condition_mapping(csv_path)
    df = _apply_player_type_mapping(df, catalog=catalog, skip_mapping=skip_mapping)
    df = _apply_data_filters(
        df,
        catalog=catalog,
        player_id=player_id,
        version_filter=version_filter,
        condition_exclude=condition_exclude,
        turn_filter=turn_filter,
        min_turn=min_turn,
        max_turn=max_turn,
        exclude_non_llm=exclude_non_llm,
    )
    df["turn_progress"] = round(df["turn"] / df["max_turn"], 2)
    return df


def load_panel_data(
    csv_path="panel_data.csv",
    catalog: Optional[Catalog] = None,
    player_id=None,
    version_filter=None,
    condition_exclude=None,
    condition_include=None,
    exclude_non_llm=False,
    skip_mapping=False,
) -> pd.DataFrame:
    df = _load_csv_with_condition_mapping(csv_path)
    df = _apply_player_type_mapping(df, catalog=catalog, skip_mapping=skip_mapping)
    df = _apply_data_filters(
        df,
        catalog=catalog,
        player_id=player_id,
        version_filter=version_filter,
        condition_exclude=condition_exclude,
        condition_include=condition_include,
        exclude_non_llm=exclude_non_llm,
    )
    return df
