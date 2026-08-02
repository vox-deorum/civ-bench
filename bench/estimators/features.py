"""Feature engineering for victory-prediction estimators.

Ported from ``../vox-deorum-analysis/models/utils/data_utils.py`` (the feature
pipeline). For the **load-only** path (stage 2) we keep exactly the deterministic
transforms that re-inference needs so the rebuilt feature matrix matches the
``metadata.selected_features`` a saved model expects:

1. raw turns CSV → city-cost adjustments (Vox Populi formula)
2. relative features (score ratio, normalized rank, turn progress)
3. competitive features (per-turn shares, gaps from leader, military utilization)
4. drop transformed/intermediate columns
5. ``prepare_features`` selects the modelling columns + ID columns

Training-only concerns (CV split generation, resampling/SMOTE) are **not** ported
here — they belong to the ``fit:"train"`` pipeline in stage 6.

``turn_progress`` used as a *feature* is the unrounded ``turn / max_turn`` that
:func:`add_relative_features` computes (the loaders' rounded ``turn_progress`` is
overwritten here), matching the source pipeline byte-for-byte.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ── feature group definitions (ported verbatim) ─────────────────────────────
FEATURE_GROUPS = {
    "shares": [
        "science_share", "culture_share", "tourism_share", "gold_share",
        "faith_share", "production_share", "food_share", "military_share",
        "cities_share", "population_share", "votes_share", "minor_allies_share",
    ],
    "gaps": [
        "technologies_gap", "policies_gap",
    ],
    "percentages": [
        "happiness_percentage", "religion_percentage",
        "military_utilization", "score_ratio",
    ],
    "progress": [
        "turn_progress",
    ],
    "diplomacy": [
        "highest_war_weariness", "active_wars", "truces", "defensive_pacts", "friendships",
    ],
}

# Default active subset for modelling (the source SELECTED_FEATURES).
SELECTED_FEATURES = [
    "science_share", "culture_share", "food_share", "cities_share", "faith_share",
    "tourism_share", "gold_share",
    "production_share", "military_share",
    "population_share", "votes_share", "minor_allies_share",
    "highest_war_weariness", "active_wars", "truces", "defensive_pacts", "friendships",
    "technologies_gap", "policies_gap",
    "happiness_percentage", "religion_percentage",
    "military_utilization", "score_ratio",
    "turn_progress",
]


def get_all_feature_names() -> List[str]:
    """Flat list of ALL feature names from FEATURE_GROUPS."""
    return [f for group in FEATURE_GROUPS.values() for f in group]


def get_selected_feature_names() -> List[str]:
    """Return a copy of SELECTED_FEATURES."""
    return list(SELECTED_FEATURES)


def needs_variant_columns(model_class) -> bool:
    """True if a model's DEFAULT_FEATURES includes columns outside SELECTED_FEATURES."""
    defaults = getattr(model_class, "DEFAULT_FEATURES", None)
    if defaults is None:
        return False
    return bool(set(defaults) - set(SELECTED_FEATURES))


def _validate_feature_config() -> None:
    all_features = set(get_all_feature_names())
    unknown = set(SELECTED_FEATURES) - all_features
    if unknown:
        raise ValueError(f"SELECTED_FEATURES contains features not in FEATURE_GROUPS: {unknown}")


_validate_feature_config()


# ── feature engineering (ported verbatim) ───────────────────────────────────
def apply_city_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the Vox Populi city scaling formula to per-turn metrics.

    ``adjusted_value = value / max(1.05 * (cities - 1), 1.0)``.
    """
    df = df.copy()
    df["city_multiplier"] = np.maximum(1.05 * (df["cities"] - 1), 1.0)
    df["science_adj"] = df["science_per_turn"] / df["city_multiplier"]
    df["culture_adj"] = df["culture_per_turn"] / df["city_multiplier"]
    df["tourism_adj"] = df["tourism_per_turn"] / df["city_multiplier"]
    df["gold_adj"] = df["gold_per_turn"] / df["city_multiplier"]
    df["faith_adj"] = df["faith_per_turn"] / df["city_multiplier"]
    df["production_adj"] = df["production_per_turn"] / df["city_multiplier"]
    df["food_adj"] = df["food_per_turn"] / df["city_multiplier"]
    df["military_adj"] = df["military_strength"] / df["city_multiplier"]
    cities = df["cities"]
    df["population_per_city"] = np.where(cities == 0, 0, df["population"] / cities.replace(0, 1))
    df["territory_per_city"] = np.where(cities == 0, 0, df["territory"] / cities.replace(0, 1))
    return df


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add position-relative-to-others features within the same turn."""
    df = df.copy()
    max_players_per_game = df.groupby("game_id")["player_id"].nunique()
    df["max_players"] = df["game_id"].map(max_players_per_game)

    max_score = df["max_score"]
    df["score_ratio"] = np.where(max_score == 0, 0, df["score"] / max_score.replace(0, 1))
    df["rank_normalized"] = (df["max_players"] + 1 - df["rank"]) / df["max_players"]
    df["turn_progress"] = df["turn"] / df["max_turn"]
    return df


def add_competitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add competitive-position features (relative shares + gaps from leader)."""
    df = df.copy()
    group_key = ["game_id", "turn"]

    share_metrics = {
        "science_adj": "science_share",
        "culture_adj": "culture_share",
        "tourism_adj": "tourism_share",
        "gold_adj": "gold_share",
        "faith_adj": "faith_share",
        "production_adj": "production_share",
        "food_adj": "food_share",
        "military_strength": "military_share",
        "cities": "cities_share",
        "population": "population_share",
        "votes": "votes_share",
        "minor_allies": "minor_allies_share",
    }
    for source_col, target_col in share_metrics.items():
        group_sum = df.groupby(group_key)[source_col].transform("sum")
        df[target_col] = np.where(
            group_sum == 0, 0,
            (df[source_col] / group_sum.replace(0, 1)) * df["max_players"] * 100,
        )

    gap_metrics = {
        "technologies": "technologies_gap",
        "policies": "policies_gap",
    }
    for source_col, target_col in gap_metrics.items():
        group_max = df.groupby(group_key)[source_col].transform("max")
        df[target_col] = group_max - df[source_col]

    if "military_units" in df.columns and "military_supply" in df.columns:
        supply = df["military_supply"]
        df["military_utilization"] = np.where(supply == 0, 0, df["military_units"] / supply.replace(0, 1))

    return df


def add_raw_share_features(df: pd.DataFrame) -> pd.DataFrame:
    """Competitive shares from raw (non-city-adjusted) per-turn values."""
    df = df.copy()
    group_key = ["game_id", "turn"]
    raw_share_metrics = {
        "science_per_turn": "science_raw_share",
        "culture_per_turn": "culture_raw_share",
        "tourism_per_turn": "tourism_raw_share",
        "gold_per_turn": "gold_raw_share",
        "faith_per_turn": "faith_raw_share",
        "production_per_turn": "production_raw_share",
        "food_per_turn": "food_raw_share",
    }
    for source_col, target_col in raw_share_metrics.items():
        group_sum = df.groupby(group_key)[source_col].transform("sum")
        df[target_col] = np.where(
            group_sum == 0, 0,
            (df[source_col] / group_sum.replace(0, 1)) * df["max_players"] * 100,
        )
    return df


def drop_transformed_columns(df: pd.DataFrame, keep_variants: bool = False) -> pd.DataFrame:
    """Drop original/intermediate columns superseded by the transformations."""
    if keep_variants:
        cols_to_drop = [
            "city_multiplier", "max_players",
            "rationale",
            "military_units", "military_supply",
            "territory",
            "score", "max_score", "rank",
        ]
    else:
        cols_to_drop = [
            "science_per_turn", "culture_per_turn", "tourism_per_turn",
            "gold_per_turn", "faith_per_turn", "production_per_turn", "food_per_turn",
            "military_strength",
            "military_units", "military_supply",
            "science_adj", "culture_adj", "tourism_adj", "gold_adj",
            "faith_adj", "production_adj", "food_adj", "military_adj",
            "city_multiplier", "max_players",
            "cities", "population", "territory", "votes", "minor_allies",
            "technologies", "policies",
            "score", "max_score", "rank",
            "rationale",
        ]
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])


def prepare_features(
    df: pd.DataFrame,
    keep_ids: bool = True,
    use_variant_columns: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target vector y for modelling.

    Mirrors the source: ID columns (``game_id``/``turn``/``player_id``) and the
    ``_is_zero_score`` marker ride along with the selected features so grouped
    torch models that need them receive them directly. The estimator's own
    ``_filter_features`` / ID-stripping narrows further at predict time.
    """
    if use_variant_columns:
        meta_cols = {
            "game_id", "turn", "player_id", "experiment", "civilization",
            "is_winner", "is_changed", "max_turn", "_is_zero_score",
        }
        feature_cols = [
            c for c in df.columns
            if c not in meta_cols
            and not c.startswith("flavor_")
            and c != "grand_strategy"
        ]
    else:
        feature_cols = [c for c in SELECTED_FEATURES if c in df.columns]

    if keep_ids:
        id_cols = ["game_id", "turn", "player_id"]
        meta_passthrough = [c for c in ["_is_zero_score"] if c in df.columns]
        X = df[id_cols + meta_passthrough + feature_cols].copy()
    else:
        X = df[feature_cols].copy()

    y = df["is_winner"].copy()
    return X, y


# ── load-only feature frame ──────────────────────────────────────────────────
def build_feature_frame(
    turns_csv: str,
    use_variants: bool = False,
    filter_zero_score: bool = False,
    preloaded_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load the canonical turns CSV and run the full feature pipeline.

    This is the inference-side counterpart to the source ``load_and_prepare_base_data``
    + the front half of ``run_full_prediction``: it produces the engineered frame a
    saved model can be re-applied to. ``filter_zero_score`` defaults to ``False`` so
    predictions cover every row (including eliminated players / opening turns), exactly
    as the training repo's full-prediction mode did. The per-model ``FILTER_ZERO_SCORE``
    flag is irrelevant at inference time.
    """
    if preloaded_df is not None:
        df = preloaded_df.copy()
    else:
        # The canonical turns table is wide and can contain mixed telemetry
        # payload columns. Infer against the whole file so pandas does not assign
        # conflicting per-chunk dtypes (and emit DtypeWarning) on large runs.
        df = pd.read_csv(turns_csv, low_memory=False)
    if "experiment" in df.columns and "condition" not in df.columns:
        df["condition"] = df["experiment"]

    # Mark eliminated/inactive players before any filtering (matches source).
    df["_is_zero_score"] = df["score"] == 0
    if filter_zero_score:
        df = df[df["score"] != 0]

    df = apply_city_adjustments(df)
    df = add_relative_features(df)
    if use_variants:
        df = add_raw_share_features(df)
    df = add_competitive_features(df)
    df = drop_transformed_columns(df, keep_variants=use_variants)
    return df
