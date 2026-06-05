"""Config-shaped dataframe filters."""

from __future__ import annotations

import pandas as pd

from bench.catalog import Catalog
from bench.data import apply_filter_spec


def _catalog() -> Catalog:
    return Catalog(
        {
            "vanilla_label": "Vanilla",
            "null_label": "Null",
            "strategist_models": [],
            "strategist_variants": {},
        },
        {
            "vanilla_experiments": ["vanilla"],
            "null_ai_experiments": ["null"],
        },
    )


def test_apply_filter_spec_merges_global_and_stage_filters():
    df = pd.DataFrame(
        {
            "condition": ["staff", "staff", "staff", "staff", "other", "vanilla"],
            "player_type": ["A", "A", "A", "B", "A", "Vanilla"],
            "game_id": [1, 1, 2, 1, 3, 4],
            "turn": [100, 220, 230, 250, 260, 300],
        }
    )

    out = apply_filter_spec(
        df,
        catalog=_catalog(),
        filter_spec={"experiments": ["staff", "other", "vanilla"], "min_games": 2},
        stage_filter={"experiments": ["staff"], "turn_range": [200, None]},
    )

    assert out["player_type"].tolist() == ["A", "A"]
    assert out["turn"].tolist() == [220, 230]


def test_apply_filter_spec_only_llm_drops_baselines():
    df = pd.DataFrame(
        {
            "condition": ["staff", "staff", "vanilla", "null"],
            "player_type": ["A", "Vanilla", "Vanilla", "Null"],
            "game_id": [1, 1, 2, 3],
        }
    )

    out = apply_filter_spec(df, catalog=_catalog(), filter_spec={"only_llm": True})

    assert out["player_type"].tolist() == ["A"]
