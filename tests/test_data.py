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


# ── global condition-completeness filter (data.filter.min_condition_completeness) ──

def _grid_frame():
    """Controlled rows whose union grid is seed {0, 1} × rotation {0, 1}.

    ``full`` covers every slot (completeness 1.0); ``partial`` covers only
    rotation 0 (completeness 0.5); ``plain`` is uncontrolled (never counted).
    """
    rows = []
    for exp, cells in (
        ("full", [(0, 0), (0, 1), (1, 0), (1, 1)]),
        ("partial", [(0, 0), (1, 0)]),
    ):
        for g, (seed, rot) in enumerate(cells):
            rows.append({
                "experiment": exp, "game_id": f"{exp}-{g}", "player_type": "A",
                "seed": seed, "seating_rotation": rot,
            })
    rows.append({"experiment": "plain", "game_id": "plain-0", "player_type": "A",
                 "seed": -1, "seating_rotation": -1})
    return pd.DataFrame(rows)


def test_condition_completeness_and_incomplete_experiments():
    from bench.data import condition_completeness, incomplete_experiments

    df = _grid_frame()
    assert condition_completeness(df) == {"full": 1.0, "partial": 0.5}
    assert incomplete_experiments(df, 1.0) == {"partial"}
    assert incomplete_experiments(df, 0.5) == set()
    assert condition_completeness(df[["experiment", "game_id"]]) == {}


def test_apply_filter_spec_drops_incomplete_conditions_from_grid():
    df = _grid_frame()

    pruned = apply_filter_spec(df, filter_spec={"min_condition_completeness": 1.0})
    assert set(pruned["experiment"]) == {"full", "plain"}

    kept = apply_filter_spec(df, filter_spec={"min_condition_completeness": 0.5})
    assert set(kept["experiment"]) == {"full", "partial", "plain"}


def test_apply_filter_spec_uses_precomputed_incomplete_conditions_on_gridless_tables():
    # tokens/panel/turns carry no seed/seating_rotation, so the caller resolves the
    # incomplete set once from the games table and passes it down by experiment.
    tokens = pd.DataFrame({
        "experiment": ["full", "full", "partial", "partial", "plain"],
        "game_id": [1, 2, 3, 4, 5],
        "player_type": ["A", "B", "A", "B", "A"],
    })

    out = apply_filter_spec(
        tokens,
        filter_spec={"min_condition_completeness": 1.0},
        condition_incomplete={"partial"},
    )
    assert set(out["experiment"]) == {"full", "plain"}

