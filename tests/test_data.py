"""Config-shaped dataframe filters."""

from __future__ import annotations

import pandas as pd
import pytest

from bench.catalog import Catalog
from bench.data import apply_filter_spec
from bench.data.failures import decision_failure_games, failed_game_ids


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


# ── decision-failure game filter ────────────────────────────────────────────

def _failure_tokens(rows):
    return pd.DataFrame([
        {
            "experiment": exp, "game_id": game, "player_id": player,
            "valid_turn_count": valid, "failed_turn_count": failed,
            "failure_pct": failed / valid if valid else 0.0,
        }
        for exp, game, player, valid, failed in rows
    ])


def test_decision_failure_default_is_exact_and_unrounded():
    tokens = _failure_tokens([
        ("exp", "exact", 0, 5, 1),
        ("exp", "below", 0, 100000, 19999),
    ])
    tokens["failure_pct"] = tokens["failure_pct"].round(4)
    assert failed_game_ids(tokens, {}) == {"exact"}


def test_decision_failure_excludes_any_player_even_when_game_average_is_low():
    tokens = _failure_tokens([
        ("exp", "bad", 0, 5, 1),
        ("exp", "bad", 1, 100, 0),
    ])
    games = decision_failure_games(tokens).set_index("game_id")
    assert games.loc["bad", "failure_pct"] < 0.2
    assert failed_game_ids(tokens, {}) == {"bad"}


def test_decision_failure_duplicate_model_rows_do_not_change_result():
    tokens = _failure_tokens([
        ("exp", "bad", 0, 10, 2),
        ("exp", "bad", 0, 10, 2),
        ("exp", "good", 0, 10, 1),
        ("exp", "good", 0, 10, 1),
    ])
    assert failed_game_ids(tokens, {}) == {"bad"}


@pytest.mark.parametrize("resolved, expected", [
    ({"max_decision_failure_pct": None}, set()),
    ({"max_decision_failure_pct": 0.3}, {"high"}),
])
def test_decision_failure_null_and_custom_cutoff(resolved, expected):
    tokens = _failure_tokens([
        ("exp", "high", 0, 10, 4),
        ("exp", "medium", 0, 10, 1),
    ])
    assert failed_game_ids(tokens, resolved) == expected


def test_decision_failure_missing_legacy_and_zero_telemetry_are_retained():
    missing = pd.DataFrame([{"experiment": "exp", "game_id": "g", "player_id": 0}])
    legacy = pd.DataFrame([{
        "experiment": "exp", "game_id": "g", "player_id": 0,
        "valid_turn_count": 10,
    }])
    zero = _failure_tokens([("exp", "zero", 0, 0, 0)])
    assert failed_game_ids(missing, {}) == set()
    assert failed_game_ids(legacy, {}) == set()
    assert failed_game_ids(zero, {}) == set()


def test_completeness_excludes_failed_games_and_keeps_design_reference_grid():
    from bench.data import condition_completeness, incomplete_experiments

    games = pd.DataFrame([
        {"experiment": "full", "game_id": "f0", "seed": 1, "seating_rotation": 0},
        {"experiment": "full", "game_id": "f1", "seed": 1, "seating_rotation": 1},
        {"experiment": "gone", "game_id": "g0", "seed": 1, "seating_rotation": 0},
        {"experiment": "gone", "game_id": "g1", "seed": 1, "seating_rotation": 1},
    ])
    assert condition_completeness(games, excluded_game_ids={"g0", "g1"}) == {
        "full": 1.0, "gone": 0.0,
    }
    assert incomplete_experiments(games, 1.0, excluded_game_ids={"g0", "g1"}) == {"gone"}


def test_precomputed_failure_ids_filter_gridless_tables_before_min_games():
    tokens = pd.DataFrame({
        "experiment": ["exp", "exp", "exp"],
        "game_id": ["bad", "good-1", "good-2"],
        "player_type": ["A", "A", "A"],
    })
    out = apply_filter_spec(
        tokens,
        filter_spec={"min_games": 2},
        decision_failure_ids={"bad"},
    )
    assert out["game_id"].tolist() == ["good-1", "good-2"]
    out = apply_filter_spec(
        tokens, filter_spec={"min_games": 3}, decision_failure_ids={"bad"},
    )
    assert out.empty
