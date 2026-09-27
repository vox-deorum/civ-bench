from types import SimpleNamespace

import pandas as pd
import pytest

from bench.analyses.errors import AnalysisError
from bench.analyses.performance.usage import (
    cache_accounting,
    compute_game_costs,
    cost_note,
    plot_usage_figures,
    summarize_game_costs,
)


class FakeCatalog:
    vanilla_label = "Vanilla"
    null_label = "Null"

    def __init__(self, estimate=(), cache=None):
        self._estimate = set(estimate)
        # {model: (read_ratio, estimate)}; None turns cache accounting off.
        self._cache = cache

    def cache_policy(self, model):
        if self._cache is None:
            return None
        return self._cache.get(model, (0.1, True))

    def pricing_per_million(self):
        return {"priced": {"input_per_million": 1.0, "output_per_million": 2.0}}

    def reasoning_estimate_models(self):
        return set(self._estimate)

    def canonicalize_model_name(self, name):
        return str(name).lower()

    def strategist_model_colors(self):
        return {"priced": "#123456", "free": "#654321"}

    def split_player_type(self, value):
        return {"model_id": str(value).split("-")[0], "variant": ""}

    def split_condition_suffix(self, value, suffixes):
        value = str(value)
        for suffix in suffixes:
            if value.endswith(suffix):
                return value[: -len(suffix)], suffix
        return value, ""


def test_unpriced_tokens_still_have_token_averages():
    catalog = FakeCatalog()
    raw = pd.DataFrame([
        {"game_id": "g1", "player_type": "priced", "model_name": "priced",
         "input_tokens": 100, "output_tokens": 10, "reasoning_tokens": 0},
        {"game_id": "g1", "player_type": "free", "model_name": "free",
         "input_tokens": 200, "output_tokens": 20, "reasoning_tokens": 0},
    ])
    game = compute_game_costs(raw, catalog)
    summary = summarize_game_costs(game, ["player_type"])
    free = summary.set_index("player_type").loc["free"]
    assert free["avg_input"] == 200
    assert free["avg_output"] == 20
    assert pd.isna(free["avg_cost_per_player_game"])
    assert free["token_complete_player_games"] == 1
    assert free["cost_complete_player_games"] == 0
    assert free["token_na_player_games"] == 0


def _estimate_rows():
    # "priced" fully recorded 30 reasoning over 20 output across two games
    # (ratio 1.5). Its g2 has 5 partly reported reasoning tokens that the estimate
    # replaces. "free" has the same shape but is not flagged.
    return pd.DataFrame([
        {"game_id": "g1", "player_type": "priced", "model_name": "priced",
         "input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 30,
         "reasoning_recorded_tokens": 30, "reasoning_recorded_output_tokens": 20},
        {"game_id": "g2", "player_type": "priced", "model_name": "priced",
         "input_tokens": 100, "output_tokens": 40, "reasoning_tokens": 5,
         "reasoning_recorded_tokens": 0, "reasoning_recorded_output_tokens": 0},
        {"game_id": "g1", "player_type": "free", "model_name": "free",
         "input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 30,
         "reasoning_recorded_tokens": 30, "reasoning_recorded_output_tokens": 20},
    ])


def test_reasoning_estimate_pools_recorded_calls_across_games():
    game = compute_game_costs(_estimate_rows(), FakeCatalog(estimate={"priced"}))
    rows = game.set_index(["model", "game_id"])
    # g1: 30 + 1.5 * (50 - 20) = 75 reasoning, plus 50 output.
    assert rows.loc[("priced", "g1"), "combined_output_tokens"] == 125
    # g2: the partial 5 is replaced by 1.5 * 40 = 60 reasoning, plus 40 output.
    assert rows.loc[("priced", "g2"), "combined_output_tokens"] == 100
    assert rows.loc[("priced", "g2"), "total_cost"] == pytest.approx(100 / 1e6 * 1.0 + 100 / 1e6 * 2.0)
    assert rows.loc[("priced", "g2"), "reasoning_estimated"]
    # Unflagged models keep their raw telemetry.
    assert rows.loc[("free", "g1"), "combined_output_tokens"] == 80
    assert not rows.loc[("free", "g1"), "reasoning_estimated"]


def test_reasoning_estimate_keeps_raw_values_without_recorded_calls():
    raw = _estimate_rows()
    raw["reasoning_recorded_tokens"] = 0
    raw["reasoning_recorded_output_tokens"] = 0
    game = compute_game_costs(raw, FakeCatalog(estimate={"priced"}))
    rows = game.set_index(["model", "game_id"])
    assert rows.loc[("priced", "g1"), "combined_output_tokens"] == 80
    assert rows.loc[("priced", "g2"), "combined_output_tokens"] == 45
    assert not game["reasoning_estimated"].any()


def test_reasoning_estimate_requires_recorded_output_column():
    raw = _estimate_rows().drop(columns=["reasoning_recorded_tokens"])
    with pytest.raises(AnalysisError, match="civ-bench extract"):
        compute_game_costs(raw, FakeCatalog(estimate={"priced"}))
    # Without a flagged model the old table shape still works.
    compute_game_costs(raw, FakeCatalog())


def _cache_rows(g2_cached=0, g2_repeated=1500):
    return pd.DataFrame([
        {"game_id": "g1", "player_type": "priced", "model_name": "priced",
         "input_tokens": 1000, "cached_input_tokens": 100, "repeated_input_tokens": 400,
         "output_tokens": 10, "reasoning_tokens": 0},
        {"game_id": "g2", "player_type": "priced", "model_name": "priced",
         "input_tokens": 1000, "cached_input_tokens": g2_cached,
         "repeated_input_tokens": g2_repeated,
         "output_tokens": 0, "reasoning_tokens": 0},
    ])


def test_cached_input_prices_the_larger_count_at_the_read_rate():
    rows = compute_game_costs(_cache_rows(), FakeCatalog(cache={})).set_index("game_id")
    # g1: max(reported 100, overlap 400) cached; the other 600 stay at full price.
    assert rows.loc["g1", "cached_input_tokens"] == 400
    assert rows.loc["g1", "total_cost"] == pytest.approx(
        (600 + 400 * 0.1) / 1e6 * 1.0 + 10 / 1e6 * 2.0
    )
    # Overlap beyond the row's input is capped at input, so g2 is all cache-read.
    assert rows.loc["g2", "cached_input_tokens"] == 1000
    assert rows.loc["g2", "total_cost"] == pytest.approx(1000 * 0.1 / 1e6 * 1.0)


def test_reported_only_family_ignores_the_step_overlap():
    catalog = FakeCatalog(cache={"priced": (0.05, False)})
    rows = compute_game_costs(_cache_rows(), catalog).set_index("game_id")
    # Only the provider-reported 100 counts, priced at the family ratio 0.05.
    assert rows.loc["g1", "cached_input_tokens"] == 100
    assert rows.loc["g1", "total_cost"] == pytest.approx(
        (900 + 100 * 0.05) / 1e6 * 1.0 + 10 / 1e6 * 2.0
    )


def test_switches_off_reproduce_full_price_raw_costs():
    raw = _estimate_rows()
    raw["cached_input_tokens"] = 100
    raw["repeated_input_tokens"] = 400
    game = compute_game_costs(
        raw, FakeCatalog(estimate={"priced"}, cache={}),
        estimate_reasoning=False, cached_input=False,
    )
    rows = game.set_index(["model", "game_id"])
    # Raw reasoning is kept even for the flagged model.
    assert rows.loc[("priced", "g1"), "combined_output_tokens"] == 80
    assert rows.loc[("priced", "g2"), "combined_output_tokens"] == 45
    assert not game["reasoning_estimated"].any()
    # The cache telemetry is ignored, so every input token is at full price.
    assert rows.loc[("priced", "g2"), "cached_input_tokens"] == 0
    assert rows.loc[("priced", "g2"), "total_cost"] == pytest.approx(
        100 / 1e6 * 1.0 + 45 / 1e6 * 2.0
    )


def test_cached_input_requires_the_cache_columns():
    raw = _estimate_rows()  # no cached_input_tokens / repeated_input_tokens
    with pytest.raises(AnalysisError, match="Cached-input estimates need"):
        compute_game_costs(raw, FakeCatalog(cache={}))
    # A catalog without cache_pricing still works on the old table shape.
    compute_game_costs(raw, FakeCatalog())


def test_summarize_game_costs_exposes_avg_cached_input():
    records = compute_game_costs(_cache_rows(g2_repeated=200), FakeCatalog(cache={}))
    summary = summarize_game_costs(records, ["player_type"])
    row = summary.set_index("player_type").loc["priced"]
    # g1 caches 400, g2 caches 200.
    assert row["avg_cached_input"] == 300


def test_cache_accounting_follows_the_catalog_not_only_the_switch():
    catalog = FakeCatalog(cache={"priced": (0.05, False)})
    assert cache_accounting(["priced", "free"], catalog, True) == (True, ["priced"])
    assert cache_accounting(["priced"], catalog, False) == (False, [])
    # Without cache_pricing nothing is discounted, even with the switch on.
    assert cache_accounting(["priced"], FakeCatalog(), True) == (False, [])
    assert "priced uses only provider-reported" in cost_note(True, ["priced"])
    assert "a, b use only provider-reported" in cost_note(True, ["a", "b"])
    assert "provider-reported cache counts." not in cost_note(True)


def test_catalog_rejects_non_boolean_reasoning_flag():
    from bench.catalog import Catalog
    from bench.config.errors import ConfigError

    models = {"strategist_models": [
        {"id": "A", "color": "#000000", "estimate_reasoning": True},
        {"id": "B", "color": "#000000"},
    ]}
    assert Catalog(models, {}).reasoning_estimate_models() == {"A"}
    models["strategist_models"][1]["estimate_reasoning"] = "yes"
    with pytest.raises(ConfigError, match="estimate_reasoning"):
        Catalog(models, {})


def test_usage_figures_share_three_keys_and_exclude_baselines():
    table = pd.DataFrame([
        {"player_type": "Vanilla", "avg_cost_per_player_game": 1, "avg_input": 1, "avg_output": 1},
        {"player_type": "priced", "avg_cost_per_player_game": 2, "avg_input": 2, "avg_output": 3},
        {"player_type": "free", "avg_cost_per_player_game": float("nan"), "avg_input": 4, "avg_output": 5},
    ])
    ctx = SimpleNamespace(catalog=FakeCatalog(), condition_pairing=lambda: None)
    figures = plot_usage_figures(table, ctx)
    assert set(figures) == {"cost", "input_tokens", "output_tokens"}
    assert all("Vanilla" not in ax.get_yticklabels()[0].get_text() for ax in [figures["cost"].axes[0]])


def test_paired_usage_figures_order_base_identities_by_elo():
    from bench.plotting.pairing import PairingSpec

    table = pd.DataFrame([
        {"player_type": "A", "avg_cost_per_player_game": 1, "avg_input": 1, "avg_output": 1},
        {"player_type": "A-Per-5", "avg_cost_per_player_game": 2, "avg_input": 2, "avg_output": 2},
        {"player_type": "B", "avg_cost_per_player_game": 3, "avg_input": 3, "avg_output": 3},
        {"player_type": "B-Per-5", "avg_cost_per_player_game": 4, "avg_input": 4, "avg_output": 4},
    ])
    ratings = pd.DataFrame([
        {"player_type": "A", "elo": 1700},
        {"player_type": "B", "elo": 1600},
    ])
    ctx = SimpleNamespace(
        catalog=FakeCatalog(),
        condition_pairing=lambda: PairingSpec(("-Per-5",), "base", "Base"),
    )
    figures = plot_usage_figures(table, ctx, ratings)
    assert list(figures["cost"].axes[0].get_yticklabels())
    labels = [tick.get_text() for tick in figures["cost"].axes[0].get_yticklabels()]
    assert labels[:2] == ["A", "B"]
