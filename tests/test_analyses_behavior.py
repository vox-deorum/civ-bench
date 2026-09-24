"""``behavior.*`` analyses on a small synthetic controlled design.

Design: a baseline experiment where every seat is the in-game AI, and a
treatment experiment where seat 0 is an LLM. Both play seeds 1 and 2. The
treatment also has one uncontrolled game and one controlled game on seed 3,
which the baseline never played (so its players have no matched cell).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.analyses.behavior.commitment import turn_metrics
from bench.analyses.behavior.policies import _earliest
from bench.catalog import Catalog
from bench.config import ConfigError, load_config

BASELINE = "vanilla-standard"
TREATMENT = "llm-standard"
LLM = "Kimi-K2.5"
SURVIVAL = 99  # 100 turns alive, so per-100-turn rates equal raw counts
BEHAVIOR_SPEC = {
    "stats": ["avg"],
    "flavor": ["Offense"],
    "persona": ["Friendliness", "Loyalty"],
    "events": ["wars_declared"],
    "policies": ["policy_changes", "tradition", "progress", "freedom", "order"],
}

# (game_id, experiment, seed, seating_rotation)
GAMES = [
    ("B1a", BASELINE, 1, 0), ("B1b", BASELINE, 1, 1), ("B2a", BASELINE, 2, 0),
    ("T1", TREATMENT, 1, 0), ("T2", TREATMENT, 2, 0),
    ("TU", TREATMENT, -1, -1),   # uncontrolled
    ("T3", TREATMENT, 3, 0),     # controlled, but no baseline game on seed 3
]
# Baseline wars per seat by game; treatment values below.
BASE_WARS = {"B1a": [1, 0, 2], "B1b": [3, 0, 2], "B2a": [2, 2, 2]}
TREAT_WARS = {"T1": [5, 1, 2], "T2": [2, 2, 2], "TU": [9, 9, 9], "T3": [4, 4, 4]}


def _behavior_rows():
    rows = []
    for gid, exp, _seed, _rot in GAMES:
        wars = (BASE_WARS if exp == BASELINE else TREAT_WARS)[gid]
        for pid in range(3):
            llm = exp == TREATMENT and pid == 0
            rows.append({
                "experiment": exp, "game_id": gid, "player_id": pid,
                "player_type": LLM if llm else "Vanilla", "civilization": f"Civ{pid}",
                "survival_turn": SURVIVAL,
                "flavor_offense_avg": 8.0 if llm else 5.0 + pid,
                "persona_friendliness_avg": 3.0 if llm else 6.0,
                "persona_loyalty_avg": 5.0,
                "wars_declared": wars[pid],
                "policy_changes": 10 if llm else 0,
                "tradition": 20 if pid != 1 else "N/A",
                "progress": 15 if pid == 1 else "N/A",
                "freedom": 150 if llm else "N/A",
                "order": "N/A" if llm else 170,
                "relationship_changes": 4 if llm else 0,
                "relationship_targets": 2 if llm else 0,
                "stance_public_avg": 10.0 if llm else "",
                "stance_private_avg": -5.0 if llm else "",
                "stance_net_avg": 5.0 if llm else "",
                "stance_masked_hostility_share": 0.25 if llm else "",
                "stance_masked_goodwill_share": 0.05 if llm else "",
            })
    return rows


def _turn_rows():
    rows = []
    for gid, exp, _seed, _rot in GAMES:
        for pid in range(3):
            llm = exp == TREATMENT and pid == 0
            plan = ["Conquest", "Conquest", "Culture", "Culture"] if llm else ["Spaceship"] * 4
            for turn, gs in enumerate(plan):
                rows.append({
                    "experiment": exp, "game_id": gid, "player_id": pid,
                    "player_type": LLM if llm else "Vanilla", "turn": turn,
                    "is_decision": 1 if llm else 0, "is_changed": int(llm and turn == 2),
                    "grand_strategy": gs, "rationale": "long text",
                })
    return rows


def _panel_rows():
    rows = []
    for gid, exp, _seed, _rot in GAMES:
        for pid in range(3):
            llm = exp == TREATMENT and pid == 0
            rows.append({
                "experiment": exp, "game_id": gid, "player_id": pid,
                "player_type": LLM if llm else "Vanilla", "civilization": f"Civ{pid}",
                "survival_turn": SURVIVAL, "strategy_changes": 6 if llm else 0,
                "persona_changes": 2 if llm else 0, "research_changes": 1,
            })
    return rows


@pytest.fixture
def behavior_env(tmp_path, dev_spec, write_spec):
    tables = {
        "games": pd.DataFrame(GAMES, columns=["game_id", "experiment", "seed", "seating_rotation"]),
        "behavior": pd.DataFrame(_behavior_rows()),
        "turns": pd.DataFrame(_turn_rows()),
        "panel": pd.DataFrame(_panel_rows()),
    }
    paths = {}
    for name, frame in tables.items():
        if name == "games":
            frame = frame.assign(timestamp="2026-01-01")
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)

    def make(stages, *, behavior_spec=BEHAVIOR_SPEC, games=True, adjust_baseline=None):
        spec = json.loads(json.dumps(dev_spec))
        spec["output"] = {"root": str(tmp_path / "out"), "suffix": ""}
        spec["data"]["extract"]["behavior"] = behavior_spec
        spec["data"]["extract"]["issues_path"] = str(tmp_path / "import_issues.csv")
        spec["data"]["tables"] = {k: v for k, v in paths.items() if games or k != "games"}
        spec.pop("estimators", None)
        spec.pop("adjust", None)
        if adjust_baseline:
            spec["estimators"] = dev_spec["estimators"]
            spec["adjust"] = [{**dev_spec["adjust"][0],
                               "params": {"baseline_experiment": adjust_baseline}}]
        spec["analyses"] = stages
        return load_config(write_spec(spec))

    def run(module, params=None, sid="t", **kwargs):
        stage = {"id": sid, "module": module, "enabled": True, "params": params or {}}
        cfg = make([stage], **kwargs)
        result = run_analysis(cfg, cfg.analyses[0].raw, catalog=Catalog.from_run_config(cfg))
        return result, cfg

    run.make = make
    return run


def _table(result, name):
    return pd.read_csv(result.table_paths[name])


def _cell(table, group, metric, column="mean"):
    row = table[(table["player_type"] == group) & (table["metric"] == metric)]
    assert len(row) == 1, (group, metric, table)
    return float(row[column].iloc[0])


# ── profiles ───────────────────────────────────────────────────────────────────
PROFILE = {"baseline_experiment": BASELINE, "metrics": ["wars_declared", "flavor_offense_avg"],
           "bootstrap_n": 50}


def test_relative_subtracts_the_matched_seed_seat_baseline(behavior_env):
    result, _cfg = behavior_env("behavior.profiles", PROFILE)
    rel = _table(result, "profile_relative")
    # Seat 0 baseline: seed 1 → mean(1, 3) = 2, seed 2 → 2. LLM: T1 = 5, T2 = 2.
    assert _cell(rel, LLM, "wars_declared") == pytest.approx(((5 - 2) + (2 - 2)) / 2)
    assert _cell(rel, LLM, "wars_declared", "n_players") == 2
    # Treatment AI seats: T1 seat 1 is 1 vs 0, seat 2 is 2 vs 2; T2 seats are 2 vs 2.
    assert _cell(rel, "Vanilla", "wars_declared") == pytest.approx(1 / 4)
    assert _cell(rel, LLM, "flavor_offense_avg") == pytest.approx(3.0)

    absolute = _table(result, "profile_absolute")
    # Absolute keeps every filtered row, including the uncontrolled and unmatched games.
    assert _cell(absolute, LLM, "wars_declared", "n_players") == 4
    assert _cell(absolute, LLM, "wars_declared") == pytest.approx((5 + 2 + 9 + 4) / 4)

    meta = result.metadata
    assert meta["n_unmatched_controlled_players"] == 3      # the seed-3 game
    assert meta["n_baseline_players"] == 9
    assert list(meta["views"]) == ["relative", "absolute"]
    assert meta["views"]["relative"]["figures"] == ["profile_relative"]
    assert result.summary.startswith("Against the matched in-game AI")


def test_rate_normalizes_counts_by_turns_alive(behavior_env, tmp_path):
    frame = pd.read_csv(tmp_path / "behavior.csv")
    frame["survival_turn"] = 49                                # 50 turns alive
    frame.to_csv(tmp_path / "behavior.csv", index=False)
    per_100, _ = behavior_env("behavior.profiles", PROFILE, sid="a")
    per_game, _ = behavior_env("behavior.profiles", {**PROFILE, "rate": "per_game"}, sid="b")
    a = _table(per_100, "profile_absolute")
    b = _table(per_game, "profile_absolute")
    assert _cell(a, LLM, "wars_declared") == pytest.approx(2 * _cell(b, LLM, "wars_declared"))
    # Levels are never rate-scaled.
    assert _cell(a, LLM, "flavor_offense_avg") == _cell(b, LLM, "flavor_offense_avg")


def test_baseline_defaults_to_the_strength_stage(behavior_env):
    params = {k: v for k, v in PROFILE.items() if k != "baseline_experiment"}
    result, _ = behavior_env("behavior.profiles", params, adjust_baseline=BASELINE)
    assert result.metadata["baseline_experiment"] == BASELINE
    assert "profile_relative" in result.table_paths


@pytest.mark.parametrize("kwargs, note", [
    ({}, "no baseline_experiment is configured"),
    ({"games": False}, "no controlled games"),
])
def test_without_a_baseline_only_the_absolute_view_is_declared(behavior_env, kwargs, note):
    params = {k: v for k, v in PROFILE.items() if k != "baseline_experiment"}
    if kwargs.get("games") is False:
        params["baseline_experiment"] = BASELINE
    result, _ = behavior_env("behavior.profiles", params, **kwargs)
    assert list(result.metadata["views"]) == ["absolute"]
    assert result.metadata["relative_view"] == note
    assert "profile_relative" not in result.table_paths
    assert result.summary.startswith("The most distinctive value")


def test_baseline_survives_player_filters(behavior_env):
    unfiltered, _ = behavior_env("behavior.profiles", PROFILE)
    # Keep only the LLM rows: the baseline pool is still read from the unfiltered table.
    cfg = behavior_env.make([{"id": "f", "module": "behavior.profiles", "enabled": True,
                              "params": PROFILE, "filter": {"players": [LLM]}}])
    filtered = run_analysis(cfg, cfg.analyses[0].raw, catalog=Catalog.from_run_config(cfg))
    rel = _table(filtered, "profile_relative")
    assert set(rel["player_type"]) == {LLM}
    expected = _cell(_table(unfiltered, "profile_relative"), LLM, "wars_declared")
    assert _cell(rel, LLM, "wars_declared") == expected


def test_tables_are_byte_stable(behavior_env):
    first, _ = behavior_env("behavior.profiles", PROFILE, sid="one")
    second, _ = behavior_env("behavior.profiles", PROFILE, sid="one")
    for name in ("profile_relative", "profile_absolute"):
        with open(first.table_paths[name], "rb") as a, open(second.table_paths[name], "rb") as b:
            assert a.read() == b.read()
    ci = _table(first, "profile_absolute")[["ci_lower", "ci_upper"]]
    assert ci.notna().all().all()


# ── diplomacy ──────────────────────────────────────────────────────────────────
def test_diplomacy_drops_stance_from_the_relative_view(behavior_env):
    result, _ = behavior_env("behavior.diplomacy", {"baseline_experiment": BASELINE,
                                                     "traits": ["Friendliness", "Loyalty"],
                                                     "bootstrap_n": 20})
    rel = _table(result, "diplomacy_relative")
    assert _cell(rel, LLM, "persona_friendliness_avg") == pytest.approx(-3.0)
    assert not rel["metric"].str.startswith("stance_").any()
    dropped = result.metadata["relative_metrics_without_baseline"]
    assert "stance_net_avg" in dropped and "stance_masked_hostility_share" in dropped
    # relationship_changes is 0 for the AI, so it is a valid relative metric.
    assert _cell(rel, LLM, "relationship_changes") == pytest.approx(4.0)

    mix = _table(result, "stance_mix_absolute")
    assert list(mix["player_type"]) == [LLM]
    row = mix.iloc[0]
    assert row["masked hostility"] == pytest.approx(0.25)
    assert row["aligned"] == pytest.approx(0.70)
    assert "stance_mix_absolute" in result.metadata["views"]["absolute"]["figures"]
    assert "masked hostility" in result.summary


# ── commitment ─────────────────────────────────────────────────────────────────
def test_turn_metrics_count_switches_and_pivots():
    turns = pd.DataFrame({
        "game_id": ["g"] * 5, "player_id": [0] * 5, "turn": [4, 0, 1, 2, 3],
        "is_decision": [1, 1, 0, 1, 1], "is_changed": [0, 1, 0, 0, 0],
        "grand_strategy": ["Conquest", "Culture", "Culture", np.nan, "Conquest"],
    })
    m = turn_metrics(turns, ["Conquest", "Culture", "Spaceship"]).iloc[0]
    assert m["decision_rate"] == pytest.approx(0.8)
    assert m["change_rate"] == pytest.approx(0.2)
    # Recorded strategies by turn: Culture, Culture, Conquest, Conquest -> one switch in 4.
    assert m["grand_strategy_switches"] == pytest.approx(25.0)
    assert m["main_strategy_share"] == pytest.approx(0.5)
    assert m["first_pivot_turn"] == 3
    assert m["grand_strategy_share_spaceship"] == 0


def test_commitment_views(behavior_env):
    result, _ = behavior_env("behavior.commitment", {"baseline_experiment": BASELINE,
                                                      "bootstrap_n": 20})
    rel = _table(result, "commitment_relative")
    assert _cell(rel, LLM, "decision_rate") == pytest.approx(1.0)
    assert _cell(rel, LLM, "grand_strategy_switches") == pytest.approx(25.0)
    assert _cell(rel, LLM, "strategy_changes") == pytest.approx(6.0)
    shares = _table(result, "strategy_shares_relative")
    assert _cell(shares, LLM, "grand_strategy_share_spaceship") == pytest.approx(-1.0)
    mix = _table(result, "strategy_mix_absolute").set_index("player_type")
    assert mix.loc[LLM, "Conquest"] == pytest.approx(0.5)
    assert result.metadata["grand_strategies"] == ["Conquest", "Culture", "Spaceship"]


# ── policies ───────────────────────────────────────────────────────────────────
def test_earliest_branch_uses_turns_and_none():
    df = pd.DataFrame({"a": [5, np.nan, np.nan, 3], "b": [2, 7, np.nan, 3]})
    assert list(_earliest(df, ["a", "b"])) == ["b", "b", "none", "a"]


def test_policies_views(behavior_env):
    result, _ = behavior_env("behavior.policies", {"baseline_experiment": BASELINE,
                                                    "bootstrap_n": 20})
    rel = _table(result, "adoption_relative")
    # Seat 0 in the baseline always adopts order and never freedom.
    assert _cell(rel, LLM, "adopted_freedom") == pytest.approx(1.0)
    assert _cell(rel, LLM, "adopted_order") == pytest.approx(-1.0)
    turns = _table(result, "adoption_turn_absolute")
    assert _cell(turns, LLM, "adoption_turn_freedom") == pytest.approx(150)
    ideologies = _table(result, "ideologies_absolute").set_index("player_type")
    assert ideologies.loc[LLM, "freedom"] == pytest.approx(1.0)
    openings = _table(result, "openings_absolute").set_index("player_type")
    # 17 in-game AI rows (9 baseline, 8 treatment); seat 1 opens with progress in all 7 games.
    assert openings.loc["Vanilla", "n_players"] == 17
    assert openings.loc["Vanilla", "progress"] == pytest.approx(7 / 17)
    assert openings.loc[LLM, "tradition"] == pytest.approx(1.0)
    figures = result.metadata["views"]["absolute"]["figures"]
    assert figures[:2] == ["adoption_absolute", "adoption_turn_absolute"]


# ── config validation ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("module, params, match", [
    ("behavior.profiles", {}, "params.metrics: required"),
    ("behavior.profiles", {"metrics": ["wars_received"]}, "not columns of the behavior table"),
    ("behavior.profiles", {"metrics": ["wars_declared"], "rate": "per_turn"}, "must be one of"),
    ("behavior.profiles", {"metrics": ["wars_declared"], "bogus": 1}, "unknown key"),
    ("behavior.diplomacy", {"traits": ["Charm"]}, "unknown persona trait"),
    ("behavior.diplomacy", {"traits": ["Boldness"]}, "need data.extract.behavior.persona"),
    ("behavior.policies", {"branches": ["artistry"]}, "not selected"),
    ("behavior.policies", {"rate": "per_game"}, "unknown key"),
    ("behavior.commitment", {"bootstrap_n": 0}, "integer >= 1"),
])
def test_config_rejects_bad_behavior_params(behavior_env, module, params, match):
    with pytest.raises(ConfigError, match=match):
        behavior_env.make([{"id": "x", "module": module, "enabled": True, "params": params}])


def test_profiles_metrics_are_not_prediction_metrics(behavior_env):
    cfg = behavior_env.make([{"id": "x", "module": "behavior.profiles", "enabled": True,
                              "params": {"metrics": ["wars_declared", "tradition"]}}])
    assert cfg.analyses[0].raw["params"]["metrics"] == ["wars_declared", "tradition"]
