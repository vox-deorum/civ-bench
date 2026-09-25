"""``behavior.*`` analyses on a small synthetic controlled design.

Design: a baseline experiment where every seat is the in-game AI, and a
treatment experiment where seat 0 is an LLM. Both play seeds 1 and 2. The
treatment also has one uncontrolled game and one controlled game on seed 3,
which the baseline never played (so its players have no matched cell).
Neither fills the whole controlled grid; :func:`_add_complete_experiments`
adds two that do, for the ``"completed"`` baseline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.analyses.behavior import common as C
from bench.analyses.behavior.commitment import turn_metrics
from bench.analyses.behavior.policies import _earliest
from bench.catalog import Catalog
from bench.config import ConfigError, load_config
from bench.config.behavior import flavor_gate_text

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


def _add_complete_experiments(tmp_path):
    """Two experiments on every controlled slot: an LLM one and a Null one.

    Controlled slots are (1, 0), (1, 1), (2, 0), and (3, 0). ``LLM2`` sets Offense
    to 40 / 60 / 30 / 70 on seat 0; Null holds 50. Their AI opponents set nothing.
    """
    slots = [(1, 0), (1, 1), (2, 0), (3, 0)]
    games = pd.read_csv(tmp_path / "games.csv")
    behavior = pd.read_csv(tmp_path / "behavior.csv")
    new_games, new_rows = [], []
    for exp, strategist, values in (
        (COMPLETE, LLM2, [40.0, 60.0, 30.0, 70.0]),
        (NULL_EXP, "Null", [50.0] * 4),
    ):
        for (seed, rot), value in zip(slots, values):
            gid = f"{exp}-{seed}{rot}"
            new_games.append({"game_id": gid, "experiment": exp, "seed": seed,
                              "seating_rotation": rot, "timestamp": "2026-01-01"})
            for pid in range(3):
                new_rows.append({
                    "experiment": exp, "game_id": gid, "player_id": pid,
                    "player_type": strategist if pid == 0 else "Vanilla",
                    "civilization": f"Civ{pid}", "survival_turn": SURVIVAL,
                    "flavor_offense_avg": value if pid == 0 else np.nan,
                })
    pd.concat([games, pd.DataFrame(new_games)]).to_csv(tmp_path / "games.csv", index=False)
    pd.concat([behavior, pd.DataFrame(new_rows)]).to_csv(tmp_path / "behavior.csv", index=False)


def _cell(table, group, metric, column="mean"):
    row = table[(table["player_type"] == group) & (table["metric"] == metric)]
    assert len(row) == 1, (group, metric, table)
    return float(row[column].iloc[0])


# ── flavors and the shared view machinery ──────────────────────────────────────
COMPLETE = "llm2-standard"
NULL_EXP = "null-standard"
LLM2 = "GLM-5"
FLAVORS = {"baseline": BASELINE, "bootstrap_n": 50}
OFFENSE = "flavor_offense_avg"


def test_relative_subtracts_the_matched_seed_seat_baseline(behavior_env):
    result, _cfg = behavior_env("behavior.flavors", FLAVORS)
    rel = _table(result, "flavors_relative")
    # Seat 0 baseline is 5 on both seeds, the LLM sets 8; AI seats match their own seat.
    assert _cell(rel, LLM, OFFENSE) == pytest.approx(3.0)
    assert _cell(rel, LLM, OFFENSE, "n_players") == 2
    assert _cell(rel, "Vanilla", OFFENSE) == pytest.approx(0.0)

    absolute = _table(result, "flavors_absolute")
    # Absolute keeps every filtered row, including the uncontrolled and unmatched games.
    assert _cell(absolute, LLM, OFFENSE, "n_players") == 4
    assert _cell(absolute, LLM, OFFENSE) == pytest.approx(8.0)

    meta = result.metadata
    assert meta["n_unmatched_controlled_players"] == 3      # the seed-3 game
    assert meta["n_baseline_players"] == 9
    assert meta["baseline_experiments"] == [BASELINE]
    assert list(meta["views"]) == ["relative", "absolute"]
    assert meta["views"]["relative"]["tables"] == ["flavors_relative"]
    assert meta["views"]["relative"]["label"] == "Relative"
    assert meta["views"]["relative"]["tip"] == f"Relative to matched '{BASELINE}'"
    assert result.summary.startswith(f"Against the matched '{BASELINE}' on the same map and seat")
    assert f"**{LLM}** on Offense (**+3**)" in result.summary


def test_flavor_heatmaps_carry_layout_and_colors(behavior_env, tmp_path):
    frame = pd.read_csv(tmp_path / "behavior.csv")
    frame["flavor_offense_min"] = frame["flavor_offense_avg"] - 2
    frame["flavor_offense_max"] = frame["flavor_offense_avg"] + 2
    frame.to_csv(tmp_path / "behavior.csv", index=False)
    spec = {**BEHAVIOR_SPEC, "stats": ["min", "avg", "max"]}
    result, _ = behavior_env("behavior.flavors", FLAVORS, behavior_spec=spec)
    spec = result.metadata["heatmaps"]["flavors_absolute"]
    assert spec["column_labels"] == {OFFENSE: "Off"}
    assert spec["column_names"] == {OFFENSE: "Offense"}
    assert spec["column_tips"][OFFENSE].startswith("Offense\nPivots the military")
    assert spec["column_group"] == "metric_group"
    assert spec["range_columns"] == ["mean_min", "mean_max"]
    absolute = _table(result, "flavors_absolute")
    # Absolute colors sit on the fixed 0 to 100 flavor scale.
    assert _cell(absolute, LLM, OFFENSE, "color_position") == pytest.approx(0.08)
    assert _cell(absolute, LLM, OFFENSE, "mean_min") == pytest.approx(6.0)
    assert _cell(absolute, LLM, OFFENSE, "mean_max") == pytest.approx(10.0)
    assert set(absolute["metric_group"]) == {"Military"}
    rel = _table(result, "flavors_relative")
    # Relative colors are the difference in pool SDs, full color at 2 SDs.
    sd = float(pd.Series([5.0, 6.0, 7.0] * 3).std(ddof=1))
    expected = 0.5 + 0.5 * min(1.0, 3.0 / sd / 2.0)
    assert _cell(rel, LLM, OFFENSE, "color_position") == pytest.approx(expected)
    assert "mean_min" not in rel.columns
    assert result.metadata["heatmaps"]["flavors_relative"]["signed"] is True


def test_completed_baseline_averages_complete_strategist_experiments(behavior_env, tmp_path):
    _add_complete_experiments(tmp_path)
    result, _ = behavior_env("behavior.flavors", {"bootstrap_n": 20})
    meta = result.metadata
    assert meta["baseline"] == "completed-experiment average"
    # Null's and the AI opponents' rows are not part of the pool.
    assert meta["n_baseline_experiments"] == 1
    assert meta["n_baseline_players"] == 4
    rel = _table(result, "flavors_relative")
    # Seat-0 cells: (1) mean(40, 60) = 50, (2) 30, (3) 70. The LLM sets 8 on T1, T2, T3.
    assert _cell(rel, LLM, OFFENSE) == pytest.approx(((8 - 50) + (8 - 30) + (8 - 70)) / 3)
    # A completed experiment is part of its own baseline.
    assert _cell(rel, LLM2, OFFENSE) == pytest.approx(0.0)
    # The Null strategist never moves off 50, so it is left out of both tables.
    absolute = _table(result, "flavors_absolute")
    assert "Null" not in set(rel["player_type"]) | set(absolute["player_type"])
    assert meta["views"]["relative"]["label"] == "Relative"
    assert meta["views"]["relative"]["tip"] == "Relative to completed-experiment average"
    # Both views pin the baseline pool itself as the first row, in absolute
    # terms on the 0 to 100 scale, without a CI or a median.
    pinned = "Completed-experiment average"
    pool = pd.Series([40.0, 60.0, 30.0, 70.0])
    for name, table in (("flavors_relative", rel), ("flavors_absolute", absolute)):
        base = table[table["row_kind"] == "baseline"]
        assert len(base) == 1, name
        row = base.iloc[0]
        assert row["player_type"] == row["strategist"] == row["row_label"] == pinned
        assert pd.isna(row["condition"]) and row["metric"] == OFFENSE   # "" reads back as NaN
        assert row["mean"] == pytest.approx(50.0)
        assert row["sd"] == pytest.approx(pool.std())
        assert row["n_players"] == 4 and row["n_games"] == 4
        assert row["color_position"] == pytest.approx(0.5)   # 50 on the fixed 0-100 scale
        assert np.isnan(row["ci_lower"]) and np.isnan(row["ci_upper"])
        assert np.isnan(row["median"])
        spec = meta["heatmaps"][name]
        assert spec["baseline_rows"] == [pinned]
        # Null is gone from the tables, so it leaves the reference rows too.
        assert spec["reference_rows"] == [pinned, "Vanilla"]
        assert spec["row_order"][0] == pinned
        expected_label = "Difference" if name == "flavors_relative" else "Average"
        assert spec["value_label"] == expected_label
    # The headline leaves Null and the pinned baseline row out.
    assert "**Null**" not in result.summary
    assert pinned not in result.summary


def test_condition_pairing_labels_rows_like_matched_maps(behavior_env, tmp_path):
    _add_complete_experiments(tmp_path)
    pairing = {"enabled": True, "suffixes": ["-Per-5"]}
    result, _ = behavior_env("behavior.flavors", {"bootstrap_n": 20, "condition_pairing": pairing})
    absolute = _table(result, "flavors_absolute")
    labels = dict(zip(absolute["player_type"], absolute["row_label"]))
    assert labels[LLM] == f"{LLM} | Base"
    assert "Null" not in labels                      # Null is left out entirely
    pinned = "Completed-experiment average"
    assert labels[pinned] == pinned                  # the baseline row keeps its own label
    assert list(absolute.columns[:5]) == ["row_kind", "player_type", "strategist",
                                          "condition", "row_label"]
    assert result.metadata["heatmaps"]["flavors_absolute"]["row_order"][0] == pinned


def test_rate_normalizes_counts_by_turns_alive(behavior_env, tmp_path):
    frame = pd.read_csv(tmp_path / "behavior.csv")
    frame["survival_turn"] = 49                                # 50 turns alive
    frame.to_csv(tmp_path / "behavior.csv", index=False)
    params = {"baseline": BASELINE, "traits": ["Friendliness"], "bootstrap_n": 20}
    per_100, _ = behavior_env("behavior.diplomacy", params, sid="a")
    per_game, _ = behavior_env("behavior.diplomacy", {**params, "rate": "per_game"}, sid="b")
    a = _table(per_100, "diplomacy_absolute")
    b = _table(per_game, "diplomacy_absolute")
    count = "relationship_changes"
    assert _cell(a, LLM, count) == pytest.approx(2 * _cell(b, LLM, count))
    # Levels are never rate-scaled.
    level = "persona_friendliness_avg"
    assert _cell(a, LLM, level) == _cell(b, LLM, level)


def test_baseline_defaults_to_the_strength_stage(behavior_env):
    result, _ = behavior_env("behavior.diplomacy", {"traits": ["Friendliness"], "bootstrap_n": 20},
                             adjust_baseline=BASELINE)
    assert result.metadata["baseline_experiments"] == [BASELINE]
    assert result.metadata["views"]["relative"]["tip"] == "Relative to matched in-game AI"
    assert result.summary.startswith("Against the matched in-game AI")


@pytest.mark.parametrize("module, params, kwargs, note", [
    ("behavior.diplomacy", {}, {}, "no baseline is configured"),
    ("behavior.diplomacy", {"baseline": BASELINE}, {"games": False}, "no controlled games"),
    ("behavior.flavors", {}, {}, "no experiment is complete"),
])
def test_without_a_baseline_only_the_absolute_view_is_declared(behavior_env, module, params, kwargs, note):
    result, _ = behavior_env(module, {**params, "bootstrap_n": 20}, **kwargs)
    assert list(result.metadata["views"]) == ["absolute"]
    assert result.metadata["relative_view"] == note
    assert not any(name.endswith("_relative") for name in result.table_paths)
    assert result.summary.startswith("The most distinctive value")


def test_baseline_survives_player_filters(behavior_env):
    unfiltered, _ = behavior_env("behavior.flavors", FLAVORS)
    # Keep only the LLM rows: the baseline pool is still read from the unfiltered table.
    cfg = behavior_env.make([{"id": "f", "module": "behavior.flavors", "enabled": True,
                              "params": FLAVORS, "filter": {"players": [LLM]}}])
    filtered = run_analysis(cfg, cfg.analyses[0].raw, catalog=Catalog.from_run_config(cfg))
    rel = _table(filtered, "flavors_relative")
    assert set(rel.loc[rel["row_kind"] == "group", "player_type"]) == {LLM}
    expected = _cell(_table(unfiltered, "flavors_relative"), LLM, OFFENSE)
    assert _cell(rel, LLM, OFFENSE) == expected
    # The pinned baseline row holds the unfiltered pool: 5, 6, 7 over 9 players.
    base = rel[rel["row_kind"] == "baseline"]
    assert len(base) == 1
    assert base["player_type"].iloc[0] == f"Matched '{BASELINE}'"
    assert base["mean"].iloc[0] == pytest.approx(6.0)
    assert base["n_players"].iloc[0] == 9 and base["n_games"].iloc[0] == 3


def test_tables_are_byte_stable(behavior_env):
    first, _ = behavior_env("behavior.flavors", FLAVORS, sid="one")
    second, _ = behavior_env("behavior.flavors", FLAVORS, sid="one")
    for name in ("flavors_relative", "flavors_absolute"):
        with open(first.table_paths[name], "rb") as a, open(second.table_paths[name], "rb") as b:
            assert a.read() == b.read()
    absolute = _table(first, "flavors_absolute")
    ci = absolute.loc[absolute["row_kind"] == "group", ["ci_lower", "ci_upper"]]
    assert ci.notna().all().all()


def test_flavors_write_matched_map_tables_per_seed_and_seat(behavior_env, tmp_path):
    _add_complete_experiments(tmp_path)
    result, _ = behavior_env("behavior.flavors", {"bootstrap_n": 20})
    assert result.metadata["matched_maps"] == {
        "label": "Strategic", "tip": "Strategic settings: average flavor settings",
        "seed_table": "flavors_by_seed", "seat_table": "flavors_by_seat",
    }
    pinned = "Completed-experiment average"
    seed = _table(result, "flavors_by_seed")
    seat = _table(result, "flavors_by_seat")
    # Each seed (and seat) pins its own baseline average first: seed 1 pools
    # the complete experiment's 40 and 60 on seat 0.
    first = seed[seed["seed"] == 1].iloc[0]
    assert (first["row_kind"], first["row_label"], first["mean"]) == ("baseline", pinned, 50.0)
    llm = seed[(seed["seed"] == 1) & (seed["player_type"] == LLM)].iloc[0]
    assert llm["mean"] == pytest.approx(8.0)
    assert llm["difference"] == pytest.approx(8.0 - 50.0)
    assert llm["color_position"] == pytest.approx(0.08)
    assert set(seat.columns) >= {"seed", "player_id", "difference", "color_position"}
    assert set(seat.loc[seat["row_kind"] == "baseline", "player_id"]) == {0}
    spec = result.metadata["heatmaps"]["flavors_by_seat"]
    assert spec["reference_rows"] == [pinned] and spec["baseline_rows"] == [pinned]
    assert spec["tip_rows"][0]["column"] == "difference"
    # The seed and seat tables stay downloads, off the behavior page.
    assert "flavors_by_seed" not in sum((v["tables"] for v in result.metadata["views"].values()), [])


def test_uncontrolled_flavors_offer_no_matched_map_tab(behavior_env):
    result, _ = behavior_env("behavior.flavors", FLAVORS, games=False)
    assert "matched_maps" not in result.metadata
    assert "flavors_by_seed" not in result.table_paths


def test_flavors_skip_columns_an_older_extract_lacks(behavior_env):
    spec = {**BEHAVIOR_SPEC, "flavor": ["Offense", "Science"]}
    result, _ = behavior_env("behavior.flavors", FLAVORS, behavior_spec=spec)
    assert result.metadata["flavors"] == ["Offense"]
    assert result.metadata["flavors_not_extracted"] == [f for f in C.S.FLAVOR_INFO if f != "Offense"]


def test_flavor_columns_follow_group_order(behavior_env, tmp_path):
    frame = pd.read_csv(tmp_path / "behavior.csv")
    for name in ("naval", "science", "espionage", "use_nuke"):
        frame[f"flavor_{name}_avg"] = frame["flavor_offense_avg"]
    frame.to_csv(tmp_path / "behavior.csv", index=False)
    spec = {**BEHAVIOR_SPEC, "flavor": ["Espionage", "Science", "Naval", "UseNuke", "Offense"]}
    result, _ = behavior_env("behavior.flavors", FLAVORS, behavior_spec=spec)
    assert result.metadata["flavors"] == ["Offense", "UseNuke", "Naval", "Science", "Espionage"]
    groups = _table(result, "flavors_absolute").drop_duplicates("metric")["metric_group"]
    assert list(groups) == ["Military", "Nuclear", "Naval and air", "Economy", "Other"]


def test_null_rows_are_left_out_of_both_flavor_tables(behavior_env, tmp_path):
    _add_complete_experiments(tmp_path)          # adds Null players holding 50
    result, _ = behavior_env("behavior.flavors", FLAVORS)
    for name in ("flavors_relative", "flavors_absolute"):
        table = _table(result, name)
        assert "Null" not in set(table["player_type"])
        assert "Vanilla" in set(table["player_type"])
    # Null's explanatory row tip is gone with the row.
    assert result.metadata["heatmaps"]["flavors_absolute"]["row_tips"] == {}


def test_uncontrolled_run_has_no_pinned_baseline_row(behavior_env):
    # Nothing is complete, so there is no relative view and nothing to pin.
    result, _ = behavior_env("behavior.flavors", {"bootstrap_n": 20})
    assert list(result.metadata["views"]) == ["absolute"]
    absolute = _table(result, "flavors_absolute")
    assert set(absolute["row_kind"]) == {"group"}
    assert "baseline_rows" not in result.metadata["heatmaps"]["flavors_absolute"]


def test_gated_flavor_tips_carry_the_gate_sentence(behavior_env, tmp_path):
    frame = pd.read_csv(tmp_path / "behavior.csv")
    frame["flavor_use_nuke_avg"] = frame["flavor_offense_avg"]
    frame.to_csv(tmp_path / "behavior.csv", index=False)
    spec = {**BEHAVIOR_SPEC, "flavor": ["Offense", "UseNuke"]}
    result, _ = behavior_env("behavior.flavors", FLAVORS, behavior_spec=spec)
    heat = result.metadata["heatmaps"]["flavors_absolute"]
    use_nuke = "flavor_use_nuke_avg"
    # UseNuke carries the default nuclear-tech gate; Offense does not.
    assert heat["column_tips"][use_nuke] == "\n".join([
        "UseNuke", C.S.FLAVOR_INFO["UseNuke"][2],
        flavor_gate_text(C.S.DEFAULT_FLAVOR_GATES["UseNuke"]),
    ])
    assert heat["column_tips"][OFFENSE] == "Offense\n" + C.S.FLAVOR_INFO["Offense"][2]
    assert "count only from the turn a player gains access" in heat["help"]


# ── diplomacy ──────────────────────────────────────────────────────────────────
def test_diplomacy_drops_stance_from_the_relative_view(behavior_env):
    result, _ = behavior_env("behavior.diplomacy", {"baseline": BASELINE,
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
    result, _ = behavior_env("behavior.commitment", {"baseline": BASELINE,
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
    result, _ = behavior_env("behavior.policies", {"baseline": BASELINE,
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
    ("behavior.flavors", {"flavors": ["Charm"]}, "unknown flavor"),
    ("behavior.flavors", {"flavors": ["Science"]}, "need data.extract.behavior.flavor"),
    ("behavior.flavors", {"rate": "per_game"}, "unknown key"),
    ("behavior.flavors", {"baseline": 3}, "must be \"completed\""),
    ("behavior.flavors", {"baseline": []}, "baseline"),
    ("behavior.diplomacy", {"traits": ["Charm"]}, "unknown persona trait"),
    ("behavior.diplomacy", {"traits": ["Boldness"]}, "need data.extract.behavior.persona"),
    ("behavior.diplomacy", {"rate": "per_turn"}, "must be one of"),
    ("behavior.diplomacy", {"baseline_experiment": BASELINE}, "unknown key"),
    ("behavior.policies", {"branches": ["artistry"]}, "not selected"),
    ("behavior.policies", {"rate": "per_game"}, "unknown key"),
    ("behavior.commitment", {"bootstrap_n": 0}, "integer >= 1"),
])
def test_config_rejects_bad_behavior_params(behavior_env, module, params, match):
    with pytest.raises(ConfigError, match=match):
        behavior_env.make([{"id": "x", "module": module, "enabled": True, "params": params}])


def test_baseline_accepts_a_list_of_experiments(behavior_env):
    result, _ = behavior_env("behavior.flavors", {"baseline": [BASELINE, TREATMENT],
                                                  "bootstrap_n": 20})
    assert result.metadata["baseline"] == "matched average of 2 experiments"
    # Both named experiments leave the relative view, so no player is left to compare.
    assert result.metadata["relative_view"] == (
        "no controlled player has a matched average of 2 experiments cell"
    )
