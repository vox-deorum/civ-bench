"""``world.*`` analyses on a small synthetic controlled design.

Design: a baseline experiment where every seat is the in-game AI (the strength
stage's baseline), a treatment experiment where seat 0 is an LLM, and a Null
experiment where seat 0 is the Null strategist. The baseline plays seeds 1 (two
games) and 2; the treatment plays seeds 1 and 2 plus one uncontrolled game; the
Null experiment plays seed 1. Baseline game B1b ends at turn 3, one turn early,
and the treatment's Vanilla civ in seat 1 of T2 is eliminated at turn 3.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.catalog import Catalog
from bench.config import ConfigError, load_config
from bench.reports import run_report
from bench.reports.runner import report_dir
from bench.config import schema as S

BASELINE = "vanilla-standard"
TREATMENT = "llm-standard"
NULL_EXP = "null-standard"
LLM = "Kimi-K2.5"
CHECKPOINTS = [2, 4]

# (game_id, experiment, seed, seating_rotation)
GAMES = [
    ("B1a", BASELINE, 1, 0), ("B1b", BASELINE, 1, 1), ("B2a", BASELINE, 2, 0),
    ("T1", TREATMENT, 1, 0), ("T2", TREATMENT, 2, 0),
    ("TU", TREATMENT, -1, -1),   # uncontrolled
    ("N1", NULL_EXP, 1, 0),
]
FINAL = {gid: 3 if gid == "B1b" else 4 for gid, *_ in GAMES}
# Wars declared per seat (behavior table) and per game (world table).
WARS = {"B1a": [1, 0, 2], "B1b": [3, 0, 2], "B2a": [2, 2, 2], "T1": [5, 1, 2],
        "T2": [2, 2, 2], "TU": [9, 9, 9], "N1": [0, 1, 1]}
WORLD_WARS = {"B1a": 3, "B1b": 5, "B2a": 6, "T1": 10, "T2": 6, "TU": 20, "N1": 2}
POPULATION = {"B1a": [10, 10, 10], "B1b": [20, 20, 20], "B2a": [10, 10, 10],
              "T1": [30, 15, 15], "T2": [5, 10, 10], "TU": [50, 50, 50], "N1": [0, 15, 15]}
# (victory type, winner seat)
VICTORY = {"B1a": ("Science", 0), "B1b": ("Domination", 2), "B2a": ("N/A", None),
           "T1": ("Domination", 0), "T2": ("Science", 1), "TU": ("Time", 0),
           "N1": ("Science", 1)}
STOCKS = [m for m in S.WORLD_PROGRESS_METRICS if m != "civs_alive"]


def _player_type(exp, pid):
    if pid == 0 and exp == TREATMENT:
        return LLM
    if pid == 0 and exp == NULL_EXP:
        return "Null"
    return "Vanilla"


def _keys(gid, exp, pid):
    return {"experiment": exp, "game_id": gid, "player_id": pid,
            "player_type": _player_type(exp, pid)}


def _eliminated(gid, pid, turn):
    return gid == "T2" and pid == 1 and turn >= 3


def _tables():
    panel, behavior, turns, world = [], [], [], []
    for gid, exp, _seed, _rot in GAMES:
        victory, winner = VICTORY[gid]
        world.append({
            "experiment": exp, "game_id": gid, "final_turn": FINAL[gid], "n_civs": 3,
            **{m: 0 for m in S.WORLD_CONFLICT_EVENTS},
            "wars_declared": WORLD_WARS[gid], "peak_wars": 1,
            "civs_eliminated": 1 if gid == "T2" else 0,
        })
        for pid in range(3):
            keys = _keys(gid, exp, pid)
            panel.append({**keys, "civilization": f"Civ{pid}", "survival_turn": FINAL[gid],
                          "victory_type": victory, "is_winner": int(pid == winner)})
            behavior.append({**keys, **{m: 0 for m in S.WORLD_CONFLICT_EVENTS},
                             "wars_declared": WARS[gid][pid],
                             "wars_on_city_states": int(gid == "T1" and pid == 0)})
            llm = keys["player_type"] == LLM
            for turn in range(FINAL[gid] + 1):
                turns.append({
                    **keys, "turn": turn, "score": 0 if _eliminated(gid, pid, turn) else 100,
                    **{m: 10 for m in STOCKS},
                    "population": POPULATION[gid][pid],
                    "technologies": 20 if llm else 10,
                    "active_wars": 2 if llm and turn >= 2 else 0,
                })
    return {
        "games": pd.DataFrame(GAMES, columns=["game_id", "experiment", "seed", "seating_rotation"])
                   .assign(timestamp="2026-01-01"),
        "panel": pd.DataFrame(panel),
        "behavior": pd.DataFrame(behavior),
        "turns": pd.DataFrame(turns),
        "world": pd.DataFrame(world),
    }


@pytest.fixture
def world_env(tmp_path, dev_spec, write_spec):
    paths = {}
    for name, frame in _tables().items():
        path = tmp_path / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = str(path)

    def make(module, params=None, *, games=True, in_game_ai=True, tables=None, events=None,
             stages=None, report=None):
        spec = json.loads(json.dumps(dev_spec))
        spec["output"] = {"root": str(tmp_path / "out"), "suffix": ""}
        spec["data"]["extract"]["issues_path"] = str(tmp_path / "import_issues.csv")
        if events is not None:
            spec["data"]["extract"]["behavior"] = {"events": events}
        outputs = spec["data"]["extract"].get("outputs")
        if outputs is not None and "world" not in outputs:
            outputs.append("world")
        spec["data"]["tables"] = {k: v for k, v in paths.items()
                                  if (games or k != "games") and (tables is None or k in tables)}
        spec.pop("estimators", None)
        spec.pop("adjust", None)
        if in_game_ai:
            spec["estimators"] = dev_spec["estimators"]
            spec["adjust"] = [{**dev_spec["adjust"][0],
                               "params": {"baseline_experiment": BASELINE}}]
        stage = {"id": "w", "module": module, "enabled": True,
                 "params": {"bootstrap_n": 20, **(params or {})}}
        spec["analyses"] = stages or [stage]
        if report is not None:
            spec["report"] = {"out_dir": str(tmp_path / "out") + "/", "sections": None,
                              "section_overrides": {}, "title": None,
                              "include_disabled": False, **report}
        return load_config(write_spec(spec))

    def run(module, params=None, **kwargs):
        cfg = make(module, params, **kwargs)
        return run_analysis(cfg, cfg.analyses[0].raw, catalog=Catalog.from_run_config(cfg))

    run.make = make
    return run


def _table(result, name):
    return pd.read_csv(result.table_paths[name], keep_default_na=False, na_values=[""])


def _cell(table, group, metric, column="mean"):
    row = table[(table["player_type"] == group) & (table["metric"] == metric)]
    assert len(row) == 1, (group, metric, table)
    return row[column].iloc[0]


PROGRESS = {"checkpoints": CHECKPOINTS}


# ── wars and conquest ──────────────────────────────────────────────────────────
def test_conflict_pairs_the_civ_with_its_seat_and_the_world_with_its_map(world_env):
    result = world_env("world.conflict")
    table = _table(result, "world_conflict_relative")
    # Seat 0 on seed 1: (5 - 1 on city-states) - mean(1, 3) = +2; on seed 2: 2 - 2 = 0.
    assert _cell(table, LLM, "p__wars_declared") == pytest.approx(1.0)
    assert _cell(table, LLM, "p__wars_declared", "cell_text") == "+1.0"
    # Seed 1: 10 - mean(3, 5) = +6; seed 2: 6 - 6 = 0.
    assert _cell(table, LLM, "w__wars_declared") == pytest.approx(3.0)
    # At war on 3 of 5 turns against never; the world: 3 of 15 and 3 of 13 civ-turns.
    assert _cell(table, LLM, "p__at_war_share") == pytest.approx(60.0)
    assert _cell(table, LLM, "p__at_war_share", "cell_text") == "+60 pts"
    assert _cell(table, LLM, "w__at_war_share") == pytest.approx((20.0 + 300 / 13) / 2)
    # Elimination of a Vanilla civ in T2 shows in the World column only.
    assert _cell(table, LLM, "p__eliminated") == pytest.approx(0.0)
    assert _cell(table, LLM, "w__eliminated") == pytest.approx(0.5)
    # Vanilla rows leave (the pinned row stands for them); Null stays.
    assert set(table["player_type"]) == {"AI-only world", LLM, "Null"}
    pinned = table[table["row_kind"] == "baseline"]
    assert _cell(pinned, "AI-only world", "w__wars_declared") == pytest.approx(14 / 3)
    assert _cell(pinned, "AI-only world", "p__at_war_share", "cell_text") == "0%"


def test_conflict_heatmap_pairs_player_and_world_columns(world_env):
    result = world_env("world.conflict", {"metrics": ["wars_declared", "at_war_share"]})
    spec = result.metadata["heatmaps"]["world_conflict_relative"]
    assert spec["column_order"] == ["p__wars_declared", "w__wars_declared",
                                    "p__at_war_share", "w__at_war_share"]
    assert spec["column_labels"]["w__wars_declared"] == "World"
    assert spec["text_column"] == "cell_text"
    assert spec["baseline_rows"] == ["AI-only world"]
    assert list(result.metadata["views"]) == ["relative", "absolute"]
    assert result.metadata["matched_maps"]["label"] == "World: conflict"
    assert {"world_conflict_by_seed", "world_conflict_by_seat"} <= set(result.table_paths)
    assert "the whole world" in result.summary


def test_conflict_needs_a_world_table(world_env):
    with pytest.raises(ConfigError, match="data.extract.outputs"):
        world_env("world.conflict", tables={"games", "panel", "behavior", "turns"})


def test_uncontrolled_conflict_shows_only_absolute_values(world_env):
    result = world_env("world.conflict", games=False)
    assert list(result.metadata["views"]) == ["absolute"]
    assert "matched_maps" not in result.metadata
    table = _table(result, "world_conflict_absolute")
    assert "AI-only world" not in set(table["player_type"])


# ── world growth ───────────────────────────────────────────────────────────────
def test_progress_reads_percent_changes_at_each_checkpoint(world_env):
    result = world_env("world.progress", PROGRESS)
    t2 = _table(result, "world_progress_t2_relative")
    # Seat 0: 30 vs mean(10, 20) = +100%, 5 vs 10 = -50%.
    assert _cell(t2, LLM, "p__population@2") == pytest.approx(25.0)
    assert _cell(t2, LLM, "p__population@2", "cell_text") == "+25%"
    # Sums: 60 vs mean(30, 60) = +33%, 25 vs 30 = -17%.
    assert _cell(t2, LLM, "w__population@2") == pytest.approx(25 / 3)
    assert _cell(t2, "Null", "p__population@2") == pytest.approx(-100.0)
    t4 = _table(result, "world_progress_t4_relative")
    # B1b ended at turn 3, so seed 1 compares with B1a alone; the civ eliminated
    # in T2 counts as 0 there.
    assert _cell(t4, LLM, "p__population@4") == pytest.approx(75.0)
    assert _cell(t4, LLM, "w__population@4") == pytest.approx(25.0)
    assert _cell(t4, LLM, "w__civs_alive@4") == pytest.approx(-0.5)
    # Techs average over surviving civs: 40/3 vs 10, then 15 vs 10.
    assert _cell(t4, LLM, "w__technologies@4") == pytest.approx((100 / 3 + 50) / 2)


def test_progress_endings_compare_win_shares_in_points(world_env):
    result = world_env("world.progress", PROGRESS)
    table = _table(result, "world_endings_relative")
    assert _cell(table, LLM, "p__win_domination") == pytest.approx(50.0)
    assert _cell(table, LLM, "w__win_domination") == pytest.approx(25.0)
    assert _cell(table, LLM, "w__no_winner", "cell_text") == "-50 pts"
    assert not (table["metric"] == "p__no_winner").any()
    spec = result.metadata["heatmaps"]["world_endings_relative"]
    assert spec["column_order"][:2] == ["p__win_domination", "w__win_domination"]


def test_progress_tabs_follow_the_overview_turns_endings_absolute_order(world_env):
    result = world_env("world.progress", PROGRESS)
    views = result.metadata["views"]
    assert list(views) == ["overview", "turn_2", "turn_4", "endings", "all_absolute"]
    assert [views[v]["label"] for v in views] == ["Overview", "Turn 2", "Turn 4", "Endings",
                                                  "Absolute"]
    assert views["overview"]["tables"] == ["world_progress_overview_relative"]
    assert views["all_absolute"]["tables"] == [
        "world_progress_t2_absolute", "world_progress_t4_absolute", "world_endings_absolute",
    ]
    spec = result.metadata["heatmaps"]["world_progress_overview_relative"]
    assert spec["column_order"][:4] == ["p__population@2", "w__population@2",
                                        "p__population@4", "w__population@4"]
    overview = _table(result, "world_progress_overview_relative")
    assert set(overview.loc[overview["metric"] == "p__population@4", "metric_group"]) == {
        "Pop · T4"}
    assert result.metadata["matched_maps"]["label"] == "World: progress"
    seed = result.metadata["heatmaps"]["world_progress_by_seed"]
    assert seed["column_order"][0] == "p__population@4"
    assert seed["tip_rows"][0]["units"]["p__population@4"] == "%"


def test_uncontrolled_progress_shows_absolute_values_in_every_tab(world_env):
    result = world_env("world.progress", PROGRESS, games=False)
    views = result.metadata["views"]
    assert list(views) == ["overview", "turn_2", "turn_4", "endings"]
    assert views["turn_2"]["tables"] == ["world_progress_t2_absolute"]
    assert views["overview"]["tables"] == ["world_progress_overview_absolute"]


def test_world_tables_are_byte_stable(world_env, tmp_path):
    first = world_env("world.progress", PROGRESS)
    contents = {name: open(path, "rb").read() for name, path in first.table_paths.items()}
    second = world_env("world.progress", PROGRESS)
    for name, path in second.table_paths.items():
        assert open(path, "rb").read() == contents[name], name
    assert np.isfinite(_cell(_table(second, "world_progress_t2_relative"), LLM,
                             "p__population@2"))


# ── config validation ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("module, params, kwargs, match", [
    ("world.conflict", {"metrics": ["population"]}, {}, "unknown metric"),
    ("world.conflict", {"metrics": []}, {}, "metrics"),
    ("world.conflict", {"rate": "per_game"}, {}, "unknown key"),
    ("world.conflict", {"checkpoints": [100]}, {}, "unknown key"),
    ("world.conflict", {}, {"events": ["wars_declared"]}, "wars_on_city_states"),
    ("world.progress", {"metrics": ["wars_declared"]}, {}, "unknown metric"),
    ("world.progress", {"metrics": ["cities"], "overview_metrics": ["population"]}, {},
     "not in params.metrics"),
    ("world.progress", {"checkpoints": []}, {}, "checkpoints"),
    ("world.progress", {"checkpoints": [200, 100]}, {}, "increasing"),
    ("world.progress", {"checkpoints": [0, 100]}, {}, "checkpoints"),
    ("world.progress", {"checkpoints": [True]}, {}, "checkpoints"),
    ("world.progress", {"baseline": 3}, {}, "baseline"),
])
def test_config_rejects_bad_world_params(world_env, module, params, kwargs, match):
    with pytest.raises(ConfigError, match=match):
        world_env.make(module, params, **kwargs)


def test_progress_needs_no_world_table(world_env):
    cfg = world_env.make("world.progress", PROGRESS,
                         tables={"games", "panel", "behavior", "turns"})
    assert cfg.analyses[0].raw["module"] == "world.progress"


def test_conflict_reads_only_the_event_columns_its_metrics_need(world_env):
    result = world_env("world.conflict", {"metrics": ["units_lost", "turns"]},
                       events=["units_lost"])
    assert result.metadata["metrics"] == ["units_lost", "turns"]


# ── report ─────────────────────────────────────────────────────────────────────
def test_world_page_renders_both_sections_with_their_tabs(world_env):
    stages = [
        {"id": "wld_conflict", "module": "world.conflict", "enabled": True,
         "params": {"bootstrap_n": 20}},
        {"id": "wld_progress", "module": "world.progress", "enabled": True,
         "params": {"bootstrap_n": 20, **PROGRESS}},
    ]
    cfg = world_env.make("world.conflict", stages=stages, report={
        "formats": ["md", "html"], "overview_sections": ["wld_conflict", "wld_progress"]})
    catalog = Catalog.from_run_config(cfg)
    for stage in cfg.analyses:
        run_analysis(cfg, stage.raw, catalog=catalog)
    run_report(cfg)
    out = report_dir(cfg)
    page = (out / "world.html").read_text(encoding="utf-8")
    assert "Wars and conquest" in page and "World growth" in page
    assert page.count('<div class="view-switch"') == 2
    for label in ("Overview", "Turn 2", "Turn 4", "Endings"):
        assert f">{label}</button>" in page
    assert page.index(">Overview</button>") < page.index(">Turn 2</button>") < page.index(
        ">Endings</button>")
    assert ">AI-only world<" in page
    assert ">+25%<" in page
    # The Matched Maps tables are not drawn inline on the World page.
    assert "World: progress" not in page
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## World" in md


def test_world_columns_match_the_completed_pool_by_map_not_by_seat():
    """A completed pool holds only strategist seats; World values still match by map.

    The completed games on seed 1 total 10 and 30 wars, with their strategists in
    different seats (two of them in C1). A treatment game totaling 20 sits at the
    map's average, though C1 alone shares its seat; its Player value still
    matches seat by seat.
    """
    from bench.analyses.behavior import common as C

    def frame(records):
        return pd.DataFrame(records, columns=["experiment", "game_id", "player_id", "player_type",
                                              "p__wars", "w__wars"])

    pool = frame([("done", "C1", 0, LLM, 3.0, 10.0), ("done", "C1", 2, LLM, 1.0, 10.0),
                  ("done", "C2", 1, LLM, 4.0, 30.0)])
    rows = frame([("treat", "T", 0, LLM, 5.0, 20.0)])
    cells = pd.DataFrame({"game_id": ["C1", "C2", "T"], "seed": [1, 1, 1]})
    baseline = C.Baseline(C.COMPLETED, ("done",), "completed-experiment average")

    views = C.build_views(rows, pool, ["p__wars", "w__wars"], cells, baseline,
                          per_game={"w__wars"})
    assert views.relative[["p__wars", "w__wars"]].to_dict("records") == [
        {"p__wars": 2.0, "w__wars": 0.0}]
    assert views.baseline_summary["w__wars"]["mean"] == 20.0
    assert views.baseline_summary["w__wars"]["n_games"] == 2
    assert views.baseline_summary["w__wars"]["n_players"] == 2
