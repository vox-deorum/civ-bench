"""``agentic.*`` analyses on a small synthetic ``agentic`` table.

Three player types, one player-game each:

* ``Kimi-K2.5`` (game G1, seat 0) uses the workspace in runs 1 and 2 and
  makes no call in run 3 (a ``none`` row). Run 1 looks, then acts; run 2 looks
  and acts in one step, then looks again. Every run carries its reply count in
  ``run_steps`` (3, 4, and 0), which need not match its call rows;
* ``GLM-5`` (game G2, seat 1) makes one workspace call, keeps a note, and
  replies once more without a call;
* ``Qwen3.5`` (game G1, seat 1) only acts, so every module hides it.

Every metric below is worked by hand from these rows.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.analyses.agentic.base import NO_WORKSPACE, REQUIRED_COLUMNS
from bench.analyses.agentic.foraging import REFERENCE, SECTIONS
from bench.analyses.errors import AnalysisError
from bench.catalog import Catalog
from bench.config import ConfigError, load_config
from bench.reports import run_report
from bench.reports.runner import report_dir

EXP = "llm-agentic"
USER, NOTER, ACTOR = "Kimi-K2.5", "GLM-5", "Qwen3.5"


def _ws(**kw):
    return {"kind": "workspace", "tool": "bash", "outcome": "ok", "read_style": "full",
            "out_chars": 0, "truncated_chars": 0, **kw}


def _act(tool, **kw):
    return {"kind": "action", "tool": tool, **kw}


def _rows():
    user = [
        # run 1, turn 0: look (step 1), then act (step 2); a third reply made no call
        dict(run=1, turn=0, step=1, call=1, run_steps=3, **_ws(out_chars=2000, src_records=1,
                                                               rec_events_all=1, lookback=4)),
        dict(run=1, turn=0, step=2, call=2, run_steps=3, **_act("set-research", option="Pottery",
                                                                action_ok=1, grounded=1)),
        dict(run=1, turn=0, step=2, call=3, run_steps=3, **_act("set-research", option="Mining",
                                                                action_ok=0, grounded=0)),
        # run 2, turn 10: a failed look while acting (mixed step), then a search;
        # 4 replies in all, so the last two made no call
        dict(run=2, turn=10, step=1, call=1, run_steps=4, **_ws(outcome="missing_file",
                                                                read_style="none",
                                                                src_notes=1, note_missing=1)),
        dict(run=2, turn=10, step=1, call=2, run_steps=4, **_act("set-policy", option="Liberty",
                                                                 action_ok=1, grounded=0)),
        dict(run=2, turn=10, step=2, call=3, run_steps=4, **_ws(outcome="no_match",
                                                                read_style="targeted",
                                                                out_chars=300, truncated_chars=50,
                                                                src_reference=1, ref_policies=1,
                                                                writes_rejected=2)),
        # run 3, turn 20: no calls, and it ended before any reply
        dict(run=3, turn=20, kind="none", run_steps=0),
    ]
    noter = [
        dict(run=1, turn=0, step=1, call=1, run_steps=3, **_ws(src_tmp=1, writes_persisted=1)),
        dict(run=1, turn=0, step=2, call=2, run_steps=3, **_act("keep-status-quo", action_ok=1)),
    ]
    actor = [
        dict(run=1, turn=0, step=1, call=1, run_steps=1,
             **_act("set-research", option="Pottery", action_ok=1, grounded=0)),
    ]
    rows = []
    for player_type, game, seat, calls in ((USER, "G1", 0, user), (NOTER, "G2", 1, noter),
                                          (ACTOR, "G1", 1, actor)):
        for call in calls:
            rows.append({"experiment": EXP, "game_id": game, "player_id": seat,
                         "player_type": player_type, "agent": "simple-strategist",
                         "capability": "write", **call})
    columns = list(dict.fromkeys([*REQUIRED_COLUMNS, *SECTIONS, *REFERENCE]))
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    workspace = frame["kind"] == "workspace"
    flags = [c for c in columns if c.startswith(("src_", "rec_", "ref_", "writes_"))
             or c in ("guide_edit", "note_missing")]
    frame.loc[workspace, flags] = frame.loc[workspace, flags].fillna(0)
    return frame


def _player(turns, last=None):
    """A player with one workspace call per run at ``turns``, ending at ``last``."""
    rows = [dict(run=i + 1, turn=t, step=1, call=1, run_steps=1, **_ws())
            for i, t in enumerate(turns)]
    if last is not None:
        rows.append(dict(run=len(turns) + 1, turn=last, kind="none", run_steps=0))
    frame = pd.DataFrame([{"experiment": EXP, "game_id": "G9", "player_id": 0,
                           "player_type": USER, "agent": "simple-strategist", **r} for r in rows])
    template = _rows().iloc[0:0]
    return pd.concat([template, frame], ignore_index=True)[template.columns]


@pytest.fixture
def agentic_env(tmp_path, dev_spec, write_spec):
    path = tmp_path / "agentic.csv"
    _rows().to_csv(path, index=False)

    def make(stages, *, tables=True, outputs=None, report=None):
        spec = json.loads(json.dumps(dev_spec))
        spec["output"] = {"root": str(tmp_path / "out"), "suffix": ""}
        spec["data"]["extract"]["issues_path"] = str(tmp_path / "import_issues.csv")
        if outputs is not None:
            spec["data"]["extract"]["outputs"] = outputs
        elif "agentic" not in (spec["data"]["extract"].get("outputs") or ["agentic"]):
            spec["data"]["extract"]["outputs"].append("agentic")
        spec["data"]["tables"] = {"agentic": str(path)} if tables else {}
        spec.pop("estimators", None)
        spec.pop("adjust", None)
        spec["analyses"] = stages
        if report is not None:
            spec["report"] = {"out_dir": str(tmp_path / "out") + "/", "sections": None,
                              "section_overrides": {}, "title": None,
                              "include_disabled": False, **report}
        return load_config(write_spec(spec))

    def run(module, params=None, sid="a", **kwargs):
        stage = {"id": sid, "module": module, "enabled": True,
                 "params": {"bootstrap_n": 20, **(params or {})}}
        cfg = make([stage], **kwargs)
        return run_analysis(cfg, cfg.analyses[0].raw, catalog=Catalog.from_run_config(cfg))

    run.make = make
    run.path = path
    return run


def _table(result, name):
    return pd.read_csv(result.table_paths[name])


def _cell(table, group, metric):
    row = table[(table["player_type"] == group) & (table["metric"] == metric)]
    assert len(row) == 1, (group, metric, table)
    return float(row["mean"].iloc[0])


def _table_of(calls, game="G9"):
    """A full-column frame of one USER player-game from ``calls`` row dicts."""
    frame = pd.DataFrame([{"experiment": EXP, "game_id": game, "player_id": 0,
                           "player_type": USER, "agent": "simple-strategist", **c}
                          for c in calls])
    template = _rows().iloc[0:0]
    return pd.concat([template, frame], ignore_index=True)[template.columns]


# ── hide rule ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("module", ["agentic.engagement", "agentic.foraging",
                                    "agentic.memory", "agentic.reliability"])
def test_groups_without_workspace_calls_are_hidden(agentic_env, module):
    result = agentic_env(module)
    assert result.metadata["hidden"] == [ACTOR]
    for name in result.metadata["heatmaps"]:
        groups = set(_table(result, name)["player_type"])
        assert USER in groups and ACTOR not in groups, name
    assert list(result.metadata["views"]) == ["absolute"]


def test_no_workspace_use_gives_an_empty_result(agentic_env):
    frame = _rows()
    frame[frame["player_type"] == ACTOR].to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.engagement")
    assert result.empty
    assert result.summary == NO_WORKSPACE
    assert result.metadata["hidden"] == [ACTOR]


# ── metrics ─────────────────────────────────────────────────────────────────────
def test_engagement_metrics(agentic_env):
    result = agentic_env("agentic.engagement")
    table = _table(result, "engagement_absolute")
    # Three runs (the none run counts), workspace calls in two of them.
    assert _cell(table, USER, "use_share") == pytest.approx(200 / 3)
    assert _cell(table, USER, "calls_per_run") == pytest.approx(1.0)
    # Run 1 looked before its first action; run 2 looked in the acting step.
    assert _cell(table, USER, "look_first_share") == pytest.approx(50.0)
    # Look-up-only steps: run 1 step 1 and run 2 step 2.
    assert _cell(table, USER, "gather_steps_per_run") == pytest.approx(2 / 3)
    assert _cell(table, USER, "kb_per_run") == pytest.approx(2.3 / 3)
    assert _cell(table, USER, "mixed_share") == pytest.approx(100 / 3)
    # Each run's reply count comes from its run_steps (3, 4, and 0), not from
    # its call rows: the mean of those, not a row-weighted 21 / 7.
    assert _cell(table, USER, "steps_per_run") == pytest.approx(7 / 3)
    assert _cell(table, NOTER, "steps_per_run") == pytest.approx(3.0)
    assert _cell(table, NOTER, "use_share") == pytest.approx(100.0)


def test_steps_per_run_averages_runs_not_calls(agentic_env):
    # Run 1 has two call steps but three replies (its last reply made no call);
    # run 2 is one reply; the no-call run 3 ended before any reply. Each run
    # counts once: (3 + 1 + 0) / 3, not the row-weighted (9 + 1 + 0) / 5.
    calls = [
        dict(run=1, turn=0, step=1, call=1, run_steps=3, **_ws()),
        dict(run=1, turn=0, step=2, call=2, run_steps=3, **_ws()),
        dict(run=1, turn=0, step=2, call=3, run_steps=3, **_ws()),
        dict(run=2, turn=1, step=1, call=1, run_steps=1, **_ws()),
        dict(run=3, turn=2, kind="none", run_steps=0),
    ]
    _table_of(calls).to_csv(agentic_env.path, index=False)
    table = _table(agentic_env("agentic.engagement"), "engagement_absolute")
    assert _cell(table, USER, "steps_per_run") == pytest.approx(4 / 3)


def test_steps_per_run_averages_player_games_equally(agentic_env):
    # One game had a single nine-step run; the other had three one-step runs.
    # The cell is the mean of the two player-games, as every metric is, not
    # the mean over the four runs.
    big = _table_of([dict(run=1, turn=0, step=1, call=1, run_steps=9, **_ws())], game="G1")
    small = _table_of([
        dict(run=1, turn=0, step=1, call=1, run_steps=1, **_ws()),
        dict(run=2, turn=1, step=1, call=1, run_steps=1, **_ws()),
        dict(run=3, turn=2, step=1, call=1, run_steps=1, **_ws()),
    ], game="G2")
    pd.concat([big, small], ignore_index=True).to_csv(agentic_env.path, index=False)
    table = _table(agentic_env("agentic.engagement"), "engagement_absolute")
    assert _cell(table, USER, "steps_per_run") == pytest.approx(5.0)


def test_engagement_needs_run_steps(agentic_env):
    _rows().drop(columns=["run_steps"]).to_csv(agentic_env.path, index=False)
    with pytest.raises(AnalysisError,
                       match=r"lacks column\(s\) \['run_steps'\]; re-run extract"):
        agentic_env("agentic.engagement")
    # Only engagement reads the column; the other modules still run.
    assert not agentic_env("agentic.memory", sid="mem").empty


def test_engagement_progress_bins_split_the_game(agentic_env):
    result = agentic_env("agentic.engagement", {"progress_bins": 2})
    progress = _table(result, "progress")
    user = progress[progress["player_type"] == USER].set_index("bin")
    # Turn 0 of 20 falls in the first half; turns 10 and 20 in the second.
    assert user.loc[0, "use_share"] == pytest.approx(100.0)
    assert user.loc[0, "calls_per_run"] == pytest.approx(1.0)
    assert user.loc[1, "use_share"] == pytest.approx(50.0)
    assert user.loc[1, "calls_per_run"] == pytest.approx(1.0)
    assert list(result.metadata["views"]["absolute"]["figures"]) == ["progress"]
    assert "progress" in result.figure_paths


def test_foraging_metrics(agentic_env):
    result = agentic_env("agentic.foraging")
    sources = _table(result, "sources_absolute")
    for column in ("src_records", "src_notes", "src_reference"):
        assert _cell(sources, USER, column) == pytest.approx(100 / 3)
    assert _cell(sources, USER, "src_guide") == 0.0
    assert _cell(_table(result, "sections_absolute"), USER, "rec_events_all") == 100.0
    reference = _table(result, "reference_absolute")
    assert _cell(reference, USER, "ref_policies") == 100.0
    assert _cell(reference, USER, "ref_run_share") == pytest.approx(100 / 3)
    style = _table(result, "read_style_absolute")
    for column in ("style_full", "style_targeted", "style_none"):
        assert _cell(style, USER, column) == pytest.approx(100 / 3)
    lookback = _table(result, "lookback_absolute")
    assert _cell(lookback, USER, "lookback_share") == 100.0
    assert _cell(lookback, USER, "lookback_mean") == 4.0
    # Only successful choices count: the failed, ungrounded Mining pick is left out.
    choices = _table(result, "choices_absolute")
    assert _cell(choices, USER, "grounded_research_share") == 100.0
    assert _cell(choices, USER, "grounded_policy_share") == 0.0


def test_memory_metrics_and_persistence_summary(agentic_env):
    result = agentic_env("agentic.memory")
    table = _table(result, "memory_absolute")
    assert _cell(table, USER, "writes_rejected_per_100_runs") == pytest.approx(200 / 3)
    assert _cell(table, USER, "persist_share") == 0.0
    assert _cell(table, NOTER, "persist_share") == 100.0
    assert _cell(table, NOTER, "writes_persisted_per_100_runs") == 100.0
    assert _cell(table, USER, "note_reads_per_run") == pytest.approx(1 / 3)
    assert _cell(table, USER, "note_missing_share") == 100.0
    assert "Keeps most notes" in result.summary


def test_memory_without_writes_leaves_persistence_blank(agentic_env):
    frame = _rows()
    frame = frame[frame["player_type"] != NOTER]
    frame["writes_rejected"] = frame["writes_rejected"].where(frame["kind"] != "workspace", 0)
    frame.to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.memory")
    table = _table(result, "memory_absolute")
    assert table[table["metric"] == "persist_share"].empty
    assert result.summary == "No strategist wrote to the workspace."


def test_memory_summary_names_refused_writes(agentic_env):
    frame = _rows()
    frame[frame["player_type"] != NOTER].to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.memory")
    assert result.summary == f"No strategist kept a note; **{USER}** had **2** writes refused."


def test_memory_guide_reads_count_as_note_reads(agentic_env):
    calls = [
        dict(run=1, turn=0, step=1, call=1, **_ws(src_guide=1, src_notes=0, note_missing=0)),
        dict(run=1, turn=0, step=1, call=2,
             **_ws(outcome="missing_file", src_guide=1, src_notes=0, note_missing=1)),
        # One command that reads both the notes and the guide counts as one read.
        dict(run=1, turn=0, step=1, call=3, **_ws(src_guide=1, src_notes=1, note_missing=0)),
        dict(run=1, turn=0, step=1, call=4,
             **_ws(outcome="missing_file", src_guide=0, src_notes=1, note_missing=1)),
    ]
    _table_of(calls).to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.memory")
    table = _table(result, "memory_absolute")
    # All four calls read the notes or the guide, so the run has 4 reads and
    # two of them found nothing.
    assert _cell(table, USER, "note_reads_per_run") == pytest.approx(4.0)
    assert _cell(table, USER, "note_missing_share") == pytest.approx(50.0)


def test_memory_summary_names_sparse_note_keeper(agentic_env):
    # Three games of one player type, all with workspace calls, but only one
    # keeps a note; the summary must still name that player.
    games = (("G1", _ws()), ("G2", _ws(writes_persisted=1)), ("G3", _ws()))
    calls = [dict(run=1, turn=0, step=1, call=1, **ws) for _, ws in games]
    rows = _table_of(calls)
    rows["game_id"] = [game for game, _ in games]
    rows.to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.memory")
    table = _table(result, "memory_absolute")
    assert table.loc[table["metric"] == "persist_share", "n_players"].item() == 1
    assert "Keeps most notes" in result.summary
    assert USER in result.summary


def test_reliability_metrics(agentic_env):
    result = agentic_env("agentic.reliability")
    table = _table(result, "reliability_absolute")
    # A search that found nothing is not a failure.
    assert _cell(table, USER, "error_share") == pytest.approx(100 / 3)
    assert _cell(table, USER, "no_match_share") == pytest.approx(100 / 3)
    assert _cell(table, USER, "truncated_share") == pytest.approx(100 / 3)
    # The failed call 1 of run 2 is followed by call 3.
    assert _cell(table, USER, "recovery_share") == 100.0
    # One missing-file failure in three commands; the kinds add up to the failure rate.
    kinds = _table(result, "error_kinds_absolute")
    assert _cell(kinds, USER, "error_missing_file_per_100_calls") == pytest.approx(100 / 3)
    assert _cell(kinds, USER, "error_syntax_per_100_calls") == 0.0
    user_kinds = kinds[kinds["player_type"] == USER]
    assert user_kinds["mean"].sum() == pytest.approx(_cell(table, USER, "error_share"))
    assert _cell(table, NOTER, "error_share") == 0.0
    # A player without failures has none of any kind, not a blank.
    assert _cell(kinds, NOTER, "error_missing_file_per_100_calls") == 0.0


def test_progress_bins_keep_boundary_runs_in_their_own_slice(agentic_env):
    # 4 of 7 turns at 7 bins is 3.9999999999999996 in floating point.
    _player([4], last=7).to_csv(agentic_env.path, index=False)
    progress = _table(agentic_env("agentic.engagement", {"progress_bins": 7}), "progress")
    assert progress.loc[progress["use_share"] == 100.0, "bin"].tolist() == [4]


@pytest.mark.parametrize("module, table_name, metric, expected", [
    ("agentic.engagement", "engagement_absolute", "calls_per_run", 2.0),
    ("agentic.foraging", "sources_absolute", "src_notes", 50.0),
    ("agentic.memory", "memory_absolute", "writes_rejected_per_100_runs", 200.0),
    ("agentic.reliability", "reliability_absolute", "error_share", 50.0),
])
def test_grouping_by_turn_keeps_numeric_groups(agentic_env, module, table_name, metric, expected):
    result = agentic_env(module, {"by": "turn"})
    # Only the ``none`` run at turn 20 lacks a workspace call; its hidden
    # label is a string.
    assert result.metadata["hidden"] == ["20"]
    for name in result.metadata["heatmaps"]:
        table = _table(result, name)
        assert pd.api.types.is_numeric_dtype(table["turn"]), name
        # Turn 20 is gone; a turn that measured nothing for a table is left
        # out of that table, never relabeled as a string.
        assert set(table["turn"]) <= {0, 10}, name
        assert not table.empty, name
    # Turn 10 holds only USER's run 2: two workspace calls, one a failed
    # note read, one a no-match reference read with 2 refused writes.
    table = _table(result, table_name)
    mean = table.loc[(table["turn"] == 10) & (table["metric"] == metric), "mean"].item()
    assert mean == pytest.approx(expected)
    assert set(table["turn"]) == {0, 10}


def test_progress_by_turn_uses_the_full_player_timeline(agentic_env):
    _player([0, 10], last=20).to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.engagement", {"by": "turn", "progress_bins": 4})
    assert result.metadata["hidden"] == ["20"]
    progress = _table(result, "progress")
    assert pd.api.types.is_numeric_dtype(progress["turn"])
    # Turn 20 is hidden as a group but still marks the player's last turn,
    # so turn 10 is halfway through the game.
    assert len(progress) == 2
    bins = progress.set_index("turn")["bin"]
    assert int(bins.loc[0]) == 0
    assert int(bins.loc[10]) == 2
    assert set(progress["use_share"]) == {100.0}
    assert set(progress["calls_per_run"]) == {1.0}


def test_progress_has_game_level_cis_and_a_table_tooltip(agentic_env):
    # Two games: the second half of the game is used in one and not the other.
    used, skipped = _player([0, 10]), _player([0], last=10).assign(game_id="G8")
    pd.concat([used, skipped], ignore_index=True).to_csv(agentic_env.path, index=False)
    result = agentic_env("agentic.engagement", {"progress_bins": 2})
    late = _table(result, "progress").set_index("bin").loc[1]
    assert late["use_share"] == 50.0 and late["n_games"] == 2
    for metric in ("use_share", "calls_per_run"):
        assert late[f"{metric}_ci_lower"] <= late[metric] <= late[f"{metric}_ci_upper"]
    assert late["use_share_ci_lower"] < late["use_share_ci_upper"]
    html = open(result.figure_paths["progress"], encoding="utf-8").read()
    assert "table_tooltip" in html and "95% CI" in html


def test_a_missing_column_asks_to_re_run_extract(agentic_env):
    _rows().drop(columns=["writes_lost"]).to_csv(agentic_env.path, index=False)
    with pytest.raises(AnalysisError, match=r"lacks column\(s\) \['writes_lost'\]; re-run extract"):
        agentic_env("agentic.memory")


@pytest.mark.parametrize("module", ["agentic.engagement", "agentic.foraging",
                                    "agentic.memory", "agentic.reliability"])
def test_tables_are_byte_stable(agentic_env, module):
    first = agentic_env(module, sid="one")
    contents = {name: open(path, "rb").read() for name, path in first.table_paths.items()}
    second = agentic_env(module, sid="one")
    for name, path in second.table_paths.items():
        assert open(path, "rb").read() == contents[name], name


# ── config and report ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("module, params, match", [
    ("agentic.engagement", {"progress_bins": 1}, "progress_bins"),
    ("agentic.engagement", {"progress_bins": True}, "progress_bins"),
    ("agentic.foraging", {"progress_bins": 4}, "unknown key"),
    ("agentic.memory", {"baseline": "completed"}, "unknown key"),
    ("agentic.reliability", {"rate": "per_game"}, "unknown key"),
    ("agentic.reliability", {"by": 3}, "by"),
])
def test_config_rejects_bad_agentic_params(agentic_env, module, params, match):
    with pytest.raises(ConfigError, match=match):
        agentic_env.make([{"id": "a", "module": module, "enabled": True, "params": params}])


def test_config_requires_the_agentic_table(agentic_env):
    stage = {"id": "a", "module": "agentic.memory", "enabled": True, "params": {}}
    with pytest.raises(ConfigError, match="data.tables.agentic"):
        agentic_env.make([stage], tables=False)
    with pytest.raises(ConfigError, match="data.extract.outputs"):
        agentic_env.make([stage], outputs=["turns", "panel", "games", "tokens"])


def test_agentic_page_renders(agentic_env):
    stages = [{"id": f"agt_{n}", "module": f"agentic.{n}", "enabled": True,
               "params": {"bootstrap_n": 20}}
              for n in ("engagement", "foraging", "memory", "reliability")]
    cfg = agentic_env.make(stages, report={"formats": ["md", "html"],
                                           "overview_sections": ["agt_engagement"]})
    catalog = Catalog.from_run_config(cfg)
    for stage in cfg.analyses:
        run_analysis(cfg, stage.raw, catalog=catalog)
    run_report(cfg)
    out = report_dir(cfg)
    page = (out / "agentic.html").read_text(encoding="utf-8")
    for title in ("Workspace use", "What strategists read", "Notes and memory",
                  "Command failures", "Informed choices", "Steps per run"):
        assert title in page, title
    # The hidden group is named only in each section's details note, never as a row.
    assert page.count(ACTOR) == page.count(f"hidden: {ACTOR}") == 4
    assert "relative_view" not in page
    assert "## Agentic" in (out / "report.md").read_text(encoding="utf-8")
