"""Front-page updates use saved results and dates, independent of heading copy."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bench.config import load_config
from bench.reports.announcements import build_announcements
from bench.reports.context import ReportBuildContext
from bench.reports.model import Section
from bench.reports.render import render_html_site, render_markdown
from bench.reports.runner import run_report
from bench.reports.templates import default_template


@pytest.fixture
def announcement_context(tmp_path):
    ctx = ReportBuildContext(meta={
        "title": "Benchmark", "run_name": "test", "seed": 42,
        "config_path": "fixture.json", "output_root": str(tmp_path),
        "condition_completeness_filter": True,
    })
    ctx.sections = [
        Section("strategy", "ratings.bradley_terry", metadata={"group_by": ["player_type", "strategy"]}),
        Section("rating", "ratings.bradley_terry", metadata={"group_by": ["player_type"]}),
        Section("coverage", "performance.experiment_completeness"),
    ]
    tables = {
        ("strategy", "ratings"): pd.DataFrame({"player_type": ["Recent-Per-5"], "elo": [9999]}),
        ("rating", "ratings"): pd.DataFrame({
            "player_type": ["Leader", "Recent-Per-5", "Pending", "Undated",
                            "Recent", "Leader-Per-5", "Vanilla", "Null", "Invalid"],
            "elo": [2100, 1637, 2500, 1800, 1430, 1637, 9999, 9998, float("nan")],
        }),
        ("coverage", "condition_progress"): pd.DataFrame([
            ["old", "Leader", 24, 24, "2026-09-10T00:00:00+00:00"],
            ["recent", "Recent-Per-5", 24, 24, "2026-09-17T00:00:00.125+00:00"],
            ["pending", "Pending", 20, 24, "2026-09-18T00:00:00+00:00"],
            ["undated", "Undated", 24, 24, None],
        ], columns=["experiment", "player_type", "completed_games", "required_games", "completed_at"]),
    }
    for (sid, name), frame in tables.items():
        path = tmp_path / f"{sid}-{name}.csv"
        frame.to_csv(path, index=False)
        ctx.record_table(sid, name, tmp_path, path.name)
    ctx.meta["rating_labels"] = {"rating": {
        identity: {
            "model": identity.removesuffix("-Per-5"),
            "condition": "-Per-5" if identity.endswith("-Per-5") else "",
            "label": "Per-5" if identity.endswith("-Per-5") else "Every turn",
            "baseline": identity in {"Vanilla", "Null"},
        }
        for identity in tables[("rating", "ratings")]["player_type"]
    }}
    return ctx


def _replace_progress(ctx, frame):
    source = ctx._table_sources[("coverage", "condition_progress")]
    frame.to_csv(source.analysis_dir / source.rel_file, index=False)


@pytest.mark.parametrize("latest_identity", ["Recent", "Recent-Per-5"])
def test_announcement_selects_newest_complete_condition_not_highest_rating(
    announcement_context, latest_identity,
):
    progress = announcement_context.load_table("coverage", "condition_progress")
    progress.loc[progress["experiment"] == "recent", "player_type"] = latest_identity
    _replace_progress(announcement_context, progress)
    result, pending = build_announcements(announcement_context)
    assert result.kind == "result"
    assert "**Recent**" in result.text
    assert "**1637 Elo** (Per-5, #1)" in result.text
    assert "**1430 Elo** (Every turn, #4)" in result.text
    assert "2100" not in result.text
    assert "2500" not in result.text
    assert "9999" not in result.text
    assert result.date == "2026-09-17"
    assert pending.kind == "testing"
    assert "**Pending** (20/24)" in pending.text


@pytest.mark.parametrize("filter_enabled,all_complete", [(False, False), (True, True)])
def test_testing_announcement_visibility(announcement_context, filter_enabled, all_complete):
    ctx = announcement_context
    ctx.meta["condition_completeness_filter"] = filter_enabled
    if all_complete:
        frame = ctx.load_table("coverage", "condition_progress")
        frame["completed_games"] = frame["required_games"]
        _replace_progress(ctx, frame)
    announcements = build_announcements(ctx)
    assert [item.kind for item in announcements] == ["result"]


def test_pending_conditions_need_no_rating_or_completion_date(announcement_context):
    ctx = announcement_context
    ctx.sections = [ctx.section("coverage")]
    pending, = build_announcements(ctx)
    assert pending.kind == "testing"
    assert "**Pending** (20/24)" in pending.text


def test_legacy_coverage_artifacts_omit_announcements(announcement_context):
    ctx = announcement_context
    ctx._table_sources.pop(("coverage", "condition_progress"))
    assert build_announcements(ctx) == []


def test_announcements_use_full_saved_table(announcement_context):
    ctx = announcement_context
    frame = ctx.load_table("coverage", "condition_progress")
    older = frame.iloc[[0]].copy()
    filler = pd.concat([older.assign(experiment=f"old-{i}") for i in range(110)])
    _replace_progress(ctx, pd.concat([filler, frame], ignore_index=True))
    result = build_announcements(ctx)[0]
    assert "**Recent**" in result.text
    assert result.date == "2026-09-17"


def test_render_updates_only_on_front_page_and_escapes_labels(announcement_context):
    doc = default_template(announcement_context)
    doc.announcements[0].title = "A revised heading"
    doc.announcements[0].text += " <script>unsafe</script>"
    pages = render_html_site(doc)
    front = pages["index.html"]
    assert 'data-announcement="result"' in front
    assert 'data-announcement="testing"' in front
    assert '<time datetime="2026-09-17">' in front
    assert "<strong>Recent</strong> scored <strong>1430 Elo</strong> (Every turn, #4)" in front
    assert "<strong>1637 Elo</strong> (Per-5, #1)" in front
    assert "<strong>Pending</strong> (20/24)" in front
    assert "A revised heading" in front
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in front
    assert "<script>unsafe</script>" not in front
    assert front.index('data-announcement="result"') < front.index('id="overview-heading"')
    assert 'data-announcement=' not in pages["ratings.html"]
    assert 'href="index.html"' in pages["ratings.html"]
    assert "**Recent** scored **1430 Elo** (Every turn, #4)" in render_markdown(doc)
    assert render_html_site(doc) == pages


def test_report_loads_announcements_from_artifacts_with_filter_preset(
    tmp_path, dev_spec, write_spec, announcement_context,
):
    spec = dev_spec
    root = str(tmp_path / "output")
    spec["output"] = {"root": root, "suffix": ""}
    spec["data"]["extract"]["enabled"] = False
    spec["filters"]["complete"] = {"min_condition_completeness": 1}
    spec["data"]["filter"] = "complete"
    spec["presentation"] = {"condition_pairing": {
        "enabled": True, "base_label": "Every turn", "suffixes": ["-Per-5"],
    }}
    spec["analyses"] = [
        stage for stage in spec["analyses"]
        if stage["id"] in {"bt_main", "perf_experiment_completeness"}
    ]
    spec["report"] = {"out_dir": root, "formats": ["html"], "sections": None}
    cfg = load_config(write_spec(spec))
    for sid, source, module, table in [
        ("bt_main", "rating", "ratings.bradley_terry", "ratings"),
        ("perf_experiment_completeness", "coverage", "performance.experiment_completeness", "condition_progress"),
    ]:
        artifact_dir = Path(root) / "analyses" / sid
        artifact_dir.mkdir(parents=True)
        announcement_context.load_table(source, table).to_csv(artifact_dir / f"{table}.csv", index=False)
        manifest = {
            "id": sid, "module": module, "metadata": {},
            "tables": [{"name": table, "file": f"{table}.csv"}],
        }
        (artifact_dir / "result.json").write_text(json.dumps(manifest), encoding="utf-8")
    previous = Path(root) / cfg.name
    previous.mkdir()
    (previous / "report.html").write_text("Previous overview", encoding="utf-8")
    result = run_report(cfg)
    front = Path(result.report_dir, "index.html").read_text(encoding="utf-8")
    assert "<strong>Recent</strong> scored <strong>1430 Elo</strong> (Every turn, #4)" in front
    assert "<strong>1637 Elo</strong> (Per-5, #1)" in front
    assert "<strong>Pending</strong> (20/24)" in front
    assert 'datetime="2026-09-17"' in front
    assert not Path(result.report_dir, "report.html").exists()
    run_report(cfg)
    assert Path(result.report_dir, "index.html").read_text(encoding="utf-8") == front
