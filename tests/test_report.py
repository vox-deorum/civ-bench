"""Stage 5 report tests.

Exercise the report stage on fabricated analysis manifests (no machine data roots,
per AGENTS.md): section resolution (null = enabled analyses in canonical family
order; explicit list = priority order, then remaining enabled analyses), the manifest → document → md/html render,
asset copying into a self-contained tree, empty-section handling, determinism
(byte-stable re-render), and the loud error when a manifest is missing.

The report reads each analysis's ``result.json`` from disk, so we fabricate those
directly rather than running the (heavy) analysis modules; that keeps the suite
fast and hermetic while testing exactly the report-stage contract.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import pandas as pd
import pytest
import plotly.graph_objects as go

from bench.analyses.base import AnalysisResult
from bench.analyses.runner import run_analysis
from bench.config import load_config
from bench.reports import (
    ReportError,
    render_html,
    render_html_site,
    render_markdown,
    run_report,
)
from bench.reports.model import FamilyGroup
from bench.reports.runner import _analyses_dir, _resolve_section_ids, report_dir
from bench.reports.templates import _summarize_family

_FAKE_PNG = b"\x89PNG\r\n\x1a\n-- not a real image, copied verbatim --"
_FAKE_HTML = "<!doctype html><title>Interactive figure</title>"


class _PageText(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.text = []
        self.links = []
        self.feed(source)

    def handle_data(self, data):
        self.text.append(data)

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.append(dict(attrs))


def _emit(cfg, sid, module, *, summary="", metadata=None, tables=None, figures=None,
          artifacts=None, empty=False, module_name="", module_description=""):
    """Fabricate one analysis's persisted artifacts + ``result.json`` manifest."""
    d = _analyses_dir(cfg, sid)
    d.mkdir(parents=True, exist_ok=True)
    tnames, fnames, anames = [], [], []
    if not empty:
        for name, frame in (tables or {}).items():
            frame.to_csv(d / f"{name}.csv", index=False)
            tnames.append({"name": name, "file": f"{name}.csv"})
        for figure in figures or []:
            if isinstance(figure, str):
                name, format_ = figure, "png"
            else:
                name = figure["name"]
                format_ = figure.get("format", "png")
            suffix = ".html" if format_ == "plotly" else ".png"
            path = d / f"{name}{suffix}"
            if format_ == "plotly":
                path.write_text(_FAKE_HTML, encoding="utf-8")
            else:
                path.write_bytes(_FAKE_PNG)
            fnames.append({"name": name, "file": path.name})
        for rel, content in (artifacts or {}).items():
            p = d / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            anames.append({"name": rel, "file": rel})
    manifest = {
        "id": sid, "module": module, "summary": summary,
        "module_name": module_name, "module_description": module_description,
        "metadata": metadata or {}, "empty": empty,
        "tables": tnames, "figures": fnames, "artifacts": anames,
    }
    (d / "result.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def report_env(tmp_path, write_spec, dev_spec):
    """A loaded RunConfig with four enabled analyses across three families + emitted
    manifests; returns the cfg. The report output root is the tmp dir."""
    root = str(tmp_path / "out")
    spec = dev_spec
    spec["output"] = {"root": root, "suffix": ""}
    spec["data"]["extract"]["enabled"] = False
    spec["adjust"] = []
    spec["analyses"] = [
        {"id": "pred_metrics", "module": "prediction.evaluate", "enabled": True,
         "params": {"metrics": ["roc_auc"]}},
        {"id": "pred_compare", "module": "prediction.compare", "enabled": True, "params": {}},
        {"id": "cal_reliability", "module": "calibration.reliability", "enabled": True,
         "params": {"n_bins": 5}},
        {"id": "perf_usage_efficiency", "module": "performance.usage_efficiency",
         "enabled": True, "uses": {"tables": ["tokens"]},
         "params": {}},
    ]
    # out_dir authored as the tmp root so it re-roots there; <name> is appended.
    spec["report"] = {"out_dir": root + "/", "formats": ["md", "html"],
                      "sections": None,
                      "overview_sections": [
                          "pred_metrics", "cal_reliability", "perf_usage_efficiency"
                      ],
                      "section_overrides": {},
                      "title": None, "include_disabled": False}

    cfg = load_config(write_spec(spec))

    _emit(cfg, "pred_metrics", "prediction.evaluate",
          summary="Evaluated 3 estimator(s); best **roc_auc** = 0.87 (`attention`).",
          metadata={"metrics": ["roc_auc"], "n_models": 3},
          tables={"metrics": pd.DataFrame({"model": ["attention", "score"],
                                           "roc_auc": [0.87, 0.82]})},
          figures=["metrics"])
    _emit(cfg, "pred_compare", "prediction.compare", summary="(none)", empty=True)
    _emit(cfg, "cal_reliability", "calibration.reliability",
          summary="ECE = 0.031 over 5 bins.",
          tables={"reliability": pd.DataFrame({"bin": [0, 1], "freq": [0.1, 0.9]}),
                  "ece": pd.DataFrame({"ece": [0.031]})},
          figures=["reliability"])
    _emit(cfg, "perf_usage_efficiency", "performance.usage_efficiency",
          summary="Total spend $12.34 across 192 games.",
          tables={"token_costs": pd.DataFrame({"model": ["a"], "total_cost": [12.34]})},
          artifacts={"seating/ctrl.seating.json": '{"totalSeats": 2, "cells": {}}'})
    return cfg


# ── end-to-end render ──────────────────────────────────────────────────────────
@pytest.fixture
def replay_env(tmp_path, write_spec, dev_spec):
    spec = dev_spec
    root = str(tmp_path / "out")
    runs = tmp_path / "source"
    spec["output"] = {"root": root, "suffix": ""}
    spec["data"]["extract"].update(enabled=False, runs_dir=str(runs))
    spec["estimators"] = []
    spec["adjust"] = []
    spec["analyses"] = [{"id": "game_log", "module": "performance.game_log", "params": {}}]
    spec["report"] = {"out_dir": root, "formats": ["html", "md"], "replay": {"enabled": True}}
    cfg = load_config(write_spec(spec))
    games = pd.DataFrame([
        {"game_id": gid, "timestamp": timestamp, "date_utc": "2026-09-18",
         "experiment": "private-experiment-name", "label": label,
         "seed": seed, "seating_rotation": 1, "controlled": seed != -1,
         "turns": 312, "victory_type": "Science", "winner_player_id": winner,
         "winner_civilization": "Rome" if winner == 0 else "Persia",
         "winner_is_vanilla": winner == 0}
        for gid, timestamp, seed, winner, label in [
            ("game-new", 3000, 2, 5, "Strategist | Every-turn"),
            ("game-missing", 2000, 2, 0, "Strategist | Every-turn"),
            ("game-free", 1000, -1, 0, "VPAI"),
        ]
    ])
    players = pd.DataFrame([
        {"game_id": game.game_id, "player_id": pid,
         "strategist": "Vanilla" if vanilla else "Strategist", "condition": "Every-turn",
         "player_type": "Vanilla" if vanilla else "Strategist", "is_vanilla": vanilla,
         "civilization": "Rome" if pid == 0 else "Persia", "is_winner": pid == game.winner_player_id}
        for game in games.itertuples() for pid in [0, 5]
        for vanilla in [pid == 0 or game.seed == -1]
    ])
    _emit(cfg, "game_log", "performance.game_log", tables={"games": games, "game_players": players},
          metadata={"latest_game_id": "game-new", "vanilla_label": "Vanilla"})
    experiment = runs / "private-experiment-name"
    experiment.mkdir(parents=True)
    for filename, content in [("game-new_111.Civ5Save", b"earlier"),
                              ("game-new_222.Civ5Save", b"latest save"),
                              ("game-free_111.Civ5Save", b"free save")]:
        (experiment / filename).write_bytes(content)
    return cfg


def test_replay_report_copies_saves_and_renders_game_links(replay_env):
    result = run_report(replay_env)
    out = report_dir(replay_env)
    assert (out / "saves/private-experiment-name/game-new.Civ5Save").read_bytes() == b"latest save"
    assert (out / "saves/private-experiment-name/game-free.Civ5Save").is_file()
    assert any("1 of 3 saves not found" in warning and "game-missing" in warning for warning in result.warnings)
    games = (out / "games.html").read_text(encoding="utf-8")
    page = _PageText(games)
    assert "private-experiment-name" not in "".join(page.text)
    links = [link for link in page.links if link.get("class") == "replay-link"]
    assert parse_qs(links[0]["data-query"]) == {"player5": ["Strategist | Every-turn"], "winner": ["5"]}
    assert parse_qs(links[1]["data-query"]) == {"winner": ["0"]}
    assert "no replay" in games
    assert "Player 5 (Persia, Won)" in games
    assert "Winner: Player 0 (Rome, VPAI)" in games
    assert "Player 0 (Rome)" not in games
    assert 'data-seed="2"' in games and 'data-seed="-"' in games
    assert 'data-timestamp="3000"' in games and 'data-player="' in games
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Latest game" in index and "Player 5 (Persia, Won)" in index
    assert "Player 0" not in index
    assert "games.html" not in index.split("</aside>")[0]
    assert 'src="assets/report-common.js"' in index
    markdown = (out / "report.md").read_text(encoding="utf-8")
    assert "## Game Log" in markdown and "game_players (CSV)" in markdown
    assert "vox-deorum-replay/?file=" not in markdown


def test_replay_report_is_byte_stable_with_cached_saves(replay_env):
    run_report(replay_env)
    out = report_dir(replay_env)
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    run_report(replay_env)
    after = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    assert before == after


def test_replay_scope_and_absolute_base_url(replay_env):
    replay_env.report["replay"].update(saves="controlled", base_url="https://example.org/report dir/")
    run_report(replay_env)
    out = report_dir(replay_env)
    assert not (out / "saves/private-experiment-name/game-free.Civ5Save").exists()
    page = _PageText((out / "games.html").read_text(encoding="utf-8"))
    link = next(link for link in page.links if link.get("class") == "replay-link")
    query = parse_qs(urlsplit(link["href"]).query)
    assert query["file"] == ["https://example.org/report dir/saves/private-experiment-name/game-new.Civ5Save"]
    assert query["player5"] == ["Strategist | Every-turn"]
    assert link["target"] == "_blank" and link["data-direct"] == "true"
    assert "[Watch replay](https://vox-deorum.github.io/vox-deorum-replay/?file=" in (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("runs_dir", [None, "nonexistent-runs"])
def test_replay_missing_runs_dir_warns(replay_env, runs_dir):
    replay_env.data["extract"]["runs_dir"] = runs_dir
    result = run_report(replay_env)
    assert any("runs_dir" in warning and "missing" in warning for warning in result.warnings)
    assert "no replay" in (report_dir(replay_env) / "games.html").read_text(encoding="utf-8")


def test_replay_without_latest_card_and_disabled(replay_env):
    replay_env.report["replay"]["latest_game"] = False
    run_report(replay_env)
    out = report_dir(replay_env)
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Latest game" not in index and "Browse recent games" in index
    replay_env.report["replay"]["enabled"] = False
    run_report(replay_env)
    assert not (out / "saves").exists()
    assert "no replay" in (out / "games.html").read_text(encoding="utf-8")


def test_replay_copy_rejects_unsafe_paths(replay_env):
    artifact = _analyses_dir(replay_env, "game_log") / "games.csv"
    games = pd.read_csv(artifact)
    games.loc[0, "experiment"] = "../outside"
    games.loc[2, "game_id"] = "../outside"
    games.to_csv(artifact, index=False)
    result = run_report(replay_env)
    assert sum("unsafe save path" in warning for warning in result.warnings) == 2
    assert not (report_dir(replay_env) / "saves").exists()


def test_replay_cached_hardlinks_allow_in_place_publish(replay_env, monkeypatch):
    run_report(replay_env)
    out = report_dir(replay_env)
    original = Path.replace

    def deny_report_rename(path, target):
        if path == out:
            raise PermissionError("directory is open")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", deny_report_rename)
    result = run_report(replay_env)
    assert any("updated its files in place" in warning for warning in result.warnings)
    assert not any("could not update report file" in warning for warning in result.warnings)
    assert (out / "saves/private-experiment-name/game-new.Civ5Save").read_bytes() == b"latest save"


def test_run_report_writes_md_and_html(report_env):
    result = run_report(report_env)
    out = report_dir(report_env)
    assert result.n_sections == 4
    expected = {
        "report.md", "index.html", "prediction.html", "calibration.html",
        "performance.html", "assets/report.css", "assets/report-help.js",
    }
    assert expected == {
        str(path.relative_to(out)).replace("\\", "/")
        for path in map(type(out), result.written)
    }
    assert all((out / rel).exists() for rel in expected)
    assert set(result.formats) == {"md", "html"}

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# civbench-dev")
    assert "regenerated every result" not in md
    assert "Run **civbench-dev**" not in md
    # Family chapters, canonical order: performance → prediction → calibration.
    assert md.index("## Performance") < md.index("## Prediction") < md.index("## Calibration")
    prediction_summary = _summarize_family(
        FamilyGroup(key="prediction", title="Prediction")
    )
    assert prediction_summary in md
    # Section content surfaced from the manifest.
    assert "### pred_metrics" in md and "best **roc_auc**" in md
    assert "metrics" in md and "0.87" in md  # inline table value
    # Empty section is labelled, not silently dropped.
    assert "### pred_compare" in md and "produced no artifacts" in md

    overview = (out / "index.html").read_text(encoding="utf-8")
    assert "regenerated every result" not in overview
    assert "<p></p>" not in overview
    assert overview.count('class="overview-card"') == 3
    assert "pred_compare" in overview  # present in shared navigation
    assert "<table" not in overview and "<img" not in overview

    prediction = (out / "prediction.html").read_text(encoding="utf-8")
    assert "pred_metrics" in prediction and "pred_compare" in prediction
    assert prediction_summary in prediction
    assert '<h2 id="section-cal-reliability">' not in prediction
    assert 'href="assets/report.css"' in prediction
    assert 'aria-current="page">Prediction</a>' in prediction


@pytest.mark.parametrize("explicit_null", [False, True])
def test_default_footer_and_citation(report_env, explicit_null):
    from bench.reports.content import CITATION_BIBTEX

    if explicit_null:
        report_env.report["footer"] = None
        report_env.report["benchmark_citation"] = None
    run_report(report_env)
    out = report_dir(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert f"```bibtex\n{CITATION_BIBTEX}\n```" in md
    assert "@misc{civbench_results," not in md
    assert md.index("## Citation") < md.index("Generated by")
    for page in out.glob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert html.count('<footer class="report-footer">') == 1
        assert 'Generated by <a href="https://github.com/vox-deorum/civ-bench">CivBench</a>' in html
        assert '<a href="https://arxiv.org/abs/2604.07733">Chen, 2026</a>' in html
        assert ("@article{chen2026civbench," in html) == (page.name == "index.html")
        assert "@misc{civbench_results," not in html


def test_benchmark_results_citation(report_env):
    report_env.report["benchmark_citation"] = {
        "title": "Controlled CivBench results",
        "url": "https://example.com/results/vp-5.2.7",
    }
    report_env.report["footer"] = ""
    run_report(report_env)
    out = report_dir(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "### Paper" in md and "### Benchmark results" in md
    assert "@article{chen2026civbench," in md
    benchmark = (
        "@misc{civbench_results,\n"
        "  title={Controlled CivBench results},\n"
        "  url={https://example.com/results/vp-5.2.7}\n"
        "}"
    )
    assert f"```bibtex\n{benchmark}\n```" in md
    assert md.index("@article{") < md.index("@misc{")
    for page in out.glob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert (benchmark in html) == (page.name == "index.html")
        if page.name == "index.html":
            assert html.count('<code class="language-bibtex">') == 2
            assert "@article{chen2026civbench," in html
    before = {page.name: page.read_bytes() for page in out.glob("report.*")}
    run_report(report_env)
    assert before == {page.name: page.read_bytes() for page in out.glob("report.*")}


def test_benchmark_citation_escapes_configured_text(report_env):
    report_env.report["benchmark_citation"] = {
        "title": "Results {preview}\n```\n<script>alert(1)</script>",
        "url": "https://example.com/results?a=1&b=2",
    }
    run_report(report_env)
    out = report_dir(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert r"title={Results \{preview\} ``` <script>alert(1)</script>}" in md
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "https://example.com/results?a=1&amp;b=2" in html
    assert html.count('<code class="language-bibtex">') == 2


def test_custom_markdown_footer(report_env):
    footer = "Copyright **Example**\n\n- [Site](https://example.com)\n- *Contact* `team`\n\n<script>alert(1)</script>"
    report_env.report["footer"] = footer
    run_report(report_env)
    out = report_dir(report_env)
    assert (out / "report.md").read_text(encoding="utf-8").endswith(footer + "\n")
    for page in out.glob("*.html"):
        html = page.read_text(encoding="utf-8")
        assert "Copyright <strong>Example</strong>" in html
        assert '<li><a href="https://example.com">Site</a></li>' in html
        assert "<em>Contact</em> <code>team</code>" in html
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "Generated by" not in html


@pytest.mark.parametrize("footer", ["", " \n "])
def test_empty_footer_keeps_citation(report_env, footer):
    report_env.report["footer"] = footer
    run_report(report_env)
    out = report_dir(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## Citation" in md and "Generated by" not in md
    for page in out.glob("*.html"):
        assert "<footer" not in page.read_text(encoding="utf-8")
    assert "@article{chen2026civbench," in (out / "index.html").read_text(encoding="utf-8")


def test_assets_copied_self_contained(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    assert (out / "assets" / "pred_metrics" / "metrics.png").exists()
    assert (out / "assets" / "pred_metrics" / "metrics.csv").exists()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "assets/pred_metrics/metrics.png" in md
    assert "Figure: pred_metrics" in md  # compact default keeps the PNG as a download


def test_plotly_figure_persistence_is_self_contained_and_deterministic(report_env, monkeypatch):
    class PlotlyAnalysis:
        default_all_estimators = False

        def __init__(self, stage_id, params):
            self.stage_id = stage_id

        def run(self, ctx):
            return AnalysisResult(
                figures={"cost_vs_rating": go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))}
            )

        def report_identity(self):
            return "Plotly test", "A test figure."

    monkeypatch.setattr("bench.analyses.runner.get_analysis", lambda module: PlotlyAnalysis)
    stage = {"id": "plotly_test", "module": "test.plotly", "params": {}}
    first = run_analysis(report_env, stage, catalog=object())
    path = Path(first.figure_paths["cost_vs_rating"])
    content = path.read_bytes()
    manifest = json.loads((path.parent / "result.json").read_text(encoding="utf-8"))

    assert path.name == "cost_vs_rating.html"
    assert manifest["figures"] == [{
        "name": "cost_vs_rating", "file": "cost_vs_rating.html",
    }]
    assert b'id="plotly-cost_vs_rating"' in content
    assert b'src="https://cdn.plot.ly' not in content

    second = run_analysis(report_env, stage, catalog=object())
    assert Path(second.figure_paths["cost_vs_rating"]).read_bytes() == content


def test_plotly_figures_embed_in_html_and_link_in_markdown(report_env):
    _emit(
        report_env,
        "pred_metrics",
        "prediction.evaluate",
        summary="Interactive cost and skill figure.",
        figures=[{"name": "cost_vs_rating", "format": "plotly"}],
    )
    report_env.report["section_overrides"] = {
        "pred_metrics": {"figures": ["cost_vs_rating"]}
    }
    run_report(report_env)
    out = report_dir(report_env)
    html = (out / "prediction.html").read_text(encoding="utf-8")
    md = (out / "report.md").read_text(encoding="utf-8")

    asset = out / "assets" / "pred_metrics" / "cost_vs_rating.html"
    assert asset.read_text(encoding="utf-8") == _FAKE_HTML
    assert (
        '<iframe class="interactive-figure" '
        'src="assets/pred_metrics/cost_vs_rating.html" '
        'title="pred_metrics: cost_vs_rating" loading="lazy"></iframe>'
    ) in html
    assert 'href="assets/pred_metrics/cost_vs_rating.html">Open interactive figure (HTML)</a>' in html
    assert (
        "[Figure: pred_metrics: cost_vs_rating (interactive HTML)]"
        "(assets/pred_metrics/cost_vs_rating.html)"
    ) in md
    assert "![](" not in md
    overview = (out / "index.html").read_text(encoding="utf-8")
    assert "interactive-figure" not in overview


def test_usage_efficiency_default_keeps_only_interactive_figure_inline(report_env):
    _emit(
        report_env,
        "perf_usage_efficiency",
        "performance.usage_efficiency",
        summary="Usage and skill figures.",
        figures=[
            "cost",
            "input_tokens",
            "output_tokens",
            {"name": "usage_vs_rating", "format": "plotly"},
        ],
    )
    run_report(report_env)
    out = report_dir(report_env)
    html = (out / "performance.html").read_text(encoding="utf-8")

    assert html.count('class="interactive-figure"') == 1
    assert 'src="assets/perf_usage_efficiency/usage_vs_rating.html"' in html
    for name in ("cost", "input_tokens", "output_tokens"):
        asset = out / "assets" / "perf_usage_efficiency" / f"{name}.png"
        assert asset.exists()
        assert f'href="assets/perf_usage_efficiency/{name}.png"' in html
        assert f'<img src="assets/perf_usage_efficiency/{name}.png"' not in html


def test_report_tables_present_vpai_label_but_keep_csv_identity(report_env):
    _emit(
        report_env,
        "pred_metrics",
        "prediction.evaluate",
        summary="Table result.",
        tables={"metrics": pd.DataFrame({"Vanilla": [0.4], "score": [1]})},
    )
    run_report(report_env)
    out = report_dir(report_env)
    markdown = (out / "report.md").read_text(encoding="utf-8")
    html = (out / "prediction.html").read_text(encoding="utf-8")
    csv = (out / "assets" / "pred_metrics" / "metrics.csv").read_text(encoding="utf-8")
    assert "VPAI" in markdown and "Vanilla" not in markdown
    assert "<th>VPAI</th>" in html and "<th>Vanilla</th>" not in html
    assert csv.startswith("Vanilla,score")


def test_artifacts_copied_and_linked(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    # Subdir tree mirrored under assets/<id>/.
    asset = out / "assets" / "perf_usage_efficiency" / "seating" / "ctrl.seating.json"
    assert asset.exists()
    assert asset.read_text(encoding="utf-8") == '{"totalSeats": 2, "cells": {}}'
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "**Downloads and supporting files**" in md
    assert "[ctrl.seating.json](assets/perf_usage_efficiency/seating/ctrl.seating.json)" in md
    html = (out / "performance.html").read_text(encoding="utf-8")
    assert 'href="assets/perf_usage_efficiency/seating/ctrl.seating.json"' in html


def test_html_family_pages_use_compact_module_defaults(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    prediction = (out / "prediction.html").read_text(encoding="utf-8")
    calibration = (out / "calibration.html").read_text(encoding="utf-8")

    assert "<table" in prediction and "0.87" in prediction
    assert '<img src="assets/pred_metrics/metrics.png"' not in prediction
    assert 'href="assets/pred_metrics/metrics.png"' in prediction
    assert "<strong>roc_auc</strong>" in prediction

    assert '<img src="assets/cal_reliability/reliability.png"' in calibration
    assert "<table" in calibration and "0.031" in calibration
    assert 'href="assets/cal_reliability/reliability.csv"' in calibration


def test_rerender_is_byte_stable(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    first = {
        str(path.relative_to(out)): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    run_report(report_env)  # re-render from the same artifacts
    second = {
        str(path.relative_to(out)): path.read_bytes()
        for path in out.rglob("*")
        if path.is_file()
    }
    assert second == first


# ── publishing a previous report directory ─────────────────────────────────────
def _deny_renames(monkeypatch):
    """Simulate Windows denying every directory rename (WinError 5)."""
    def denied(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(Path, "replace", denied)


def test_locked_report_dir_falls_back_to_in_place_update(report_env, monkeypatch):
    # A report directory held open by another program cannot be renamed away,
    # so publishing must overwrite its files in place instead of failing.
    run_report(report_env)
    out = report_dir(report_env)
    fresh_md = (out / "report.md").read_bytes()
    (out / "report.md").write_bytes(b"stale bytes")
    (out / "stale.html").write_text("old", encoding="utf-8")

    _deny_renames(monkeypatch)
    result = run_report(report_env)

    assert any("updated its files in place" in w for w in result.warnings)
    assert (out / "report.md").read_bytes() == fresh_md
    assert not (out / "stale.html").exists()  # obsolete file removed
    assert not any(p.name.startswith(f".{out.name}.tmp") for p in out.parent.iterdir())


def test_locked_obsolete_file_is_a_warning_not_a_failure(report_env, monkeypatch):
    run_report(report_env)
    out = report_dir(report_env)
    stale = out / "stale.html"
    stale.write_text("old", encoding="utf-8")

    real_unlink = Path.unlink

    def denied_unlink(self, missing_ok=False):
        if self.name == "stale.html":
            raise PermissionError(5, "Access is denied")
        return real_unlink(self, missing_ok=missing_ok)

    _deny_renames(monkeypatch)
    monkeypatch.setattr(Path, "unlink", denied_unlink)
    result = run_report(report_env)

    assert any(
        "could not remove obsolete report file" in w and "stale.html" in w
        for w in result.warnings
    )
    assert stale.exists()  # kept because it is locked, but the run succeeded
    assert (out / "report.md").exists()


# ── section ordering ────────────────────────────────────────────────────────────
def test_partial_sections_prioritize_and_include_the_rest(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    assert (out / "calibration.html").exists()

    report_env.report["sections"] = ["perf_usage_efficiency", "pred_metrics"]
    report_env.report["overview_sections"] = ["perf_usage_efficiency", "pred_metrics"]
    result = run_report(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert result.n_sections == 4
    assert "cal_reliability" in md
    # Authored order respected across families: performance before prediction.
    assert md.index("## Performance") < md.index("## Prediction")
    assert (out / "calibration.html").exists()
    assert md.index("## Prediction") < md.index("## Calibration")


@pytest.mark.parametrize("sections", [None, [], ["perf_usage_efficiency", "pred_compare"]])
def test_section_remainder_uses_default_order(report_env, sections):
    report_env.report["sections"] = sections
    expected = ["perf_usage_efficiency", "pred_metrics", "pred_compare", "cal_reliability"]
    if sections:
        expected = sections + [sid for sid in expected if sid not in sections]
    assert _resolve_section_ids(report_env, []) == expected


@pytest.mark.parametrize("include_disabled", [False, True])
def test_section_autofill_skips_disabled_and_deduplicates(report_env, include_disabled):
    report_env.analyses[2].enabled = False
    report_env._resolved_graph = None
    report_env.report["include_disabled"] = include_disabled
    report_env.report["sections"] = ["pred_compare", "pred_compare"]
    warnings = []
    assert _resolve_section_ids(report_env, warnings) == [
        "pred_compare", "perf_usage_efficiency", "pred_metrics",
    ]
    assert any("listed more than once" in warning for warning in warnings)
    report_env.report["sections"] = ["cal_reliability"]
    expected = ["perf_usage_efficiency", "pred_metrics", "pred_compare"]
    if include_disabled:
        expected.insert(0, "cal_reliability")
    assert _resolve_section_ids(report_env, []) == expected


def test_html_uses_one_shared_responsive_stylesheet(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    pages = [out / name for name in (
        "index.html", "prediction.html", "calibration.html", "performance.html"
    )]
    for page in pages:
        html = page.read_text(encoding="utf-8")
        assert '<link rel="stylesheet" href="assets/report.css">' in html
        assert "<style" not in html

    css = (out / "assets" / "report.css").read_text(encoding="utf-8")
    assert ".sidebar { position: fixed" in css
    assert "@media (max-width: 820px)" in css


def test_unknown_section_id_is_loud(report_env):
    report_env.report["sections"] = ["pred_metrics", "nope"]
    with pytest.raises(ReportError, match="not an analysis stage id"):
        run_report(report_env)


def test_overview_section_must_be_in_resolved_sections(report_env):
    report_env.analyses[2].enabled = False
    report_env._resolved_graph = None
    report_env.report["sections"] = ["pred_metrics"]
    report_env.report["overview_sections"] = ["cal_reliability"]
    with pytest.raises(ReportError, match="overview_sections"):
        run_report(report_env)


def test_overview_sections_null_uses_every_resolved_section(report_env):
    report_env.report["overview_sections"] = None
    run_report(report_env)
    overview = (report_dir(report_env) / "index.html").read_text(encoding="utf-8")
    assert overview.count('class="overview-card"') == 4


def test_section_override_replaces_one_dimension_and_inherits_the_other(report_env):
    report_env.report["section_overrides"] = {
        "pred_metrics": {"figures": ["metrics", "not_emitted"]}
    }
    result = run_report(report_env)
    prediction = (report_dir(report_env) / "prediction.html").read_text(encoding="utf-8")

    assert '<img src="assets/pred_metrics/metrics.png"' in prediction
    assert "<table" in prediction  # tables inherit prediction.evaluate's default
    assert any("not_emitted" in warning for warning in result.warnings)


def test_empty_override_hides_inline_artifact_but_keeps_download(report_env):
    report_env.report["section_overrides"] = {
        "pred_metrics": {"tables": [], "figures": []}
    }
    run_report(report_env)
    prediction = (report_dir(report_env) / "prediction.html").read_text(encoding="utf-8")

    assert "<table" not in prediction and "<img" not in prediction
    assert 'href="assets/pred_metrics/metrics.csv"' in prediction
    assert 'href="assets/pred_metrics/metrics.png"' in prediction


def test_unknown_section_override_id_is_loud(report_env):
    report_env.report["section_overrides"] = {"nope": {"tables": []}}
    with pytest.raises(ReportError, match="section_overrides"):
        run_report(report_env)


def test_missing_manifest_is_loud(report_env):
    # Remove one section's manifest → the report must fail loud, not skip silently.
    (_analyses_dir(report_env, "cal_reliability") / "result.json").unlink()
    with pytest.raises(ReportError, match="no result manifest"):
        run_report(report_env)


def test_failed_rerender_preserves_previous_site(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    previous = (out / "index.html").read_bytes()
    (_analyses_dir(report_env, "cal_reliability") / "result.json").unlink()

    with pytest.raises(ReportError, match="no result manifest"):
        run_report(report_env)

    assert (out / "index.html").read_bytes() == previous
    assert not list(out.parent.glob(f".{out.name}.tmp-*"))


def test_manifest_asset_cannot_escape_analysis_tree(report_env):
    manifest_path = _analyses_dir(report_env, "pred_metrics") / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"][0]["file"] = "../outside.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outside = manifest_path.parent.parent / "outside.csv"
    outside.write_text("secret\nvalue\n", encoding="utf-8")

    result = run_report(report_env)

    assert any("escapes the analysis tree" in warning for warning in result.warnings)
    assert not (report_dir(report_env) / "outside.csv").exists()


def test_malformed_manifest_cleans_unique_staging_directory(report_env):
    manifest_path = _analyses_dir(report_env, "pred_metrics") / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["module"] = 7
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = report_dir(report_env)

    with pytest.raises(ReportError, match="non-string module"):
        run_report(report_env)

    assert not list(out.parent.glob(f".{out.name}.tmp-*"))


def test_unsupported_format_is_loud(report_env):
    report_env.report["formats"] = ["md", "pdf"]
    with pytest.raises(ReportError, match="pdf"):
        run_report(report_env)


# ── renderer units ──────────────────────────────────────────────────────────────
def test_render_markdown_and_html_from_document(report_env):
    from bench.reports.model import ReportDocument, FamilyGroup, Section, Table

    doc = ReportDocument(
        title="T", run_name="r", seed=1, config_path="c.json", output_root="reports",
        intro="hi **there**",
        groups=[FamilyGroup(
            key="prediction",
            title="Prediction",
            summary="This page summarizes the prediction result.",
            sections=[
            Section(id="s", module="prediction.evaluate", summary="ok",
                    tables=[Table(name="t", frame=pd.DataFrame({"a": [1]}),
                                  n_total_rows=1, n_shown_rows=1)]),
        ])],
    )
    md = render_markdown(doc)
    pages = render_html_site(doc)
    html = pages["index.html"]
    assert "# T" in md and "## Prediction" in md and "### s" in md
    assert "<h1>T</h1>" in html and "<strong>there</strong>" in html
    assert "prediction.html" in pages
    assert "<table" in pages["prediction.html"]
    assert "This page summarizes the prediction result." in md
    assert "This page summarizes the prediction result." in pages["prediction.html"]
    assert render_html(doc) == html


def test_summary_markdown_renders_in_overview_and_details(report_env):
    summary = (
        "**3** models cost **12.34 USD** across **192** games; "
        "*estimated* from `**tokens**`. "
        "<script>alert(1)</script> [unsafe](javascript:alert(1))."
    )
    _emit(report_env, "perf_usage_efficiency", "performance.usage_efficiency",
          summary=summary)
    run_report(report_env)
    out = report_dir(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.count(summary) == 2
    for filename in ("index.html", "performance.html"):
        html = (out / filename).read_text(encoding="utf-8")
        assert "<strong>3</strong> models cost <strong>12.34 USD</strong>" in html
        assert "<strong>192</strong> games" in html
        assert "<em>estimated</em> from <code>**tokens**</code>" in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
        assert "<script>alert(1)</script>" not in html
        assert 'href="javascript:' not in html


def test_section_without_summary_gets_one_sentence_in_every_view(report_env):
    from bench.reports.model import FamilyGroup, ReportDocument, Section

    section = Section(id="s", module="prediction.evaluate")
    doc = ReportDocument(
        title="T",
        run_name="r",
        seed=1,
        config_path="c.json",
        output_root="reports",
        groups=[FamilyGroup(key="prediction", title="Prediction", sections=[section])],
        overview_sections=[section],
    )

    sentence = "No result summary was produced for this analysis."
    md = render_markdown(doc)
    pages = render_html_site(doc)
    assert md.count(sentence) == 2
    assert sentence in pages["index.html"]
    assert sentence in pages["prediction.html"]


def test_report_renders_rating_provenance_metadata(report_env):
    from bench.reports.model import ReportDocument, FamilyGroup, Section

    section = Section(
        id="bt_main",
        module="ratings.bradley_terry",
        summary="9 identities rated.",
        metadata={
            "group_by": ["player_type"],
            "strength_estimator": "attention",
            "estimator_model": "attention_mlp",
            "adjust_block": "auto/start_cell",
        },
    )
    doc = ReportDocument(
        title="T", run_name="r", seed=1, config_path="c.json", output_root="reports",
        groups=[FamilyGroup(key="ratings", title="Ratings", sections=[section])],
    )
    md = render_markdown(doc)
    html = render_html_site(doc)["ratings.html"]
    assert "strength_estimator: attention" in md
    assert "estimator_model: attention_mlp" in md
    assert "adjust_block: auto/start_cell" in html


def test_report_help_contains_escaped_section_details_and_accessible_link():
    from bench.reports.model import FamilyGroup, ReportDocument, Section

    section = Section(
        id="s",
        module="family.module",
        display_name="Readable section",
        description="Description <with> & details",
        summary="Result summary.",
        metadata={"source": "table <main> & derived"},
    )
    doc = ReportDocument(
        title="Benchmark",
        run_name="run",
        seed=1,
        config_path="config.json",
        output_root="reports",
        description="Main benchmark description.",
        groups=[FamilyGroup(key="family", title="Family", sections=[section])],
        overview_sections=[section],
    )

    pages = render_html_site(doc)
    family = pages["family.html"]
    assert '<p class="caption">Main benchmark description.</p>' in pages["index.html"]
    assert (
        '<button type="button" class="help-toggle" aria-label="Details" '
        'aria-describedby="help-section-s"'
    ) in family
    help_start = '<span class="help-text" role="tooltip" id="help-section-s">'
    assert help_start in family
    help_text = family.split(help_start, 1)[1].split("</span>", 1)[0]
    assert "Description &lt;with&gt; &amp; details" in help_text
    assert "Module: family.module" in help_text
    assert "source: table &lt;main&gt; &amp; derived" in help_text
    assert "<with>" not in help_text

    markdown = render_markdown(doc)
    assert "<details>" in markdown and "<summary>Technical details</summary>" in markdown
    assert "*Description <with> & details*" in markdown
    assert "*Module: `family.module`*" in markdown
    assert "*source: table <main> & derived*" in markdown


def test_cached_controlled_summary_is_publicly_normalized():
    from bench.reports.content import report_summary

    summary = (
        "Controlled seed comparison covers 3 seed(s) and 8 final seat(s) with "
        "800 strategist-condition combination(s) plus the dedicated Vanilla "
        "baseline 'vanilla-standard-fixed' from 810 controlled game(s)."
    )
    rendered = report_summary(
        summary,
        {"baseline_experiment": "vanilla-standard-fixed"},
    )
    assert rendered.startswith("The comparison covers **3** seed(s) and **8** final seat(s)")
    assert "**800** strategist-condition combination(s)" in rendered
    assert "**810** controlled game(s)" in rendered
    assert "VPAI" in rendered
    assert "vanilla-standard-fixed" not in rendered


def test_help_ids_remain_unique_when_section_anchors_overlap_help_suffix():
    from bench.reports.model import FamilyGroup, ReportDocument, Section

    sections = [Section(id="x", module="m"), Section(id="x-help", module="m")]
    doc = ReportDocument(
        title="Benchmark", run_name="run", seed=1, config_path="config.json",
        output_root="reports", groups=[FamilyGroup(key="family", title="Family", sections=sections)],
    )
    html = render_html_site(doc)["family.html"]
    assert 'id="help-section-x"' in html
    assert 'id="help-section-x-help"' in html
    assert html.count('class="help-text" role="tooltip"') == 2


# ── descriptions and report structure ─────────────────────────────────────────
def test_manifest_module_name_and_description_are_rendered(report_env):
    _emit(report_env, "pred_metrics", "prediction.evaluate",
          summary="Evaluated estimator(s).",
          tables={"metrics": pd.DataFrame({"model": ["a"], "roc_auc": [0.5]})},
          module_name="Prediction metrics",
          module_description="Scores each estimator's win-probability metrics.")
    result = run_report(report_env)
    out = report_dir(report_env)

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "### " in md
    assert "*Scores each estimator's win-probability metrics.*" in md
    assert "Module: `prediction.evaluate`" in md
    # The TOC and overview contain a link and a section card.
    assert "- [" in md
    assert "- **" in md

    overview = (out / "index.html").read_text(encoding="utf-8")
    assert "<h1>civbench-dev</h1>" in overview
    assert 'class="overview-card"' in overview
    assert "<h2>" in overview

    prediction = (out / "prediction.html").read_text(encoding="utf-8")
    assert "<h2 id=\"section-pred-metrics\">" in prediction
    assert 'href="prediction.html#section-pred-metrics"' in prediction
    assert "Scores each estimator" in prediction
    assert 'class="report-help"' in prediction
    assert result.n_sections == 4


def test_manifest_without_module_name_falls_back_to_stage_id(report_env):
    # A manifest from before friendly names still renders a section heading.
    _emit(report_env, "pred_metrics", "prediction.evaluate",
          summary="ok.",
          tables={"metrics": pd.DataFrame({"model": ["a"], "roc_auc": [0.5]})})
    run_report(report_env)
    md = (report_dir(report_env) / "report.md").read_text(encoding="utf-8")
    stage = next(s for s in report_env.analyses if s.id == "pred_metrics")
    assert f"### {stage.id}" in md


def test_section_name_description_override_beats_module_defaults(report_env):
    _emit(report_env, "pred_metrics", "prediction.evaluate",
          summary="ok.",
          tables={"metrics": pd.DataFrame({"model": ["a"], "roc_auc": [0.5]})},
          module_name="Module default name",
          module_description="Module default description.")
    stage = next(s for s in report_env.analyses if s.id == "pred_metrics")
    stage.raw["name"] = "Configured heading"
    stage.raw["description"] = "Configured description."
    run_report(report_env)
    md = (report_dir(report_env) / "report.md").read_text(encoding="utf-8")
    assert f"### {stage.raw['name']}" in md
    assert "*Configured description.*" in md


def test_config_friendly_name_and_description_show_on_report_page(report_env):
    report_env.friendly_name = "Staff benchmark 2026"
    report_env.description = "Staff line-up, standard 8-seat map."
    run_report(report_env)
    out = report_dir(report_env)

    overview = (out / "index.html").read_text(encoding="utf-8")
    assert "<h1>" in overview
    assert 'class="caption">Staff line-up, standard 8-seat map.</p>' in overview

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# Staff benchmark 2026")
    assert "*Staff line-up, standard 8-seat map.*" in md


# ── section views (relative / absolute toggle) ───────────────────────────────
@pytest.fixture
def views_env(tmp_path, write_spec, dev_spec):
    root = str(tmp_path / "out")
    spec = dev_spec
    spec["output"] = {"root": root, "suffix": ""}
    spec["data"]["extract"]["enabled"] = False
    spec["adjust"] = []
    spec["analyses"] = [
        {"id": "perf_usage_efficiency", "module": "performance.usage_efficiency",
         "enabled": True, "uses": {"tables": ["tokens"]}, "params": {}},
        {"id": "beh_two", "module": "behavior.commitment", "enabled": True,
         "params": {}},
        {"id": "beh_one", "module": "behavior.commitment", "enabled": True,
         "params": {}},
    ]
    # The behavior pages draw HTML heatmaps now; these sections inline PNGs on request.
    figures = {"tables": [], "figures": ["commitment_relative", "commitment_absolute"]}
    spec["report"] = {"out_dir": root + "/", "formats": ["md", "html"], "sections": None,
                      "overview_sections": None,
                      "section_overrides": {"beh_two": figures, "beh_one": figures},
                      "title": None, "include_disabled": False}
    cfg = load_config(write_spec(spec))
    _emit(cfg, "perf_usage_efficiency", "performance.usage_efficiency", summary="Usage.",
          figures=[{"name": "usage_efficiency", "format": "plotly"}])
    frame = pd.DataFrame({"player_type": ["Kimi"], "metric": ["wars_declared"], "mean": [1.5]})
    views = {
        "relative": {"label": "Relative to matched in-game AI",
                     "tables": [], "figures": ["commitment_relative"]},
        "absolute": {"label": "Absolute", "tables": [], "figures": ["commitment_absolute"]},
    }
    _emit(cfg, "beh_two", "behavior.commitment", summary="Two views.",
          metadata={"baseline_experiment": "vanilla", "views": views},
          tables={"commitment_relative": frame, "commitment_absolute": frame},
          figures=["commitment_relative", "commitment_absolute"])
    _emit(cfg, "beh_one", "behavior.commitment", summary="One view.",
          metadata={"views": {"absolute": views["absolute"]}},
          tables={"commitment_absolute": frame}, figures=["commitment_absolute"])
    return cfg


def test_views_render_a_toggle_with_the_first_view_selected(views_env):
    run_report(views_env)
    out = report_dir(views_env)
    page = (out / "behavior.html").read_text(encoding="utf-8")
    assert page.count('<div class="view-switch"') == 1          # only the two-view section
    relative_button = page.index('data-view="relative" aria-controls=')
    absolute_button = page.index('data-view="absolute" aria-controls=')
    assert relative_button < absolute_button
    assert 'data-view="relative" aria-controls="section-beh-two-view-relative" aria-pressed="true"' in page
    assert 'aria-pressed="false">Absolute</button>' in page
    assert page.count('class="view-panel"') == 3
    assert page.index('src="assets/beh_two/commitment_relative.png"') < page.index(
        'src="assets/beh_two/commitment_absolute.png"')
    assert "views:" not in page                                # layout keys stay out of tooltips
    assert ".views-ready .view-switch" in (out / "assets/report.css").read_text(encoding="utf-8")
    assert "views-ready" in (out / "assets/report-help.js").read_text(encoding="utf-8")

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.index("## Performance") < md.index("## Behavior")
    assert md.index("**Relative to matched in-game AI**") < md.index("**Absolute**")
    assert "views:" not in md
    # Tables stay downloads (module defaults inline figures only).
    assert "assets/beh_two/commitment_relative.csv" in md


def test_malformed_views_fall_back_to_a_single_view(views_env):
    _emit(views_env, "beh_two", "behavior.commitment", summary="Broken.",
          metadata={"views": ["relative"]}, figures=["commitment_relative"])
    run_report(views_env)
    page = (report_dir(views_env) / "behavior.html").read_text(encoding="utf-8")
    assert page.count('<div class="view-switch"') == 0
    assert 'src="assets/beh_two/commitment_relative.png"' in page


# ── HTML heatmap tables ────────────────────────────────────────────────────────
BASELINE_ROW = "Completed-experiment average"


def _heat_frame(view: str, n_rows: int = 60) -> pd.DataFrame:
    """A long heatmap table: the pinned baseline row, Null, and ``n_rows - 1``
    strategists over two flavors. The baseline row holds the pool's absolute
    means (no median or CI), like ``behavior.flavors`` writes."""
    records = []
    for metric, group_name in (("flavor_offense_avg", "Military"),
                               ("flavor_science_avg", "Economy")):
        records.append({
            "row_kind": "baseline", "player_type": BASELINE_ROW, "strategist": BASELINE_ROW,
            "condition": "", "row_label": BASELINE_ROW,
            "metric": metric, "metric_group": group_name,
            "mean": 55.0, "sd": 12.0, "n_players": 12, "n_games": 5,
            "color_position": 0.55,
        })
    for i in range(n_rows):
        group = "Null" if i == 0 else f"Model-{i:02d}"
        for metric, group_name, position in (
            ("flavor_offense_avg", "Military", 0.0 if i == 1 else 0.5),
            ("flavor_science_avg", "Economy", 1.0 if i == 1 else 0.5),
        ):
            records.append({
                "row_kind": "group",
                "player_type": group, "strategist": group, "condition": "", "row_label": group,
                "metric": metric, "metric_group": group_name,
                "mean": 12.4 if view == "relative" else 62.4, "median": 12.0,
                "ci_lower": 10.0, "ci_upper": 15.0, "n_players": 48, "n_games": 48,
                "color_position": position,
            })
    return pd.DataFrame(records)


def _heat_spec(view: str) -> dict:
    return {
        "view": view, "title": f"Flavors {view}", "help": "How to read it.",
        "row": "row_label", "column": "metric", "value": "mean",
        "row_heading": "Strategist | Condition",
        "row_order": [BASELINE_ROW, "Null", "Model-02", "Model-01"],
        "reference_rows": [BASELINE_ROW, "Null"], "row_tips": {"Null": "Never changes flavors."},
        "baseline_rows": [BASELINE_ROW],
        "column_order": ["flavor_offense_avg", "flavor_science_avg"],
        "column_names": {"flavor_offense_avg": "Offense", "flavor_science_avg": "Science"},
        "column_labels": {"flavor_offense_avg": "Off", "flavor_science_avg": "Sci"},
        "column_tips": {"flavor_offense_avg": "Offense\nPivots the military.",
                        "flavor_science_avg": "Science\nPrioritizes science."},
        "column_group": "metric_group", "decimals": 0, "signed": view == "relative",
        "value_label": "Difference" if view == "relative" else "Average",
        "ci_level": 0.95, "legend": [[0, "low"], [0.5, "balanced"], [1, "high"]],
    }


@pytest.fixture
def heatmap_env(tmp_path, write_spec, dev_spec):
    root = str(tmp_path / "out")
    spec = dev_spec
    spec["output"] = {"root": root, "suffix": ""}
    spec["data"]["extract"]["enabled"] = False
    spec["adjust"] = []
    spec["analyses"] = [{"id": "beh_flavors", "module": "behavior.flavors", "enabled": True,
                         "params": {}}]
    spec["report"] = {"out_dir": root + "/", "formats": ["md", "html"], "sections": None,
                      "overview_sections": None, "section_overrides": {}, "title": None,
                      "include_disabled": False}
    cfg = load_config(write_spec(spec))
    views = {
        "relative": {"label": "Relative", "tip": "Relative to completed-experiment average",
                     "tables": ["flavors_relative"], "figures": []},
        "absolute": {"label": "Absolute", "tables": ["flavors_absolute"], "figures": []},
    }
    _emit(cfg, "beh_flavors", "behavior.flavors", summary="Flavors.",
          metadata={"views": views, "heatmaps": {
              "flavors_relative": _heat_spec("relative"),
              "flavors_absolute": _heat_spec("absolute")}},
          tables={"flavors_relative": _heat_frame("relative"),
                  "flavors_absolute": _heat_frame("absolute")})
    return cfg


def test_heatmap_tables_render_in_the_matched_maps_style(heatmap_env):
    run_report(heatmap_env)
    out = report_dir(heatmap_env)
    page = (out / "behavior.html").read_text(encoding="utf-8")
    assert page.count('<table class="heatmap">') == 2
    assert page.count('<div class="view-switch"') == 1
    # Short headers carry the full name and description in their tooltip.
    assert '<span tabindex="0" data-tip="Offense\nPivots the military.">Off</span>' in page
    assert 'colspan="1" class="heat-group">Military</th>' in page
    # The baseline pool and Null are pinned in the reference body, before the order.
    assert '<tbody class="vanilla-body"><tr class="vanilla-row">' in page
    assert (page.index(f">{BASELINE_ROW}</th>") < page.index('data-tip="Never changes flavors.">Null')
            < page.index(">Model-02<") < page.index(">Model-01<"))
    # Positions 0, 0.5, and 1 take the red, yellow, and blue anchors.
    assert re.search(r'background-color:#a50026;color:#ffffff" data-value="[^"]+" '
                     r'data-tip="Model-01\nOffense', page)
    assert re.search(r'background-color:#313695;color:#ffffff" data-value="[^"]+" '
                     r'data-tip="Model-01\nScience', page)
    # Short tab labels carry the long wording as their tooltip.
    assert 'data-tip="Relative to completed-experiment average">Relative</button>' in page
    # Headers carry their column position for the shared click-to-sort script.
    assert 'data-col="1"><span tabindex="0" data-tip="Offense' in page
    assert "sortableTable" in (out / "assets/report-help.js").read_text(encoding="utf-8")
    assert "background-color:#ffffbf" in page
    assert ">+12</td>" in page and ">62</td>" in page
    # Cell tooltips are grid lines: label, value, CI, and counts (decimals + 1).
    assert (
        'data-tip="Model-01\nOffense\nDifference\t+12.4\n95% CI\t\t+10.0 to +15.0\n'
        'Players\t48\tin 48 games"' in page
    )
    # The baseline row reads unsigned even in the signed relative table, and
    # shows its SD (one legend step) and absolute-average label.
    assert (
        'data-tip="' + BASELINE_ROW + '\nOffense\nAverage\t55.0\n'
        'SD\t12.0\tone legend step\nPlayers\t12\tin 5 games"' in page
    )
    # The absolute table drops the SD line and the cell never takes a sign.
    assert (
        'data-tip="' + BASELINE_ROW + '\nOffense\nAverage\t55.0\n'
        'Players\t12\tin 5 games"' in page
    )
    assert ">+55<" not in page
    assert f">{BASELINE_ROW}</th>" in page  # the row label itself carries no sign logic
    # The whole long table survives the inline row cap (60 rows x 2 flavors).
    assert page.count(">Model-59<") == 2
    assert "heatmaps:" not in page
    help_js = (out / "assets/report-help.js").read_text(encoding="utf-8")
    assert "data-tip" in help_js
    assert 'tip-number-" + color' in help_js
    assert 'sign === "+" ? "positive"' in help_js
    assert 'sign === "-"' in help_js and '"unsigned"' in help_js
    css = (out / "assets/report.css").read_text(encoding="utf-8")
    assert "table.heatmap .row-label { position: sticky" in css
    assert ".heat-tooltip .tip-number-positive { color: #237a45; }" in css
    assert ".heat-tooltip .tip-number-negative { color: #c0392b; }" in css
    assert ".heat-tooltip .tip-number-unsigned { color: #6b3fa0; }" in css

    md = (out / "report.md").read_text(encoding="utf-8")
    assert r"| Strategist \| Condition | Off | Sci |" in md
    assert "| Null | 62 | 62 |" in md
    # The baseline row renders unsigned in both the relative and absolute grids.
    assert md.count(f"| {BASELINE_ROW} | 55 | 55 |") == 2
    assert "+55" not in md
    assert "_Columns: Off = Offense; Sci = Science._" in md


def test_heatmap_range_uses_the_table_number_format():
    from bench.reports.heatmap import render_heatmap_html

    frame = pd.DataFrame([{"row_label": "Kimi", "metric": "persona_loyalty_avg", "mean": -3.0,
                           "mean_min": -4.0, "mean_max": 0.25, "color_position": 0.2}])
    spec = {"row": "row_label", "column": "metric", "value": "mean", "decimals": 1,
            "signed": True, "value_label": "Difference",
            "range_columns": ["mean_min", "mean_max"], "range_label": "Range",
            "range_note": "mean min to max, vs. baseline"}
    # A relative range is signed, in the table's decimals, with its own note.
    assert "Range		-4.0 to +0.2 (mean min to max, vs. baseline)" in render_heatmap_html(frame, spec)
    spec = {**spec, "signed": False}
    del spec["range_note"]
    frame = frame.assign(mean_min=2.0, mean_max=4.0)
    assert "Range		2.0 to 4.0 (mean min to max)" in render_heatmap_html(frame, spec)


def test_heatmap_columns_carry_their_own_format_units_and_labels():
    import html as _html

    from bench.reports.heatmap import render_heatmap_html, render_heatmap_md

    frame = pd.DataFrame([
        {"row_label": "Base", "metric": "acts", "mean": 64.94, "sd": 30.0, "color_position": 0.5},
        {"row_label": "Base", "metric": "step", "mean": 9.87, "sd": 3.0, "color_position": 0.5},
        {"row_label": "Kimi", "metric": "acts", "mean": 28.44, "ci_lower": 25.0, "ci_upper": 31.4,
         "color_position": 0.9},
        {"row_label": "Kimi", "metric": "step", "mean": -1.25, "color_position": 0.4,
         "shift": 30.0},
    ])
    spec = {"row": "row_label", "column": "metric", "decimals": 0, "signed": True,
            "value_label": "Difference", "baseline_rows": ["Base"], "ci_level": 0.95,
            "column_decimals": {"step": 1},
            "column_units": {"acts": "pts", "step": "points"},
            "baseline_units": {"acts": "%", "step": "points"},
            "column_value_labels": {"step": "Change"},
            "tip_rows": [{"column": "shift", "label": "Total", "unit": "points", "decimals": 1}]}
    page = _html.unescape(render_heatmap_html(frame, spec))
    # Cell text follows each column's decimals; tooltips add one place and the unit.
    assert ">+28</td>" in page and ">-1.2</td>" in page
    assert "Difference\t+28.4 pts" in page
    assert "95% CI\t\t+25.0 pts to +31.4 pts" in page
    assert "Change\t-1.25 points" in page and "Total\t30.0 points" in page
    # The pinned baseline row is absolute, so it takes the baseline units.
    assert "Average\t64.9%" in page
    assert "| Kimi | +28 | -1.2 |" in render_heatmap_md(frame, spec)


def test_heatmap_category_cells_take_their_category_color():
    import html as _html

    from bench.reports.heatmap import category_background, render_heatmap_html

    frame = pd.DataFrame([
        {"row_label": "Kimi", "metric": "main", "mean": 62.0, "color_position": 0.62,
         "category": "Spaceship", "text": "Spaceship 62%", "strategy": "Spaceship"},
        {"row_label": "Kimi", "metric": "switches", "mean": 0.5, "color_position": float("nan"),
         "category": "", "text": "", "strategy": ""},
    ])
    spec = {"row": "row_label", "column": "metric", "decimals": 0, "value_label": "Share of turns",
            "text_column": "text", "category_column": "category",
            "category_colors": {"Spaceship": "#4e9b4e"}, "column_decimals": {"switches": 2},
            "column_units": {"main": "%"},
            "tip_rows": [{"column": "strategy", "label": "Strategy", "text": True}]}
    page = _html.unescape(render_heatmap_html(frame, spec))
    assert f"background-color:{category_background('#4e9b4e', 0.62)}" in page
    assert ">Spaceship 62%</td>" in page and "Strategy\tSpaceship" in page
    # A cell without a category or color stays plain.
    assert '<td class="heat-cell heat-cell-value" data-value="0.5"' in page and ">0.50</td>" in page


def test_a_heatmap_spec_that_does_not_fit_renders_a_plain_table(heatmap_env):
    frame = _heat_frame("absolute", n_rows=3).drop(columns=["color_position"])
    _emit(heatmap_env, "beh_flavors", "behavior.flavors", summary="Flavors.",
          metadata={"heatmaps": {"flavors_absolute": _heat_spec("absolute")}},
          tables={"flavors_absolute": frame})
    run_report(heatmap_env)
    page = (report_dir(heatmap_env) / "behavior.html").read_text(encoding="utf-8")
    assert '<table class="heatmap">' not in page
    assert "<strong>flavors_absolute</strong>" in page
