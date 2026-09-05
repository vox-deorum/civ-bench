"""Stage 5 report tests.

Exercise the report stage on fabricated analysis manifests (no machine data roots,
per AGENTS.md): section resolution (null = enabled analyses in canonical family
order; explicit list = authored order), the manifest → document → md/html render,
asset copying into a self-contained tree, empty-section handling, determinism
(byte-stable re-render), and the loud error when a manifest is missing.

The report reads each analysis's ``result.json`` from disk, so we fabricate those
directly rather than running the (heavy) analysis modules; that keeps the suite
fast and hermetic while testing exactly the report-stage contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bench.config import load_config
from bench.reports import (
    ReportError,
    render_html,
    render_html_site,
    render_markdown,
    run_report,
)
from bench.reports.runner import _analyses_dir, report_dir

_FAKE_PNG = b"\x89PNG\r\n\x1a\n-- not a real image, copied verbatim --"


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
        for name in figures or []:
            (d / f"{name}.png").write_bytes(_FAKE_PNG)
            fnames.append({"name": name, "file": f"{name}.png"})
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
        {"id": "explore_token_costs", "module": "exploratory.model_token_costs",
         "enabled": True, "uses": {"tables": ["tokens"]}, "params": {}},
    ]
    # out_dir authored as the tmp root so it re-roots there; <name> is appended.
    spec["report"] = {"out_dir": root + "/", "formats": ["md", "html"],
                      "sections": None,
                      "overview_sections": [
                          "pred_metrics", "cal_reliability", "explore_token_costs"
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
    _emit(cfg, "explore_token_costs", "exploratory.model_token_costs",
          summary="Total spend $12.34 across 192 games.",
          tables={"token_costs": pd.DataFrame({"model": ["a"], "total_cost": [12.34]})},
          artifacts={"seating/ctrl.seating.json": '{"totalSeats": 2, "cells": {}}'})
    return cfg


# ── end-to-end render ──────────────────────────────────────────────────────────
def test_run_report_writes_md_and_html(report_env):
    result = run_report(report_env)
    out = report_dir(report_env)
    assert result.n_sections == 4
    expected = {
        "report.md", "report.html", "prediction.html", "calibration.html",
        "exploratory.html", "assets/report.css",
    }
    assert expected == {
        str(path.relative_to(out)).replace("\\", "/")
        for path in map(type(out), result.written)
    }
    assert all((out / rel).exists() for rel in expected)
    assert set(result.formats) == {"md", "html"}

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# civbench-dev")
    assert (
        "Run **civbench-dev** (seed 42) produced 4 analysis sections across 3 "
        "families" in md
    )
    # Family chapters, canonical order: prediction → calibration → exploratory.
    assert md.index("## Prediction") < md.index("## Calibration") < md.index("## Exploratory")
    assert (
        "This page brings together the results for pred_metrics and pred_compare."
        in md
    )
    # Section content surfaced from the manifest.
    assert "### pred_metrics" in md and "best **roc_auc**" in md
    assert "metrics" in md and "0.87" in md  # inline table value
    # Empty section is labelled, not silently dropped.
    assert "### pred_compare" in md and "produced no artifacts" in md

    overview = (out / "report.html").read_text(encoding="utf-8")
    assert overview.count('class="overview-card"') == 3
    assert "pred_compare" in overview  # present in shared navigation
    assert "<table" not in overview and "<img" not in overview

    prediction = (out / "prediction.html").read_text(encoding="utf-8")
    assert "pred_metrics" in prediction and "pred_compare" in prediction
    assert "This page brings together the results for pred_metrics and pred_compare." in prediction
    assert '<h2 id="section-cal-reliability">' not in prediction
    assert 'href="assets/report.css"' in prediction
    assert 'aria-current="page">Prediction</a>' in prediction


def test_assets_copied_self_contained(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    assert (out / "assets" / "pred_metrics" / "metrics.png").exists()
    assert (out / "assets" / "pred_metrics" / "metrics.csv").exists()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "assets/pred_metrics/metrics.png" in md
    assert "Figure: pred_metrics" in md  # compact default keeps the PNG as a download


def test_artifacts_copied_and_linked(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    # Subdir tree mirrored under assets/<id>/.
    asset = out / "assets" / "explore_token_costs" / "seating" / "ctrl.seating.json"
    assert asset.exists()
    assert asset.read_text(encoding="utf-8") == '{"totalSeats": 2, "cells": {}}'
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "**Downloads and supporting files**" in md
    assert "[ctrl.seating.json](assets/explore_token_costs/seating/ctrl.seating.json)" in md
    html = (out / "exploratory.html").read_text(encoding="utf-8")
    assert 'href="assets/explore_token_costs/seating/ctrl.seating.json"' in html


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


# ── section curation ────────────────────────────────────────────────────────────
def test_explicit_sections_curate_and_reorder(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    assert (out / "calibration.html").exists()

    report_env.report["sections"] = ["explore_token_costs", "pred_metrics"]
    report_env.report["overview_sections"] = ["explore_token_costs", "pred_metrics"]
    result = run_report(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert result.n_sections == 2
    assert "cal_reliability" not in md  # curated out
    # Authored order respected across families: exploratory before prediction.
    assert md.index("## Exploratory") < md.index("## Prediction")
    assert not (out / "calibration.html").exists()


def test_html_uses_one_shared_responsive_stylesheet(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    pages = [out / name for name in (
        "report.html", "prediction.html", "calibration.html", "exploratory.html"
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
    report_env.report["sections"] = ["pred_metrics"]
    report_env.report["overview_sections"] = ["cal_reliability"]
    with pytest.raises(ReportError, match="overview_sections"):
        run_report(report_env)


def test_overview_sections_null_uses_every_resolved_section(report_env):
    report_env.report["overview_sections"] = None
    run_report(report_env)
    overview = (report_dir(report_env) / "report.html").read_text(encoding="utf-8")
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
    previous = (out / "report.html").read_bytes()
    (_analyses_dir(report_env, "cal_reliability") / "result.json").unlink()

    with pytest.raises(ReportError, match="no result manifest"):
        run_report(report_env)

    assert (out / "report.html").read_bytes() == previous
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
    html = pages["report.html"]
    assert "# T" in md and "## Prediction" in md and "### s" in md
    assert "<h1>T</h1>" in html and "<strong>there</strong>" in html
    assert "prediction.html" in pages
    assert "<table" in pages["prediction.html"]
    assert "This page summarizes the prediction result." in md
    assert "This page summarizes the prediction result." in pages["prediction.html"]
    assert render_html(doc) == html


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
    assert sentence in pages["report.html"]
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


# ── friendly names + descriptions ─────────────────────────────────────────────
def test_manifest_module_name_and_description_are_rendered(report_env):
    _emit(report_env, "pred_metrics", "prediction.evaluate",
          summary="Evaluated estimator(s).",
          tables={"metrics": pd.DataFrame({"model": ["a"], "roc_auc": [0.5]})},
          module_name="Prediction metrics",
          module_description="Scores each estimator's win-probability metrics.")
    result = run_report(report_env)
    out = report_dir(report_env)

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "### Prediction metrics" in md
    assert "*Scores each estimator's win-probability metrics.*" in md
    assert "Module: `prediction.evaluate`" in md
    # TOC and overview use the friendly name too.
    assert "- [Prediction metrics]" in md
    assert "- **Prediction metrics** (Prediction):" in md

    overview = (out / "report.html").read_text(encoding="utf-8")
    assert "<h1>civbench-dev</h1>" in overview
    assert 'class="overview-card"' in overview
    assert "<h2>Prediction metrics</h2>" in overview

    prediction = (out / "prediction.html").read_text(encoding="utf-8")
    assert "<h2 id=\"section-pred-metrics\">Prediction metrics</h2>" in prediction
    assert "Prediction metrics" in prediction  # sidebar entry
    assert 'class="caption">Scores each estimator' in prediction
    assert result.n_sections == 4


def test_manifest_without_module_name_falls_back_to_stage_id(report_env):
    # A manifest from before friendly names still renders the stage id as heading.
    _emit(report_env, "pred_metrics", "prediction.evaluate",
          summary="ok.",
          tables={"metrics": pd.DataFrame({"model": ["a"], "roc_auc": [0.5]})})
    run_report(report_env)
    md = (report_dir(report_env) / "report.md").read_text(encoding="utf-8")
    assert "### pred_metrics" in md
    assert "### Prediction metrics" not in md


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
    assert "### Configured heading" in md
    assert "*Configured description.*" in md
    assert "Module default name" not in md


def test_config_friendly_name_and_description_show_on_report_page(report_env):
    report_env.friendly_name = "Staff benchmark 2026"
    report_env.description = "Staff line-up, standard 8-seat map."
    run_report(report_env)
    out = report_dir(report_env)

    overview = (out / "report.html").read_text(encoding="utf-8")
    assert "<h1>Staff benchmark 2026</h1>" in overview
    assert 'class="caption">Staff line-up, standard 8-seat map.</p>' in overview

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# Staff benchmark 2026")
    assert "*Staff line-up, standard 8-seat map.*" in md
