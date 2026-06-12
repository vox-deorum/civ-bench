"""Stage 5 — report tests.

Exercise the report stage on fabricated analysis manifests (no machine data roots,
per AGENTS.md): section resolution (null = enabled analyses in canonical family
order; explicit list = authored order), the manifest → document → md/html render,
asset copying into a self-contained tree, empty-section handling, determinism
(byte-stable re-render), and the loud error when a manifest is missing.

The report reads each analysis's ``result.json`` from disk, so we fabricate those
directly rather than running the (heavy) analysis modules — keeping the suite fast
and hermetic while testing exactly the report-stage contract.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from bench.config import load_config
from bench.reports import ReportError, render_html, render_markdown, run_report
from bench.reports.runner import _analyses_dir, report_dir

_FAKE_PNG = b"\x89PNG\r\n\x1a\n-- not a real image, copied verbatim --"


def _emit(cfg, sid, module, *, summary="", metadata=None, tables=None, figures=None, empty=False):
    """Fabricate one analysis's persisted artifacts + ``result.json`` manifest."""
    d = _analyses_dir(cfg, sid)
    d.mkdir(parents=True, exist_ok=True)
    tnames, fnames = [], []
    if not empty:
        for name, frame in (tables or {}).items():
            frame.to_csv(d / f"{name}.csv", index=False)
            tnames.append({"name": name, "file": f"{name}.csv"})
        for name in figures or []:
            (d / f"{name}.png").write_bytes(_FAKE_PNG)
            fnames.append({"name": name, "file": f"{name}.png"})
    manifest = {
        "id": sid, "module": module, "summary": summary,
        "metadata": metadata or {}, "empty": empty, "tables": tnames, "figures": fnames,
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
    spec["report"] = {"template": "default", "out_dir": root + "/", "formats": ["md", "html"],
                      "sections": None, "title": None, "include_disabled": False}

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
          tables={"token_costs": pd.DataFrame({"model": ["a"], "total_cost": [12.34]})})
    return cfg


# ── end-to-end render ──────────────────────────────────────────────────────────
def test_run_report_writes_md_and_html(report_env):
    result = run_report(report_env)
    out = report_dir(report_env)
    assert result.n_sections == 4
    assert (out / "report.md").exists() and (out / "report.html").exists()
    assert set(result.formats) == {"md", "html"}

    md = (out / "report.md").read_text(encoding="utf-8")
    assert md.startswith("# civbench-dev")
    # Family chapters, canonical order: prediction → calibration → exploratory.
    assert md.index("## Prediction") < md.index("## Calibration") < md.index("## Exploratory")
    # Section content surfaced from the manifest.
    assert "### pred_metrics" in md and "best **roc_auc**" in md
    assert "metrics" in md and "0.87" in md  # inline table value
    # Empty section is labelled, not silently dropped.
    assert "### pred_compare" in md and "produced no artifacts" in md


def test_assets_copied_self_contained(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    assert (out / "assets" / "pred_metrics" / "metrics.png").exists()
    assert (out / "assets" / "pred_metrics" / "metrics.csv").exists()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "assets/pred_metrics/metrics.png" in md  # report-relative reference


def test_html_has_table_and_image(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "<table" in html and "<img" in html
    assert "<title>civbench-dev</title>" in html
    # inline-markdown subset converted in summaries
    assert "<strong>roc_auc</strong>" in html


def test_rerender_is_byte_stable(report_env):
    run_report(report_env)
    out = report_dir(report_env)
    first_md = (out / "report.md").read_bytes()
    first_html = (out / "report.html").read_bytes()
    run_report(report_env)  # re-render from the same artifacts
    assert (out / "report.md").read_bytes() == first_md
    assert (out / "report.html").read_bytes() == first_html


# ── section curation ────────────────────────────────────────────────────────────
def test_explicit_sections_curate_and_reorder(report_env):
    report_env.report["sections"] = ["explore_token_costs", "pred_metrics"]
    result = run_report(report_env)
    out = report_dir(report_env)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert result.n_sections == 2
    assert "cal_reliability" not in md  # curated out
    # Authored order respected across families: exploratory before prediction.
    assert md.index("## Exploratory") < md.index("## Prediction")


def test_unknown_section_id_is_loud(report_env):
    report_env.report["sections"] = ["pred_metrics", "nope"]
    with pytest.raises(ReportError, match="not an analysis stage id"):
        run_report(report_env)


def test_missing_manifest_is_loud(report_env):
    # Remove one section's manifest → the report must fail loud, not skip silently.
    (_analyses_dir(report_env, "cal_reliability") / "result.json").unlink()
    with pytest.raises(ReportError, match="no result manifest"):
        run_report(report_env)


def test_unsupported_format_is_loud(report_env):
    report_env.report["formats"] = ["md", "pdf"]
    with pytest.raises(ReportError, match="pdf"):
        run_report(report_env)


def test_unknown_template_is_loud(report_env):
    report_env.report["template"] = "fancy"
    with pytest.raises(ReportError, match="unknown report template"):
        run_report(report_env)


# ── renderer units ──────────────────────────────────────────────────────────────
def test_render_markdown_and_html_from_document(report_env):
    from bench.reports.model import ReportDocument, FamilyGroup, Section, Table

    doc = ReportDocument(
        title="T", run_name="r", seed=1, config_path="c.json", output_root="reports",
        intro="hi **there**",
        groups=[FamilyGroup(key="prediction", title="Prediction", sections=[
            Section(id="s", module="prediction.evaluate", summary="ok",
                    tables=[Table(name="t", frame=pd.DataFrame({"a": [1]}),
                                  n_total_rows=1, n_shown_rows=1)]),
        ])],
    )
    md = render_markdown(doc)
    html = render_html(doc)
    assert "# T" in md and "## Prediction" in md and "### s" in md
    assert "<h1>T</h1>" in html and "<strong>there</strong>" in html


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
    html = render_html(doc)
    assert "strength_estimator: attention" in md
    assert "estimator_model: attention_mlp" in md
    assert "adjust_block: auto/start_cell" in html
