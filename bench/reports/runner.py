"""Report-stage orchestrator (stage 5).

Walks each produced analysis's ``result.json`` manifest (written by the analysis
runner) in dependency order, copies its figures/tables into a self-contained
``assets/`` tree under the report directory, builds a :class:`ReportDocument` via
the selected template, and renders it to the configured formats (md / html).

This is the milestone stage: it proves the full **extract → estimators → adjust →
analyses → report** pipeline end-to-end. Because it reads artifacts from disk (not
in-memory results), ``civ-bench report`` re-renders the document from existing
artifacts without re-running any analysis (invariant 3).

Import-light relative to the analysis runner: it needs pandas + the stdlib only
(figures are already PNGs on disk), so it pulls neither matplotlib nor R.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import RunConfig
from ..pipeline import build_dag
from .errors import ReportError
from .model import Download, Figure, Section, Table
from .render import render_html, render_markdown
from .templates import family_of, family_sort_index, get_template

# Inline tables are capped so a large audit trail (e.g. cell baselines) does not
# bloat the document; the full data is always one click away via the copied CSV.
MAX_TABLE_ROWS = 100

# Renderers we ship today. `pdf` is reserved in the schema but not yet rendered.
_RENDERERS = {"md": render_markdown, "html": render_html}
_EXT = {"md": "md", "html": "html"}


@dataclass
class ReportRunResult:
    report_dir: str
    formats: list[str]
    written: list[str] = field(default_factory=list)
    n_sections: int = 0
    warnings: list[str] = field(default_factory=list)


# ── path resolution ───────────────────────────────────────────────────────────
def _analyses_dir(cfg: RunConfig, stage_id: str) -> Path:
    authored = f"{cfg.output.root}/analyses/{stage_id}"
    return Path(cfg.output.resolve(authored))


def report_dir(cfg: RunConfig) -> Path:
    """The directory the report is written to: ``<resolved-out_dir>/<name>``.

    ``report.out_dir`` is authored under the base output root (default
    ``"reports/"``) and re-rooted by ``output.suffix`` (§2.1); the run name is
    appended as the per-run subdir, so two runs never clobber each other.
    """
    authored = cfg.report.get("out_dir") or f"{cfg.output.root}/"
    resolved = cfg.output.resolve(authored)
    return Path(resolved) / cfg.name


# ── section resolution ─────────────────────────────────────────────────────────
def _ordered_analysis_ids(cfg: RunConfig) -> list[str]:
    """Enabled analysis ids for the default (``sections:null``) document.

    Canonical family order (ratings → prediction → calibration → performance →
    exploratory), then the authored config order within each family so the report
    reads like the run-spec. The topo pass still limits us to enabled analysis
    nodes validated by the DAG.
    """
    dag = build_dag(cfg)
    module_of = {s.id: (s.module or "") for s in cfg.analyses}
    pos = {s.id: i for i, s in enumerate(cfg.analyses)}
    topo = [nid for nid in dag.order if dag.nodes[nid].kind == "analyses"]
    return sorted(
        topo,
        key=lambda nid: (family_sort_index(family_of(module_of[nid])), pos[nid]),
    )


def _resolve_section_ids(cfg: RunConfig, warnings: list[str]) -> list[str]:
    enabled = {s.id for s in cfg.analyses if s.enabled}
    all_ids = {s.id for s in cfg.analyses}
    include_disabled = bool(cfg.report.get("include_disabled", False))

    sections = cfg.report.get("sections")
    if sections is None:
        # null ⇒ every enabled analysis, in dependency order.
        return _ordered_analysis_ids(cfg)

    out: list[str] = []
    for sid in sections:
        if sid not in all_ids:
            raise ReportError(
                f"report.sections references '{sid}', which is not an analysis "
                f"stage id {sorted(all_ids)}."
            )
        if sid in out:
            warnings.append(f"section '{sid}' listed more than once in report.sections — kept once.")
            continue
        if sid not in enabled:
            if include_disabled:
                out.append(sid)
            else:
                warnings.append(
                    f"section '{sid}' is disabled and include_disabled is false — skipped."
                )
            continue
        out.append(sid)
    return out


# ── manifest → section ─────────────────────────────────────────────────────────
def _read_manifest(cfg: RunConfig, stage_id: str) -> dict:
    path = _analyses_dir(cfg, stage_id) / "result.json"
    if not path.exists():
        raise ReportError(
            f"analysis '{stage_id}' has no result manifest at '{path}'. "
            f"Run `civ-bench run` (or `--only {stage_id}`) to produce its "
            f"artifacts before rendering the report."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_section(
    cfg: RunConfig, stage_id: str, assets_root: Path, warnings: list[str]
) -> Section:
    manifest = _read_manifest(cfg, stage_id)
    src_dir = _analyses_dir(cfg, stage_id)
    module = manifest.get("module", "")
    section = Section(
        id=stage_id,
        module=module,
        summary=manifest.get("summary", ""),
        metadata=manifest.get("metadata") or {},
        empty=bool(manifest.get("empty", False)),
    )
    if section.empty:
        return section

    asset_dir = assets_root / stage_id
    for fig in manifest.get("figures", []):
        rel = _copy_asset(src_dir / fig["file"], asset_dir, stage_id, fig["file"], warnings)
        if rel is not None:
            section.figures.append(Figure(caption=_caption(stage_id, fig["name"]), rel_path=rel))

    for tbl in manifest.get("tables", []):
        src = src_dir / tbl["file"]
        if not src.exists():
            warnings.append(f"section '{stage_id}': table file '{src}' is missing — skipped.")
            continue
        frame = pd.read_csv(src)
        rel_csv = _copy_asset(src, asset_dir, stage_id, tbl["file"], warnings)
        shown = frame.head(MAX_TABLE_ROWS)
        section.tables.append(
            Table(
                name=tbl["name"],
                frame=shown,
                rel_csv=rel_csv,
                n_total_rows=int(len(frame)),
                n_shown_rows=int(len(shown)),
            )
        )

    for art in manifest.get("artifacts", []):
        rel_within = str(art["file"])  # relative path under the analysis dir, subdirs kept
        copied = _copy_asset(src_dir / rel_within, asset_dir, stage_id, rel_within, warnings)
        if copied is not None:
            section.downloads.append(Download(label=Path(copied).name, rel_path=copied))
    return section


def _caption(stage_id: str, name: str) -> str:
    return f"{stage_id} — {name}" if name != stage_id else stage_id


def _copy_asset(
    src: Path, asset_dir: Path, stage_id: str, rel_within: str, warnings: list[str]
) -> Optional[str]:
    """Copy an artifact into ``assets/<id>/<rel_within>``; return its report-relative path.

    ``rel_within`` may contain subdirs (e.g. ``seating/<exp>.seating.json``); the
    tree under ``assets/<id>/`` mirrors the analysis dir so links resolve.
    """
    if not src.exists():
        warnings.append(f"section '{stage_id}': asset '{src}' is missing — skipped.")
        return None
    rel_norm = rel_within.replace("\\", "/")
    dst = asset_dir / rel_norm
    # Containment guard: a manifest-supplied rel path must stay under assets/<id>/;
    # a stray ``..`` or absolute component would otherwise write outside the report
    # tree and return a broken link.
    asset_root = asset_dir.resolve()
    resolved = dst.resolve()
    if resolved != asset_root and asset_root not in resolved.parents:
        warnings.append(
            f"section '{stage_id}': asset path '{rel_within}' escapes the asset tree — skipped."
        )
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    # report-relative: assets/<id>/<rel_within>
    return f"assets/{stage_id}/{rel_norm}"


# ── entry point ────────────────────────────────────────────────────────────────
def run_report(cfg: RunConfig) -> ReportRunResult:
    """Render the report for ``cfg`` from the produced analysis artifacts."""
    report_cfg = cfg.report or {}
    template_name = report_cfg.get("template") or "default"
    template = get_template(template_name)

    formats = list(report_cfg.get("formats") or ["md", "html"])
    unsupported = [f for f in formats if f not in _RENDERERS]
    if unsupported:
        raise ReportError(
            f"report.formats: {unsupported} not implemented yet (md/html only); "
            f"`pdf` is schema-reserved. Remove it from report.formats."
        )

    warnings: list[str] = []
    section_ids = _resolve_section_ids(cfg, warnings)
    if not section_ids:
        raise ReportError(
            "report has no sections to render (no enabled analyses, or every "
            "requested section was disabled). Enable an analysis or adjust "
            "report.sections."
        )

    out_dir = report_dir(cfg)
    assets_root = out_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build into a staging tree and swap it in only once every section succeeds, so a
    # mid-build failure (e.g. a missing result.json manifest) leaves the previous
    # report and its assets/ intact instead of deleting them and then aborting.
    # Swapping still rebuilds assets/ from scratch, so a removed section/figure leaves
    # no orphan in the "self-contained" tree.
    staging = out_dir / "assets.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        sections = [_build_section(cfg, sid, staging, warnings) for sid in section_ids]
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if assets_root.exists():
        shutil.rmtree(assets_root)
    staging.replace(assets_root)

    title = report_cfg.get("title") or cfg.name
    meta = {
        "title": title,
        "run_name": cfg.name,
        "seed": cfg.seed,
        "config_path": str(cfg.config_path),
        "output_root": cfg.output.resolved_root,
    }
    document = template(meta, sections)

    written: list[str] = []
    for fmt in formats:
        text = _RENDERERS[fmt](document)
        path = out_dir / f"report.{_EXT[fmt]}"
        path.write_text(text, encoding="utf-8")
        written.append(str(path))

    return ReportRunResult(
        report_dir=str(out_dir),
        formats=formats,
        written=written,
        n_sections=len(sections),
        warnings=warnings,
    )
