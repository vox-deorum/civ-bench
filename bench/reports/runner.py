"""Report-stage orchestrator (stage 5).

Walks each produced analysis's ``result.json`` manifest (written by the analysis
runner) in dependency order, copies its figures/tables into a self-contained
``assets/`` tree under the report directory, builds the report document
(family chapters, plus the controlled-seed heatmap annex when the report
carries that analysis), and renders it to the configured formats (md / html).

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
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import RunConfig
from ..config import schema as S
from ..config.analysis_metadata import analysis_report_defaults
from ..pipeline import build_dag
from .context import ReportBuildContext
from .errors import ReportError
from .model import Download, Figure, Section, Table
from .render import render_html_site, render_markdown, render_stylesheet
from .templates import default_template, family_of, family_sort_index

# Inline tables are capped so a large audit trail (e.g. cell baselines) does not
# bloat the document; the full data is always one click away via the copied CSV.
MAX_TABLE_ROWS = 100

# Formats we ship today. `pdf` is reserved in the schema but not yet rendered.
_SUPPORTED_FORMATS = {"md", "html"}


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


def _resolve_overview_ids(
    cfg: RunConfig, section_ids: list[str], warnings: list[str]
) -> list[str]:
    overview = cfg.report.get("overview_sections")
    if overview is None:
        return list(section_ids)

    selected = set(section_ids)
    out: list[str] = []
    for sid in overview:
        if sid not in selected:
            raise ReportError(
                f"report.overview_sections references '{sid}', which is not in "
                "the resolved report.sections."
            )
        if sid in out:
            warnings.append(
                f"section '{sid}' listed more than once in "
                "report.overview_sections; kept once."
            )
            continue
        out.append(sid)
    return out


def _resolve_section_overrides(cfg: RunConfig) -> dict[str, dict]:
    overrides = cfg.report.get("section_overrides") or {}
    all_ids = {stage.id for stage in cfg.analyses}
    unknown = sorted(set(overrides) - all_ids)
    if unknown:
        raise ReportError(
            f"report.section_overrides references analysis stage id(s) {unknown}; "
            f"known ids are {sorted(all_ids)}."
        )
    return overrides


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
    cfg: RunConfig,
    stage_id: str,
    assets_root: Path,
    warnings: list[str],
    override: Optional[dict] = None,
    context: Optional[ReportBuildContext] = None,
) -> Section:
    manifest = _read_manifest(cfg, stage_id)
    src_dir = _analyses_dir(cfg, stage_id)
    module = manifest.get("module", "")
    if not isinstance(module, str):
        raise ReportError(
            f"analysis '{stage_id}' has an invalid non-string module in its manifest."
        )
    stage_raw = next((s.raw for s in cfg.analyses if s.id == stage_id), {})
    display_name = (
        stage_raw.get("name")
        or str(manifest.get("module_name") or "")
        or stage_id
    )
    description = stage_raw.get("description") or str(manifest.get("module_description") or "")
    section = Section(
        id=stage_id,
        module=module,
        display_name=display_name,
        description=description,
        summary=manifest.get("summary", ""),
        metadata=manifest.get("metadata") or {},
        empty=bool(manifest.get("empty", False)),
    )

    table_entries = list(manifest.get("tables", []))
    figure_entries = list(manifest.get("figures", []))
    inline_tables = _inline_artifact_names(
        stage_id,
        module,
        "tables",
        [entry["name"] for entry in table_entries],
        override,
        warnings,
    )
    inline_figures = _inline_artifact_names(
        stage_id,
        module,
        "figures",
        [entry["name"] for entry in figure_entries],
        override,
        warnings,
    )
    if section.empty:
        return section

    asset_dir = assets_root / stage_id
    for fig in figure_entries:
        rel = _copy_asset(
            src_dir / fig["file"],
            asset_dir,
            stage_id,
            fig["file"],
            warnings,
            source_root=src_dir,
        )
        if rel is not None:
            caption = _caption(stage_id, fig["name"])
            if fig["name"] in inline_figures:
                section.figures.append(Figure(caption=caption, rel_path=rel))
            else:
                section.downloads.append(
                    Download(label=f"Figure: {caption} (PNG)", rel_path=rel)
                )

    for tbl in table_entries:
        src = src_dir / tbl["file"]
        rel_csv = _copy_asset(
            src,
            asset_dir,
            stage_id,
            tbl["file"],
            warnings,
            source_root=src_dir,
        )
        if rel_csv is None:
            continue
        # Every successfully copied table is loadable in full through the build
        # context (a specialized template may need more than the capped inline
        # frame).
        if context is not None:
            context.record_table(stage_id, tbl["name"], src_dir, tbl["file"])
        if tbl["name"] in inline_tables:
            frame = pd.read_csv(src)
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
        else:
            section.downloads.append(
                Download(label=f"Table: {tbl['name']} (CSV)", rel_path=rel_csv)
            )

    for art in manifest.get("artifacts", []):
        rel_within = str(art["file"])  # relative path under the analysis dir, subdirs kept
        copied = _copy_asset(
            src_dir / rel_within,
            asset_dir,
            stage_id,
            rel_within,
            warnings,
            source_root=src_dir,
        )
        if copied is not None:
            section.downloads.append(Download(label=Path(copied).name, rel_path=copied))
    return section


def _inline_artifact_names(
    stage_id: str,
    module: str,
    kind: str,
    available: list[str],
    override: Optional[dict],
    warnings: list[str],
) -> set[str]:
    """Resolve inline artifact names from a stage override or module defaults."""
    explicit = override is not None and kind in override
    defaults = analysis_report_defaults(module)
    if explicit:
        requested = list(override[kind])
    elif defaults is None or kind not in defaults:
        return set(available)
    else:
        requested = list(defaults[kind])

    if explicit:
        missing = [name for name in requested if name not in available]
        for name in missing:
            warnings.append(
                f"section '{stage_id}': report.section_overrides requested {kind[:-1]} "
                f"'{name}', but it was not emitted; skipped."
            )
    return set(requested) & set(available)


def _caption(stage_id: str, name: str) -> str:
    return f"{stage_id} — {name}" if name != stage_id else stage_id


def _copy_asset(
    src: Path,
    asset_dir: Path,
    stage_id: str,
    rel_within: str,
    warnings: list[str],
    source_root: Optional[Path] = None,
) -> Optional[str]:
    """Copy an artifact into ``assets/<id>/<rel_within>``; return its report-relative path.

    ``rel_within`` may contain subdirs (e.g. ``seating/<exp>.seating.json``); the
    tree under ``assets/<id>/`` mirrors the analysis dir so links resolve.
    """
    if source_root is not None:
        resolved_source_root = source_root.resolve()
        resolved_src = src.resolve()
        if (
            resolved_src != resolved_source_root
            and resolved_source_root not in resolved_src.parents
        ):
            warnings.append(
                f"section '{stage_id}': asset path '{rel_within}' escapes the "
                "analysis tree; skipped."
            )
            return None
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

    # An omitted `formats` list defaults to md + html; a present list must stick
    # to the formats the report actually renders (`pdf` is schema-reserved).
    formats = list(report_cfg.get("formats") or S.REPORT_DEFAULT_FORMATS)
    unsupported = [f for f in formats if f not in _SUPPORTED_FORMATS]
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
    overview_ids = _resolve_overview_ids(cfg, section_ids, warnings)
    section_overrides = _resolve_section_overrides(cfg)

    out_dir = report_dir(cfg)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.tmp-", dir=str(out_dir.parent))
    )
    assets_root = staging / "assets"
    assets_root.mkdir()
    title = report_cfg.get("title") or cfg.friendly_name or cfg.name
    meta = {
        "title": title,
        "run_name": cfg.name,
        "seed": cfg.seed,
        "config_path": str(cfg.config_path),
        "output_root": cfg.output.resolved_root,
        "description": cfg.description,
        "overview_section_ids": overview_ids,
        "formats": formats,
    }
    context = ReportBuildContext(meta=meta)
    try:
        sections = [
            _build_section(
                cfg,
                sid,
                assets_root,
                warnings,
                override=section_overrides.get(sid),
                context=context,
            )
            for sid in section_ids
        ]
        context.sections = sections
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    written_rel: list[Path] = []
    try:
        document = default_template(context)

        if "md" in formats:
            path = staging / "report.md"
            path.write_text(render_markdown(document), encoding="utf-8")
            written_rel.append(Path("report.md"))
        if "html" in formats:
            stylesheet = assets_root / "report.css"
            stylesheet.write_text(render_stylesheet(), encoding="utf-8")
            written_rel.append(Path("assets") / "report.css")
            # The family pages plus, when the document carries it, the
            # controlled-seed heatmap annex.
            for filename, text in render_html_site(document).items():
                path = staging / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                written_rel.append(Path(filename))
        elif document.controlled_seed is not None:
            warnings.append(
                "report.formats excludes html; the controlled-seed heatmap "
                "pages are HTML-only and were skipped."
            )

        warnings.extend(_replace_report_dir(staging, out_dir))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ReportRunResult(
        report_dir=str(out_dir),
        formats=formats,
        written=[str(out_dir / rel) for rel in written_rel],
        n_sections=len(sections),
        warnings=warnings,
    )


def _replace_report_dir(staging: Path, out_dir: Path) -> list[str]:
    """Publish the staged report site, replacing any previous report.

    Prefers the atomic route: rename the previous report away, move the staged
    site into its place, then delete the old copy. Windows denies renaming a
    directory that another program holds open (an Explorer window, an editor,
    a terminal), so a denied rename falls back to copying the staged files over
    the previous report in place and removing obsolete ones. A file that stays
    locked becomes a warning, never a failed run.
    """
    backup = staging.with_name(staging.name + ".bak")

    had_previous = out_dir.exists()
    if had_previous:
        try:
            out_dir.replace(backup)
        except OSError as exc:
            return _publish_in_place(staging, out_dir, reason=str(exc))
    try:
        staging.replace(out_dir)
    except BaseException:
        if had_previous and backup.exists() and not out_dir.exists():
            backup.replace(out_dir)
        raise
    warnings: list[str] = []
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            warnings.append(f"could not remove prior report backup '{backup}': {exc}")
    return warnings


def _publish_in_place(staging: Path, out_dir: Path, reason: str) -> list[str]:
    """Copy the staged site over the previous report and drop obsolete files.

    The fallback for a report directory that cannot be renamed away: each
    staged file overwrites its previous copy, files the new report no longer
    contains are removed, and a file another program keeps locked is reported
    as a warning instead of failing the run.
    """
    warnings = [
        f"could not rename the previous report directory ({reason}); "
        "updated its files in place"
    ]
    keep: set[Path] = set()
    for src in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = src.relative_to(staging)
        keep.add(rel)
        dst = out_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        except OSError as exc:
            warnings.append(f"could not update report file '{dst}': {exc}")
    for existing in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if existing.relative_to(out_dir) in keep:
            continue
        try:
            existing.unlink()
        except OSError as exc:
            warnings.append(f"could not remove obsolete report file '{existing}': {exc}")
    # Children sort after their parents, so reverse order removes a directory
    # only after obsolete files have left it empty.
    for directory in sorted((p for p in out_dir.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    shutil.rmtree(staging, ignore_errors=True)
    return warnings
