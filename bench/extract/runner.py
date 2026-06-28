"""Extract-stage orchestrator: raw game DBs under ``runs_dir`` → canonical CSVs.

Wires ``data.extract`` / ``data.tables`` (benchmark.md §3) to the four exporters,
applies the **skip-if-newer** shortcut, and threads a single :class:`Catalog` (for
the orthodox ``player_type`` composition) through panel / turn / token extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..catalog import Catalog
from ..config import RunConfig
from .errors import ExtractError
from .extract_games import export_game_data
from .extract_model_tokens import export_model_token_data
from .extract_panel import export_panel_data
from .extract_turns import export_turn_data
from .issues import DEFAULT_ISSUES_PATH, ImportIssueLog
from .utilities import find_all_databases, outputs_are_fresh


# Canonical table key → default CSV path (used when `data.tables` omits a key).
DEFAULT_TABLE_PATHS = {
    "turns": "runs/turn_data.csv",
    "panel": "runs/panel_data.csv",
    "games": "runs/game_data.csv",
    "tokens": "runs/model_token_usage.csv",
}


@dataclass
class ExtractResult:
    skipped: bool = False
    reason: str = ""
    new_rows: dict = field(default_factory=dict)
    output_paths: dict = field(default_factory=dict)
    issues: ImportIssueLog = field(default_factory=ImportIssueLog)
    issues_path: str = ""


def _table_path(cfg: RunConfig, key: str) -> str:
    tables = cfg.data.get("tables", {}) or {}
    return tables.get(key) or DEFAULT_TABLE_PATHS[key]


def run_extract(
    cfg: RunConfig,
    catalog: Optional[Catalog] = None,
    force_rebuild: bool = False,
) -> ExtractResult:
    """Run the extract stage for ``cfg`` and return a summary of what was written.

    ``force_rebuild`` (e.g. the CLI ``--force-rebuild`` flag) overrides the
    config's ``data.extract.force_rebuild`` when either is set.
    """
    extract_cfg = cfg.data.get("extract", {}) or {}

    if not cfg.extract_enabled:
        return ExtractResult(
            skipped=True,
            reason="data.extract.enabled is false; loaders read data.tables directly.",
        )

    runs_dir = extract_cfg.get("runs_dir", "runs/")
    outputs = list(extract_cfg.get("outputs", list(DEFAULT_TABLE_PATHS)))
    max_dbs = extract_cfg.get("max_dbs")
    prune_only = bool(extract_cfg.get("prune_missing", False))
    force_rebuild = force_rebuild or bool(extract_cfg.get("force_rebuild", False))
    issues_path = extract_cfg.get("issues_path", DEFAULT_ISSUES_PATH)

    output_paths = {key: _table_path(cfg, key) for key in outputs}

    print("=" * 60)
    print("civ-bench extract")
    print("=" * 60)
    print(f"runs_dir: {runs_dir}")
    print(f"outputs:  {outputs}")

    db_files, available_game_ids = find_all_databases(runs_dir)
    db_files = sorted(db_files)
    print(f"Found {len(db_files)} database files with {len(available_game_ids)} unique games")

    selected_db_files = db_files
    capped = max_dbs is not None and max_dbs < len(db_files)
    if max_dbs is not None:
        selected_db_files = db_files[: max_dbs]
        print(f"Limiting extraction to {len(selected_db_files)} database(s) via max_dbs={max_dbs}")

    # Skip-if-newer: every output exists and is newer than every source DB.
    # The mtime check can only attest "outputs written after the DBs", not
    # "outputs cover every DB", so it is unsafe whenever max_dbs caps the run
    # below the available DB count — those outputs are a subset and skipping
    # would strand the remaining games permanently. Only trust the skip when
    # the run would process every discovered DB. The issues report is one of the
    # outputs: a missing/stale report forces a (re)build so it is never stranded
    # behind already-fresh tables.
    freshness_paths = list(output_paths.values()) + [issues_path]
    if not force_rebuild and not prune_only and not capped \
            and outputs_are_fresh(freshness_paths, db_files):
        return ExtractResult(
            skipped=True,
            reason="all outputs exist and are newer than the source DBs "
                   "(pass force_rebuild to override).",
            output_paths=output_paths,
            issues_path=issues_path,
        )

    catalog = catalog or Catalog.from_run_config(cfg)

    # One issue log threaded through every exporter; malformed-DB failures are
    # recorded here (deduped by game_id). It is seeded from the existing report so
    # the run reconciles rather than clobbers — an issue on a game skipped this run
    # (e.g. it still produced some rows) is carried forward, not lost.
    issues = ImportIssueLog()
    issues.load(issues_path)

    new_rows: dict = {}
    for key in outputs:
        path = output_paths[key]
        print("\n" + "-" * 60)
        print(f"EXTRACTING {key} → {path}")
        print("-" * 60)
        if key == "games":
            new_rows[key] = export_game_data(selected_db_files, available_game_ids, path, prune_only=prune_only, issues=issues)
        elif key == "panel":
            new_rows[key] = export_panel_data(selected_db_files, available_game_ids, path, catalog=catalog, prune_only=prune_only, issues=issues)
        elif key == "turns":
            new_rows[key] = export_turn_data(selected_db_files, available_game_ids, path, catalog=catalog, prune_only=prune_only, issues=issues)
        elif key == "tokens":
            new_rows[key] = export_model_token_data(selected_db_files, available_game_ids, path, catalog, prune_only=prune_only, issues=issues)

    # Reconcile this run's findings with the prior report (carry forward issues for
    # games no stage re-examined; drop those whose DB is gone) and persist. This
    # also gives prune-only correct behavior: it inspects no DBs, so it simply drops
    # issues for removed games and keeps the rest — never clobbering with a clean
    # header. Fail loud if persistence fails (else the CLI would claim issues were
    # "recorded" when nothing was written).
    issues.reconcile(available_game_ids)
    if not issues.write_csv(issues_path):
        raise ExtractError(f"failed to write import-issues report to {issues_path}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    for key in outputs:
        print(f"  {key}: {new_rows.get(key, 0)} new rows → {output_paths[key]}")

    if issues:
        print(f"\nPROBLEM DATABASES ({len(issues)}) → {issues_path}")
        for line in issues.summary_lines():
            print(line)

    return ExtractResult(
        skipped=False,
        new_rows=new_rows,
        output_paths=output_paths,
        issues=issues,
        issues_path=issues_path,
    )
