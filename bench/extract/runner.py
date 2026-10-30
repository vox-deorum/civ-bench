"""Extract-stage orchestrator: raw game DBs under ``runs_dir`` → canonical CSVs.

Wires ``data.extract`` / ``data.tables`` (benchmark.md §3) to the seven exporters,
applies the **skip-if-newer** shortcut, and threads a single :class:`Catalog` (for
the orthodox ``player_type`` composition) through panel / turn / token extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bench.catalog import Catalog
from bench.config import RunConfig
from bench.config.behavior import resolve_behavior_spec
from bench.config.schema import DEFAULT_RUNS_DIR
from bench.extract.errors import ExtractError
from bench.extract.extract_agentic import AGENTIC_FIELDNAMES, export_agentic_data
from bench.extract.extract_behavior import behavior_fieldnames, export_behavior_data
from bench.extract.extract_games import GAME_FIELDNAMES, export_game_data
from bench.extract.extract_model_tokens import MODEL_TOKEN_FIELDNAMES, export_model_token_data
from bench.extract.extract_panel import PANEL_FIELD_MAPPINGS, export_panel_data
from bench.extract.extract_turns import TURN_FIELD_MAPPINGS, export_turn_data
from bench.extract.extract_world import WORLD_FIELDNAMES, export_world_data
from bench.extract.issues import DEFAULT_ISSUES_PATH, ImportIssueLog
from bench.extract.prune import PruneGate, read_game_rows
from bench.extract.sources import scan_sources
from bench.extract.tiered import TieredExtraction
from bench.extract.utilities import csv_header_matches, outputs_are_fresh


# Canonical table key → default CSV path (used when `data.tables` omits a key).
DEFAULT_TABLE_PATHS = {
    "turns": "runs/turn_data.csv",
    "panel": "runs/panel_data.csv",
    "games": "runs/game_data.csv",
    "tokens": "runs/model_token_usage.csv",
    "behavior": "runs/behavior_data.csv",
    "world": "runs/world_data.csv",
    "agentic": "runs/agentic_data.csv",
}


def expected_fieldnames(key: str, behavior_spec: dict) -> list[str]:
    """The CSV header the ``key`` exporter writes (and accepts when pruning)."""
    if key == "games":
        return GAME_FIELDNAMES
    if key == "panel":
        return list(PANEL_FIELD_MAPPINGS)
    if key == "turns":
        return list(TURN_FIELD_MAPPINGS)
    if key == "tokens":
        return MODEL_TOKEN_FIELDNAMES
    if key == "world":
        return WORLD_FIELDNAMES
    if key == "agentic":
        return AGENTIC_FIELDNAMES
    return behavior_fieldnames(behavior_spec)


@dataclass
class ExtractResult:
    skipped: bool = False
    reason: str = ""
    new_rows: dict = field(default_factory=dict)
    output_paths: dict = field(default_factory=dict)
    issues: ImportIssueLog = field(default_factory=ImportIssueLog)
    issues_path: str = ""
    # Games with no DB file whose rows were dropped, or kept because pruning was
    # declined (or, on a skipped run, never reviewed).
    pruned_game_ids: set = field(default_factory=set)
    kept_missing_game_ids: set = field(default_factory=set)
    # Cached games without a selected source that may have copies in offline cold
    # storage, so they are kept and never offered for pruning.
    protected_game_ids: set = field(default_factory=set)


def table_path(cfg: RunConfig, key: str) -> str:
    """Canonical CSV path for ``key``: ``data.tables`` or the extract default."""
    tables = cfg.data.get("tables", {}) or {}
    return tables.get(key) or DEFAULT_TABLE_PATHS[key]


def run_extract(
    cfg: RunConfig,
    catalog: Optional[Catalog] = None,
    force_rebuild: bool = False,
    prune_missing: Optional[bool] = None,
    prune_gate: Optional[PruneGate] = None,
) -> ExtractResult:
    """Run the extract stage for ``cfg`` and return a summary of what was written.

    ``force_rebuild`` (e.g. the CLI ``--force-rebuild`` flag) overrides the
    config's ``data.extract.force_rebuild`` when either is set. ``prune_missing``
    (``None`` = use the config value) lets a caller force a full re-import: the
    auto-fix re-import passes ``prune_missing=False`` so it actually re-inspects the
    repaired DBs instead of only pruning.

    Rows for games with no DB file are dropped only when ``prune_gate`` approves
    them. The CLI passes a gate that lists the games and asks first; without one,
    every missing game is pruned.
    """
    extract_cfg = cfg.data.get("extract", {}) or {}

    if not cfg.extract_enabled:
        return ExtractResult(
            skipped=True,
            reason="data.extract.enabled is false; loaders read data.tables directly.",
        )

    runs_dir = extract_cfg.get("runs_dir", DEFAULT_RUNS_DIR)
    outputs = list(extract_cfg.get("outputs", list(DEFAULT_TABLE_PATHS)))
    max_dbs = extract_cfg.get("max_dbs")
    if prune_missing is None:
        prune_only = bool(extract_cfg.get("prune_missing", False))
    else:
        prune_only = bool(prune_missing)
    force_rebuild = force_rebuild or bool(extract_cfg.get("force_rebuild", False))
    issues_path = extract_cfg.get("issues_path", DEFAULT_ISSUES_PATH)
    behavior_spec = resolve_behavior_spec(extract_cfg.get("behavior"))

    output_paths = {key: table_path(cfg, key) for key in outputs}
    checkpoint_paths = {**output_paths, "_issues": issues_path}

    print("=" * 60)
    print("civ-bench extract")
    print("=" * 60)
    print(f"runs_dir: {runs_dir}")
    print(f"outputs:  {outputs}")

    inventory = scan_sources(extract_cfg)
    db_files, available_game_ids = inventory.db_files, inventory.game_ids
    games_path = table_path(cfg, "games")
    cold = inventory.tiers.get("cold")
    tiered = TieredExtraction.load(inventory, games_path) if cold else None
    if cold:
        print(f"cold_runs_dir: {cold.root} ({'available' if cold.available else 'unavailable'})")
    duplicates = sum(scan.duplicates for scan in inventory.tiers.values())
    if duplicates:
        print(f"  Note: Found {duplicates} duplicate database(s), keeping latest per game")
    print(f"Found {len(db_files)} database files with {len(available_game_ids)} unique games")

    # game_data is small, so read it up front: it names the games that would be
    # pruned (with experiment and seed for the prompt) before any big table is read.
    game_rows = read_game_rows(games_path)
    protected_game_ids = set()
    if tiered:
        protected_game_ids, missing_game_ids = tiered.split_absent(set(game_rows))
    else:
        missing_game_ids = set(game_rows) - available_game_ids

    selected_db_files = db_files
    capped = max_dbs is not None and max_dbs < len(db_files)
    if max_dbs is not None:
        selected_db_files = db_files[: max_dbs]
        print(f"Limiting extraction to {len(selected_db_files)} database(s) via max_dbs={max_dbs}")

    # Skip-if-newer: every output exists and is newer than every source DB.
    # The mtime check can only attest "outputs written after the DBs", not
    # "outputs cover every DB", so it is unsafe whenever max_dbs caps the run
    # below the available DB count: those outputs are a subset, and skipping
    # would strand the remaining games permanently. Only trust the skip when
    # the run would process every discovered DB. The issues report is one of the
    # outputs: a missing/stale report forces a (re)build so it is never stranded
    # behind already-fresh tables. An output whose header no longer matches its
    # schema (a changed behavior selection, a dropped column) is stale too, since
    # mtimes cannot see a config change.
    freshness_paths = list(output_paths.values()) + [issues_path]
    headers_current = all(
        csv_header_matches(path, expected_fieldnames(key, behavior_spec))
        for key, path in output_paths.items()
    )
    if not force_rebuild and not prune_only and not capped and headers_current \
            and outputs_are_fresh(freshness_paths, db_files) \
            and (tiered is None or tiered.is_current(checkpoint_paths)):
        reason = ("all outputs exist and are newer than the source DBs "
                  "(pass force_rebuild to override).")
        if missing_game_ids:
            # Removing a DB does not make the outputs stale, so say how to drop its rows.
            reason += (f" {len(missing_game_ids)} game(s) in {games_path} have no "
                       f"database file; pass --prune-missing to review and remove them.")
        return ExtractResult(
            skipped=True,
            reason=reason,
            output_paths=output_paths,
            issues_path=issues_path,
            kept_missing_game_ids=missing_game_ids,
            protected_game_ids=protected_game_ids,
        )

    prune_gate = prune_gate or PruneGate()
    prune_gate.add_details(game_rows)
    removed = set()
    retained_game_ids = available_game_ids
    if tiered:
        # Removals cover every table on disk, so every cached id is needed now.
        all_paths = {key: table_path(cfg, key) for key in DEFAULT_TABLE_PATHS}
        tiered.read_tables(all_paths, issues_path)
        protected_game_ids = tiered.protected
        if protected_game_ids:
            print(f"Protecting {len(protected_game_ids)} cached game(s) while cold storage is unavailable")
        prune_gate.add_details(read_game_rows(issues_path))
        prune_gate.add_details({
            gid: {"game_id": gid, "experiment": entry["experiment"]}
            for gid, entry in tiered.state.games.items()
        })
        removed = tiered.review(prune_gate)
        fields = {key: expected_fieldnames(key, behavior_spec) for key in DEFAULT_TABLE_PATHS}
        tiered.preflight(output_paths, all_paths, fields, selected_db_files, prune_only, removed)
        tiered.journal(removed, all_paths, checkpoint_paths)
        tiered.prune(removed, all_paths)
        retained_game_ids = available_game_ids | tiered.protected | (tiered.missing - removed)
    elif "games" in outputs and missing_game_ids:
        # Ask about every game the games table already knows is missing in one go;
        # the other tables only prompt for games this list did not cover.
        prune_gate.review(missing_game_ids)

    catalog = catalog or Catalog.from_run_config(cfg)

    # One issue log threaded through every exporter; malformed-DB failures are
    # recorded here (deduped by game_id). It is seeded from the existing report so
    # the run reconciles rather than clobbers: an issue on a game skipped this run
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
            new_rows[key] = export_game_data(selected_db_files, retained_game_ids, path, prune_only=prune_only, issues=issues, prune_gate=prune_gate)
        elif key == "panel":
            new_rows[key] = export_panel_data(selected_db_files, retained_game_ids, path, catalog=catalog, prune_only=prune_only, issues=issues, prune_gate=prune_gate)
        elif key == "turns":
            new_rows[key] = export_turn_data(selected_db_files, retained_game_ids, path, catalog=catalog, prune_only=prune_only, issues=issues, prune_gate=prune_gate)
        elif key == "tokens":
            new_rows[key] = export_model_token_data(selected_db_files, retained_game_ids, path, catalog, prune_only=prune_only, issues=issues, prune_gate=prune_gate)
        elif key == "behavior":
            new_rows[key] = export_behavior_data(selected_db_files, retained_game_ids, path, behavior_spec, catalog=catalog, prune_only=prune_only, issues=issues, prune_gate=prune_gate)
        elif key == "world":
            new_rows[key] = export_world_data(selected_db_files, retained_game_ids, path, prune_only=prune_only, issues=issues, prune_gate=prune_gate)
        elif key == "agentic":
            new_rows[key] = export_agentic_data(selected_db_files, retained_game_ids, path, catalog, prune_only=prune_only, issues=issues, prune_gate=prune_gate)

    # Reconcile this run's findings with the prior report (carry forward issues for
    # games no stage re-examined; drop those whose DB is gone) and persist. This
    # also gives prune-only correct behavior: it inspects no DBs, so it simply drops
    # issues for removed games and keeps the rest, never clobbering with a clean
    # header. Fail loud if persistence fails (else the CLI would claim issues were
    # "recorded" when nothing was written). Declined games keep their rows, so they
    # keep their issues too.
    issues.reconcile(retained_game_ids | prune_gate.declined)
    if not issues.write_csv(issues_path):
        raise ExtractError(f"failed to write import-issues report to {issues_path}")

    if tiered:
        tiered.finish(removed, checkpoint_paths, complete=not capped and not prune_only)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    for key in outputs:
        print(f"  {key}: {new_rows.get(key, 0)} new rows → {output_paths[key]}")
    if prune_gate.approved:
        print(f"  pruned {len(prune_gate.approved)} game(s) without database files")
    if prune_gate.declined:
        print(f"  kept {len(prune_gate.declined)} game(s) without database files (pruning declined)")

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
        pruned_game_ids=set(prune_gate.approved),
        kept_missing_game_ids=set(prune_gate.declined),
        protected_game_ids=set(protected_game_ids),
    )
