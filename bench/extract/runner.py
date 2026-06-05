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
from .extract_games import export_game_data
from .extract_model_tokens import export_model_token_data
from .extract_panel import export_panel_data
from .extract_turns import export_turn_data
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


def _table_path(cfg: RunConfig, key: str) -> str:
    tables = cfg.data.get("tables", {}) or {}
    return tables.get(key) or DEFAULT_TABLE_PATHS[key]


def run_extract(cfg: RunConfig, catalog: Optional[Catalog] = None) -> ExtractResult:
    """Run the extract stage for ``cfg`` and return a summary of what was written."""
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
    force_rebuild = bool(extract_cfg.get("force_rebuild", False))

    output_paths = {key: _table_path(cfg, key) for key in outputs}

    print("=" * 60)
    print("civ-bench extract")
    print("=" * 60)
    print(f"runs_dir: {runs_dir}")
    print(f"outputs:  {outputs}")

    db_files, available_game_ids = find_all_databases(runs_dir)
    db_files = sorted(db_files)
    print(f"Found {len(db_files)} database files with {len(available_game_ids)} unique games")

    # Skip-if-newer: every output exists and is newer than every source DB.
    if not force_rebuild and not prune_only and outputs_are_fresh(list(output_paths.values()), db_files):
        return ExtractResult(
            skipped=True,
            reason="all outputs exist and are newer than the source DBs "
                   "(pass force_rebuild to override).",
            output_paths=output_paths,
        )

    selected_db_files = db_files
    if max_dbs is not None:
        selected_db_files = db_files[: max_dbs]
        print(f"Limiting extraction to {len(selected_db_files)} database(s) via max_dbs={max_dbs}")

    catalog = catalog or Catalog.from_run_config(cfg)

    new_rows: dict = {}
    for key in outputs:
        path = output_paths[key]
        print("\n" + "-" * 60)
        print(f"EXTRACTING {key} → {path}")
        print("-" * 60)
        if key == "games":
            new_rows[key] = export_game_data(selected_db_files, available_game_ids, path, prune_only=prune_only)
        elif key == "panel":
            new_rows[key] = export_panel_data(selected_db_files, available_game_ids, path, catalog=catalog, prune_only=prune_only)
        elif key == "turns":
            new_rows[key] = export_turn_data(selected_db_files, available_game_ids, path, catalog=catalog, prune_only=prune_only)
        elif key == "tokens":
            new_rows[key] = export_model_token_data(selected_db_files, available_game_ids, path, catalog, prune_only=prune_only)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    for key in outputs:
        print(f"  {key}: {new_rows.get(key, 0)} new rows → {output_paths[key]}")

    return ExtractResult(skipped=False, new_rows=new_rows, output_paths=output_paths)
