"""Shared extraction helpers (ported from ``../vox-deorum-analysis/extract/utilities.py``).

Two departures from the source, per ``plans/stage1.md``:

1. **No hardcoded roots.** DB discovery takes an explicit ``root_dir`` (driven by
   ``data.extract.runs_dir``); nothing reaches into a ``shared`` package.
2. **Controlled-design metadata.** :func:`extract_seeding_fields` reads the
   ``configured*RandSeed`` / ``seating*`` keys the game runner now records in each
   DB's ``GameMetadata`` and reduces them to the lean ``seed`` / ``seating_rotation``
   / per-player ``config_slot`` facts (benchmark.md §3, §3.3, rule 14).
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from dataclasses import dataclass, field

from .errors import ExtractError


# ── domain constants (ported verbatim) ──────────────────────────────────────
POLICY_BRANCHES = [
    "tradition", "authority", "progress",      # Ancient era
    "fealty", "statecraft", "artistry",        # Classical/Medieval era
    "industry", "imperialism", "rationalism",  # Renaissance/Industrial era
    "freedom", "autocracy", "order",           # Modern era
]

STRATEGY_MAPPINGS = {
    "Conquest": "domination_ratio",
    "Culture": "culture_ratio",
    "UnitedNations": "diplomatic_ratio",
    "Spaceship": "science_ratio",
}

CHANGE_FIELDS = [
    "strategy_changes",
    "persona_changes",
    "research_changes",
    "policy_changes",
]

PLAYER_CORE_FIELDS = [
    "civilization",
    "score",
    "survival_turn",
    "score_ratio",
    "is_winner",
    "nuke",
    "use_nuke",
]

# Uncontrolled sentinel for `seed` / `seating_rotation` (rule 14). Controlled
# seeds are ≥ 1 (0 is Civ's "pick random"); rotations are ≥ 0.
UNCONTROLLED = -1


# ── controlled-design metadata (§3.3, rule 14) ──────────────────────────────
@dataclass
class SeedingInfo:
    """The lean controlled-design facts pulled from a game's ``GameMetadata``."""

    seed: int = UNCONTROLLED
    seating_rotation: int = UNCONTROLLED
    # player_id → config_slot, inverted from ``seatingMap``. Only the controlled
    # (treatment) seats appear here.
    config_slots: dict = field(default_factory=dict)

    def config_slot(self, player_id: int) -> int:
        # A seat absent from the seatingMap is not part of the controlled seating
        # (an in-game-AI opponent) → the uncontrolled sentinel, NOT its player_id,
        # so downstream can tell treatment seats from the field.
        return self.config_slots.get(player_id, UNCONTROLLED)

    @property
    def controlled(self) -> bool:
        return self.seed != UNCONTROLLED and self.seating_rotation != UNCONTROLLED


def is_decision_changes(changes_json) -> bool:
    """True when a ``Changes`` value records a strategist **decision**.

    The strategist writes a ``FlavorChanges``/``StrategyChanges`` row every turn it
    acts, listing the fields it touched. A status-quo turn — the agent chose to
    keep everything the same but still gave a rationale — is ``'["Rationale"]'``:
    no flavor numbers changed, yet it *is* a decision. Only a truly empty row
    (``'[]'``/null) is not a decision (no agent output, e.g. carry-forward).
    """
    if changes_json is None:
        return False
    return changes_json not in ("", "[]")


def _maybe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def invert_seating_map(seating_map_raw, where: str = "seatingMap") -> dict:
    """Invert ``{config_slot: player_id}`` to ``{player_id: config_slot}``.

    The runner records ``seatingMap`` as a JSON object keyed by config slot
    (string) whose value is the actual game player id. Downstream we want the
    inverse so a row keyed by ``player_id`` can recover its slot.
    """
    if seating_map_raw is None or seating_map_raw == "":
        return {}
    if isinstance(seating_map_raw, str):
        try:
            seating_map = json.loads(seating_map_raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ExtractError(f"{where}: not valid JSON ({exc}).")
    else:
        seating_map = seating_map_raw
    if not isinstance(seating_map, dict):
        raise ExtractError(f"{where}: expected an object, got {type(seating_map).__name__}.")

    inverted: dict = {}
    for slot_str, player_id in seating_map.items():
        pid = _maybe_int(player_id)
        slot = _maybe_int(slot_str)
        if pid is None or slot is None:
            raise ExtractError(f"{where}: non-integer entry {slot_str!r} → {player_id!r}.")
        inverted[pid] = slot
    return inverted


def extract_seeding_fields(metadata: dict, where: str = "game") -> SeedingInfo:
    """Reduce controlled-design ``GameMetadata`` to a :class:`SeedingInfo`.

    Policy (benchmark.md rule 14): Vox Deorum can record distinct sync/map seeds,
    but the controlled-design benchmark requires matched starts, so when both
    ``configuredSyncRandSeed`` and ``configuredMapRandSeed`` are present they must
    be equal — a mismatch **aborts extraction**. The agreed value becomes ``seed``
    (``-1`` when uncontrolled or ``0`` = Civ "pick random"). ``seatingRotation`` is
    read straight through (``-1`` when absent); ``seatingMap`` inverts to per-player
    ``config_slot``.
    """
    sync = _maybe_int(metadata.get("configuredSyncRandSeed"))
    map_ = _maybe_int(metadata.get("configuredMapRandSeed"))

    if sync is not None and map_ is not None and sync != map_:
        raise ExtractError(
            f"{where}: controlled run has mismatched seeds "
            f"configuredSyncRandSeed={sync} != configuredMapRandSeed={map_}; "
            f"civ-bench's controlled design requires matched starts."
        )

    configured = sync if sync is not None else map_
    seed = configured if (configured is not None and configured >= 1) else UNCONTROLLED

    rotation = _maybe_int(metadata.get("seatingRotation"))
    seating_rotation = rotation if (rotation is not None and rotation >= 0) else UNCONTROLLED

    config_slots = invert_seating_map(metadata.get("seatingMap"), f"{where}.seatingMap")
    return SeedingInfo(seed=seed, seating_rotation=seating_rotation, config_slots=config_slots)


# ── DB discovery (ported) ────────────────────────────────────────────────────
def find_all_databases(root_dir):
    """Find game ``*.db`` files under ``root_dir``, keeping the latest per game.

    Game files follow ``{uuid}_{timestamp_ms}.db``; player-trace exports
    (``*-player-*.db``) and non-game files are skipped. Returns
    ``(db_files, game_ids)``.
    """
    best_per_game = {}
    duplicates_found = 0

    for root, _dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith(".db"):
                continue
            if "-player-" in file:
                continue

            parts = file[:-3].split("_")
            if len(parts) < 2:
                continue

            game_id = parts[0]
            try:
                timestamp = int(parts[1])
            except (ValueError, IndexError):
                continue

            full_path = os.path.join(root, file)
            if game_id in best_per_game:
                duplicates_found += 1
                if timestamp > best_per_game[game_id][0]:
                    best_per_game[game_id] = (timestamp, full_path)
            else:
                best_per_game[game_id] = (timestamp, full_path)

    if duplicates_found > 0:
        print(f"  Note: Found {duplicates_found} duplicate database(s), keeping latest per game")

    db_files = [path for _, path in best_per_game.values()]
    game_ids = set(best_per_game.keys())
    return db_files, game_ids


def get_player_info_cache(cursor):
    """Cache ``player_id → {civilization, is_major}`` from ``PlayerInformations``."""
    cursor.execute("SELECT Key, Civilization, IsMajor FROM PlayerInformations")
    player_info = {}
    for player_id, civilization, is_major in cursor.fetchall():
        player_info[player_id] = {
            "civilization": civilization if civilization else "N/A",
            "is_major": bool(is_major),
        }
    return player_info


def get_major_players(cursor):
    """Return the sorted list of major-civilization player ids."""
    cursor.execute("SELECT Key FROM PlayerInformations WHERE IsMajor = 1 ORDER BY Key")
    return [row[0] for row in cursor.fetchall()]


def read_game_metadata(cursor) -> dict:
    """Read the ``GameMetadata`` Key→Value table into a dict."""
    cursor.execute("SELECT Key, Value FROM GameMetadata")
    return dict(cursor.fetchall())


# ── existing-CSV reconciliation (ported) ─────────────────────────────────────
def read_existing_csv(filepath, expected_fields):
    """Read an existing CSV, validating its header against ``expected_fields``.

    Returns ``(rows, game_ids, structure_matches)``; a header mismatch returns
    empty data with ``structure_matches=False`` so the caller does a full rewrite.
    """
    if not os.path.exists(filepath):
        return [], set(), True

    existing_data = []
    existing_game_ids = set()

    with open(filepath, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        existing_fieldnames = reader.fieldnames

        if existing_fieldnames != expected_fields:
            print(f"WARNING: Column mismatch in {filepath}")
            print(f"  Expected {len(expected_fields)} columns")
            print(f"  Found {len(existing_fieldnames or [])} columns")
            missing = set(expected_fields) - set(existing_fieldnames or [])
            extra = set(existing_fieldnames or []) - set(expected_fields)
            if missing:
                print(f"  Missing columns: {missing}")
            if extra:
                print(f"  Extra columns: {extra}")
            return [], set(), False

        for row in reader:
            existing_data.append(row)
            game_id = row.get("game_id")
            if game_id and game_id != "N/A":
                existing_game_ids.add(game_id)

    return existing_data, existing_game_ids, True


def filter_existing_data(existing_data, available_game_ids):
    """Drop existing rows whose game no longer has a DB file.

    Returns ``(filtered, kept_ids, pruned_rows, pruned_ids)``.
    """
    filtered_data = []
    kept_game_ids = set()
    pruned_rows = 0
    pruned_game_ids = set()

    for row in existing_data:
        game_id = row.get("game_id")
        if game_id and game_id != "N/A":
            if game_id in available_game_ids:
                filtered_data.append(row)
                kept_game_ids.add(game_id)
            else:
                pruned_rows += 1
                pruned_game_ids.add(game_id)
        else:
            filtered_data.append(row)

    return filtered_data, kept_game_ids, pruned_rows, pruned_game_ids


def should_skip_game(game_id, existing_game_ids):
    """True if ``game_id`` has already been extracted."""
    return game_id in existing_game_ids


def get_game_id_from_path(db_path):
    """Extract the game uuid (first ``_``-segment) from a DB filename."""
    db_filename = os.path.basename(db_path)
    if db_filename.endswith(".db"):
        parts = db_filename[:-3].split("_")
        if len(parts) >= 1:
            return parts[0]
    return None


def get_timestamp_from_path(db_path):
    """Extract the millisecond timestamp (second ``_``-segment) from a filename."""
    db_filename = os.path.basename(db_path)
    if db_filename.endswith(".db"):
        parts = db_filename[:-3].split("_")
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except (ValueError, IndexError):
                return None
    return None


def get_experiment_from_path(db_path):
    """Experiment name = the DB file's parent folder name."""
    return os.path.basename(os.path.dirname(db_path))


def open_database_readonly(db_path):
    """Open a SQLite DB read-only + immutable; ``(conn, cursor)`` or ``(None, None)``."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
        cursor = conn.cursor()
        return conn, cursor
    except Exception as exc:
        print(f"Error opening {os.path.basename(db_path)}: {exc}")
        return None, None


def write_csv_file(filepath, fieldnames, data_rows):
    """Write ``data_rows`` to ``filepath`` with a header (full rewrite)."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
        return True
    except Exception as exc:
        print(f"Error writing to {filepath}: {exc}")
        return False


def append_csv_file(filepath, fieldnames, data_rows):
    """Append ``data_rows`` to ``filepath`` (writing a header only if new/empty)."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        write_header = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
        with open(filepath, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(data_rows)
        return True
    except Exception as exc:
        print(f"Error appending to {filepath}: {exc}")
        return False


# ── skip-if-newer (§3) ───────────────────────────────────────────────────────
def newest_mtime(paths) -> float | None:
    """Newest modification time over ``paths`` (missing paths ignored)."""
    times = [os.path.getmtime(p) for p in paths if os.path.exists(p)]
    return max(times) if times else None


def outputs_are_fresh(output_paths, db_files) -> bool:
    """True when every output CSV exists and is newer than every source DB.

    With no DBs the outputs are trivially "fresh" (nothing to (re)build). A
    missing output forces a (re)build.
    """
    if not output_paths:
        return True
    for path in output_paths:
        if not os.path.exists(path):
            return False
    newest_db = newest_mtime(db_files)
    if newest_db is None:
        return True
    oldest_output = min(os.path.getmtime(p) for p in output_paths)
    return oldest_output >= newest_db
