"""Map workspace paths onto the folders the agentic host offers.

| Folder | Access | Holds |
| --- | --- | --- |
| ``/workspace/records`` | read-only | per-turn reports (``turn-<N>/...``), ``game/``, ``CATALOG.md`` |
| ``/workspace/reference`` | read-only | rules: technologies, policies, buildings, units |
| ``/workspace/game`` | writable, kept | notes, plus the editable guide ``AGENTS.md`` |
| ``/tmp`` | writable, per game | scratch |

Writes anywhere else vanish after the command.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Optional, Set, Tuple

RECORDS = "/workspace/records"
REFERENCE = "/workspace/reference"
GAME = "/workspace/game"
GUIDE = "/workspace/game/AGENTS.md"
TMP = "/tmp"

# Record sections, by file stem inside ``turn-<N>/`` (plus the game-level files).
TURN_SECTIONS = (
    "events-all", "events-economic", "events-military", "events-diplomatic",
    "events-others", "options", "strategies", "victory", "players", "cities", "military",
)
GAME_SECTIONS = ("situation", "civilization")
RECORD_SECTIONS = ("catalog",) + GAME_SECTIONS + TURN_SECTIONS + ("other",)
REFERENCE_CATEGORIES = ("technologies", "policies", "buildings", "units", "other")
SOURCES = ("records", "reference", "notes", "guide", "tmp", "other")

_TURN = re.compile(r"^turn-(.+)$")


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def source_of(path: str) -> Optional[str]:
    """The workspace area a path belongs to, ``"outside"``, or ``None`` for ``/dev``."""
    if _within(path, "/dev"):
        return None
    if _within(path, RECORDS):
        return "records"
    if _within(path, REFERENCE):
        return "reference"
    if path == GUIDE:
        return "guide"
    if _within(path, GAME):
        return "notes"
    if _within(path, TMP):
        return "tmp"
    if _within(path, "/workspace"):
        return "other"
    return "outside"


def _stem(name: str) -> str:
    for suffix in (".md", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def record_sections(path: str) -> Tuple[Set[str], Optional[int]]:
    """Record sections a path names and the turn folder it is in (if numeric).

    A folder alone (``turn-12/``) names no section. Globs such as
    ``turn-*/events-*.md`` name every section they match and no single turn.
    """
    rel = path[len(RECORDS):].lstrip("/")
    if not rel:
        return set(), None
    parts = rel.split("/")
    head = parts[0]
    if head == "CATALOG.md":
        return {"catalog"}, None
    if head == "game":
        if len(parts) == 1:
            return set(), None
        stem = _stem(parts[1])
        matches = {s for s in GAME_SECTIONS if fnmatch.fnmatchcase(s, stem)}
        return (matches or {"other"}), None
    match = _TURN.match(head)
    if not match:
        return {"other"}, None
    turn = int(match.group(1)) if match.group(1).isdigit() else None
    if len(parts) == 1:
        return set(), turn
    stem = _stem(parts[1])
    matches = {s for s in TURN_SECTIONS if fnmatch.fnmatchcase(s, stem)}
    return (matches or {"other"}), turn


def reference_category(path: str) -> Optional[str]:
    """``technologies``/``policies``/``buildings``/``units``/``other``, or ``None`` for the root."""
    rel = path[len(REFERENCE):].lstrip("/")
    if not rel:
        return None
    head = rel.split("/")[0]
    head = _stem(head)
    return head if head in REFERENCE_CATEGORIES[:-1] else "other"


def write_placement(path: str, outcome: str) -> Optional[str]:
    """Where a write lands: ``persisted``, ``scratch``, ``lost``, or ``rejected``.

    Writes to the read-only folders are rejected. Writes to ``/workspace/game`` are
    kept when the command succeeded and rejected otherwise. Writes to ``/tmp`` last only for the game.
    Writes anywhere else, including relative paths, vanish after the command.
    """
    source = source_of(path) if path.startswith("/") else "outside"
    if source is None:
        return None
    if source in ("records", "reference"):
        return "rejected"
    if source in ("notes", "guide"):
        return "persisted" if outcome in ("ok", "no_match") else "rejected"
    if source == "tmp":
        return "scratch"
    return "lost"
