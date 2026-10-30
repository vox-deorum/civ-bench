"""Complete source scans and durable ownership for optional cold storage."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from bench.extract.errors import ExtractError
from bench.config.errors import ConfigError
from bench.config.validation import resolve_source_roots


@dataclass
class TierScan:
    root: str
    available: bool
    files: list[str] = field(default_factory=list)
    databases: dict[str, str] = field(default_factory=dict)
    duplicates: int = 0
    error: str = ""


@dataclass
class Inventory:
    """Scans of each source tier. The views are cached, so treat scans as frozen."""

    tiers: dict[str, TierScan]

    @cached_property
    def selected(self) -> dict[str, str]:
        # Assign hot last so it wins independently of the snapshot timestamp.
        selected = {}
        for name in ("cold", "hot"):
            if name in self.tiers:
                selected.update(self.tiers[name].databases)
        return selected

    @cached_property
    def game_ids(self) -> set[str]:
        return set(self.selected)

    @cached_property
    def db_files(self) -> list[str]:
        hot = self.tiers["hot"].databases
        return sorted(hot.values()) + sorted(
            path for gid, path in self.selected.items() if gid not in hot
        )

    @cached_property
    def signature(self) -> str:
        payload = {
            name: {"available": scan.available, "databases": scan.databases}
            for name, scan in self.tiers.items()
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _scan(root: str) -> TierScan:
    """Keep the newest ``{uuid}_{timestamp}.db`` per game; any walk error fails the scan."""
    scan = TierScan(str(root), False)
    best: dict[str, tuple[int, str]] = {}

    def failed(exc: OSError) -> None:
        raise exc

    try:
        if not os.path.isdir(scan.root):
            raise OSError(f"not an accessible directory: {scan.root}")
        for directory, dirs, files in os.walk(scan.root, onerror=failed):
            dirs.sort()
            for name in sorted(files):
                path = os.path.join(directory, name)
                scan.files.append(path)
                if not name.endswith(".db") or "-player-" in name:
                    continue
                parts = name[:-3].split("_")
                if len(parts) < 2:
                    continue
                try:
                    timestamp = int(parts[1])
                except ValueError:
                    continue
                candidate = (timestamp, path)
                if parts[0] in best:
                    scan.duplicates += 1
                if candidate > best.get(parts[0], (-1, "")):
                    best[parts[0]] = candidate
        scan.available = True
        scan.databases = {gid: path for gid, (_, path) in sorted(best.items())}
    except OSError as exc:
        scan = TierScan(scan.root, False, error=str(exc))
    return scan


def scan_sources(extract_cfg: dict, require_hot: bool = True) -> Inventory:
    """Scan every configured root. A failed scan keeps no partial inventory.

    An incomplete hot scan is an error unless ``require_hot`` is false, in which
    case the hot tier is reported as unavailable with no databases.
    """
    try:
        roots = resolve_source_roots(extract_cfg)
    except ConfigError as exc:
        raise ExtractError(str(exc)) from exc
    tiers = {name: _scan(str(root)) for name, root in roots.items()}
    if require_hot and not tiers["hot"].available:
        raise ExtractError(f"cannot scan runs_dir: {tiers['hot'].error}")
    return Inventory(tiers)


def atomic_json(path: str, data: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, sort_keys=True, indent=2)
            fh.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass
class SourceState:
    path: str
    roots: dict
    games: dict = field(default_factory=dict)
    checkpoints: dict = field(default_factory=dict)
    pending_removals: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, games_path: str, inventory: Inventory) -> SourceState:
        roots = {name: scan.root for name, scan in inventory.tiers.items()}
        state = cls(str(games_path) + ".sources.json", roots)
        if not Path(state.path).exists():
            return state
        try:
            data = json.loads(Path(state.path).read_text(encoding="utf-8"))
            if data["version"] != 1 or not isinstance(data["roots"], dict):
                raise ValueError("unsupported version or invalid roots")
            games, checkpoints, pending = data["games"], data["checkpoints"], data["pending_removals"]
            if not isinstance(games, dict) or not isinstance(checkpoints, dict) or not isinstance(pending, list):
                raise ValueError("invalid ownership, checkpoints or removals")
            for gid, entry in games.items():
                if (not isinstance(gid, str) or not isinstance(entry, dict)
                        or not isinstance(entry.get("experiment"), str)
                        or not isinstance(entry.get("tiers"), list)
                        or not entry["tiers"] or any(t not in {"hot", "cold"} for t in entry["tiers"])):
                    raise ValueError("invalid game ownership")
            if any(not isinstance(k, str) or not isinstance(v, str) for k, v in checkpoints.items()):
                raise ValueError("invalid checkpoints")
            if any(not isinstance(gid, str) for gid in pending):
                raise ValueError("invalid pending removals")
            if data["roots"] == roots:
                state.games = games
                state.checkpoints = checkpoints
                state.pending_removals = set(pending)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ExtractError(f"cannot read source ownership index {state.path}: {exc}") from exc
        return state

    def unavailable_copies(self, game_ids: set[str], inventory: Inventory) -> set[str]:
        """Games that may have copies in an unavailable tier, even if hot exists."""
        offline = {name for name, scan in inventory.tiers.items() if not scan.available}
        if not offline:
            return set()
        return {
            gid for gid in game_ids
            if gid not in self.games or offline.intersection(self.games[gid]["tiers"])
        }

    def protected(self, game_ids: set[str], inventory: Inventory) -> set[str]:
        return self.unavailable_copies(game_ids, inventory) - inventory.game_ids

    def observe(self, inventory: Inventory, unknown_cached_ids: set[str] | None = None) -> None:
        unknown_cached_ids = unknown_cached_ids or set()
        offline = any(not scan.available for scan in inventory.tiers.values())
        for gid, path in inventory.selected.items():
            # Seeing an existing cached game in hot cannot establish that it has
            # no cold copy while cold is offline. Keep that ownership unknown.
            if offline and gid in unknown_cached_ids:
                continue
            owners = {name for name, scan in inventory.tiers.items() if gid in scan.databases}
            old = self.games.get(gid, {})
            owners.update(name for name in old.get("tiers", []) if not inventory.tiers[name].available)
            self.games[gid] = {"experiment": Path(path).parent.name, "tiers": sorted(owners)}
        self.pending_removals.difference_update(inventory.game_ids)

    def save(self) -> None:
        atomic_json(self.path, {
            "version": 1, "roots": self.roots, "games": self.games,
            "checkpoints": self.checkpoints, "pending_removals": sorted(self.pending_removals),
        })


def csv_game_ids(path: str) -> set[str]:
    """Read IDs even when an old table's other columns no longer match."""
    if not Path(path).exists():
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        if "game_id" not in header:
            raise ExtractError(f"cannot identify cached games in {path}: missing game_id header")
        # Only the one column is needed, so skip building a dict per row.
        column = header.index("game_id")
        ids = {row[column] for row in reader if len(row) > column}
    ids.discard("")
    ids.discard("N/A")
    return ids


def prune_csv(path: str, removed: set[str]) -> None:
    """Atomically drop approved IDs, keeping the original header and row order."""
    if not removed or not Path(path).exists():
        return
    destination = Path(path)
    temporary = None
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            if not fields or "game_id" not in fields:
                raise ExtractError(f"cannot prune {path}: missing game_id header")
            fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                for row in reader:
                    if row.get("game_id") not in removed:
                        writer.writerow(row)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
