"""Structured collection of malformed-DB import issues (benchmark.md §3).

When a game's SQLite DB is corrupt, the four exporters can no longer pull its
rows. Rather than print an opaque per-player ``database disk image is malformed``
line that names neither the game nor its experiment/seed/rotation, each failure
site records an :class:`ImportIssue` here. The log dedups **by game_id** — a game
that fails in several stages becomes one entry whose ``stages`` set lists each —
so cross-stage records also recover ``seed``/``seating_rotation`` from whichever
stage *could* read the metadata.

Mirrors the ``warnings``-accumulation pattern already used by ``adjust``/``report``:
the orchestrator (:func:`bench.extract.runner.run_extract`) owns one log, threads
it through every exporter, prints a grouped summary, and persists it to a durable
``import_issues.csv``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .utilities import (
    extract_seeding_fields,
    get_experiment_from_path,
    get_game_id_from_path,
    write_csv_file,
)

# Durable record of malformed-DB import issues (overridable via ``data.extract``).
# Defined here (rather than in ``runner.py``) so both the extract orchestrator that
# *writes* it and the analysis layer that *reads* it to exclude problem games share
# one definition.
DEFAULT_ISSUES_PATH = "runs/import_issues.csv"

# ``seed``/``seating_rotation`` value used when the DB was too corrupt to read its
# ``GameMetadata`` — kept distinct from the legitimate ``-1`` UNCONTROLLED sentinel.
UNKNOWN = "unknown"

ISSUE_FIELDNAMES = [
    "game_id",
    "experiment",
    "seed",
    "seating_rotation",
    "stages",
    "players",
    "db_name",
    "message",
]


def resolve_issues_path(data: dict) -> str:
    """The effective import-issues report path for a run's ``data`` block.

    Falls back to :data:`DEFAULT_ISSUES_PATH` when ``data.extract.issues_path`` is
    unset (or ``data.extract`` itself is null/absent). Shared by the extract stage
    that *writes* the ledger and every adjust/analysis consumer that *reads* it to
    exclude problem games, so a configured override is honoured in one place.
    """
    extract = (data or {}).get("extract") or {}
    return extract.get("issues_path") or DEFAULT_ISSUES_PATH


def read_problem_game_ids(path: str) -> set[str]:
    """Return the set of ``game_id``s recorded in an ``import_issues.csv``.

    The downstream consumer of the malformed-DB log: analyses read this to drop
    problem games (whose stale, identity-less rows otherwise poison ratings) from
    their inputs. A missing/empty/unreadable report yields an empty set — never an
    error — so analyses still run when no issues were ever recorded.
    """
    if not path or not Path(path).exists():
        return set()
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return {
                gid
                for row in csv.DictReader(fh)
                if (gid := (row.get("game_id") or "").strip())
            }
    except Exception as exc:  # a corrupt report must not abort analyses
        print(f"WARNING: could not read import-issues report {path}: {exc}")
        return set()


@dataclass
class ImportIssue:
    """One problem game: where it came from, which stage(s)/player(s) it hit."""

    game_id: str
    experiment: str
    seed: int | str  # int (incl. -1 UNCONTROLLED) when readable, else ``UNKNOWN``
    seating_rotation: int | str
    db_name: str
    stages: set[str] = field(default_factory=set)
    players: set[int] = field(default_factory=set)
    message: str = ""

    def as_row(self) -> dict:
        """Flatten to a CSV row (``stages``/``players`` pipe-joined, sorted)."""
        return {
            "game_id": self.game_id,
            "experiment": self.experiment,
            "seed": self.seed,
            "seating_rotation": self.seating_rotation,
            "stages": "|".join(sorted(self.stages)),
            "players": "|".join(str(p) for p in sorted(self.players)),
            "db_name": self.db_name,
            "message": self.message,
        }


def _split_set(value) -> set:
    """Parse a pipe-joined CSV cell back to a set (empty cell → empty set)."""
    return {part for part in (value or "").split("|") if part}


def _coerce_seed(value) -> int | str:
    """Parse a persisted ``seed``/``seating_rotation`` cell back to int or UNKNOWN."""
    if value is None or value == "" or value == UNKNOWN:
        return UNKNOWN
    try:
        return int(value)
    except (ValueError, TypeError):
        return UNKNOWN


def _best_effort_seeding(metadata) -> tuple:
    """``(seed, seating_rotation)`` from metadata, or ``(UNKNOWN, UNKNOWN)``.

    ``extract_seeding_fields`` can itself raise (an ``ExtractError`` on a
    controlled-seed mismatch, or anything if ``metadata`` is junk) — for issue
    reporting we only want a best-effort read, never a second failure.
    """
    if not metadata:
        return UNKNOWN, UNKNOWN
    try:
        seeding = extract_seeding_fields(metadata)
        return seeding.seed, seeding.seating_rotation
    except Exception:
        return UNKNOWN, UNKNOWN


class ImportIssueLog:
    """Dedup-by-game collector of :class:`ImportIssue` records.

    The log is reconciled against the prior report rather than clobbering it: it
    carries an existing report's issues forward except where a stage *re-examined*
    the game this run (then the fresh verdict wins) or the game's DB is gone. This
    keeps issues durable across incremental runs — e.g. a corrupt player-trace on a
    game that still produced token rows is skipped next run, so without carry-over
    its issue would silently vanish.
    """

    def __init__(self) -> None:
        self._by_game: dict = {}        # issues found *this run* (game_id → ImportIssue)
        self._prior: dict = {}          # issues loaded from the existing report
        self._evaluated: dict = {}      # game_id → {stage} re-examined this run
        self._fresh: set = set()        # game_ids RECORDED this run (never carried-forward)

    def load(self, path: str) -> None:
        """Seed the prior-report state from an existing ``import_issues.csv``.

        A missing/empty/unreadable report is not fatal — extraction must still run;
        we just start with no history (and re-discover anything still failing).
        """
        if not Path(path).exists():
            return
        try:
            with open(path, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    gid = (row.get("game_id") or "").strip()
                    if not gid:
                        continue
                    self._prior[gid] = ImportIssue(
                        game_id=gid,
                        experiment=row.get("experiment", "") or "",
                        seed=_coerce_seed(row.get("seed")),
                        seating_rotation=_coerce_seed(row.get("seating_rotation")),
                        db_name=row.get("db_name", "") or "",
                        stages=_split_set(row.get("stages")),
                        players={int(p) for p in _split_set(row.get("players")) if p.isdigit()},
                        message=row.get("message", "") or "",
                    )
        except Exception as exc:  # a corrupt report must not abort extraction
            print(f"WARNING: could not read existing import-issues report {path}: {exc}")
            self._prior = {}

    def mark_evaluated(self, stage: str, game_id: str | None) -> None:
        """Note that ``stage`` re-read ``game_id`` this run (success *or* failure).

        Reconciliation uses this to know which prior verdicts are stale: a stage
        that re-examined a game owns its fresh result; one that skipped it does not.
        """
        if game_id:
            self._evaluated.setdefault(game_id, set()).add(stage)

    def reconcile(self, available_game_ids: set) -> None:
        """Fold the prior report into this run's findings (call once, post-extract).

        For each prior issue: drop it if the game's DB is gone; otherwise keep only
        the stages that were **not** re-examined this run (a re-examined stage's
        fresh pass/fail is authoritative), merging into any fresh issue for the game.
        """
        for gid, prior in self._prior.items():
            if gid not in available_game_ids:
                continue  # DB removed → drop, mirroring row pruning
            carry = prior.stages - self._evaluated.get(gid, set())
            if not carry:
                continue
            fresh = self._by_game.get(gid)
            if fresh is None:
                self._by_game[gid] = ImportIssue(
                    game_id=gid,
                    experiment=prior.experiment,
                    seed=prior.seed,
                    seating_rotation=prior.seating_rotation,
                    db_name=prior.db_name,
                    stages=set(carry),
                    players=set(prior.players),
                    message=prior.message,
                )
            else:
                fresh.stages |= carry
                fresh.players |= prior.players
                if fresh.seed == UNKNOWN and prior.seed != UNKNOWN:
                    fresh.seed = prior.seed
                if fresh.seating_rotation == UNKNOWN and prior.seating_rotation != UNKNOWN:
                    fresh.seating_rotation = prior.seating_rotation

    def record(
        self,
        *,
        stage: str,
        db_path: str,
        message: str,
        game_id: str | None = None,
        experiment: str | None = None,
        metadata: dict | None = None,
        player_id: int | None = None,
    ) -> None:
        """Record (or merge into) the issue for the game behind ``db_path``.

        ``game_id``/``experiment`` are derived from ``db_path`` (or ``metadata``)
        when not given; pass them explicitly for player-trace DBs whose filename
        does not follow the ``{uuid}_{timestamp}.db`` game pattern. ``seed``/
        ``seating_rotation`` come best-effort from ``metadata``.
        """
        gid = game_id or (metadata or {}).get("gameId") \
            or get_game_id_from_path(db_path) or Path(db_path).name
        exp = experiment if experiment is not None else get_experiment_from_path(db_path)
        seed, rotation = _best_effort_seeding(metadata)

        # Mark this game as freshly failed THIS run. reconcile() never touches
        # _fresh, so a carried-forward prior issue stays non-fresh — which is what
        # lets auto-fix fire only on genuinely-new failures (WS3).
        self._fresh.add(gid)

        issue = self._by_game.get(gid)
        if issue is None:
            issue = ImportIssue(
                game_id=gid,
                experiment=exp,
                seed=seed,
                seating_rotation=rotation,
                db_name=Path(db_path).name,
            )
            self._by_game[gid] = issue
        else:
            # Upgrade placeholders once any stage manages to read the metadata.
            if issue.seed == UNKNOWN and seed != UNKNOWN:
                issue.seed = seed
            if issue.seating_rotation == UNKNOWN and rotation != UNKNOWN:
                issue.seating_rotation = rotation

        issue.stages.add(stage)
        if player_id is not None:
            issue.players.add(player_id)
        if message:
            issue.message = message

    def __len__(self) -> int:
        return len(self._by_game)

    def __bool__(self) -> bool:
        return bool(self._by_game)

    @property
    def fresh_game_ids(self) -> set:
        """game_ids that FAILED during this run (excludes carried-forward priors).

        This is the auto-fix trigger set: repairing a game that a prior run already
        flagged (but that no stage re-examined this run) would re-attempt the same
        unrecoverable DB every invocation. ``--force-rebuild`` re-examines a stale
        ledger, which re-records still-broken games as fresh.
        """
        return set(self._fresh)

    @property
    def has_fresh_issues(self) -> bool:
        """True when a malformed DB was recorded *this run* (not just carried over)."""
        return bool(self._fresh)

    def issues(self) -> list:
        """Issues sorted by ``(experiment, game_id)`` for stable output."""
        return sorted(
            self._by_game.values(), key=lambda i: (i.experiment, i.game_id)
        )

    def summary_lines(self) -> list:
        """Human-readable grouped lines for the end-of-extract console summary."""
        lines = []
        for issue in self.issues():
            stages = "|".join(sorted(issue.stages))
            players = (
                " players=" + ",".join(str(p) for p in sorted(issue.players))
                if issue.players else ""
            )
            lines.append(
                f"  {issue.game_id} ({issue.experiment}) "
                f"seed={issue.seed} rotation={issue.seating_rotation} "
                f"[{stages}]{players}: {issue.message}"
            )
        return lines

    def write_csv(self, path: str) -> bool:
        """Persist all issues to ``path`` (header-only when there are none)."""
        rows = [issue.as_row() for issue in self.issues()]
        return write_csv_file(path, ISSUE_FIELDNAMES, rows)


def record_db_failure(
    issues: ImportIssueLog | None,
    *,
    stage: str,
    db_path: str,
    exc: Exception,
    metadata: dict | None = None,
    game_id: str | None = None,
    experiment: str | None = None,
    player_id: int | None = None,
) -> None:
    """Print and (if a log is present) record a malformed/locked-DB read failure.

    Centralizes the print-and-record pair every exporter repeats at its game/trace
    catch sites, so the message and the recorded issue stay in lock-step.
    """
    print(f"Error processing {Path(db_path).name}: {exc}")
    if issues is not None:
        issues.record(
            stage=stage, db_path=db_path, message=str(exc), metadata=metadata,
            game_id=game_id, experiment=experiment, player_id=player_id,
        )
