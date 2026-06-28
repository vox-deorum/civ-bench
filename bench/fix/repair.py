"""Pure-Python recovery for malformed game SQLite DBs (``civ-bench fix`` core).

``runs/import_issues.csv`` lists games whose DB raises ``database disk image is
malformed``. Such files still open in DB Browser and most of their data is readable
— the damage is usually a single corrupt page or a corrupt index, not a wholesale
loss. There is no ``sqlite3`` CLI on this machine, so the shell ``.recover`` command
is unavailable; :func:`repair_database` recovers as much as possible into a fresh
database using only the stdlib ``sqlite3`` module.

The source is always opened **read-only + immutable** — ``immutable=1`` tells SQLite
to trust the on-disk image as-is rather than try to lock or roll back a journal,
which is exactly what lets a malformed image be read page-by-page (the same reason a
viewer can open it). The source file is never mutated; recovery writes only ``dst``.

Strategy is layered, cheapest first:

* **iterdump** — replay a clean SQL dump into a fresh DB. Complete and pristine when
  the corruption does not sit on any page the dump must read.
* **tolerant rebuild** — recreate each table from ``sqlite_master``, copy its rows
  while skipping corrupt pages (recovering rows on *both* sides of a bad page via a
  rowid range walk), then rebuild indexes/triggers/views from the recovered data.

Acceptance is **best-effort**: a result is kept when it opens, passes
``PRAGMA quick_check``, and recovered at least one table — even if a few unreadable
rows had to be dropped. Completeness is then measured honestly against a raw page scan
of the source (``rows_on_disk``): how many user rows physically exist versus how many the
walk reached. That ground truth, not the walk's own skip counters (which silently miss
rows leapt over on a corrupt page), is what the caller reports — typically ~100%, with
only the handful of rows on a physically corrupt page lost.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field

# Rows fetched/inserted per batch on the fast bulk-copy path.
_BATCH = 500
# Bytes of dump SQL buffered before flushing a transaction (bounds peak memory: a
# blob-heavy DB dumps to many times its on-disk size, so the whole script is never held).
_DUMP_CHUNK = 8 << 20
# Initial rowid window for the range-scan recovery path; narrowed on a corrupt page.
_RANGE_STEP = 1000
# Empty windows grow (×4, capped) to leap across gaps fast; after this many consecutive
# empties the walk stops — corruption can yield a garbage MAX(rowid) (billions), so we
# cannot trust the upper bound and instead detect "past the real data" by the empty run.
_GAP_GROW = 4
_GAP_MAX = 1 << 16
_EMPTY_LIMIT = 64
# A corrupt page returns an *error* (not an empty window), which the walk crosses one
# unreadable rowid at a time. After this many consecutive single-rowid failures with no
# row in between, the b-tree is unreadable from here on — stop rather than grind toward
# the open upper bound. A localized bad page is a handful of rows, far below this.
_SKIP_LIMIT = 10000
# Open upper bound for the rowid walk when MAX(rowid) is itself unreadable; the empty-run
# heuristic, not this sentinel, is what actually ends the walk.
_ROWID_MAX = (1 << 63) - 1
# Sidecars SQLite may write next to a DB file; cleared around the temp ``dst`` (and,
# by the runner, beside a freshly-swapped DB). Single source of truth for both.
SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


@dataclass
class RepairReport:
    """What a single :func:`repair_database` attempt produced.

    Two distinct kinds of loss are reported separately, because they are not the same
    severity:

    * ``tables_partial`` — tables that **were** recovered but lost some rows to a corrupt
      page (the table and the rest of its rows are intact). ``rows_skipped`` is the total
      row count across these.
    * ``objects_skipped`` — schema objects **dropped** entirely: a table whose CREATE
      failed, or an index/trigger/view that could not be rebuilt from the recovered data.

    Keeping them apart is why a few unreadable rows in ``GameEvents`` no longer reads as
    "dropped the GameEvents table".

    ``rows_on_disk`` is an independent ground-truth count of how many *user-table* rows are
    physically present in the source file (a raw page scan, not a b-tree walk, with schema
    and ``sqlite_%`` shadow rows subtracted), so the caller can report recovery completeness
    honestly — ``rows_recovered`` of ``rows_on_disk`` — instead of guessing from the walk's
    skip counters, which silently miss rows leapt over on corrupt pages. ``0`` if uncomputed.
    """

    success: bool = False
    strategy: str = "failed"          # "iterdump" | "tolerant-rebuild" | "failed"
    tables_recovered: list[str] = field(default_factory=list)
    rows_recovered: int = 0
    rows_skipped: int = 0
    rows_on_disk: int = 0             # table rows physically present (raw page scan); 0 = unknown
    tables_partial: list[str] = field(default_factory=list)
    objects_skipped: list[str] = field(default_factory=list)
    integrity: str = ""               # PRAGMA quick_check result
    error: str = ""


# ── small helpers ────────────────────────────────────────────────────────────
def _q(identifier: str) -> str:
    """Quote a SQL identifier (double internal quotes), so odd table names are safe."""
    return '"' + identifier.replace('"', '""') + '"'


def _connect_readonly_immutable(path: str) -> sqlite3.Connection:
    """Open ``path`` read-only + immutable (mirrors ``open_database_readonly``)."""
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def _remove(path: str) -> None:
    """Best-effort delete of ``path`` and any SQLite sidecars beside it."""
    for candidate in (path, *(path + suffix for suffix in SIDECAR_SUFFIXES)):
        try:
            if os.path.exists(candidate):
                os.remove(candidate)
        except OSError:
            pass


def _count_rows_on_disk(path: str) -> int | None:
    """Total table rows physically present in the SQLite file, by a raw page scan.

    Independent of the b-tree (which the corruption damaged): every page is classified by
    its first byte and a *table-leaf* page (``0x0D``) carries its live cell count in bytes
    3–4 of its header. Summing those counts gives the ground-truth number of rows on disk —
    the honest denominator for "recovered N of M". Page 1's b-tree header sits just after
    the 100-byte database header. Overflow/interior/index/freelist pages are not table
    leaves and are correctly ignored. Returns ``None`` if the file cannot be read as SQLite.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(100)
            if len(header) < 100 or header[:16] != b"SQLite format 3\x00":
                return None
            page_size = int.from_bytes(header[16:18], "big")
            if page_size == 1:  # the format's escape value for a 64 KiB page
                page_size = 65536
            if page_size < 512:
                return None
            total = 0
            first = True
            while True:
                page = fh.read(page_size) if not first else header + fh.read(page_size - 100)
                if len(page) < page_size:
                    break
                off = 100 if first else 0  # b-tree header offset within the page
                if page[off] == 0x0D:      # table b-tree leaf
                    total += int.from_bytes(page[off + 3:off + 5], "big")
                first = False
            return total
    except OSError:
        return None


def _count_internal_rows(src: sqlite3.Connection) -> int:
    """Rows the page scan counts but the user-table recovery does not: ``sqlite_master``
    plus any ``sqlite_%`` shadow tables (``sqlite_sequence``/``sqlite_stat*``).

    Subtracting these makes ``rows_on_disk`` directly comparable to ``rows_recovered`` (both
    user tables only), so a clean data recovery reads as 100% instead of being dinged for
    the schema rows it faithfully reproduces. The schema b-tree is intact whenever we got
    this far, so these counts are reliable; any unreadable one is skipped.
    """
    total = 0
    try:
        total += src.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()[0]
    except sqlite3.DatabaseError:
        return 0
    try:
        shadow = src.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'sqlite_%'"
        ).fetchall()
    except sqlite3.DatabaseError:
        return total
    for (name,) in shadow:
        try:
            total += src.execute(f"SELECT COUNT(*) FROM {_q(name)}").fetchone()[0]
        except sqlite3.DatabaseError:
            pass
    return total


def _quick_check(conn: sqlite3.Connection) -> str:
    """``"ok"`` when the DB is structurally sound, else a joined error summary."""
    rows = conn.execute("PRAGMA quick_check").fetchall()
    messages = [str(r[0]) for r in rows]
    return "ok" if messages == ["ok"] else "; ".join(messages[:5])


def source_quick_check(path: str) -> str:
    """``"ok"`` if the on-disk DB at ``path`` is structurally sound, else why not.

    Lets the orchestrator tell an already-healthy related DB (leave it alone, no
    backup) from one that needs recovery. Opened read-only + immutable so a malformed
    image is read as-is; a DB that cannot even be opened/scanned counts as needing
    recovery (its error message is returned, never raised).
    """
    try:
        conn = _connect_readonly_immutable(path)
    except sqlite3.Error as exc:
        return f"unreadable: {exc}"
    try:
        return _quick_check(conn)
    except sqlite3.DatabaseError as exc:
        return f"unreadable: {exc}"
    finally:
        conn.close()


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    return [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    ]


def _count_rows(conn: sqlite3.Connection, tables: list[str]) -> int:
    total = 0
    for table in tables:
        try:
            total += conn.execute(f"SELECT COUNT(*) FROM {_q(table)}").fetchone()[0]
        except sqlite3.DatabaseError:
            pass
    return total


# ── strategy A: clean SQL dump ───────────────────────────────────────────────
def _try_iterdump(src: sqlite3.Connection, dst_path: str, report: RepairReport) -> bool:
    """Replay ``src.iterdump()`` into a fresh DB in bounded chunks; True iff quick_check ok.

    Statements are flushed in ~``_DUMP_CHUNK`` transactions instead of materializing the
    whole dump. The dump's own BEGIN/COMMIT are dropped so each chunk is one fast
    transaction. ``iterdump`` raises mid-stream on a corrupt page; the partial ``dst`` is
    discarded by the caller.
    """
    dst = sqlite3.connect(dst_path)
    try:
        buf: list[str] = []
        size = 0

        def _flush() -> None:
            if buf:
                dst.executescript("BEGIN;\n" + "\n".join(buf) + "\nCOMMIT;")
                buf.clear()

        for stmt in src.iterdump():
            head = stmt.lstrip()
            if head.startswith("BEGIN TRANSACTION") or head.startswith("COMMIT"):
                continue
            buf.append(stmt)
            size += len(stmt)
            if size >= _DUMP_CHUNK:
                _flush()
                size = 0
        _flush()
        dst.commit()

        integrity = _quick_check(dst)
        if integrity != "ok":
            report.error = f"iterdump produced a still-malformed DB ({integrity})"
            return False
        tables = _list_tables(dst)
        if not tables:
            report.error = "iterdump recovered no tables"
            return False
        report.strategy = "iterdump"
        report.tables_recovered = tables
        report.rows_recovered = _count_rows(dst, tables)
        report.integrity = integrity
        report.success = True
        return True
    except sqlite3.DatabaseError as exc:
        report.error = str(exc)
        return False
    finally:
        dst.close()


# ── strategy B: tolerant table-by-table rebuild ──────────────────────────────
def _read_master(src: sqlite3.Connection) -> list[tuple]:
    """``(type, name, sql)`` for every user object (raises if the schema is unreadable)."""
    return src.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
    ).fetchall()


def _column_count(src: sqlite3.Connection, name: str) -> int:
    """Number of columns in ``name`` (reads the parsed schema, not data pages)."""
    return len(src.execute(f"PRAGMA table_info({_q(name)})").fetchall())


def _rowid_bounds(src: sqlite3.Connection, name: str):
    """``(min_rowid, max_rowid)`` or ``(None, None)`` when the boundaries are unreadable."""
    try:
        lo, hi = src.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {_q(name)}").fetchone()
    except sqlite3.DatabaseError:
        return None, None
    if lo is None or hi is None:
        return None, None
    return lo, hi


def _bulk_copy(src: sqlite3.Connection, dst: sqlite3.Connection, name: str, insert: str) -> int:
    """Copy every row of ``name`` in batches; raises on the first corrupt page."""
    cur = src.execute(f"SELECT * FROM {_q(name)}")
    copied = 0
    while True:
        batch = cur.fetchmany(_BATCH)
        if not batch:
            break
        dst.executemany(insert, batch)
        copied += len(batch)
    dst.commit()
    return copied


def _tolerant_scan(src: sqlite3.Connection, dst: sqlite3.Connection, name: str, insert: str):
    """Row-by-row copy that stops at the first corrupt page → ``(copied, truncated)``.

    Used when there is no rowid to range-walk (WITHOUT ROWID, or unreadable bounds);
    ``truncated`` is True when a corrupt page cut the scan short (so the tail is lost).
    """
    try:
        cur = src.execute(f"SELECT * FROM {_q(name)}")
    except sqlite3.DatabaseError:
        return 0, True
    copied = 0
    truncated = False
    while True:
        try:
            row = cur.fetchone()
        except sqlite3.DatabaseError:
            truncated = True
            break
        if row is None:
            break
        dst.execute(insert, row)
        copied += 1
    dst.commit()
    return copied, truncated


def _copy_by_rowid(src: sqlite3.Connection, dst: sqlite3.Connection, name: str, insert: str):
    """Walk ``name`` in rowid windows → ``(copied, skipped, truncated)``.

    The window adapts: a read error narrows it (``//8``) to localize the damage; once it
    is a single rowid that still fails, that one rowid is skipped (one query per bad rowid,
    no re-bisecting a run). A window with rows resets to the data batch size; an *empty*
    window grows (×4) to leap across gaps. Because a corrupt page can return a garbage
    ``MAX(rowid)``, the upper bound is not trusted: the walk stops after ``_EMPTY_LIMIT``
    consecutive empty windows — game-DB rowids are contiguous, so empties only pile up once
    we are past the real data. Rows on **both** sides of a bad page are recovered.

    A rowid table can *always* be range-walked, so unreadable ``MIN/MAX(rowid)`` is not a
    reason to truncate: it just means the corrupt page sits on the b-tree's leftmost or
    rightmost path. We default the lower bound to 1 (game event logs are append-only from
    rowid 1) and leave the upper bound open. The walk only gives up — reporting
    ``truncated`` — after ``_SKIP_LIMIT`` consecutive unreadable rowids, i.e. the b-tree is
    damaged beyond a single bad page and cannot be read further.
    """
    dst.execute(f"DELETE FROM {_q(name)}")  # restart clean — a corrupt cursor can't resume
    dst.commit()

    lo, hi = _rowid_bounds(src, name)
    start = lo if lo is not None else 1
    walk_hi = hi if hi is not None else _ROWID_MAX

    copied = skipped = empty_streak = skip_streak = pending = 0
    step = _RANGE_STEP
    truncated = False
    while start <= walk_hi and empty_streak < _EMPTY_LIMIT:
        end = min(start + step - 1, walk_hi)
        try:
            rows = src.execute(
                f"SELECT * FROM {_q(name)} WHERE rowid BETWEEN ? AND ?", (start, end)
            ).fetchall()
        except sqlite3.DatabaseError:
            if step > 1:
                step = max(1, step // 8)  # narrow toward the bad rowid
                continue
            # Single unreadable rowid. Hold it as *pending* rather than counting it lost
            # outright: only skips that are followed by more recovered rows are real
            # interior losses. A run of skips with no data after is end-probing past the
            # last good row (a corrupt page makes a lookup error instead of returning
            # empty), so it is discarded when the walk ends — never reported as lost rows.
            pending += 1
            skip_streak += 1
            start += 1
            if skip_streak >= _SKIP_LIMIT:  # b-tree unreadable from here — stop, don't grind
                truncated = True
                break
            continue
        if rows:
            dst.executemany(insert, rows)
            copied += len(rows)
            skipped += pending            # confirm the pending skips: real data follows them
            pending = empty_streak = skip_streak = 0
            step = _RANGE_STEP
        else:
            empty_streak += 1
            step = min(_GAP_MAX, step * _GAP_GROW)  # leap across empty space
        dst.commit()
        start = end + 1
    return copied, skipped, truncated  # trailing `pending` discarded — speculative, not lost


def _copy_table(
    src: sqlite3.Connection, dst: sqlite3.Connection, name: str, without_rowid: bool
):
    """Best-effort copy of one table → ``(copied, skipped, note)``; never raises."""
    ncols = _column_count(src, name)
    if ncols == 0:
        return 0, 0, None
    insert = f"INSERT INTO {_q(name)} VALUES ({','.join('?' * ncols)})"

    try:
        return _bulk_copy(src, dst, name, insert), 0, None
    except sqlite3.DatabaseError:
        pass  # a page in this table is corrupt — drop to recovery

    if without_rowid:
        try:
            dst.execute(f"DELETE FROM {_q(name)}")
            dst.commit()
        except sqlite3.DatabaseError:
            pass
        copied, truncated = _tolerant_scan(src, dst, name, insert)
        note = (
            f"{name}: kept {copied} row(s); WITHOUT ROWID tail past a corrupt page unreadable"
            if truncated else None
        )
        return copied, 0, note

    copied, skipped, truncated = _copy_by_rowid(src, dst, name, insert)
    note = _partial_note(name, copied, skipped, truncated)
    return copied, skipped, note


def _partial_note(name: str, copied: int, skipped: int, truncated: bool) -> str | None:
    """A one-line note for a *recovered* table that lost some rows, else ``None``.

    Phrased so it reads as partial row loss within a kept table, never as a dropped table.
    The overall recovered-of-on-disk figure (see ``rows_on_disk``) carries the precise
    completeness; this note just says which table was affected and how.
    """
    if truncated:
        # The b-tree became unreadable part-way: recovered what the walk could reach.
        extra = f" ({skipped} more on a corrupt page)" if skipped else ""
        return f"{name}: recovered {copied} row(s); a corrupt region could not be read{extra}"
    if skipped:
        return f"{name}: recovered {copied} row(s); {skipped} on a corrupt page were unreadable"
    return None


def _try_tolerant_rebuild(
    src: sqlite3.Connection, dst_path: str, report: RepairReport
) -> bool:
    """Recreate the schema and copy what is readable; True when quick_check is ok."""
    try:
        master = _read_master(src)
    except sqlite3.DatabaseError as exc:
        report.error = f"schema unreadable: {exc}"
        return False

    tables, others, skipped_objects = [], [], []
    for type_, name, sql in master:
        if not sql:
            skipped_objects.append(f"{type_}:{name} (no schema SQL)")
            continue
        if type_ == "table":
            tables.append((name, sql))
        elif type_ in ("index", "trigger", "view"):
            others.append((type_, name, sql))

    if not tables:
        report.error = "no readable tables in schema"
        report.objects_skipped = skipped_objects
        return False

    dst = sqlite3.connect(dst_path)
    try:
        dst.execute("PRAGMA foreign_keys=OFF")
        recovered, partial_tables, rows_recovered, rows_skipped = [], [], 0, 0

        for name, sql in tables:
            try:
                dst.execute(sql)
            except sqlite3.DatabaseError as exc:
                skipped_objects.append(f"table:{name} (create failed: {exc})")
                continue
            without_rowid = "WITHOUT ROWID" in sql.upper()
            copied, skipped, note = _copy_table(src, dst, name, without_rowid)
            recovered.append(name)
            rows_recovered += copied
            rows_skipped += skipped
            if note:  # table kept, but some rows were lost — record it as partial, not dropped
                partial_tables.append(note)

        # Rebuild indexes/triggers/views last, from the recovered data — a corrupt
        # index is a common root cause, so rebuilding it clean is the actual fix.
        for type_, name, sql in others:
            try:
                dst.execute(sql)
            except sqlite3.DatabaseError as exc:
                skipped_objects.append(f"{type_}:{name} (rebuild failed: {exc})")
        dst.commit()

        integrity = _quick_check(dst)
        report.strategy = "tolerant-rebuild"
        report.tables_recovered = recovered
        report.rows_recovered = rows_recovered
        report.rows_skipped = rows_skipped
        report.tables_partial = partial_tables
        report.objects_skipped = skipped_objects
        report.integrity = integrity
        report.success = integrity == "ok" and bool(recovered)
        if not report.success and not report.error:
            report.error = f"rebuilt DB still fails integrity ({integrity})"
        return report.success
    finally:
        dst.close()


# ── public entry point ───────────────────────────────────────────────────────
def repair_database(src_path: str, dst_path: str) -> RepairReport:
    """Recover ``src_path`` into a fresh DB at ``dst_path`` (which must not exist).

    Returns a :class:`RepairReport`; ``success`` is False (never an exception) when
    the file is too damaged to recover. ``src_path`` is opened read-only + immutable
    and is never modified. On failure ``dst_path`` is removed; the caller owns it on
    success.
    """
    report = RepairReport()
    try:
        src = _connect_readonly_immutable(src_path)
    except sqlite3.Error as exc:
        report.error = f"cannot open source: {exc}"
        return report

    try:
        if _try_iterdump(src, dst_path, report):
            return report
        _remove(dst_path)  # discard the failed attempt before the next strategy

        report.error = ""  # supersede the iterdump note with the rebuild's verdict
        if _try_tolerant_rebuild(src, dst_path, report):
            # Ground-truth denominator: how many *user* rows physically exist vs. how many
            # the b-tree walk could reach. The raw page scan counts schema/shadow rows too,
            # which the rebuild reproduces, so subtract them for an apples-to-apples figure.
            scanned = _count_rows_on_disk(src_path)
            if scanned is not None:
                report.rows_on_disk = max(0, scanned - _count_internal_rows(src))
            return report
        _remove(dst_path)
        report.strategy = "failed"
        report.success = False
        if not report.error:
            report.error = "no recoverable data"
        return report
    finally:
        src.close()
