"""Per-call ``agentic`` extraction: how strategists use a workspace.

Some conditions give the strategist a shell over a workspace of game records,
rules, and notes (benchmark.md §3, ``agentic`` table). This extractor reads each
player-trace DB and writes one row per tool call the strategist made inside an
agent run: workspace calls (``simple-tool.*``) and game actions (``mcp-tool.*``).
Action rows give every run a row, so per-run rates need no second table.

A player is exported only when its first agent spans show a workspace: a step
with ``host.capability``, or a run whose ``context.files`` describes workspace
folders. Conditions without one produce no rows, and a new condition that adds a
workspace is picked up without code changes.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Dict, List, Optional, Tuple

from ..catalog import Catalog
from .agentic.commands import CommandAccess, parse_command
from .agentic.layout import (
    GUIDE, RECORD_SECTIONS, REFERENCE_CATEGORIES, SOURCES,
    record_sections, reference_category, source_of, write_placement,
)
from .agentic.outcomes import (
    action_succeeded, classify_outcome, parse_tool_output, truncated_chars,
)
from .export_common import run_table_export
from .issues import record_db_failure
from .player_traces import (
    fetch_valid_traces, find_player_trace_databases, game_player_types,
    is_top_level_agent_span, parse_player_trace_path,
)
from .utilities import get_experiment_from_path, get_game_id_from_path, open_database_readonly

# Simple tools that are host helpers rather than workspace access.
NON_WORKSPACE_TOOLS = ("focus-briefer",)
# Actions whose choice can be traced back to what the strategist read.
GROUNDED_ACTIONS = {"set-research": "Technology", "set-policy": "Policy"}
WRITE_PLACEMENTS = ("persisted", "scratch", "lost", "rejected")
# Agent spans the gate reads: a run span and its first steps.
GATE_SPANS = 8

STEP_SPAN = re.compile(r"^agent\..+\.step\.(\d+)$")


def _column(name: str) -> str:
    return name.replace("-", "_")


SOURCE_COLUMNS = [f"src_{s}" for s in SOURCES]
RECORD_COLUMNS = [f"rec_{_column(s)}" for s in RECORD_SECTIONS]
REFERENCE_COLUMNS = [f"ref_{c}" for c in REFERENCE_CATEGORIES]
WRITE_COLUMNS = [f"writes_{p}" for p in WRITE_PLACEMENTS]

AGENTIC_FIELDNAMES = [
    "experiment", "game_id", "player_id", "player_type", "turn", "agent", "run", "step",
    "call", "tool", "kind", "capability",
    "option", "action_ok", "grounded",
    "exit_code", "outcome", "out_chars", "truncated_chars", "duration_ms", "read_style",
    *SOURCE_COLUMNS, *RECORD_COLUMNS, *REFERENCE_COLUMNS,
    "lookback", *WRITE_COLUMNS, "guide_edit", "note_missing",
]
WORKSPACE_COLUMNS = AGENTIC_FIELDNAMES[AGENTIC_FIELDNAMES.index("exit_code"):]
ACTION_COLUMNS = ["option", "action_ok", "grounded"]


def _shows_workspace(attributes) -> bool:
    """Whether an agent span's attributes describe a workspace.

    Steps carry ``host.capability`` (such as ``"write"``). Runs carry
    ``context.files`` as a JSON-encoded object, or ``false`` without a workspace.
    """
    try:
        parsed = json.loads(attributes) if isinstance(attributes, str) else attributes
    except ValueError:
        return False
    if not isinstance(parsed, dict):
        return False
    return (isinstance(parsed.get("host.capability"), (str, dict))
            or isinstance(parsed.get("context.files"), (str, dict)))


def _has_workspace(cursor) -> bool:
    """Whether the player's first agent spans show a workspace.

    The workspace is part of the condition, so the first run tells. Reading only
    the first few spans keeps the gate cheap on the many DBs without one.
    """
    cursor.execute(
        "SELECT attributes FROM spans WHERE name LIKE 'agent.%' ORDER BY id LIMIT ?",
        (GATE_SPANS,),
    )
    return any(_shows_workspace(attributes) for (attributes,) in cursor.fetchall())


def _player_has_workspace(trace_db_path: str) -> bool:
    """Cheap gate before the game DB is read. An unreadable DB passes, so the full
    extraction records the failure."""
    conn, _ = open_database_readonly(trace_db_path)
    if not conn:
        return True
    try:
        return _has_workspace(conn.cursor())
    except sqlite3.DatabaseError:
        return True
    finally:
        conn.close()


def _fetch_spans(cursor, trace_ids: List[str]) -> List[dict]:
    if not trace_ids:
        return []
    placeholders = ",".join(["?"] * len(trace_ids))
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(spans)").fetchall()}
    duration = "durationMs" if "durationMs" in columns else "NULL"
    cursor.execute(
        f"""
        SELECT id, traceId, spanId, parentSpanId, turn, name, statusCode, startTime, {duration},
               json_extract(attributes, '$."host.capability"'),
               json_extract(attributes, '$."tool.input"'),
               json_extract(attributes, '$."tool.output"')
        FROM spans
        WHERE traceId IN ({placeholders})
          AND (name LIKE 'agent.%' OR name LIKE 'mcp-tool.%' OR name LIKE 'simple-tool.%')
        ORDER BY startTime, id
        """,
        trace_ids,
    )
    keys = ("id", "trace_id", "span_id", "parent_span_id", "turn", "name", "status_code",
            "start_time", "duration_ms", "capability", "tool_input", "tool_output")
    return [dict(zip(keys, row)) for row in cursor.fetchall()]


def _tool_kind(name: str) -> Optional[Tuple[str, str]]:
    """``(tool, kind)`` for a tool span: ``workspace``, ``action``, or ``helper``."""
    if name.startswith("mcp-tool."):
        return name[len("mcp-tool."):], "action"
    if name.startswith("simple-tool."):
        tool = name[len("simple-tool."):]
        return tool, ("helper" if tool in NON_WORKSPACE_TOOLS else "workspace")
    return None


def _load_json(raw) -> object:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _command_access(tool_input) -> CommandAccess:
    """Paths a workspace call touched. Shell tools carry a ``Command``; other tools'
    absolute path arguments count as whole-file reads."""
    if isinstance(tool_input, dict):
        command = tool_input.get("Command") or tool_input.get("command")
        if isinstance(command, str):
            return parse_command(command)
        access = CommandAccess()
        access.reads = {v for v in tool_input.values() if isinstance(v, str) and v.startswith("/")}
        access.read_style = "full" if access.reads else "none"
        return access
    if isinstance(tool_input, str):
        return parse_command(tool_input)
    return CommandAccess()


def _workspace_fields(span: dict, turn: int) -> Tuple[dict, str]:
    """Workspace columns for one call, plus its stdout (used for grounding)."""
    stdout, stderr, exit_code = parse_tool_output(span["tool_output"])
    outcome = classify_outcome(exit_code, stderr)
    access = _command_access(_load_json(span["tool_input"]))

    sources = set()
    sections = set()
    categories = set()
    turns_read = set()
    for path in access.reads:
        if not path.startswith("/"):
            continue
        source = source_of(path)
        if source in SOURCES:
            sources.add(source)
        if source == "records":
            names, read_turn = record_sections(path)
            sections |= names
            if read_turn is not None and turn is not None and read_turn <= turn:
                turns_read.add(read_turn)
        elif source == "reference":
            category = reference_category(path)
            if category:
                categories.add(category)

    placements = {placement: 0 for placement in WRITE_PLACEMENTS}
    guide_edit = 0
    for path in access.writes:
        placement = write_placement(path, outcome)
        if placement is None:
            continue
        placements[placement] += 1
        if path == GUIDE and placement == "persisted":
            guide_edit = 1

    reads_notes = bool(sources & {"notes", "guide"})
    note_missing = int(reads_notes and "No such file" in stderr and "/workspace/game" in stderr)

    fields = {
        "exit_code": "" if exit_code is None else exit_code,
        "outcome": outcome,
        "out_chars": len(stdout),
        "truncated_chars": truncated_chars(stdout),
        "duration_ms": span["duration_ms"] if span["duration_ms"] is not None else "",
        "read_style": access.read_style,
        "lookback": (turn - min(turns_read)) if turns_read else "",
        "guide_edit": guide_edit,
        "note_missing": note_missing,
    }
    fields.update({f"src_{s}": int(s in sources) for s in SOURCES})
    fields.update({f"rec_{_column(s)}": int(s in sections) for s in RECORD_SECTIONS})
    fields.update({f"ref_{c}": int(c in categories) for c in REFERENCE_CATEGORIES})
    fields.update({f"writes_{p}": n for p, n in placements.items()})
    return fields, stdout


# What search tools put before a matched line: ``12:`` or ``file.md:12:`` from
# ``grep -n`` (``-`` on context lines), ``file.md:`` from a multi-file grep, or the
# tab-separated number from ``cat -n``.
_LINE_PREFIX = r"(?:\s*\d+\t|\S*?[:-](?:\d+[:-])?)?"


def _option_seen(option: str, outputs: List[str]) -> int:
    """1 when an earlier output in the run showed ``option`` as a heading or JSON name."""
    escaped = re.escape(option.strip())
    pattern = re.compile(
        rf"^{_LINE_PREFIX}#{{1,6}}\s+{escaped}\s*$|\"Name\"\s*:\s*\"{escaped}\"",
        re.IGNORECASE | re.MULTILINE,
    )
    return int(any(pattern.search(text) for text in outputs))


def _action_fields(span: dict, tool: str, outputs: List[str]) -> dict:
    fields = {"option": "", "action_ok": action_succeeded(span["tool_output"], span["status_code"]), "grounded": ""}
    key = GROUNDED_ACTIONS.get(tool)
    if key:
        tool_input = _load_json(span["tool_input"])
        option = tool_input.get(key) if isinstance(tool_input, dict) else None
        if isinstance(option, str) and option.strip():
            fields["option"] = option.strip()
            fields["grounded"] = _option_seen(option, outputs)
    return fields


def _agent_context(span: dict, lookup: Dict[Tuple[str, str], dict]):
    """``(agent span, step span)`` above a tool span; ``(None, None)`` for host calls."""
    step = None
    parent_id = span["parent_span_id"]
    while parent_id:
        parent = lookup.get((span["trace_id"], parent_id))
        if parent is None:
            return None, None
        if step is None and STEP_SPAN.match(parent["name"]):
            step = parent
        if is_top_level_agent_span(parent["name"]):
            return parent, step
        parent_id = parent["parent_span_id"]
    return None, None


def extract_player_agentic_rows(
    trace_db_path: str,
    game_id: str,
    experiment: str,
    player_type: str,
    issues=None,
    metadata=None,
) -> List[dict]:
    """One row per tool call inside the player's agent runs, or ``[]`` without a workspace."""
    _, player_id = parse_player_trace_path(trace_db_path)
    if player_id is None:
        return []

    conn, _ = open_database_readonly(trace_db_path)
    if not conn:
        if issues is not None:
            issues.record(
                stage="agentic", db_path=trace_db_path, game_id=game_id,
                experiment=experiment, player_id=player_id, metadata=metadata,
                message="could not open trace database",
            )
        return []
    try:
        cursor = conn.cursor()
        if not _has_workspace(cursor):
            return []
        trace_ids = [trace_id for _, trace_id, _ in fetch_valid_traces(cursor)]
        spans = _fetch_spans(cursor, trace_ids)
    except sqlite3.DatabaseError as exc:
        record_db_failure(
            issues, stage="agentic", db_path=trace_db_path, exc=exc,
            game_id=game_id, experiment=experiment, player_id=player_id, metadata=metadata,
        )
        return []
    finally:
        conn.close()

    lookup = {(span["trace_id"], span["span_id"]): span for span in spans}
    runs = [span for span in spans if is_top_level_agent_span(span["name"])]
    run_index = {(span["trace_id"], span["span_id"]): i for i, span in enumerate(runs, start=1)}

    calls_by_run: Dict[int, List[Tuple[int, dict, Optional[dict], dict]]] = {}
    for span in spans:
        if _tool_kind(span["name"]) is None:
            continue
        agent, step = _agent_context(span, lookup)
        if agent is None:
            continue
        number = int(STEP_SPAN.match(step["name"]).group(1)) if step else 0
        index = run_index[(agent["trace_id"], agent["span_id"])]
        calls_by_run.setdefault(index, []).append((number, span, step, agent))

    def base_row(agent: dict, turn, index: int) -> dict:
        row = {
            "experiment": experiment,
            "game_id": game_id,
            "player_id": player_id,
            "player_type": player_type,
            "turn": "" if turn is None else int(turn),
            "agent": agent["name"][len("agent."):],
            "run": index,
            "step": "",
            "call": "",
            "tool": "",
            "kind": "",
            "capability": "",
        }
        row.update({column: "" for column in ACTION_COLUMNS + WORKSPACE_COLUMNS})
        return row

    rows: List[dict] = []
    for agent in runs:
        index = run_index[(agent["trace_id"], agent["span_id"])]
        if index not in calls_by_run:
            # A run that ended before any tool call still counts toward per-run rates.
            row = base_row(agent, agent["turn"], index)
            row["kind"] = "none"
            rows.append(row)
            continue
        calls = sorted(calls_by_run[index], key=lambda c: (c[0], c[1]["start_time"] or 0, c[1]["id"]))
        # Workspace outputs by step. Calls in one step come from one model reply, so an
        # action sees only outputs of earlier steps.
        outputs: List[Tuple[int, str]] = []
        for call, (number, span, step, _) in enumerate(calls, start=1):
            tool, kind = _tool_kind(span["name"])
            turn = span["turn"] if span["turn"] is not None else agent["turn"]
            row = base_row(agent, turn, index)
            row.update({
                "step": number or "",
                "call": call,
                "tool": tool,
                "kind": kind,
                "capability": (step or {}).get("capability") or "",
            })
            if kind == "workspace":
                fields, stdout = _workspace_fields(span, None if turn is None else int(turn))
                row.update(fields)
                outputs.append((number, stdout))
            elif kind == "action":
                seen = [text for seen_step, text in outputs if number and seen_step and seen_step < number]
                row.update(_action_fields(span, tool, seen))
            rows.append(row)
    return rows


def extract_game_agentic_data(game_db_path: str, catalog: Catalog, issues=None) -> List[dict]:
    """Agentic rows for every player-trace DB of one game."""
    experiment = get_experiment_from_path(game_db_path)
    game_id = get_game_id_from_path(game_db_path)
    if not game_id:
        return []

    trace_dbs = find_player_trace_databases(game_db_path)
    if not trace_dbs:
        return []
    player_types = None
    metadata = None
    rows: List[dict] = []
    for player_id, trace_db_path in trace_dbs:
        if not _player_has_workspace(trace_db_path):
            continue
        if player_types is None:
            player_types, metadata = game_player_types(game_db_path, catalog, issues=issues, stage="agentic")
        rows.extend(
            extract_player_agentic_rows(
                trace_db_path=trace_db_path,
                game_id=game_id,
                experiment=experiment,
                player_type=player_types.get(player_id) or "N/A",
                issues=issues,
                metadata=metadata,
            )
        )
    return rows


def _describe_agentic_db(db_file, rows) -> str:
    players = len({row["player_id"] for row in rows})
    workspace = sum(1 for row in rows if row["kind"] == "workspace")
    return f"{players} players, {workspace} workspace calls, {len(rows)} calls"


def export_agentic_data(db_files, available_game_ids, output_file, catalog: Catalog, prune_only=False, issues=None, prune_gate=None) -> int:
    """Export agentic tool-call rows to ``output_file``; returns the count of new rows."""
    def _extract_rows(db_file, issues):
        return extract_game_agentic_data(db_file, catalog, issues=issues)

    return run_table_export(
        db_files, available_game_ids, output_file,
        stage="agentic", fieldnames=AGENTIC_FIELDNAMES,
        extract_rows=_extract_rows,
        dedupe_key=lambda r: (r.get("game_id"), r.get("player_id"), r.get("run"), r.get("call")),
        describe_db=_describe_agentic_db,
        noun="agentic rows", prune_only=prune_only, issues=issues,
        prune_gate=prune_gate,
    )
