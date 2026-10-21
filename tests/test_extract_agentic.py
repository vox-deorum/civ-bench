"""Extract: the ``agentic`` table (one row per tool call in a workspace run).

Covers the shell-command parser, the outcome labels, the workspace layout map,
and the extractor on synthetic player-trace DBs with the real ``spans`` schema.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from bench.catalog import Catalog
from bench.extract.agentic.commands import MAX_RANGE, expand_braces, parse_command
from bench.extract.agentic.layout import record_sections, write_placement
from bench.extract.agentic.outcomes import action_succeeded, classify_outcome, truncated_chars
from bench.extract.extract_agentic import (
    AGENTIC_FIELDNAMES,
    export_agentic_data,
    extract_player_agentic_rows,
)

GAME_ID = "0f0e0d0c-0000-4000-8000-000000000001"


@pytest.fixture
def catalog(configs_dir) -> Catalog:
    return Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")


# ── command parser ───────────────────────────────────────────────────────────
def test_parser_expands_variables_and_reads_files():
    access = parse_command(
        'turn=294; record="/workspace/records/turn-$turn"; '
        'cat "$record/events-economic.md" "$record/events-diplomatic.md"'
    )
    assert access.reads == {
        "/workspace/records/turn-294/events-economic.md",
        "/workspace/records/turn-294/events-diplomatic.md",
    }
    assert access.writes == set()
    assert access.read_style == "full"


def test_parser_expands_for_lists_and_braces():
    access = parse_command(
        "for f in options victory; do cat /workspace/records/turn-7/$f.md | head -40; done; "
        "ls /workspace/records/turn-{5,6}"
    )
    assert access.reads == {
        "/workspace/records/turn-7/options.md",
        "/workspace/records/turn-7/victory.md",
        "/workspace/records/turn-5",
        "/workspace/records/turn-6",
    }
    # A piped head keeps the whole-file cat a targeted read.
    assert access.read_style == "targeted"


def test_parser_skips_patterns_and_heredoc_bodies():
    access = parse_command(
        "awk '/^## Chancery$/{p=1} p{print}' /workspace/reference/buildings/medieval.md; "
        "grep -n -A 16 '^## /workspace/fake' /workspace/reference/technologies/modern.md 2>/dev/null"
    )
    assert access.reads == {
        "/workspace/reference/buildings/medieval.md",
        "/workspace/reference/technologies/modern.md",
    }
    assert access.read_style == "targeted"

    access = parse_command("cat > /workspace/game/plan.md << 'EOF'\n# Plan > /workspace/x\n- a | b\nEOF")
    assert access.writes == {"/workspace/game/plan.md"}
    assert access.reads == set()


@pytest.mark.parametrize("command, writes", [
    ("echo x >> /workspace/game/plan.md", {"/workspace/game/plan.md"}),
    ("echo x | tee -a /tmp/a.md > /dev/null", {"/tmp/a.md"}),
    ("cp /workspace/records/turn-1/options.md /workspace/game/o.md", {"/workspace/game/o.md"}),
    ("mkdir -p /workspace/game/lessons", {"/workspace/game/lessons"}),
    ("sed -i 's/a/b/' /workspace/game/AGENTS.md", {"/workspace/game/AGENTS.md"}),
    ("cat > notes.txt << 'EOF'\nhi\nEOF", {"notes.txt"}),
    ("cd /workspace/game && echo x > plan.md", {"/workspace/game/plan.md"}),
    ("sed --in-place=.bak 's/a/b/' /workspace/game/plan.md", {"/workspace/game/plan.md"}),
    ("sort -o /tmp/s.md /workspace/records/turn-1/options.md", {"/tmp/s.md"}),
    ("echo x >& /tmp/log.md", {"/tmp/log.md"}),
    ("ls /workspace/game | xargs -I{} cp {} /tmp/", {"/tmp"}),
])
def test_parser_finds_write_targets(command, writes):
    assert parse_command(command).writes == writes


def test_parser_ignores_computed_for_lists():
    access = parse_command('for f in $(ls -d /workspace/records/turn-*); do cat "$f/options.md"; done')
    # The substituted ls is a real listing; "$f/options.md" cannot be resolved.
    assert access.reads == {"/workspace/records/turn-*"}
    assert access.read_style == "listing"


def test_brace_expansion_stops_at_the_limit():
    assert expand_braces("/workspace/records/turn-{5,6}") == [
        "/workspace/records/turn-5", "/workspace/records/turn-6",
    ]
    assert expand_braces("/t/{1..1000000000}")[:2] == ["/t/1", "/t/2"]
    assert len(expand_braces("/t/{1..1000000000}")) == MAX_RANGE
    many = expand_braces("/t/" + "{a,b}" * 200)
    assert len(many) == MAX_RANGE and many[0] == "/t/" + "a" * 200
    assert expand_braces("{a,{b,c}}") == ["a", "b", "a", "c"]


def test_parser_listing_style():
    access = parse_command("ls -la /workspace/records/ 2>/dev/null | head -30")
    assert access.reads == {"/workspace/records"}
    assert access.read_style == "listing"


# ── outcomes and layout ──────────────────────────────────────────────────────
@pytest.mark.parametrize("exit_code, stderr, outcome", [
    (0, "", "ok"),
    (1, "", "no_match"),
    (1, "EROFS: read-only file system, write '/turn-35/notes.md'", "read_only"),
    (1, "cat: /workspace/game/plan.md: No such file or directory", "missing_file"),
    (2, "bash: syntax error near unexpected token `do'", "syntax"),
    (127, "foo: command not found", "not_found"),
    (5, "something else", "other"),
])
def test_classify_outcome(exit_code, stderr, outcome):
    assert classify_outcome(exit_code, stderr) == outcome


def test_truncation_and_action_success():
    assert truncated_chars("abc\n[truncated 1234 chars; read the rest in pieces]") == 1234
    assert truncated_chars("plain") == 0
    assert action_succeeded('{"Success":true}', 1) == 1
    assert action_succeeded("true", 1) == 1
    assert action_succeeded('{"Success":false,"Error":{"Code":"X"}}', 1) == 0
    assert action_succeeded('{"content":[{"type":"text","text":"Error: bad tech"}]}', 1) == 0
    assert action_succeeded('{"Success":true}', 2) == 0


def test_layout_sections_and_placements():
    assert record_sections("/workspace/records/turn-12/cities/player-3.md") == ({"cities"}, 12)
    assert record_sections("/workspace/records/turn-*/events-all.md") == ({"events-all"}, None)
    assert record_sections("/workspace/records/game/situation.md") == ({"situation"}, None)
    assert record_sections("/workspace/records/game-info/techs.txt") == ({"other"}, None)
    assert record_sections("/workspace/records/turn-4") == (set(), 4)
    assert write_placement("/workspace/records/turn-1/x.md", "ok") == "rejected"
    assert write_placement("/workspace/game/plan.md", "ok") == "persisted"
    assert write_placement("/workspace/game/plan.md", "missing_file") == "rejected"
    assert write_placement("/tmp/x", "ok") == "scratch"
    # A read-only failure elsewhere in the command does not undo a /tmp write.
    assert write_placement("/tmp/x", "read_only") == "scratch"
    assert write_placement("/workspace/notes.md", "ok") == "lost"
    assert write_placement("notes.txt", "ok") == "lost"
    assert write_placement("/dev/null", "ok") is None


# ── extractor ────────────────────────────────────────────────────────────────
def _bash(command, stdout="", stderr="", exit_code=0):
    return {
        "tool.input": json.dumps({"Command": command}),
        "tool.output": json.dumps({"stdout": stdout, "stderr": stderr, "exitCode": exit_code}),
    }


def _action(payload, output='{"Success":true}'):
    return {"tool.input": json.dumps(payload), "tool.output": output}


STEP = {"host.capability": "write", "step.tools": json.dumps(["set-research", "bash"])}


def _make_trace_db(path: Path, spans) -> None:
    """A player-trace DB with the real ``spans`` schema.

    ``spans`` holds ``(trace, span_id, parent, turn, name, attributes, status)``;
    ids and start times follow list order.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE spans (id INTEGER PRIMARY KEY, contextId TEXT, turn INTEGER, "
        "traceId TEXT, spanId TEXT, parentSpanId TEXT, name TEXT, startTime INTEGER, "
        "endTime INTEGER, durationMs INTEGER, attributes TEXT, statusCode INTEGER, "
        "statusMessage TEXT)"
    )
    for i, (trace, span_id, parent, turn, name, attributes, status) in enumerate(spans, start=1):
        conn.execute(
            "INSERT INTO spans VALUES (?, 'ctx', ?, ?, ?, ?, ?, ?, ?, 7, ?, ?, NULL)",
            (i, turn, trace, span_id, parent, name, i * 10, i * 10 + 7,
             json.dumps(attributes), status),
        )
    conn.commit()
    conn.close()


def _turn(trace, turn, agent_steps, status=1, root=None):
    """One strategist turn: a host call, then an agent run whose steps hold calls."""
    root = root or f"{trace}-root"
    spans = [
        (trace, root, None, turn, f"strategist.turn.{turn}", {}, status),
        # Host prefetch outside the agent run: never a row.
        (trace, f"{trace}-host", root, turn, "mcp-tool.get-options", {"tool.output": "{}"}, 1),
        (trace, f"{trace}-agent", root, turn, "agent.simple-strategist",
         {"context.files": '{"game":"write"}'}, 1),
    ]
    for number, calls in enumerate(agent_steps, start=1):
        step_id = f"{trace}-s{number}"
        spans.append((trace, step_id, f"{trace}-agent", turn,
                      f"agent.simple-strategist.step.{number}", STEP, 1))
        for k, (name, attributes) in enumerate(calls):
            spans.append((trace, f"{step_id}-c{k}", step_id, turn, name, attributes, 1))
    return spans


def _rows(path):
    return extract_player_agentic_rows(str(path), GAME_ID, "exp-files-per-5", "Model-Agentic")


def test_extract_skips_players_without_workspace(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    _make_trace_db(trace, [
        ("t", "root", None, 1, "strategist.turn.1", {}, 1),
        # Runs without a workspace record context.files as false.
        ("t", "agent", "root", 1, "agent.simple-strategist", {"context.files": False}, 1),
        ("t", "s1", "agent", 1, "agent.simple-strategist.step.1", {}, 1),
        ("t", "c1", "s1", 1, "mcp-tool.set-research", _action({"Technology": "Pottery"}), 1),
    ])
    assert _rows(trace) == []


def test_extract_reads_and_grounding(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    _make_trace_db(trace, _turn("t", 12, [
        [
            ("simple-tool.bash", _bash(
                'turn=12; cat "/workspace/records/turn-$turn/events-economic.md" '
                "/workspace/records/turn-10/options.md",
                stdout="# Events\n" + "x" * 20 + "\n[truncated 500 chars; read the rest]",
            )),
            ("simple-tool.bash", _bash(
                "grep -n -A8 '^## Plastics' /workspace/reference/technologies/modern.md",
                stdout="## Plastics\n- Cost: 1",
            )),
        ],
        [
            ("mcp-tool.set-research", _action({"Technology": "Plastics", "Rationale": "r"})),
            ("mcp-tool.set-policy", _action({"Policy": "Liberty", "Rationale": "r"},
                                            '{"content":[{"type":"text","text":"Error: locked"}]}')),
        ],
    ]))

    rows = _rows(trace)
    assert [(r["call"], r["step"], r["tool"], r["kind"]) for r in rows] == [
        (1, 1, "bash", "workspace"),
        (2, 1, "bash", "workspace"),
        (3, 2, "set-research", "action"),
        (4, 2, "set-policy", "action"),
    ]
    first, second, research, policy = rows
    assert {r["run"] for r in rows} == {1}
    assert first["capability"] == "write"
    assert first["turn"] == 12
    assert first["outcome"] == "ok"
    assert first["src_records"] == 1 and first["src_reference"] == 0
    assert first["rec_events_economic"] == 1 and first["rec_options"] == 1
    assert first["lookback"] == 2
    assert first["truncated_chars"] == 500
    assert first["read_style"] == "full"
    assert first["duration_ms"] == 7
    assert second["ref_technologies"] == 1 and second["src_reference"] == 1
    assert second["read_style"] == "targeted"
    assert second["lookback"] == ""
    assert research["option"] == "Plastics"
    assert research["grounded"] == 1 and research["action_ok"] == 1
    assert research["outcome"] == ""
    assert policy["option"] == "Liberty"
    assert policy["grounded"] == 0 and policy["action_ok"] == 0


def test_extract_write_placements_and_errors(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    _make_trace_db(trace, _turn("t", 3, [[
        ("simple-tool.bash", _bash(
            "cat > /workspace/records/turn-3/notes.md << 'EOF'\nplan\nEOF",
            stderr="EROFS: read-only file system, write '/turn-3/notes.md'", exit_code=1,
        )),
        ("simple-tool.bash", _bash("echo a >> /workspace/game/plan.md; echo b > /tmp/s.md; echo c > n.txt")),
        ("simple-tool.bash", _bash("echo '- rule' >> /workspace/game/AGENTS.md")),
        ("simple-tool.bash", _bash(
            "cat /workspace/game/plan.md",
            stderr="cat: /workspace/game/plan.md: No such file or directory", exit_code=1,
        )),
        ("simple-tool.bash", _bash("grep -c Raze /workspace/records/turn-3/events-all.md", exit_code=1)),
    ]]))

    rejected, mixed, guide, missing, no_match = _rows(trace)
    assert rejected["outcome"] == "read_only" and rejected["writes_rejected"] == 1
    assert (mixed["writes_persisted"], mixed["writes_scratch"], mixed["writes_lost"]) == (1, 1, 1)
    assert mixed["src_notes"] == 0  # writing a note is not reading one
    assert guide["guide_edit"] == 1 and guide["writes_persisted"] == 1
    assert missing["outcome"] == "missing_file" and missing["note_missing"] == 1
    assert missing["src_notes"] == 1
    assert no_match["outcome"] == "no_match"


def test_grounding_uses_only_earlier_steps(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    read = ("simple-tool.bash", _bash(
        "grep -n '^## Plastics' /workspace/reference/technologies/modern.md",
        stdout="/workspace/reference/technologies/modern.md:12:## Plastics",
    ))
    research = ("mcp-tool.set-research", _action({"Technology": "Plastics"}))
    _make_trace_db(trace, _turn("a", 4, [[read, research]]) + _turn("b", 5, [[read], [research]]))
    rows = [r for r in _rows(trace) if r["tool"] == "set-research"]
    # The read in the same step came from the same reply, so the action could not use it.
    assert [(r["run"], r["grounded"]) for r in rows] == [(1, 0), (2, 1)]


def test_extract_keeps_runs_without_tool_calls(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    _make_trace_db(trace, _turn("a", 4, [[]]) + _turn("b", 5, [[
        ("mcp-tool.keep-status-quo", _action({}, "true")),
    ]]))
    rows = _rows(trace)
    assert [(r["run"], r["turn"], r["kind"], r["call"], r["tool"]) for r in rows] == [
        (1, 4, "none", "", ""),
        (2, 5, "action", 1, "keep-status-quo"),
    ]


def test_run_steps_counts_every_reply(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    look = ("simple-tool.bash", _bash("ls /workspace/records"))
    act = ("mcp-tool.keep-status-quo", _action({}, "true"))
    # Run 2 has a reply without calls; run 1 made no calls at all.
    _make_trace_db(trace, _turn("a", 4, [[]]) + _turn("b", 5, [[look], [], [act]]))
    assert [(r["run"], r["run_steps"]) for r in _rows(trace)] == [(1, 1), (2, 3), (2, 3)]


def test_extract_counts_only_the_latest_root_per_turn(tmp_path):
    trace = tmp_path / f"{GAME_ID}-player-1.db"
    retried = _turn("old", 5, [[("simple-tool.bash", _bash("ls /workspace/records"))]], status=2)
    latest = _turn("new", 5, [[("simple-tool.bash", _bash("ls /workspace/game"))]])
    following = _turn("next", 6, [[("mcp-tool.keep-status-quo", _action({}, "true"))]])
    _make_trace_db(trace, retried + latest + following)

    rows = _rows(trace)
    assert [(r["turn"], r["run"], r["tool"], r["src_notes"]) for r in rows] == [
        (5, 1, "bash", 1),
        (6, 2, "keep-status-quo", ""),
    ]


def _make_game_db(tmp_path, experiment):
    exp_dir = tmp_path / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    db = exp_dir / f"{GAME_ID}_1700000000000.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
    conn.execute("INSERT INTO GameMetadata VALUES ('gameId', ?)", (GAME_ID,))
    conn.execute("CREATE TABLE PlayerInformations (Key INTEGER, Civilization TEXT, IsMajor INTEGER)")
    conn.execute("INSERT INTO PlayerInformations VALUES (1, 'Rome', 1)")
    conn.commit()
    conn.close()
    return db


def test_export_is_byte_stable_and_skips_other_players(tmp_path, catalog):
    db = _make_game_db(tmp_path, "exp-files-per-5")
    _make_trace_db(db.parent / f"{GAME_ID}-player-1.db", _turn("t", 2, [[
        ("simple-tool.bash", _bash("for f in options victory; do cat /workspace/records/turn-2/$f.md; done")),
        ("mcp-tool.set-research", _action({"Technology": "Pottery"})),
    ]]))
    # A player without a workspace in the same game contributes no rows.
    _make_trace_db(db.parent / f"{GAME_ID}-player-2.db", [
        ("u", "root", None, 2, "strategist.turn.2", {}, 1),
        ("u", "agent", "root", 2, "agent.simple-strategist", {}, 1),
        ("u", "s1", "agent", 2, "agent.simple-strategist.step.1", {}, 1),
        ("u", "c1", "s1", 2, "mcp-tool.set-research", _action({"Technology": "Pottery"}), 1),
    ])

    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    assert export_agentic_data([str(db)], {GAME_ID}, str(first), catalog) == 2
    assert export_agentic_data([str(db)], {GAME_ID}, str(second), catalog) == 2
    assert first.read_bytes() == second.read_bytes()

    rows = list(csv.DictReader(first.open(encoding="utf-8")))
    assert list(rows[0]) == AGENTIC_FIELDNAMES
    assert {r["player_id"] for r in rows} == {"1"}
    assert rows[0]["experiment"] == "exp-files-per-5"
    assert rows[0]["rec_options"] == "1" and rows[0]["rec_victory"] == "1"
    # A second run skips the already-exported game.
    assert export_agentic_data([str(db)], {GAME_ID}, str(first), catalog) == 0
