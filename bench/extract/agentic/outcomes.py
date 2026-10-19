"""Turn a workspace tool's raw result into an outcome label."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

OUTCOMES = ("ok", "no_match", "missing_file", "read_only", "syntax", "not_found", "other")

_TRUNCATED = re.compile(r"\[truncated (\d+) chars")
_SYNTAX = re.compile(r"syntax error|unexpected EOF|unterminated|unexpected end of file|bad substitution", re.I)


def parse_tool_output(raw) -> Tuple[str, str, Optional[int]]:
    """``(stdout, stderr, exit_code)`` from a ``tool.output`` attribute.

    Shell tools report ``{"stdout", "stderr", "exitCode"}``. Anything else is kept
    as plain stdout with an unknown exit code.
    """
    if raw is None:
        return "", "", None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return raw, "", None
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        return str(raw), "", None
    stdout = parsed.get("stdout") or ""
    stderr = parsed.get("stderr") or ""
    exit_code = parsed.get("exitCode")
    try:
        exit_code = int(exit_code) if exit_code is not None else None
    except (TypeError, ValueError):
        exit_code = None
    return str(stdout), str(stderr), exit_code


def truncated_chars(stdout: str) -> int:
    """Characters the host cut from the output, from its ``[truncated N chars`` note."""
    return sum(int(n) for n in _TRUNCATED.findall(stdout))


def classify_outcome(exit_code: Optional[int], stderr: str) -> str:
    """One of :data:`OUTCOMES`.

    Exit 1 with nothing on stderr is how ``grep`` and ``test`` say "no match", so it
    is ``no_match`` rather than an error.
    """
    if exit_code == 0:
        return "ok"
    text = stderr.strip()
    if exit_code == 1 and not text:
        return "no_match"
    if "read-only file system" in text.lower() or "EROFS" in text:
        return "read_only"
    if "No such file or directory" in text or "cannot access" in text:
        return "missing_file"
    if _SYNTAX.search(text):
        return "syntax"
    if exit_code == 127 or "command not found" in text:
        return "not_found"
    return "other"


def action_succeeded(raw, status_code: Optional[int]) -> int:
    """1 when a game action reported success, else 0.

    Actions answer ``{"Success": true}`` or ``true``. A failure is
    ``{"Success": false, ...}``, an MCP error ``{"content": [...], "isError": ...}``,
    or an error status on the span.
    """
    if status_code == 2:
        return 0
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return 0
    if parsed is True:
        return 1
    if isinstance(parsed, dict):
        if "Success" in parsed:
            return 1 if parsed.get("Success") else 0
        if parsed.get("isError"):
            return 0
        content = parsed.get("content")
        if isinstance(content, list):
            text = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
            return 0 if text.lstrip().lower().startswith("error") else 1
    return 0
