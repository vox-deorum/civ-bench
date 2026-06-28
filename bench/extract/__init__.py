"""Stage 1 — extract: raw game SQLite DBs in ``runs/`` → canonical CSVs.

Public surface: the :func:`run_extract` orchestrator (config-driven), the
controlled-design helpers (:func:`extract_seeding_fields`, :class:`SeedingInfo`),
the orthodox identity composition (:func:`compose_identities`), and the four
per-table exporters.
"""

from __future__ import annotations

from .errors import ExtractError
from .identity import compose_identities
from .issues import ImportIssue, ImportIssueLog
from .runner import ExtractResult, run_extract
from .utilities import (
    SeedingInfo,
    extract_seeding_fields,
    find_all_databases,
    invert_seating_map,
    outputs_are_fresh,
)

__all__ = [
    "ExtractError",
    "ExtractResult",
    "ImportIssue",
    "ImportIssueLog",
    "SeedingInfo",
    "compose_identities",
    "extract_seeding_fields",
    "find_all_databases",
    "invert_seating_map",
    "outputs_are_fresh",
    "run_extract",
]
