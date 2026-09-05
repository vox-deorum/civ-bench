"""``civ-bench fix``: best-effort recovery of malformed game SQLite DBs.

Public surface: the :func:`run_fix` orchestrator (config-driven, reads the
``import_issues.csv`` ledger) and the testable :func:`repair_database` core.
"""

from __future__ import annotations

from .errors import FixError
from .repair import RepairReport, repair_database
from .runner import FixOutcome, FixResult, run_fix

__all__ = [
    "FixError",
    "FixOutcome",
    "FixResult",
    "RepairReport",
    "repair_database",
    "run_fix",
]
