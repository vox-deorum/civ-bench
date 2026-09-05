"""Fix-stage errors (distinct from config-load :class:`ConfigError`)."""

from __future__ import annotations


class FixError(Exception):
    """A hard failure in the ``fix`` stage that aborts the whole run.

    Reserved for orchestration-level problems (e.g. ``runs_dir`` unreadable); a
    single database that cannot be repaired is **not** an error: it is recorded as
    a ``failed`` outcome and the loop carries on. So this is rarely raised; most
    of ``fix`` is best-effort by design.
    """
