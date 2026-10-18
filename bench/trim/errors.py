"""Trim-command errors (distinct from config-load :class:`ConfigError`)."""

from __future__ import annotations


class TrimError(Exception):
    """A hard failure that stops ``civ-bench trim`` before any file is renamed.

    Raised for missing inputs such as an absent ``games`` table. A single game whose
    files cannot be renamed is not an error: it is recorded as a ``failed`` outcome
    and the batch carries on.
    """
