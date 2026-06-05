"""Extraction-time errors (distinct from config-load :class:`ConfigError`)."""

from __future__ import annotations


class ExtractError(Exception):
    """A hard extraction policy violation (e.g. a controlled-seed mismatch).

    Raised while reading raw game DBs — after config validation has passed — so
    it is a separate type from :class:`bench.config.errors.ConfigError`.
    """
