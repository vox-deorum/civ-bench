"""Publish-step errors (distinct from config-load :class:`ConfigError`)."""

from __future__ import annotations


class PublishError(Exception):
    """A failure while committing or pushing the report repository.

    Raised for a missing ``git``, a failed git command, a file too large for
    GitHub, or a failed push. The rendered report on disk is never touched by
    the failure; a failed push keeps its local commit.
    """
