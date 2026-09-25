"""Release the rendered report as a git repository with a GitHub Pages workflow.

Public surface: :func:`run_publish` (initialize, summarize, confirm, commit,
push) and :func:`publish_enabled` (whether ``report.publish`` is on).
"""

from __future__ import annotations

from .errors import PublishError
from .runner import PublishResult, commit_message, publish_enabled, run_publish

__all__ = [
    "PublishError",
    "PublishResult",
    "commit_message",
    "publish_enabled",
    "run_publish",
]
