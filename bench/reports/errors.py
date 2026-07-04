"""Report-stage error type (stage 5)."""

from __future__ import annotations


class ReportError(Exception):
    """Raised when the report stage cannot render (missing artifacts, bad template,
    unsupported format, unknown section id). Fail loud — never emit a partial doc."""
