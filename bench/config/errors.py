"""Configuration errors.

Every validation failure raises ``ConfigError`` with a precise, human-readable
message (path into the run-spec + what is wrong). Per invariant 1 (config over
code), unknown keys and missing required fields are hard errors, never silently
ignored.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """A run-spec failed to load or validate."""
