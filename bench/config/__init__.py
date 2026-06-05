"""Config layer: load + validate the run-spec, expose typed objects.

Kept import-light (no heavy analysis deps) so the dry-run path stays cheap.
"""

from __future__ import annotations

from .errors import ConfigError
from .loader import load_config
from .models import OutputConfig, RunConfig, Stage

__all__ = ["ConfigError", "load_config", "RunConfig", "Stage", "OutputConfig"]
