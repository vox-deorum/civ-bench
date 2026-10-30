"""Small shared validators for run-spec parsing."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from .errors import ConfigError
from bench.config.schema import DEFAULT_RUNS_DIR


def resolve_source_roots(extract: dict) -> dict[str, Path]:
    """Resolve and validate source roots without importing the extraction layer."""
    authored = {"hot": ("runs_dir", extract.get("runs_dir", DEFAULT_RUNS_DIR))}
    if extract.get("cold_runs_dir") is not None:
        authored["cold"] = ("cold_runs_dir", extract["cold_runs_dir"])
    roots = {}
    for tier, (key, value) in authored.items():
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"data.extract.{key}: expected a non-empty string.")
        try:
            roots[tier] = Path(value).resolve()
        except (OSError, RuntimeError) as exc:
            raise ConfigError(f"data.extract.{key}: cannot resolve {value!r}: {exc}.") from exc
    if "cold" in roots:
        hot, cold = roots["hot"], roots["cold"]
        if hot == cold or hot in cold.parents or cold in hot.parents:
            raise ConfigError(
                "data.extract.cold_runs_dir: the hot and cold roots overlap "
                f"({hot} and {cold}); they must be separate directories."
            )
    return roots


def coerce_bool(value: Any, where: str) -> bool:
    """Return a bool, accepting case-insensitive JSON-ish string booleans."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ConfigError(f"{where}: expected bool, got {type(value).__name__}.")
