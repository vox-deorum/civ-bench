"""Pure resolution of the ``data.extract.behavior`` selection (benchmark.md §3.0).

Kept free of the extract layer so the config loader can validate it cheaply, and
idempotent so the extract runner can resolve a config that skipped the loader.
"""

from __future__ import annotations

from typing import Any

from . import schema as S
from .errors import ConfigError


def resolve_behavior_spec(raw: Any, where: str = "data.extract.behavior") -> dict:
    """Validate ``raw`` and return ``{stats, <family>...}`` with defaults filled in."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected an object, got {type(raw).__name__}.")

    allowed = {"stats", *S.BEHAVIOR_FAMILIES}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}. Allowed: {sorted(allowed)}.")

    domains = {"stats": S.BEHAVIOR_STATS, **S.BEHAVIOR_FAMILIES}
    resolved: dict = {}
    for key, domain in domains.items():
        value = raw.get(key, S.BEHAVIOR_DEFAULTS[key])
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ConfigError(f"{where}.{key}: expected a list of strings.")
        bad = [v for v in value if v not in domain]
        if bad:
            raise ConfigError(f"{where}.{key}: unknown name(s) {bad}. Allowed: {list(domain)}.")
        dupes = sorted({v for v in value if value.count(v) > 1})
        if dupes:
            raise ConfigError(f"{where}.{key}: duplicate name(s) {dupes}.")
        resolved[key] = list(value)
    return resolved
