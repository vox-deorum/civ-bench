"""Pure resolution of the ``data.extract.behavior`` selection (benchmark.md §3.0).

Kept free of the extract layer so the config loader can validate it cheaply, and
idempotent so the extract runner can resolve a config that skipped the loader.
It is also the single source of the table's column names and their kinds, which
the extract families write and the ``behavior.*`` analyses read.
"""

from __future__ import annotations

import re
from typing import Any

from . import schema as S
from .errors import ConfigError


def snake_case(name: str) -> str:
    """``UseNuke`` → ``use_nuke``; the column-name convention of the CSV tables."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def family_columns(family: str, selection, stats) -> list[tuple[str, str]]:
    """``(column, kind)`` pairs one behavior family writes for its selection."""
    if family in ("flavor", "persona"):
        return [(f"{family}_{snake_case(name)}_{stat}", "level") for name in selection for stat in stats]
    if family == "events":
        return [(name, "count") for name in selection]
    if family == "policies":
        return [(name, "count" if name == "policy_changes" else "turn") for name in selection]
    if family == "relationships":
        return [(name, S.BEHAVIOR_RELATIONSHIP_KINDS[name]) for name in selection]
    raise ValueError(f"unknown behavior family '{family}'")


def behavior_columns(raw: Any) -> dict[str, str]:
    """Every value column of the ``behavior`` table in header order, mapped to its kind."""
    spec = resolve_behavior_spec(raw)
    columns: dict[str, str] = {}
    for family in S.BEHAVIOR_FAMILIES:
        columns.update(family_columns(family, spec[family], spec["stats"]))
    return columns


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
