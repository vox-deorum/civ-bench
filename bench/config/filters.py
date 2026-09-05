"""Pure config-filter helpers.

Filters are accepted in config as inline objects, preset names, or lists that
merge those forms left-to-right. This module keeps that resolution independent
from pandas so the config loader can validate filters without importing the data
layer, while ``bench.data`` can reuse the same semantics when applying filters.
"""

from __future__ import annotations

from typing import Any

from . import schema as S
from .errors import ConfigError
from .validation import coerce_bool


def _check_filter_keys(obj: dict, where: str) -> None:
    unknown = sorted(set(obj) - set(S.FILTER_KEYS))
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s) {unknown}. Allowed: {sorted(S.FILTER_KEYS)}."
        )


def validate_filter_object(obj: dict, where: str) -> None:
    if not isinstance(obj, dict):
        raise ConfigError(f"{where}: expected an object, got {type(obj).__name__}.")
    _check_filter_keys(obj, where)

    # The membership fields must be lists of strings. A scalar (e.g. ``5`` or
    # ``"solo"``) would be silently wrapped by ``_as_list`` at apply time and
    # match nothing / the wrong thing, dropping every row without an error,
    # exactly the kind of silent no-op the fail-loud invariant forbids (§3.1).
    for key in ("experiments", "exclude_experiments", "players"):
        if key in obj and obj[key] is not None:
            value = obj[key]
            if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
                raise ConfigError(
                    f"{where}.{key}: expected a list of strings, got {value!r}."
                )

    if "only_llm" in obj:
        obj["only_llm"] = coerce_bool(obj["only_llm"], f"{where}.only_llm")

    if "min_games" in obj and obj["min_games"] is not None:
        mg = obj["min_games"]
        if isinstance(mg, bool) or not isinstance(mg, int) or mg < 0:
            raise ConfigError(
                f"{where}.min_games: expected a non-negative integer, got {mg!r}."
            )

    tr = obj.get("turn_range")
    if tr is not None:
        if not isinstance(tr, (list, tuple)) or len(tr) != 2:
            raise ConfigError(
                f"{where}.turn_range: expected [min, max] (either bound nullable)."
            )
        lo, hi = tr
        for label, bound in (("min", lo), ("max", hi)):
            # Reject non-numeric bounds here so the comparison below can't raise a
            # raw TypeError (bools are ints in Python but not valid turn bounds).
            if bound is not None and (isinstance(bound, bool) or not isinstance(bound, (int, float))):
                raise ConfigError(
                    f"{where}.turn_range: {label} bound must be a number or null, got {bound!r}."
                )
        if lo is not None and hi is not None and lo > hi:
            raise ConfigError(
                f"{where}.turn_range: min ({lo}) must be <= max ({hi})."
            )

    # `min_condition_completeness` (null ⇒ keep every controlled condition): a number
    # in (0, 1] drops, as a whole, every controlled experiment whose occupied
    # (seed, seating_rotation) slot fraction is below it (1.0 ⇒ drop any missing slot).
    mcc = obj.get("min_condition_completeness")
    if mcc is not None:
        if isinstance(mcc, bool) or not isinstance(mcc, (int, float)) or not 0 < mcc <= 1:
            raise ConfigError(
                f"{where}.min_condition_completeness: must be null or a "
                "number in (0, 1]."
            )


def resolve_filter_spec(spec: Any, presets: dict, where: str = "filter") -> dict:
    """Resolve a filter spec to one inline object.

    ``spec`` may be ``None``, a preset name, an inline object, or a list mixing
    names/objects. Lists merge left-to-right: later entries replace earlier
    values for the same field.
    """
    if spec is None:
        return {}
    if isinstance(spec, str):
        if spec not in presets:
            raise ConfigError(
                f"{where}: references undefined filter preset '{spec}'. "
                f"Defined presets: {sorted(presets)}."
            )
        return resolve_filter_spec(presets[spec], presets, f"filters.{spec}")
    if isinstance(spec, dict):
        validate_filter_object(spec, where)
        return dict(spec)
    if isinstance(spec, list):
        merged: dict = {}
        for i, item in enumerate(spec):
            merged.update(resolve_filter_spec(item, presets, f"{where}[{i}]"))
        return merged
    raise ConfigError(
        f"{where}: a filter must be a preset name, an inline object, or a "
        f"list of those (got {type(spec).__name__})."
    )


def intersect_filter_specs(base: dict, narrow: dict) -> dict:
    """Return the effective filter produced by narrowing ``base`` with ``narrow``."""
    out = dict(base)

    for key in ("experiments", "players"):
        left = _as_set(out.get(key))
        right = _as_set(narrow.get(key))
        if left is not None and right is not None:
            out[key] = sorted(left & right)
        elif right is not None:
            out[key] = sorted(right)

    excludes = set(out.get("exclude_experiments") or [])
    excludes.update(narrow.get("exclude_experiments") or [])
    if excludes:
        out["exclude_experiments"] = sorted(excludes)

    if base.get("only_llm") or narrow.get("only_llm"):
        out["only_llm"] = True

    if "min_games" in base or "min_games" in narrow:
        out["min_games"] = max(base.get("min_games", 1), narrow.get("min_games", 1))

    # Completeness is a floor: a stage can only require more coverage than the
    # global filter, so the effective threshold is the stricter (higher) of the two.
    if "min_condition_completeness" in base or "min_condition_completeness" in narrow:
        thresholds = [
            v for v in (
                base.get("min_condition_completeness"),
                narrow.get("min_condition_completeness"),
            ) if v is not None
        ]
        if thresholds:
            out["min_condition_completeness"] = max(thresholds)

    if "turn_range" in base or "turn_range" in narrow:
        blo, bhi = _range_bounds(base.get("turn_range"))
        nlo, nhi = _range_bounds(narrow.get("turn_range"))
        lo = max(v for v in (blo, nlo) if v is not None) if any(
            v is not None for v in (blo, nlo)
        ) else None
        hi = min(v for v in (bhi, nhi) if v is not None) if any(
            v is not None for v in (bhi, nhi)
        ) else None
        out["turn_range"] = [lo, hi]
    return out


def ensure_filter_narrows(base: dict, candidate: dict, where: str) -> None:
    """Raise if ``candidate`` tries to select rows excluded by ``base``."""
    _check_subset_constraint(
        base, candidate, "experiments", "exclude_experiments", where
    )
    _check_subset_constraint(base, candidate, "players", None, where)

    if base.get("only_llm") and candidate.get("only_llm") is False:
        raise ConfigError(f"{where}.only_llm: cannot widen global only_llm=true.")

    if "min_games" in base and candidate.get("min_games", base["min_games"]) < base["min_games"]:
        raise ConfigError(
            f"{where}.min_games: cannot be lower than global min_games={base['min_games']}."
        )

    base_mcc = base.get("min_condition_completeness")
    if base_mcc is not None and candidate.get("min_condition_completeness", base_mcc) < base_mcc:
        raise ConfigError(
            f"{where}.min_condition_completeness: cannot be lower than the global "
            f"min_condition_completeness={base_mcc}."
        )

    if "turn_range" in candidate:
        blo, bhi = _range_bounds(base.get("turn_range"))
        clo, chi = _range_bounds(candidate.get("turn_range"))
        if blo is not None and (clo is None or clo < blo):
            raise ConfigError(
                f"{where}.turn_range: min cannot be lower than global min {blo}."
            )
        if bhi is not None and (chi is None or chi > bhi):
            raise ConfigError(
                f"{where}.turn_range: max cannot be higher than global max {bhi}."
            )


def _check_subset_constraint(
    base: dict,
    candidate: dict,
    include_key: str,
    exclude_key: str | None,
    where: str,
) -> None:
    selected = _as_set(candidate.get(include_key))
    if selected is None:
        return
    base_allowed = _as_set(base.get(include_key))
    if base_allowed is not None:
        outside = sorted(selected - base_allowed)
        if outside:
            raise ConfigError(
                f"{where}.{include_key}: selects values outside the global filter: "
                f"{outside}."
            )
    if exclude_key is not None:
        excluded = set(base.get(exclude_key) or [])
        blocked = sorted(selected & excluded)
        if blocked:
            raise ConfigError(
                f"{where}.{include_key}: selects values excluded globally: {blocked}."
            )


def _as_set(value: Any) -> set | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {value}
    return set(value)


def _range_bounds(value: Any) -> tuple[Any, Any]:
    if value is None:
        return None, None
    return value[0], value[1]
