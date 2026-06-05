"""Player-type styling helpers (ported from ``shared/plot_styles.py``).

The old module read a process-global catalog; here every helper takes an explicit
:class:`civ_bench.catalog.Catalog` so styling stays config-driven (invariant 1).
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..catalog import Catalog


def sort_player_types(player_types: Iterable[str]) -> list[str]:
    pinned = ["Null", "Vanilla"]
    pts = set(player_types)
    return [p for p in pinned if p in pts] + sorted(pts - set(pinned))


def get_player_color(catalog: Catalog, player_type: str) -> str:
    parsed = catalog.split_player_type(player_type)
    model_id = parsed["model_id"]
    colors = catalog.strategist_model_colors()
    return colors.get(model_id, "#000000")


def _style_key(catalog: Catalog, player_type: str) -> Optional[str]:
    patterns = catalog.prompt_patterns()
    if player_type in patterns:
        return player_type
    parsed = catalog.split_player_type(player_type)
    return catalog.get_variant_style_key(parsed["variant"])


def get_player_hatch(catalog: Catalog, player_type: str):
    key = _style_key(catalog, player_type)
    patterns = catalog.prompt_patterns()
    return patterns[key]["hatch"] if key in patterns else None


def get_player_linestyle(catalog: Catalog, player_type: str) -> str:
    key = _style_key(catalog, player_type)
    patterns = catalog.prompt_patterns()
    return patterns[key]["linestyle"] if key in patterns else "-"


def get_player_marker(catalog: Catalog, player_type: str) -> str:
    key = _style_key(catalog, player_type)
    patterns = catalog.prompt_patterns()
    return patterns[key]["marker"] if key in patterns else "o"


def get_player_alpha(catalog: Catalog, player_type: str) -> float:
    key = _style_key(catalog, player_type)
    patterns = catalog.prompt_patterns()
    return patterns[key]["alpha"] if key in patterns else 0.8


def get_all_player_styles(catalog: Catalog, player_types: Iterable[str]) -> dict:
    return {
        pt: {
            "color": get_player_color(catalog, pt),
            "hatch": get_player_hatch(catalog, pt),
            "linestyle": get_player_linestyle(catalog, pt),
            "marker": get_player_marker(catalog, pt),
            "alpha": get_player_alpha(catalog, pt),
        }
        for pt in player_types
    }
