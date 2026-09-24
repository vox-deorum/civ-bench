"""Behavior families for the ``behavior`` table (benchmark.md §3.0).

Each family turns one kind of game record into per-player columns. A family has
``columns(selection, stats)`` and ``extract(cursor, ctx, selection, stats)``,
which returns ``{player_id: {column: value}}``. Its selection comes from the
config key of the same name. To add a family, register it here and list its
allowed names in ``bench.config.schema.BEHAVIOR_FAMILIES``.
"""

from __future__ import annotations

from .context import GameContext, build_game_context, snake_case
from .events import EVENT_COUNTERS, EventCounter, EventFamily
from .policies import PolicyFamily
from .state import StateFamily

FAMILIES = {
    "flavor": StateFamily(table="FlavorChanges", prefix="flavor"),
    "persona": StateFamily(table="PersonaChanges", prefix="persona"),
    "events": EventFamily(),
    "policies": PolicyFamily(),
}

__all__ = [
    "EVENT_COUNTERS",
    "FAMILIES",
    "EventCounter",
    "EventFamily",
    "GameContext",
    "PolicyFamily",
    "StateFamily",
    "build_game_context",
    "snake_case",
]
