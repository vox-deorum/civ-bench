"""Behavior families for the ``behavior`` table (benchmark.md §3.0).

Each family turns one kind of game record into per-player columns. A family has
``extract(cursor, ctx, selection, stats)``, which returns
``{player_id: {column: value}}``. Its selection comes from the config key of the
same name, and its column names and kinds come from
``bench.config.behavior.family_columns``. To add a family, register it here,
list its allowed names in ``bench.config.schema.BEHAVIOR_FAMILIES``, and name its
columns in ``family_columns``.
"""

from __future__ import annotations

from .context import GameContext, build_game_context, snake_case
from .events import EVENT_COUNTERS, EventCounter, EventFamily
from .policies import PolicyFamily
from .relationships import RelationshipFamily
from .state import StateFamily

FAMILIES = {
    "flavor": StateFamily(table="FlavorChanges", prefix="flavor", gated=True),
    "persona": StateFamily(table="PersonaChanges", prefix="persona"),
    "events": EventFamily(),
    "policies": PolicyFamily(),
    "relationships": RelationshipFamily(),
}

__all__ = [
    "EVENT_COUNTERS",
    "FAMILIES",
    "EventCounter",
    "EventFamily",
    "GameContext",
    "PolicyFamily",
    "RelationshipFamily",
    "StateFamily",
    "build_game_context",
    "snake_case",
]
