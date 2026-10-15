"""Event family: per-player counts of selected ``GameEvents``.

Each counter is data: an event type, the payload field naming the actor, and
optional payload filters. All selected types are read in one pass per DB.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..game_events import fetch_events, payload_get, unique_events
from .context import GameContext


@dataclass(frozen=True)
class EventCounter:
    type: str
    actor: str                   # dotted payload path to the acting player (or team)
    match: tuple = ()            # ((path, value), ...) that must all hold
    require: str | None = None   # dotted payload path that must be present
    actor_is_team: bool = False  # actor is a team id; credit every major on it
    # Dotted payload path to the other side, which must be of ``target_kind``
    # ("major" or "minor") when that is set.
    target: str | None = None
    target_is_team: bool = False
    target_kind: str | None = None

    def accepts(self, payload: dict, ctx: GameContext | None = None) -> bool:
        if any(payload_get(payload, path) != value for path, value in self.match):
            return False
        if self.require is not None and payload_get(payload, self.require) in (None, ""):
            return False
        if self.target_kind is not None:
            kinds = (ctx.team_kind if self.target_is_team else ctx.player_kind) if ctx else {}
            if kinds.get(payload_get(payload, self.target)) != self.target_kind:
                return False
        return True

    def credited(self, payload: dict, ctx: GameContext) -> list:
        """Major players this event counts for (empty when it does not count)."""
        if not self.accepts(payload, ctx):
            return []
        actor = payload_get(payload, self.actor)
        players = ctx.players_on_team(actor) if self.actor_is_team else [actor]
        majors = set(ctx.major_players)
        return [pid for pid in players if pid in majors]


EVENT_COUNTERS = {
    # The aggressor's own declarations, including the automatic ones on the
    # target's defensive-pact partners; a vassal following its master is not.
    "wars_declared": EventCounter("DeclareWar", actor="OriginatingPlayerID",
                                  match=(("IsAggressor", True),)),
    "wars_received": EventCounter("DeclareWar", actor="TargetTeamID", actor_is_team=True),
    "cities_nuked": EventCounter("NuclearDetonation", actor="PlayerID", require="Plot.City"),
    "cities_razed": EventCounter("CityRazed", actor="PlayerID"),
    # The share of wars_declared aimed at city-states.
    "wars_on_city_states": EventCounter("DeclareWar", actor="OriginatingPlayerID",
                                        match=(("IsAggressor", True),),
                                        target="TargetTeamID", target_is_team=True,
                                        target_kind="minor"),
    "units_lost": EventCounter("UnitKilledInCombat", actor="KilledPlayerID"),
    # Kills of any unit: a civ's, a city-state's, or a barbarian's.
    "units_killed": EventCounter("UnitKilledInCombat", actor="KillerPlayerID"),
    "cities_captured": EventCounter("CityCaptureComplete", actor="NewOwnerID",
                                    match=(("IsConquest", True),),
                                    target="OldOwnerID", target_kind="major"),
    "capitals_captured": EventCounter("CityCaptureComplete", actor="NewOwnerID",
                                      match=(("IsConquest", True), ("IsCapital", True)),
                                      target="OldOwnerID", target_kind="major"),
    "city_state_cities_captured": EventCounter("CityCaptureComplete", actor="NewOwnerID",
                                               match=(("IsConquest", True),),
                                               target="OldOwnerID", target_kind="minor"),
    # Atomic Bombs and Nuclear Missiles, trained or bought; ``Unit`` is the
    # trained unit's metadata the game logs with the event.
    "nukes_built": EventCounter("CityTrained", actor="PlayerID", match=(("Unit.AIType", "Icbm"),)),
    "nukes_detonated": EventCounter("NuclearDetonation", actor="PlayerID"),
}


def read_events(cursor, counters) -> list:
    """The unique ``(Type, Turn, payload)`` events any of ``counters`` reads."""
    return unique_events(fetch_events(cursor, sorted({c.type for c in counters})))


def credits(events, counters: dict, ctx: GameContext):
    """Yield ``(name, players)`` for each event a counter counts, with its majors."""
    for event_type, _turn, payload in events:
        for name, counter in counters.items():
            if counter.type != event_type:
                continue
            players = counter.credited(payload, ctx)
            if players:
                yield name, players


def count_events(events, counters: dict, ctx: GameContext) -> dict:
    """``{player_id: {name: count}}`` for every major and every counter in ``counters``."""
    counts = {pid: dict.fromkeys(counters, 0) for pid in ctx.major_players}
    for name, players in credits(events, counters, ctx):
        for pid in players:
            counts[pid][name] += 1
    return counts


def count_game_events(events, counters: dict, ctx: GameContext) -> dict:
    """``{name: count}``: each counted event once, however many majors it credits."""
    counts = dict.fromkeys(counters, 0)
    for name, _players in credits(events, counters, ctx):
        counts[name] += 1
    return counts


class EventFamily:
    def extract(self, cursor, ctx: GameContext, selection, stats) -> dict:
        if not selection:
            return {}
        counters = {name: EVENT_COUNTERS[name] for name in selection}
        return count_events(read_events(cursor, counters.values()), counters, ctx)
