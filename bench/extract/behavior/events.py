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
    # Credit the majors on both sides, once per unordered pair of civ teams per
    # turn (an event both sides may log, such as a peace deal). Both teams
    # must be civ teams.
    both_sides: bool = False

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

    def pair(self, payload: dict, ctx: GameContext) -> frozenset:
        """The unordered team pair of a ``both_sides`` event."""
        actor = payload_get(payload, self.actor)
        actor_team = actor if self.actor_is_team else ctx.team_of.get(actor, actor)
        target = payload_get(payload, self.target)
        target_team = target if self.target_is_team else ctx.team_of.get(target, target)
        return frozenset((actor_team, target_team))

    def credited(self, payload: dict, ctx: GameContext) -> list:
        """Major players this event counts for (empty when it does not count)."""
        if not self.accepts(payload, ctx):
            return []
        if self.both_sides:
            # Both sides must be civ teams: a city-state's deal with a civ is
            # not a deal between civs, whichever side logged it.
            pair = self.pair(payload, ctx)
            if len(pair) != 2 or any(ctx.team_kind.get(team) != "major" for team in pair):
                return []
            return [pid for team in sorted(pair, key=str)
                    for pid in ctx.players_on_team(team)]
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
    "peace_deals": EventCounter("MakePeace", actor="OriginatingPlayerID",
                                target="TargetTeamID", target_is_team=True,
                                target_kind="major", both_sides=True),
    "units_lost": EventCounter("UnitKilledInCombat", actor="KilledPlayerID"),
    "cities_captured": EventCounter("CityCaptureComplete", actor="NewOwnerID",
                                    match=(("IsConquest", True),),
                                    target="OldOwnerID", target_kind="major"),
    "capitals_captured": EventCounter("CityCaptureComplete", actor="NewOwnerID",
                                      match=(("IsConquest", True), ("IsCapital", True)),
                                      target="OldOwnerID", target_kind="major"),
    "city_state_cities_captured": EventCounter("CityCaptureComplete", actor="NewOwnerID",
                                               match=(("IsConquest", True),),
                                               target="OldOwnerID", target_kind="minor"),
    "nukes_detonated": EventCounter("NuclearDetonation", actor="PlayerID"),
}


def read_events(cursor, counters) -> list:
    """The unique ``(Type, Turn, payload)`` events any of ``counters`` reads."""
    return unique_events(fetch_events(cursor, sorted({c.type for c in counters})))


def credits(events, counters: dict, ctx: GameContext):
    """Yield ``(name, players)`` for each event a counter counts, with its majors.

    A ``both_sides`` counter counts one event per unordered team pair per turn,
    since both sides may log the same deal (often several times).
    """
    pairs_seen = {name: set() for name, c in counters.items() if c.both_sides}
    for event_type, turn, payload in events:
        for name, counter in counters.items():
            if counter.type != event_type:
                continue
            players = counter.credited(payload, ctx)
            if not players:
                continue
            if counter.both_sides:
                key = (turn, counter.pair(payload, ctx))
                if key in pairs_seen[name]:
                    continue
                pairs_seen[name].add(key)
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
