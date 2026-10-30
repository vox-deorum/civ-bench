"""``world.progress``: how large and productive the civs and the world became.

For each metric of :data:`bench.config.schema.WORLD_PROGRESS_METRICS`, read at
each checkpoint turn of the ``turns`` table, the Player column is the
strategist's own civ and the World column the other civs in the game (a sum,
except techs, the average over surviving other civs, and civs alive, a count).
Both columns match the baseline in the same seat on the same map:

* a civ eliminated by the checkpoint counts as 0 there;
* a game with no turn at the checkpoint (it ended earlier) is left out of it.

The Endings tab reads the panel: the share of the civ's games it won by each
victory type (Player) and the share of games that ended that way (World), plus
game length.

Views, on two switch rows. The first picks **relative** (default) or
**absolute**, with the same names as the other world and behavior sections so
the choice syncs across the page. The second picks the tab:

* **Overview** (default): ``overview_metrics`` at every checkpoint;
* **Turn N**: every metric at checkpoint N;
* **Endings**: win shares and game length.

Each view is named ``<outer>/<tab>`` (such as ``relative/turn_100``) and
:func:`bench.analyses.behavior.common.nested_views_metadata` nests them. An
uncontrolled run has no relative view, so only the tab row shows.

In controlled runs the module also writes ``world_progress_by_seed`` and
``world_progress_by_seat`` for the Matched Maps chapter's World: progress tab
(the overview metrics at the last checkpoint).
"""

from __future__ import annotations

import re

import pandas as pd

from bench.config import schema as S
from bench.analyses.base import AnalysisContext, AnalysisResult
from bench.analyses.behavior import common as C
from bench.analyses.behavior.base import MetricViews
from bench.analyses.world.base import (
    PLAYER, WORLD, WorldAnalysis, merge_columns, metric_id, paired_columns,
)

SLUG = "world_progress"
OVERVIEW = "overview"
ENDINGS = "endings"
# World values averaged over the other civs alive instead of summed.
AVERAGED = {"technologies"}
NO_WINNER = "no_winner"
GAME_LENGTH = "game_length"


def turn_view(turn: int) -> str:
    return f"turn_{turn}"


def ending_id(victory_type: str) -> str:
    return "win_" + re.sub(r"[^0-9a-z]+", "_", str(victory_type).lower()).strip("_")


class WorldProgress(WorldAnalysis):
    module = "world.progress"
    friendly_name = "World growth"
    description = (
        "Reads population, cities, land, techs, yields, and armies at fixed turns, and how "
        "games ended, comparing the strategist's own civ with the other civs at each checkpoint."
    )
    report_defaults = {
        "tables": ["world_progress_overview_*", "world_progress_t*_*", "world_endings_*"],
        "figures": [],
    }

    @property
    def metrics(self) -> list[str]:
        chosen = set(self.params.get("metrics") or S.WORLD_PROGRESS_METRICS)
        return [m for m in S.WORLD_PROGRESS_METRICS if m in chosen]

    @property
    def checkpoints(self) -> list[int]:
        return sorted(int(t) for t in (self.params.get("checkpoints")
                                       or S.WORLD_DEFAULT_CHECKPOINTS))

    @property
    def overview_metrics(self) -> list[str]:
        metrics = self.metrics
        chosen = self.params.get("overview_metrics")
        if chosen is None:
            chosen = S.WORLD_DEFAULT_OVERVIEW
        return [m for m in chosen if m in metrics]

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        metrics, checkpoints, overview = self.metrics, self.checkpoints, self.overview_metrics
        players = self.players(ctx, extra=["victory_type", "is_winner"])
        stocks = [m for m in metrics if m != "civs_alive"]
        turns = self.turns(ctx, stocks)

        per_turn = {t: paired_columns(S.WORLD_PROGRESS_METRICS, metrics, suffix=f"@{t}",
                                      label=f"T{t}") for t in checkpoints}
        endings_catalog = self._endings_catalog(players)
        endings = paired_columns(endings_catalog, list(endings_catalog))
        endings = endings.subset([c for c in endings.names if c != metric_id(PLAYER, NO_WINNER)])
        overview_columns = merge_columns(*(
            per_turn[t].subset([metric_id(scope, m, f"@{t}") for scope in (PLAYER, WORLD)])
            for m in overview for t in checkpoints
        ))
        columns = merge_columns(*per_turn.values(), endings)

        # Checkpoint values (Player and World) are keyed by player; only the
        # Endings tab's World values are keyed by game.
        seat_frames = []
        for t in checkpoints:
            seat_frames.extend(self._checkpoint_values(turns, metrics, t))
        player_endings, world_endings = self._ending_values(players, turns, endings_catalog)
        seat_frames.append(player_endings)
        seat_values = _merge(seat_frames, ["game_id", "player_id"])

        rows, pool, baseline, per_game = self.frame(
            ctx, players.drop(columns=["victory_type", "is_winner"]),
            seat_values, world_endings, list(columns.names),
        )
        if rows.empty:
            return AnalysisResult(summary="No players remain after filtering.")

        cells = C.controlled_cells(ctx)
        views = C.build_views(rows, pool, list(columns.names), cells, baseline,
                              percent=columns.percent, keep_baseline=True, per_game=per_game)
        controlled = views.relative is not None
        out = MetricViews()
        # Two switch rows: relative or absolute, then the tab. Views are named
        # "<outer>/<tab>" and declared up front so they keep this order.
        tabs = [OVERVIEW, *(turn_view(t) for t in checkpoints), ENDINGS]
        outer = [C.RELATIVE, C.ABSOLUTE] if controlled else [C.ABSOLUTE]
        out.declared = {f"{o}/{tab}": {"tables": [], "figures": []} for o in outer for tab in tabs}

        player_base, world_base = self.baseline_words(views)
        versus = self.versus_words(views)
        pinned = self.pinned_label(views)
        def help_texts(relative_world, absolute_cell, absolute_world, notes):
            """Both views' help texts for one tab, around the lines all tabs share."""
            return {
                C.RELATIVE: "\n".join([
                    f"Each cell: the strategist's average compared with {versus}'s.",
                    f"Player: its own civ, vs. {player_base}.",
                    relative_world,
                    f"Top row: the {pinned}'s actual values, for scale.",
                    "Blue is higher, red is lower; full color at 2 baseline SDs.",
                    *notes[C.RELATIVE],
                ]),
                C.ABSOLUTE: "\n".join([
                    absolute_cell,
                    "Player: the strategist's own civ.",
                    absolute_world,
                    "Color: distance from the average row; full color at 2 SDs.",
                    *notes[C.ABSOLUTE],
                ]),
            }

        shared = "It excludes this player's civ and includes every other civ, including other strategists."
        counting = (
            "Techs are averaged over surviving other civs, and are blank when none survive. "
            "Other World values are sums, with eliminated civs contributing 0. "
            "Games that ended before turn N are left out."
        )
        stock_help = help_texts(
            f"World: all other civs, compared with those around {player_base}. {shared}",
            "Each cell: the average across players' games, as measured.",
            "World: the other civs. " + shared,
            {C.RELATIVE: ["Sizes, yields, and armies: % change. Civs alive: pts (Player) or "
                          "civs (World).", counting],
             C.ABSOLUTE: ["Civs alive: % of games the civ was still alive (Player) or civs "
                          "(World).", counting]},
        )
        ending_help = help_texts(
            f"World: the whole game, vs. {world_base}.",
            "Each cell: the average as measured; World cells count each game once.",
            "World: the whole game.",
            {C.RELATIVE: ["Win shares: pts. Game length: turns."],
             C.ABSOLUTE: ["Win shares: % of games. Game length: turns."]},
        )

        def emit_tab(slug, part, tab, subject, tab_help):
            self.add_paired_views(
                ctx, views, out, slug, part,
                titles={C.RELATIVE: f"{subject} vs. {versus}",
                        C.ABSOLUTE: f"{subject}, actual values"},
                help_texts=tab_help,
                view_names={o: f"{o}/{tab}" for o in outer},
            )

        if overview:
            emit_tab(f"{SLUG}_overview", overview_columns, OVERVIEW, "World growth overview",
                     stock_help)
        for t in checkpoints:
            emit_tab(f"{SLUG}_t{t}", per_turn[t], turn_view(t), f"World growth at turn {t}",
                     stock_help)
        emit_tab("world_endings", endings, ENDINGS, "How games ended", ending_help)

        metadata = {
            **views.metadata(),
            "metrics": metrics,
            "checkpoints": checkpoints,
            "overview_metrics": overview,
            "views": C.nested_views_metadata(out.declared, baseline, inner={
                OVERVIEW: ("Overview", "Selected metrics at every checkpoint"),
                **{turn_view(t): (f"Turn {t}", f"Every metric at turn {t}") for t in checkpoints},
                ENDINGS: ("Endings", "How games ended, and game length"),
            }),
            "heatmaps": out.heatmaps,
        }
        last = checkpoints[-1]
        mapped = overview or metrics
        tab = self.add_paired_matched_maps(
            ctx, views, pool, cells, out, SLUG,
            per_turn[last].subset([metric_id(scope, m, f"@{last}")
                                   for m in mapped for scope in (PLAYER, WORLD)]),
            label="World: progress",
            tip=f"World growth at turn {last}, for the strategist's civ and the other civs",
            title=f"World growth at turn {last}",
            help_text="\n".join([
                "Each cell: the strategist's average on this map (and seat), as measured.",
                "Player: its own civ. World: the other civs, excluding this player's civ.",
                f"Top row: the {pinned} on the same map (and seat).",
                "Hover a cell to see its difference from the top row.",
            ]),
        )
        if tab is not None:
            metadata["matched_maps"] = tab
        headline_slugs = [f"{SLUG}_overview"] if overview else [f"{SLUG}_t{last}"]
        headline_columns = overview_columns if overview else per_turn[last]
        return AnalysisResult(
            tables=out.tables,
            summary=self.world_headline(ctx, out, views, headline_slugs, headline_columns,
                                        world_phrase="the other civs"),
            metadata=metadata,
        )

    # ── inputs ────────────────────────────────────────────────────────────────
    @staticmethod
    def _checkpoint_values(turns: pd.DataFrame, metrics: list[str], t: int):
        """Player and World values at turn ``t``, as ``p__<metric>@t`` / ``w__<metric>@t``."""
        keys = ["game_id", "player_id"]
        stocks = [m for m in metrics if m != "civs_alive"]
        reached = turns.loc[turns["turn"] == t, "game_id"].unique()
        # Every civ of a game that reached t, with its row at t (if any).
        civs = turns.loc[turns["game_id"].isin(reached), keys].drop_duplicates()
        at = turns.loc[turns["turn"] == t, [*keys, "score", *stocks]].drop_duplicates(keys)
        at = civs.merge(at, on=keys, how="left")
        alive = at["score"].fillna(0) > 0
        values = at[keys].copy()
        for metric in stocks:
            values[metric] = at[metric].where(alive, 0.0).fillna(0.0)
        values["civs_alive"] = alive.astype(float)
        grouped = values.groupby("game_id")
        other_alive = grouped["civs_alive"].transform("sum") - values["civs_alive"]
        world = values[keys].copy()
        for metric in metrics:
            others = grouped[metric].transform("sum") - values[metric]
            world[metric_id(WORLD, metric, f"@{t}")] = (
                others / other_alive.where(other_alive > 0) if metric in AVERAGED else others
            )
        values["civs_alive"] *= 100.0
        player = values[keys].copy()
        for metric in metrics:
            player[metric_id(PLAYER, metric, f"@{t}")] = values[metric]
        return player, world

    @staticmethod
    def _endings_catalog(players: pd.DataFrame) -> dict:
        """Ending columns in the ``paired_columns`` catalog form: configured types seen, then others."""
        seen = set(_victory_types(players).dropna())
        ordered = [v for v in S.WORLD_ENDINGS if v in seen]
        ordered += sorted(seen - set(S.WORLD_ENDINGS))
        catalog = {}
        for victory_type in ordered:
            name, short = S.WORLD_ENDINGS.get(victory_type, (victory_type, victory_type))
            catalog[ending_id(victory_type)] = (
                f"{name} wins", short,
                f"Share of the civ's games it won with a {name} victory.",
                f"Share of games that ended in a {name} victory.", "pts", "pts",
            )
        catalog[NO_WINNER] = ("No winner", "None", "", "Share of games with no winner.",
                              "pts", "pts")
        catalog[GAME_LENGTH] = ("Game length", "Turns",
                                "Turns the civ survived (its last turn with a positive score).",
                                "Game length in turns.", "diff", "diff")
        return catalog

    @staticmethod
    def _ending_values(players: pd.DataFrame, turns: pd.DataFrame, catalog: dict):
        keys = ["game_id", "player_id"]
        victory = _victory_types(players)
        winner = pd.to_numeric(players["is_winner"], errors="coerce").fillna(0) > 0
        player = players[keys].copy()
        game = players[["game_id"]].assign(_type=victory)
        for victory_type in sorted(set(victory.dropna())):
            column = ending_id(victory_type)
            if column not in catalog:
                continue
            player[metric_id(PLAYER, column)] = ((victory == victory_type) & winner) * 100.0
            game[metric_id(WORLD, column)] = (victory == victory_type) * 100.0
        game[metric_id(WORLD, NO_WINNER)] = victory.isna() * 100.0
        world = game.drop(columns="_type").groupby("game_id", as_index=False).max()

        alive = turns[turns["score"] > 0]
        last_alive = alive.groupby(keys)["turn"].max().rename(metric_id(PLAYER, GAME_LENGTH))
        final = turns.groupby("game_id")["turn"].max().rename(metric_id(WORLD, GAME_LENGTH))
        player = player.merge(last_alive.reset_index(), on=keys, how="left")
        world = world.merge(final.reset_index(), on="game_id", how="left")
        return player, world


def _victory_types(players: pd.DataFrame) -> pd.Series:
    """Each row's game victory type; blanks and ``N/A`` (no winner) become NaN."""
    text = players["victory_type"].map(lambda v: "" if pd.isna(v) else str(v).strip())
    return text.where(~text.isin(["", "N/A"]))


def _merge(frames: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=keys, how="outer")
    return out
