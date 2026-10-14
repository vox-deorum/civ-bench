"""``world.progress``: how large and productive the civs and the world became.

For each metric of :data:`bench.config.schema.WORLD_PROGRESS_METRICS`, read at
each checkpoint turn of the ``turns`` table, the Player column is the
strategist's own civ and the World column the whole game (a sum over its civs,
except techs, the average over surviving civs, and civs alive, a count):

* a civ eliminated by the checkpoint counts as 0 there;
* a game with no turn at the checkpoint (it ended earlier) is left out of it.

The Endings tab reads the panel: the share of the civ's games it won by each
victory type (Player) and the share of games that ended that way (World), plus
game length.

Tabs, in a controlled run:

* **Overview** (default): ``overview_metrics`` at every checkpoint, relative;
* **Turn N**: every metric at checkpoint N, relative;
* **Endings**: win shares and game length, relative;
* **Absolute**: the absolute tables of the Turn and Endings tabs, stacked.

An uncontrolled run has no relative view, so each tab shows absolute values and
there is no separate Absolute tab. In controlled runs the module also writes
``world_progress_by_seed`` and ``world_progress_by_seat`` for the Matched Maps
chapter's World: progress tab (the overview metrics at the last checkpoint).
"""

from __future__ import annotations

import re

import pandas as pd

from ...config import schema as S
from ..base import AnalysisContext, AnalysisResult
from ..behavior import common as C
from ..behavior.base import MetricViews
from .base import (
    PLAYER, WORLD, WorldAnalysis, merge_columns, metric_id, paired_columns, world_columns,
)

SLUG = "world_progress"
OVERVIEW = "overview"
ENDINGS = "endings"
ALL_ABSOLUTE = "all_absolute"
# World values averaged over the civs alive instead of summed.
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
        "games ended, for the strategist's own civ and for the whole world."
    )
    report_defaults = {
        "tables": [f"{SLUG}_overview_*", f"{SLUG}_t*_*", "world_endings_*"],
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

        player_frames, world_frames = [], []
        for t in checkpoints:
            player_values, world_values = self._checkpoint_values(turns, metrics, t)
            player_frames.append(player_values)
            world_frames.append(world_values)
        player_endings, world_endings = self._ending_values(players, turns, endings_catalog)
        player_frames.append(player_endings)
        world_frames.append(world_endings)
        player_values = _merge(player_frames, ["game_id", "player_id"])
        world_values = _merge(world_frames, ["game_id"])

        rows, pool, baseline = self.frame(ctx, players.drop(columns=["victory_type", "is_winner"]),
                                          player_values, world_values, list(columns.names))
        if rows.empty:
            return AnalysisResult(summary="No players remain after filtering.")

        cells = C.controlled_cells(ctx)
        views = C.build_views(rows, pool, list(columns.names), cells, baseline,
                              percent=columns.percent, keep_baseline=True,
                              per_game=world_columns(columns))
        controlled = views.relative is not None
        out = MetricViews()
        # Tab order: the views fill in later calls, which keep this order.
        for name in [OVERVIEW, *(turn_view(t) for t in checkpoints), ENDINGS, ALL_ABSOLUTE]:
            out.declared[name] = {"tables": [], "figures": []}

        player_base, world_base = self.baseline_words(views)
        pinned = self.pinned_label(views)
        relative_help = (
            "Each Player cell is the mean, over the strategist's civs, of its own value "
            f"compared with {player_base}. Each World cell is the mean, over its games, of the "
            f"whole game's value compared with {world_base}. The top row is the {pinned} "
            "itself, in absolute values. Colors run from 2 SDs of the baseline below (red) to "
            "2 SDs above (blue)."
        )
        absolute_help = (
            "Each Player cell is the mean of the civ's own value and each World cell the mean "
            "of the whole game's value. Colors show each cell as a z-score across players, "
            "full color at 2 SDs."
        )
        stock_note = (
            " Sizes, yields, and armies differ in percent; civs alive in percentage points "
            "(Player) or civs (World). World values are sums over civs, except techs (the "
            "average over surviving civs). A civ eliminated by the checkpoint counts as 0, and "
            "a game that ended earlier is left out. A World value includes the strategist's "
            "own civ, and in a game with two strategists it counts for both."
        )
        ending_note = (
            " Win shares differ in percentage points and game length in turns. A World value "
            "includes the strategist's own civ, and in a game with two strategists it counts "
            "for both."
        )

        def emit_tab(slug, part, tab, title, note, *, overview_tab=False):
            if controlled:
                view_names = {C.RELATIVE: tab, C.ABSOLUTE: ALL_ABSOLUTE}
                emit = {C.RELATIVE} if overview_tab else None
            else:
                view_names, emit = {C.ABSOLUTE: tab}, {C.ABSOLUTE}
            self.add_paired_views(
                ctx, views, out, slug, part,
                titles={C.RELATIVE: f"{title}, relative", C.ABSOLUTE: title},
                help_texts={C.RELATIVE: relative_help + note, C.ABSOLUTE: absolute_help + note},
                emit=emit, view_names=view_names,
            )

        if overview:
            emit_tab(f"{SLUG}_overview", overview_columns, OVERVIEW, "World growth at a glance",
                     stock_note, overview_tab=True)
        for t in checkpoints:
            emit_tab(f"{SLUG}_t{t}", per_turn[t], turn_view(t), f"World growth by turn {t}",
                     stock_note)
        emit_tab("world_endings", endings, ENDINGS, "How games ended", ending_note)

        metadata = {
            **views.metadata(),
            "metrics": metrics,
            "checkpoints": checkpoints,
            "overview_metrics": overview,
            "views": C.views_metadata(out.declared, baseline, extra={
                OVERVIEW: ("Overview", "Selected metrics at every checkpoint"),
                **{turn_view(t): (f"Turn {t}", f"Every metric at turn {t}") for t in checkpoints},
                ENDINGS: ("Endings", "How games ended, and game length"),
                ALL_ABSOLUTE: ("Absolute", "Absolute values"),
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
            tip=f"World growth by turn {last}, for the strategist's civ and the whole world",
            title=f"World growth by turn {last}",
            help_text=(
                "Each Player cell is the mean of the strategist's own value on this map (and "
                "seat) and each World cell the mean of the whole game's value. The top row "
                f"is the {pinned} on the same map (and seat); the tooltip shows each cell's "
                "difference from it."
            ),
        )
        if tab is not None:
            metadata["matched_maps"] = tab
        headline_slugs = [f"{SLUG}_overview"] if overview else [f"{SLUG}_t{last}"]
        headline_columns = overview_columns if overview else per_turn[last]
        return AnalysisResult(
            tables=out.tables,
            summary=self.world_headline(ctx, out, views, headline_slugs, headline_columns),
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
        world = pd.DataFrame(index=grouped.size().index)
        for metric in metrics:
            if metric == "civs_alive":
                world[metric] = grouped["civs_alive"].sum()
            elif metric in AVERAGED:
                world[metric] = values[metric].where(alive).groupby(values["game_id"]).mean()
            else:
                world[metric] = grouped[metric].sum()
        values["civs_alive"] *= 100.0
        player = values[keys].copy()
        for metric in metrics:
            player[metric_id(PLAYER, metric, f"@{t}")] = values[metric]
        world = world.rename(columns={m: metric_id(WORLD, m, f"@{t}") for m in metrics})
        world.index.name = "game_id"
        return player, world.reset_index()

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
