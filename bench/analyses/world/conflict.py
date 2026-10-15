"""``world.conflict``: wars, fighting, conquest, and destruction, per game.

For each metric of :data:`bench.config.schema.WORLD_CONFLICT_METRICS`, the
Player column counts what the strategist's own civ did and the World column what
happened in the whole game (counts are per game, not per turn):

* event counts come from the behavior table (Player) and the world table
  (World), so both follow the same counting rules;
* peak wars, the share of turns at war, elimination, and game length come from
  the turns table (Player) and the world table (World; the share of turns at
  war is the mean over the game's civ-turns alive).

Views:

* **relative** (default): Player cells minus the baseline in the same seat on
  the same map, World cells minus the baseline's games on the same map, with
  the baseline itself pinned as the top row in absolute values;
* **absolute**: the values as measured.

In controlled runs the module also writes ``world_conflict_by_seed`` and
``world_conflict_by_seat`` for the Matched Maps chapter's World: conflict tab.
"""

from __future__ import annotations

import pandas as pd

from ...config import schema as S
from ..base import AnalysisContext, AnalysisResult
from ..behavior import common as C
from ..behavior.base import MetricViews
from ..errors import AnalysisError
from .base import PLAYER, WORLD, WorldAnalysis, metric_id, paired_columns, world_columns

SLUG = "world_conflict"
# Metrics read from the turns table for the Player side.
TURN_METRICS = ("peak_wars", "at_war_share", "eliminated", "turns")
# World columns read as they are from the world table, by metric.
WORLD_TABLE_COLUMNS = {
    **{m: m for m in S.WORLD_CONFLICT_EVENTS},
    "peak_wars": "peak_wars",
    "eliminated": "civs_eliminated",
    "turns": "final_turn",
}


class WorldConflict(WorldAnalysis):
    module = "world.conflict"
    friendly_name = "Wars and conquest"
    description = (
        "Counts wars, combat kills and losses, conquests, razed cities, nukes built and "
        "used, and eliminations per game, for the strategist's own civ and for the whole world."
    )
    report_defaults = {
        "tables": ["world_conflict_relative", "world_conflict_absolute"],
        "figures": [],
    }

    @property
    def metrics(self) -> list[str]:
        chosen = set(self.params.get("metrics") or S.WORLD_CONFLICT_METRICS)
        return [m for m in S.WORLD_CONFLICT_METRICS if m in chosen]

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        metrics = self.metrics
        players = self.players(ctx)
        turns = None
        if any(m in TURN_METRICS for m in metrics):
            turns = self.turns(ctx, ["active_wars"])
        player_values = self._player_values(ctx, metrics, turns)
        world_values = self._world_values(ctx, metrics, turns)
        columns = paired_columns(S.WORLD_CONFLICT_METRICS, metrics)
        rows, pool, baseline = self.frame(ctx, players, player_values, world_values,
                                          list(columns.names))
        if rows.empty:
            return AnalysisResult(summary="No players remain after filtering.")

        cells = C.controlled_cells(ctx)
        views = C.build_views(rows, pool, list(columns.names), cells, baseline,
                              percent=columns.percent, keep_baseline=True,
                              per_game=world_columns(columns))
        out = MetricViews()
        player_base, world_base = self.baseline_words(views)
        pinned = self.pinned_label(views)
        versus = self.versus_words(views)
        self.add_paired_views(
            ctx, views, out, SLUG, columns,
            titles={C.RELATIVE: f"Wars and conquest vs. {versus}",
                    C.ABSOLUTE: "Wars and conquest, actual values per game"},
            help_texts={
                C.RELATIVE: "\n".join([
                    f"Each cell: the strategist's average minus {versus}'s.",
                    f"Player: its own civ, vs. {player_base}.",
                    f"World: the whole game, vs. {world_base}. It includes the strategist's "
                    "civ (both civs when two strategists play).",
                    f"Top row: the {pinned}'s actual values, for scale.",
                    "Blue is higher, red is lower; full color at 2 baseline SDs.",
                    "Counts are per game; shares are in percentage points.",
                ]),
                C.ABSOLUTE: "\n".join([
                    "Each cell: the average per game, with nothing subtracted.",
                    "Player: the strategist's own civ. World: the whole game.",
                    "Color: distance from the average row; full color at 2 SDs.",
                    "Counts are per game; shares are in %.",
                ]),
            },
        )
        metadata = {
            **views.metadata(),
            "metrics": metrics,
            "views": C.views_metadata(out.declared, baseline),
            "heatmaps": out.heatmaps,
        }
        tab = self.add_paired_matched_maps(
            ctx, views, pool, cells, out, SLUG, columns,
            label="World: conflict",
            tip="Wars and conquest, for the strategist's civ and the whole world",
            title="Wars and conquest per game",
            help_text="\n".join([
                "Each cell: the strategist's average on this map (and seat), as measured.",
                "Player: its own civ. World: the whole game.",
                f"Top row: the {pinned} on the same map (and seat).",
                "Hover a cell to see its difference from the top row.",
            ]),
        )
        if tab is not None:
            metadata["matched_maps"] = tab
        return AnalysisResult(
            tables=out.tables,
            summary=self.world_headline(ctx, out, views, [SLUG], columns),
            metadata=metadata,
        )

    # ── inputs ────────────────────────────────────────────────────────────────
    def _player_values(self, ctx: AnalysisContext, metrics: list[str], turns) -> pd.DataFrame:
        keys = ["game_id", "player_id"]
        frames = []
        events = [m for m in metrics if m in S.WORLD_CONFLICT_EVENTS]
        if events:
            needed = set(events) | ({"wars_on_city_states"} if "wars_declared" in events else set())
            behavior = ctx.load_table("behavior", usecols=lambda c: c in set(C.KEY_COLUMNS) | needed)
            missing = sorted(needed - set(behavior.columns))
            if missing:
                raise AnalysisError(
                    f"{self.module} '{self.stage_id}': behavior table lacks column(s) {missing}; "
                    "select them in data.extract.behavior.events and re-run extract."
                )
            behavior = C.numeric(C.prepare_keys(behavior, self.stage_id, "behavior table"),
                                 sorted(needed))
            values = behavior[keys].copy()
            for metric in events:
                values[metric_id(PLAYER, metric)] = behavior[metric]
            if "wars_declared" in events:
                # The behavior counter includes wars on city-states; the world's
                # counts wars between civs only.
                values[metric_id(PLAYER, "wars_declared")] = (
                    behavior["wars_declared"] - behavior["wars_on_city_states"]
                )
            frames.append(values)
        if any(m in TURN_METRICS for m in metrics):
            frames.append(self._turn_values(turns, [m for m in metrics if m in TURN_METRICS]))
        if not frames:
            return pd.DataFrame(columns=keys)
        out = frames[0]
        for frame in frames[1:]:
            out = out.merge(frame, on=keys, how="outer")
        return out

    @staticmethod
    def _turn_values(turns: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
        """Per player: peak wars, share of turns alive at war, elimination, and turns survived."""
        keys = ["game_id", "player_id"]
        final = turns.groupby("game_id")["turn"].max().rename("final_turn")
        alive = turns[turns["score"] > 0]
        at_war = (alive["active_wars"] > 0).astype(float) * 100.0
        values = pd.DataFrame({
            metric_id(PLAYER, "peak_wars"): turns.groupby(keys)["active_wars"].max(),
            metric_id(PLAYER, "at_war_share"): at_war.groupby([alive[k] for k in keys]).mean(),
            "last_alive": alive.groupby(keys)["turn"].max(),
        })
        values.index.names = keys
        values = values.reset_index()
        values = values.join(final, on="game_id")
        values[metric_id(PLAYER, "eliminated")] = (
            (values["last_alive"].fillna(-1) < values["final_turn"]).astype(float) * 100.0
        )
        values[metric_id(PLAYER, "turns")] = values["last_alive"]
        keep = ["game_id", "player_id", *(metric_id(PLAYER, m) for m in metrics)]
        return values[keep]

    def _world_values(self, ctx: AnalysisContext, metrics: list[str], turns) -> pd.DataFrame:
        wanted = [m for m in metrics if m in WORLD_TABLE_COLUMNS]
        world = self.table(ctx, "world", {"game_id", *(WORLD_TABLE_COLUMNS[m] for m in wanted)})
        missing = sorted({WORLD_TABLE_COLUMNS[m] for m in wanted} - set(world.columns))
        if missing:
            raise AnalysisError(
                f"{self.module} '{self.stage_id}': world table lacks column(s) {missing}; "
                "re-run extract to rebuild it."
            )
        world = world.copy()
        world["game_id"] = world["game_id"].astype(str)
        world = world.drop_duplicates("game_id")
        values = world[["game_id"]].copy()
        for metric in wanted:
            values[metric_id(WORLD, metric)] = pd.to_numeric(world[WORLD_TABLE_COLUMNS[metric]],
                                                              errors="coerce")
        if "at_war_share" in metrics:
            alive = turns[turns["score"] > 0]
            share = (alive["active_wars"] > 0).groupby(alive["game_id"]).mean() * 100.0
            values = values.merge(share.rename(metric_id(WORLD, "at_war_share")).reset_index(),
                                  on="game_id", how="outer")
        return values
