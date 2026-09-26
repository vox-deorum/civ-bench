"""``behavior.diplomacy``: diplomatic persona and relationship stance.

A persona trait is a 1 to 10 value the strategist hands the in-game AI. From the
behavior table's ``persona_<trait>_avg`` (each player's turn-weighted average),
per player type:

* **relative** (default): the player's average minus the matched in-game AI at
  the same ``(seed, player_id)`` cell, colored in baseline SDs;
* **absolute**: the average value, colored on the fixed 1 to 10 scale.

Both views are HTML heatmaps with one column per trait under its short name,
grouped as in :data:`bench.config.schema.PERSONA_INFO`, the full name and the
set-persona tool's description in the header tooltip, and the baseline average
pinned as the top row. The Null strategist, whose persona stays at the default
5, is left out; against the in-game AI baseline, so are Vanilla players (the
pinned row stands for them). Cell tooltips show each player's in-game
range (mean min to max), in the relative view as a difference from the baseline.
In controlled runs the module also writes ``diplomacy_by_seed`` and
``diplomacy_by_seat`` for the Matched Maps chapter's Diplomacy tab.

The ``relationships`` family shows in one absolute heatmap outside the views,
``stance_signals_absolute``: how often a strategist changes its modifiers, the
public, private, and net stance it holds toward rivals (on fixed scales), and
how often it masks hostility or goodwill. The in-game AI and the Null
strategist set no relationship modifiers, so they are left out and the table
has no relative form.
"""

from __future__ import annotations

import pandas as pd

from ...config import schema as S
from ...config.behavior import snake_case
from ..base import AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from . import common as C
from .base import BehaviorAnalysis, MetricViews

STANCE_LEVELS = {
    "stance_public_avg": ("Public stance", "Public",
                          "The visible stance toward a rival, from -100 (hostile) to 100 (warm), "
                          "averaged over pair-turns."),
    "stance_private_avg": ("Private stance", "Private",
                           "The hidden attitude toward a rival, from -100 (hostile) to 100 (warm), "
                           "averaged over pair-turns."),
    "stance_net_avg": ("Net stance", "Net",
                       "Public plus private, the sum the game's diplomacy uses (-200 to 200)."),
}
STANCE_LEVEL_RANGES = {
    "stance_public_avg": (-50.0, 50.0),
    "stance_private_avg": (-50.0, 50.0),
    "stance_net_avg": (-100.0, 100.0),
}
# The masked shares show as percents.
STANCE_SHARES = ("stance_masked_hostility_share", "stance_masked_goodwill_share")
STANCE_COLUMNS = ["relationship_changes", *STANCE_LEVELS, *STANCE_SHARES]

PERSONA_LEGEND = [[0.0, "1 lowest"], [0.333333, "4"], [0.5, "5.5 middle"],
                  [0.666667, "7"], [1.0, "10 highest"]]
RELATIVE_LEGEND = [[0.0, "2 SD below"], [0.25, "1 SD below"], [0.5, "same as the baseline"],
                   [0.75, "1 SD above"], [1.0, "2 SD above"]]


def trait_column(trait: str, stat: str = "avg") -> str:
    return f"persona_{snake_case(trait)}_{stat}"


class BehaviorDiplomacy(BehaviorAnalysis):
    module = "behavior.diplomacy"
    friendly_name = "Diplomatic behavior"
    description = (
        "Describes diplomatic persona traits and the public and private stances "
        "strategists set toward rivals, including how often the two conflict."
    )
    report_defaults = {
        "tables": ["diplomacy_relative", "diplomacy_absolute", "stance_signals_absolute"],
        "figures": [],
    }

    @property
    def traits(self) -> list[str]:
        return list(self.params.get("traits") or S.BEHAVIOR_DIPLOMACY_TRAITS)

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        kinds = self.behavior_kinds(ctx)
        # Group order, so each group's columns sit together under one header.
        requested = [t for t in S.PERSONA_INFO if t in set(self.traits)]
        traits = [t for t in requested if trait_column(t) in kinds]
        stance = [m for m in STANCE_COLUMNS if m in kinds]
        if not traits and not stance:
            raise AnalysisError(
                f"behavior.diplomacy '{self.stage_id}': the behavior extract selects none of "
                "the diplomatic persona averages or relationship columns."
            )
        stats = [trait_column(t, s) for t in traits for s in ("min", "max") if trait_column(t, s) in kinds]
        rows, pool, rate_scaled, baseline = self.load_behavior(
            ctx, [trait_column(t) for t in traits] + stance + stats,
        )
        if rows.empty:
            return AnalysisResult(summary="No behavior rows remain after filtering.")

        out = MetricViews()
        cells = C.controlled_cells(ctx)
        views = self._persona_views(ctx, rows, pool, cells, baseline, traits, out)
        self._stance_views(ctx, rows, stance, rate_scaled, out)
        metadata = {
            **views.metadata(),
            "traits": traits,
            "rate": self.rate,
            "views": C.views_metadata(out.declared, baseline),
            "heatmaps": out.heatmaps,
        }
        if traits:
            matched_maps = self._matched_maps(ctx, views, pool, cells, traits, out)
            if matched_maps is not None:
                metadata["matched_maps"] = matched_maps
        missing = [t for t in requested if t not in traits]
        if missing:
            metadata["traits_not_extracted"] = missing
        names = {trait_column(t): t for t in traits}
        summary = self._stance_headline(ctx, out) or (
            self.heatmap_headline(ctx, out, views, "diplomacy", names, decimals=1) if traits else ""
        )
        return AnalysisResult(
            tables=out.tables,
            summary=summary or "No player had diplomacy values.",
            metadata=metadata,
        )

    # ── persona ───────────────────────────────────────────────────────────────
    def _persona_views(self, ctx, rows, pool, cells, baseline, traits, out) -> C.BehaviorViews:
        metrics = [trait_column(t) for t in traits]
        in_game_ai = baseline is not None and baseline.in_game_ai
        # Null records the default 5 on every trait, so it says nothing about persona.
        skip = {ctx.catalog.null_label} | ({ctx.catalog.vanilla_label} if in_game_ai else set())
        rows = rows[~rows["player_type"].astype(str).isin(skip)]
        ranges = {
            trait_column(t): (trait_column(t, "min"), trait_column(t, "max"))
            for t in traits
            if trait_column(t, "min") in rows.columns and trait_column(t, "max") in rows.columns
        }
        views = C.build_views(rows, pool, metrics, cells, baseline, companions=ranges)
        if not traits:
            return views
        label = baseline.label if baseline is not None else "baseline"
        note = " The Null strategist, fixed at 5 on every trait, is left out."
        if views.relative is not None and views.baseline_summary:
            note += f" The top row is the {label} itself, as an average on the 1 to 10 scale."
            if in_game_ai:
                note += " In-game AI players are left out; the top row stands for them."
        self.add_heatmap_views(
            ctx, views, out, "diplomacy",
            titles={C.RELATIVE: "Relative persona trait", C.ABSOLUTE: "Average persona trait"},
            help_texts={
                C.RELATIVE: (
                    "Each cell is the mean, over players, of the player's turn-weighted "
                    f"average trait value minus the {label} at the same map and seat. Colors "
                    "run from 2 SDs of the baseline players below (red) to 2 SDs above "
                    "(blue). The tooltip range is the player's lowest and highest value, "
                    f"also minus the baseline.{note}"
                ),
                C.ABSOLUTE: (
                    "Each cell is the mean, over players, of the player's turn-weighted "
                    "average trait value. Traits run from 1 to 10; colors are fixed from 1 "
                    f"(red) through 5.5 (yellow) to 10 (blue).{note}"
                ),
            },
            names={trait_column(t): t for t in traits},
            short_names={trait_column(t): S.PERSONA_INFO[t][0] for t in traits},
            column_tips={trait_column(t): f"{t}\n{S.PERSONA_INFO[t][2]}" for t in traits},
            groups={trait_column(t): S.PERSONA_INFO[t][1] for t in traits},
            fixed={m: S.PERSONA_RANGE for m in metrics},
            legends={C.RELATIVE: RELATIVE_LEGEND, C.ABSOLUTE: PERSONA_LEGEND},
            value_labels={C.RELATIVE: "Difference", C.ABSOLUTE: "Average"},
            decimals=1,
            ranges=ranges,
            range_label="Range",
            baseline_row=True,
        )
        return views

    def _matched_maps(self, ctx, views, pool, cells, traits, out):
        label = views.baseline.label if views.baseline is not None else "baseline"
        return self.add_matched_map_tables(
            ctx, views, pool, cells, out, "diplomacy",
            label="Diplomacy",
            tip="Diplomatic persona: average trait values",
            title="Average persona trait",
            help_text=(
                "Each cell is the mean, over players on this map (and seat), of the player's "
                f"turn-weighted average trait value. The top row is the {label} on the same "
                "map (and seat); the tooltip shows each cell's difference from it, matched "
                "seat by seat. Colors are fixed from 1 (red) through 5.5 (yellow) to 10 (blue)."
            ),
            names={trait_column(t): t for t in traits},
            short_names={trait_column(t): S.PERSONA_INFO[t][0] for t in traits},
            column_tips={trait_column(t): f"{t}\n{S.PERSONA_INFO[t][2]}" for t in traits},
            groups={trait_column(t): S.PERSONA_INFO[t][1] for t in traits},
            fixed={trait_column(t): S.PERSONA_RANGE for t in traits},
            legend=PERSONA_LEGEND,
            decimals=1,
        )

    # ── stance ────────────────────────────────────────────────────────────────
    def _stance_views(self, ctx, rows, stance, rate_scaled, out) -> None:
        """Write the untabbed stance table."""
        if not stance:
            return
        skip = {ctx.catalog.null_label, ctx.catalog.vanilla_label}
        rows = rows[~rows["player_type"].astype(str).isin(skip)].copy()
        for share in STANCE_SHARES:
            if share in rows.columns:
                rows[share] = rows[share] * 100.0
        per = "per 100 turns alive" if "relationship_changes" in rate_scaled else "per game"
        columns = {
            "relationship_changes": ("Relationship changes", "Changes", "Activity",
                                     f"How often the player sets a new public or private modifier, {per}."),
            **{m: (name, short, "Stance", tip) for m, (name, short, tip) in STANCE_LEVELS.items()},
            "stance_masked_hostility_share": ("Masked hostility %", "Hostility %", "Mixed signals",
                                              "Share of pair-turns with a warm public and a hostile private stance."),
            "stance_masked_goodwill_share": ("Masked goodwill %", "Goodwill %", "Mixed signals",
                                             "Share of pair-turns with a hostile public and a warm private stance."),
        }
        columns = {m: v for m, v in columns.items() if m in stance}
        views = C.build_views(rows, rows.iloc[0:0], list(columns), None, None)
        self.add_heatmap_views(
            ctx, views, out, "stance_signals",
            titles={C.ABSOLUTE: "Relationship changes and mixed signals"},
            help_texts={C.ABSOLUTE: (
                f"Changes counts how often a player sets a new public or private modifier, {per}. "
                "Public is the stance a strategist shows a rival and private its hidden attitude, "
                "each set from -100 (hostile) to 100 (warm) and averaged over pair-turns; net is "
                "their sum, which the game's diplomacy uses. Masked hostility is the share of "
                "pair-turns where the public stance is warm and the private one hostile; masked "
                "goodwill is the reverse. Stance colors are fixed from -50 (red) through 0 "
                "(yellow) to +50 (blue), twice that for net. The other columns show each cell as "
                "a z-score across players, full color at 2 SDs. The in-game AI and the Null "
                "strategist set no relationship modifiers, so they are left out and this table "
                "has no relative form."
            )},
            names={m: v[0] for m, v in columns.items()},
            short_names={m: v[1] for m, v in columns.items()},
            column_tips={m: f"{v[0]}\n{v[3]}" for m, v in columns.items()},
            groups={m: v[2] for m, v in columns.items()},
            fixed={m: STANCE_LEVEL_RANGES[m] for m in columns if m in STANCE_LEVEL_RANGES},
            legends={C.ABSOLUTE: [
                [0.0, "hostile: -50 (net -100), or 2 SD below the average player"],
                [0.25, "-25 (net -50), or 1 SD below"],
                [0.5, "neutral: 0, or average"],
                [0.75, "+25 (net +50), or 1 SD above"],
                [1.0, "warm: +50 (net +100), or 2 SD above"],
            ]},
            value_labels={C.ABSOLUTE: "Average"}, decimals=1,
            # No tab: the table shows under the persona views on every one of them.
            view_names={C.ABSOLUTE: None},
        )

    def _stance_headline(self, ctx, out: MetricViews) -> str:
        """The friendliest and least friendly strategists by net stance, and the most masked."""
        table = out.summaries.get(("stance_signals", C.ABSOLUTE))
        if table is None or table.empty:
            return ""
        # Masked is the combined share of masked hostility and masked goodwill.
        shares = table[table["metric"].isin(STANCE_SHARES)]
        masked = shares.groupby("row_label", sort=False).agg(
            **{self.by: (self.by, "first"), "mean": ("mean", "sum"), "n_players": ("n_players", "min")}
        ).reset_index().assign(metric="masked")
        return self.top_row_summary(
            pd.concat([table, masked], ignore_index=True),
            {"Friendliest": "stance_net_avg", "Least friendly": "stance_net_avg", "Most masked": "masked"},
            {ctx.catalog.null_label, ctx.catalog.vanilla_label}, lowest=("Least friendly",),
            formats={"Friendliest": "{:+.1f} net", "Least friendly": "{:+.1f} net",
                     "Most masked": "{:.1f}%"},
        )
