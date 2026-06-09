"""Shared scaffolding for the fitted ``ratings.*`` analyses (BT / PL).

Both Bradley-Terry and Plackett-Luce rate the adjust stage's ``strength`` table
the same way: load it, narrow it (``only_llm`` / ``min_games``, never dropping the
reference), optionally relabel identities for a multi-dimension ``group_by``
(``["player_type","strategy"]`` ⇒ per-strategy Elo via the ``strategy`` grouping,
exactly the legacy ``strategy_ratings`` composite trick), fit the MLE rating, and
optionally attach bootstrap CIs. The subclass only supplies the per-fit calculator.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from ..grouping import grouping_label
from . import bootstrap as boot

_STRENGTH_COL = "adjusted_strength"


class RatingsAnalysis(Analysis):
    """Base for fitted rating analyses; subclasses implement :meth:`_calculate`."""

    def _calculate(self, strength_df: pd.DataFrame, reference: str) -> pd.DataFrame:
        """Fit the rating model and return a per-identity DataFrame (with ``elo``)."""
        raise NotImplementedError

    def _frozen_calculator(self, strength_df: pd.DataFrame, reference: str) -> Callable:
        """A calculator closure for the bootstrap (BT freezes the margin here)."""
        return lambda df: self._calculate(df, reference)

    # ── input prep ─────────────────────────────────────────────────────────────
    def _strength_table_id(self, ctx: AnalysisContext) -> str:
        for tbl in ctx.uses_tables():
            if tbl == "strength" or any(s.id == tbl for s in ctx.config.adjust):
                return tbl
        return "strength"

    def _strength_params(self, ctx: AnalysisContext, table_id: str) -> dict:
        stage = next((s for s in ctx.config.adjust if s.id == table_id), None)
        return dict((stage.raw.get("params") or {})) if stage else {}

    def _load_and_filter(self, ctx: AnalysisContext, reference: str) -> pd.DataFrame:
        table_id = self._strength_table_id(ctx)
        panel = ctx.load_table(table_id)
        panel = ctx.apply_filter(panel)

        only_llm = bool(self.params.get("only_llm", True))
        if only_llm and "player_type" in panel.columns:
            # Drop degenerate self-play games (a single distinct player_type): they yield zero
            # cross-type comparisons and orphan high-index reference slots in the BT fit
            # (all-Vanilla games mint Vanilla_6/_7 -> NaN abilities that poison the Vanilla
            # reference and NaN out every worth/elo). player_type based so it survives
            # experiment-id naming drift; mixed baseline games (e.g. Null-vs-Vanilla) are kept
            # so Null stays a rated identity.
            distinct = panel.groupby("game_id")["player_type"].transform("nunique")
            panel = panel[distinct >= 2]

        min_games = int(self.params.get("min_games", 0) or 0)
        if min_games > 0 and "player_type" in panel.columns:
            counts = panel.groupby("player_type")["game_id"].nunique()
            keep = set(counts[counts >= min_games].index) | {reference}
            panel = panel[panel["player_type"].isin(keep)]

        if panel.empty:
            raise AnalysisError(
                f"ratings '{self.stage_id}': no rows after filtering (only_llm="
                f"{only_llm}, min_games={min_games})."
            )
        if _STRENGTH_COL not in panel.columns:
            raise AnalysisError(
                f"ratings '{self.stage_id}': strength table lacks '{_STRENGTH_COL}'."
            )
        return panel.sort_values(["game_id", "player_id"]).reset_index(drop=True)

    # ── identity relabel (group_by) ─────────────────────────────────────────────
    def _group_by(self) -> list[str]:
        gb = self.params.get("group_by") or ["player_type"]
        if gb[0] != "player_type":
            raise AnalysisError(
                f"ratings '{self.stage_id}': group_by must start with 'player_type' "
                f"(got {gb})."
            )
        return list(gb)

    def _composite_identity(self, ctx: AnalysisContext, panel: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
        """Append a ``composite_type`` column = player_type + '-' + grouping labels."""
        df = panel.copy()
        need = [d for d in dims if d not in df.columns]
        if need:
            src_cols = []
            for d in dims:
                g = ctx.config.groupings.get(d)
                if not g:
                    raise AnalysisError(f"ratings '{self.stage_id}': unknown grouping '{d}'.")
                src_cols += list(g.get("columns") or [])
            panel_src = ctx.load_table("panel")[["game_id", "player_id", *dict.fromkeys(src_cols)]]
            df = df.merge(panel_src.drop_duplicates(["game_id", "player_id"]),
                          on=["game_id", "player_id"], how="left")
        label = df["player_type"].astype(str)
        for d in dims:
            label = label + "-" + grouping_label(df, ctx.config.groupings[d], d).astype(str)
        df["composite_type"] = label
        return df

    # ── run ──────────────────────────────────────────────────────────────────────
    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        reference = self.params.get("ref") or ctx.catalog.vanilla_label
        if reference == "Vanilla":
            reference = ctx.catalog.vanilla_label
        group_by = self._group_by()
        extra_dims = group_by[1:]
        bootstrap = self.params.get("bootstrap")

        panel = self._load_and_filter(ctx, reference)

        if not extra_dims:
            ratings = self._calculate(panel, reference)
            summary = self._summary(ratings, "player_type")
            ratings, boot_summary = self._maybe_bootstrap(
                ctx, panel, ratings, reference, bootstrap, "player_type"
            )
            fig = self._forest(ratings, ctx, "player_type")
            tables = {"ratings": ratings}
            return AnalysisResult(tables=tables, figures={"ratings": fig} if fig is not None else {},
                                  summary=summary, metadata={"group_by": group_by})

        if bootstrap is not None:
            raise AnalysisError(
                f"ratings '{self.stage_id}': bootstrap with a multi-dimension group_by "
                f"({group_by}) is not supported yet; set bootstrap:null or group_by:"
                f'["player_type"].'
            )
        ratings = self._strategy_ratings(ctx, panel, extra_dims, reference)
        fig = self._forest(ratings, ctx, "composite_type")
        summary = (
            f"Per-{'/'.join(group_by)} ratings: {len(ratings)} composite identities "
            f"across {ratings['strategy'].nunique() if 'strategy' in ratings else len(extra_dims)} group(s)."
        )
        return AnalysisResult(tables={"ratings": ratings},
                              figures={"ratings": fig} if fig is not None else {},
                              summary=summary, metadata={"group_by": group_by})

    # ── strategy (multi-dim) path ────────────────────────────────────────────────
    def _strategy_ratings(self, ctx, panel, extra_dims, reference) -> pd.DataFrame:
        df = self._composite_identity(ctx, panel, extra_dims)
        game_counts = df.groupby("composite_type")["game_id"].nunique()
        min_games = int(self.params.get("min_games", 0) or 0)

        bt_input = df[["game_id", "player_id", "composite_type", _STRENGTH_COL, "civilization"]].rename(
            columns={"composite_type": "player_type"}
        )
        vanilla_comp = game_counts[game_counts.index.str.startswith(reference + "-")]
        comp_ref = vanilla_comp.idxmax() if len(vanilla_comp) else reference
        results = self._calculate(bt_input, comp_ref)

        results = results.rename(columns={"player_type": "composite_type"})
        split = results["composite_type"].str.rsplit("-", n=1, expand=True)
        results["player_type"] = split[0]
        results["strategy"] = split[1] if split.shape[1] > 1 else ""
        results["n_games"] = results["composite_type"].map(game_counts)

        vanilla_mask = results["player_type"] == reference
        if vanilla_mask.any():
            shift = 1500 - results.loc[vanilla_mask, "elo"].mean()
            results["elo"] += shift
            results["log_worth"] = (results["elo"] - 1500) / 400 * np.log(10)
            results["worth"] = np.exp(results["log_worth"])
            pos = results["se_log_worth"] > 0
            from scipy.stats import norm
            results["z_value"] = np.where(pos, results["log_worth"] / results["se_log_worth"], np.nan)
            results["p_value"] = np.where(pos, 2 * norm.sf(np.abs(results["z_value"])), np.nan)

        if min_games > 0:
            results = results[results["n_games"] >= min_games].copy()
        return results.sort_values(["strategy", "elo"], ascending=[True, False]).reset_index(drop=True)

    # ── bootstrap ──────────────────────────────────────────────────────────────
    def _maybe_bootstrap(self, ctx, panel, ratings, reference, bootstrap, identity_col):
        if bootstrap is None:
            return ratings, None
        n = int(bootstrap.get("n"))
        ci_level = float(bootstrap.get("ci_level", 0.95))
        stratified = bool(bootstrap.get("stratified", True))
        table_id = self._strength_table_id(ctx)
        adjust_params = self._strength_params(ctx, table_id)
        calculator = self._frozen_calculator(panel, reference)
        point = ratings[[identity_col, "elo"]].copy()
        summary = boot.run_bootstrap(
            panel, point, calculator, group_col=identity_col, n=n, seed=ctx.config.seed,
            ci_level=ci_level, stratified=stratified, adjust_params=adjust_params,
            catalog=ctx.catalog, refit_strength=True,
        )
        merged = ratings.merge(
            summary[[identity_col, "ci_lower", "ci_upper", "boot_se_elo", "n_valid"]],
            on=identity_col, how="left",
        )
        return merged, summary

    # ── summary + plot ───────────────────────────────────────────────────────────
    def _summary(self, ratings: pd.DataFrame, identity_col: str) -> str:
        top = ratings.sort_values("elo", ascending=False).iloc[0]
        rng = ratings["elo"].max() - ratings["elo"].min()
        return (
            f"{len(ratings)} identities rated; top = {top[identity_col]} "
            f"(Elo {top['elo']:.0f}); spread {rng:.0f} pts."
        )

    def _forest(self, ratings: pd.DataFrame, ctx: AnalysisContext, identity_col: str):
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color

        df = ratings.sort_values("elo", ascending=True).reset_index(drop=True)
        has_ci = "ci_lower" in df.columns and df["ci_lower"].notna().any()
        fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(df) + 1.5)))
        for i, row in df.iterrows():
            name = str(row[identity_col])
            color = get_player_color(ctx.catalog, name)
            x = row["elo"]
            if has_ci and pd.notna(row.get("ci_lower")):
                lo, hi = row["ci_lower"], row["ci_upper"]
                ax.errorbar(x, i, xerr=[[max(0, x - lo)], [max(0, hi - x)]], fmt="o",
                            color=color, markersize=7, capsize=4, elinewidth=1.5, ecolor=color)
            else:
                se = row.get("se_elo", 0) or 0
                ax.errorbar(x, i, xerr=1.96 * se, fmt="o", color=color, markersize=7,
                            capsize=4, elinewidth=1.5, ecolor=color)
        ax.axvline(1500, color="gray", linestyle="--", linewidth=1, label="Reference (1500)")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df[identity_col])
        ax.set_xlabel("Elo rating")
        ax.set_title(f"{self.module} ratings", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig
