"""Shared scaffolding for the fitted ``ratings.*`` analyses (BT / PL).

Both Bradley-Terry and Plackett-Luce rate the adjust stage's ``strength`` table
the same way: load it, narrow it (``only_llm`` / ``min_games``, never dropping the
reference), optionally relabel identities for a multi-dimension ``group_by``
(``["player_type","strategy"]`` ⇒ per-strategy Elo via the ``strategy`` grouping,
exactly the legacy ``strategy_ratings`` composite trick), fit the MLE rating, and
optionally attach bootstrap CIs. The subclass only supplies the per-fit calculator.

A single ``group_by`` shows an interactive Elo forest plot (figure ``ratings``);
a strategy ``group_by`` shows the ``strategy_ratings`` HTML heatmap of Elo per
player type and strategy (:mod:`bench.analyses.ratings.heatmaps`). The saved
``ratings`` table keeps its shape either way.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError
from ..grouping import grouping_label
from . import bootstrap as boot

_STRENGTH_COL = "adjusted_strength"


class RatingsAnalysis(Analysis):
    """Base for fitted rating analyses; subclasses implement :meth:`_calculate`."""

    strategy_friendly_name: str = ""
    strategy_description: str = ""

    def report_identity(self) -> tuple[str, str]:
        """Use a distinct report identity when ratings include extra groupings."""
        group_by = self._group_by()
        if group_by == ["player_type", "strategy"]:
            return self.strategy_friendly_name, self.strategy_description
        if len(group_by) > 1:
            dimensions = ", ".join(group_by[1:])
            return (
                f"{self.friendly_name} by {dimensions}",
                f"{self.description} Results are split by {dimensions}.",
            )
        return super().report_identity()

    def _calculate(self, strength_df: pd.DataFrame, reference: str) -> pd.DataFrame:
        """Fit the rating model and return a per-identity DataFrame (with ``elo``)."""
        raise NotImplementedError

    def _frozen_calculator(self, strength_df: pd.DataFrame, reference: str) -> Callable:
        """A calculator closure for the bootstrap (BT freezes the margin here)."""
        return lambda df: self._calculate(df, reference)

    # ── input prep ─────────────────────────────────────────────────────────────
    def _strength_params(self, ctx: AnalysisContext, table_id: str) -> dict:
        stage = next((s for s in ctx.config.adjust if s.id == table_id), None)
        return dict((stage.raw.get("params") or {})) if stage else {}

    def _load(self, ctx: AnalysisContext) -> pd.DataFrame:
        """The full strength panel BEFORE rating-population narrowing.

        ``load_table`` has already dropped malformed-DB games; no rating filters are
        applied here. This is the population the bootstrap resamples and re-adjusts,
        so a replicate's civ-OLS refit sees exactly the rows the adjust stage fit on
        (including the Vanilla reference that only_llm / min_games later drop, the
        cause of the vanished-Vanilla all-NaN CIs).
        """
        table_id = ctx.strength_table_id()
        panel = ctx.load_table(table_id)
        if _STRENGTH_COL not in panel.columns:
            raise AnalysisError(
                f"ratings '{self.stage_id}': strength table lacks '{_STRENGTH_COL}'."
            )
        return panel

    def _narrow(self, ctx: AnalysisContext, panel: pd.DataFrame, reference: str) -> pd.DataFrame:
        """Apply the rating-population filters to a full/resampled panel.

        config data.filter (+ stage filter) → only_llm degenerate-self-play drop →
        min_games (never dropping the reference) → sort. Applied to both the point
        panel and each bootstrap replicate *after* readjustment, so the replicate and
        point populations are narrowed identically.
        """
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

        full_panel = self._load(ctx)
        panel = self._narrow(ctx, full_panel, reference)
        table_id = ctx.strength_table_id()
        metadata = {"group_by": group_by, **ctx.strength_provenance(table_id, panel)}

        if not extra_dims:
            ratings = self._calculate(panel, reference)
            summary = self._summary(ratings, "player_type")
            ratings, boot_summary = self._maybe_bootstrap(
                ctx, full_panel, panel, ratings, reference, bootstrap, "player_type"
            )
            ci_level = float((bootstrap or {}).get("ci_level", 0.95))
            fig = self._forest(ratings, ctx, panel, reference, metadata, ci_level)
            tables = {"ratings": ratings}
            return AnalysisResult(tables=tables, figures={"ratings": fig} if fig is not None else {},
                                  summary=summary, metadata=metadata)

        if bootstrap is not None:
            raise AnalysisError(
                f"ratings '{self.stage_id}': bootstrap with a multi-dimension group_by "
                f"({group_by}) is not supported yet; set bootstrap:null or group_by:"
                f'["player_type"].'
            )
        general = self._calculate(panel, reference)
        general = self._attach_appearances(general, panel, "player_type")
        ratings = self._strategy_ratings(ctx, panel, extra_dims, reference)
        tables = {"ratings": ratings}
        if not ratings.empty:
            heat, spec = self._strategy_heatmap(ratings, general, ctx, extra_dims[0], reference)
            tables["strategy_ratings"] = heat
            metadata["heatmaps"] = {"strategy_ratings": spec}
        if ratings.empty:
            summary = (
                "No grouped ratings have enough games for a skill estimate."
            )
        else:
            top = ratings.sort_values("elo", ascending=False).iloc[0]
            groups = ratings["strategy"].nunique() if "strategy" in ratings else len(extra_dims)
            summary = (
                f"{top['composite_type']} leads **{len(ratings)}** "
                f"player and condition combinations at **{top['elo']:.0f} Elo** "
                f"across **{groups}** groups."
            )
        return AnalysisResult(tables=tables, summary=summary, metadata=metadata)

    # ── strategy (multi-dim) path ────────────────────────────────────────────────
    def _strategy_ratings(self, ctx, panel, extra_dims, reference) -> pd.DataFrame:
        df = self._composite_identity(ctx, panel, extra_dims)
        # Count distinct seats (game_id, player_id), not distinct games: a model
        # fields several seats per game (e.g. mirror pairs), each with its own
        # argmax strategy, so one game can land in two strategy cells. Seat counts
        # are the fit's actual sample size and partition cleanly across strategies
        # (every seat has exactly one strategy -> they sum to the General total).
        appearances = df.drop_duplicates(["game_id", "player_id"]).groupby("composite_type").size()
        min_games = int(self.params.get("min_games", 0) or 0)

        bt_input = df[["game_id", "player_id", "composite_type", _STRENGTH_COL, "civilization"]].rename(
            columns={"composite_type": "player_type"}
        )
        vanilla_comp = appearances[appearances.index.str.startswith(reference + "-")]
        comp_ref = vanilla_comp.idxmax() if len(vanilla_comp) else reference
        results = self._calculate(bt_input, comp_ref)

        results = results.rename(columns={"player_type": "composite_type"})
        split = results["composite_type"].str.rsplit("-", n=1, expand=True)
        results["player_type"] = split[0]
        results["strategy"] = split[1] if split.shape[1] > 1 else ""
        results["appearances"] = results["composite_type"].map(appearances)

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
            results = results[results["appearances"] >= min_games].copy()
        return results.sort_values(["strategy", "elo"], ascending=[True, False]).reset_index(drop=True)

    @staticmethod
    def _attach_appearances(ratings: pd.DataFrame, panel: pd.DataFrame, identity_col: str) -> pd.DataFrame:
        if "appearances" in ratings.columns:
            return ratings
        counts = panel.drop_duplicates(["game_id", "player_id"]).groupby(identity_col).size()
        out = ratings.copy()
        out["appearances"] = out[identity_col].map(counts)
        return out

    def _strategy_heatmap(self, ratings, general, ctx: AnalysisContext, dim: str, reference: str):
        """The ``strategy_ratings`` heatmap table and spec: General, then each strategy."""
        from .heatmaps import strategy_heatmap

        configured = list((ctx.config.groupings.get(dim) or {}).get("labels") or [])
        observed = [s for s in ratings["strategy"].dropna().astype(str).unique()]
        strategies = configured + sorted(s for s in observed if s not in configured)
        return strategy_heatmap(ctx, ratings, general, strategies, reference,
                                f"{self.module} strategy ratings")

    # ── bootstrap ──────────────────────────────────────────────────────────────
    def _maybe_bootstrap(self, ctx, full_panel, narrowed, ratings, reference, bootstrap, identity_col):
        if bootstrap is None:
            return ratings, None
        n = int(bootstrap.get("n"))
        ci_level = float(bootstrap.get("ci_level", 0.95))
        stratified = bool(bootstrap.get("stratified", True))
        table_id = ctx.strength_table_id()
        adjust_params = self._strength_params(ctx, table_id)
        # has-cell-rows detection + margin freeze use the FULL panel and the NARROWED
        # point panel respectively; the resample population is the full panel, and each
        # replicate is narrowed (via `narrow`) after readjustment.
        fixed_cell_baseline = self._fixed_cell_baseline(ctx, table_id, full_panel, adjust_params)
        calculator = self._frozen_calculator(narrowed, reference)
        narrow = lambda df: self._narrow(ctx, df, reference)  # noqa: E731
        point = ratings[[identity_col, "elo"]].copy()
        summary = boot.run_bootstrap(
            full_panel, point, calculator, group_col=identity_col, n=n, seed=ctx.config.seed,
            ci_level=ci_level, stratified=stratified, adjust_params=adjust_params,
            catalog=ctx.catalog, refit_strength=True,
            fixed_cell_baseline=fixed_cell_baseline, narrow=narrow,
        )
        merged = ratings.merge(
            summary[[identity_col, "ci_lower", "ci_upper", "boot_se_elo", "n_valid"]],
            on=identity_col, how="left",
        )
        return merged, summary

    def _fixed_cell_baseline(self, ctx, table_id: str, panel: pd.DataFrame, adjust_params: dict):
        """Fixed per-cell baseline for bootstrap readjustment (option C).

        Both pathways hold the per-cell baseline constant across replicates,
        reading it from the adjust stage's persisted ``cell_baseline.csv`` trail
        (computed once from the full, unfiltered panel) rather than recomputing
        it from each rating-filtered resample, so the replicate baseline always
        matches the point estimate and rating filters (only_llm/min_games) cannot
        drop the reference rows. Explicit baselines key on ``(seed, player_id)``;
        implicit baselines key on ``(experiment, seed, player_id)``.
        """
        explicit = adjust_params.get("baseline_experiment") is not None
        pathway = "explicit" if explicit else "implicit"
        has_cell_rows = (
            "adjust_method" in panel.columns
            and (panel["adjust_method"] == "cell").any()
        )
        path = ctx.adjust_dir(table_id) / "cell_baseline.csv"
        if not path.exists():
            if has_cell_rows:
                raise AnalysisError(
                    f"ratings '{self.stage_id}': bootstrap needs the cell "
                    f"baseline trail at '{path}'. Re-run the adjust stage."
                )
            return {}
        cb = pd.read_csv(path)
        rows = cb[cb.get("pathway") == pathway] if "pathway" in cb.columns else pd.DataFrame()
        if rows.empty:
            if has_cell_rows:
                raise AnalysisError(
                    f"ratings '{self.stage_id}': bootstrap found no {pathway} "
                    f"baseline rows in '{path}'. Re-run the adjust stage."
                )
            return {}
        if explicit:
            return {
                (int(row.seed), int(row.player_id)): float(row.cell_baseline)
                for row in rows.itertuples(index=False)
            }
        return {
            (str(row.experiment), int(row.seed), int(row.player_id)): float(row.cell_baseline)
            for row in rows.itertuples(index=False)
        }

    # ── summary + plot ───────────────────────────────────────────────────────────
    def _summary(self, ratings: pd.DataFrame, identity_col: str) -> str:
        top = ratings.sort_values("elo", ascending=False).iloc[0]
        rng = ratings["elo"].max() - ratings["elo"].min()
        return (
            f"{top[identity_col]} leads **{len(ratings)}** identities at "
            f"**{top['elo']:.0f} Elo**; rating spread: **{rng:.0f}** points."
        )

    def _forest(self, ratings: pd.DataFrame, ctx: AnalysisContext, panel: pd.DataFrame,
                reference: str, metadata: dict, ci_level: float):
        """The interactive Elo forest plot, with a table tooltip per point."""
        from ...plotting.pairing import PairingSpec, plot_paired_rows_interactive
        from .heatmaps import count_text, p_text

        df = self._attach_appearances(ratings, panel, "player_type")
        has_ci = "ci_lower" in df.columns and df["ci_lower"].notna().any()
        df["tip_elo"] = df["elo"].map(lambda v: f"{v:.0f}")
        if has_ci:
            error_label = f"{ci_level * 100:g}% CI"
            df["tip_error"] = [
                f"{lo:.0f} to {hi:.0f}" if pd.notna(lo) and pd.notna(hi) else ""
                for lo, hi in zip(df["ci_lower"], df["ci_upper"])
            ]
        else:
            error_label = "SE"
            df["tip_error"] = df["se_elo"].map(lambda v: f"+/-{v:.0f}" if pd.notna(v) else "")
        p_values = df["p_value"] if "p_value" in df.columns else pd.Series(np.nan, index=df.index)
        df["tip_p"] = [
            "" if str(identity) == reference else p_text(p)
            for identity, p in zip(df["player_type"], p_values)
        ]
        df["tip_n"] = df["appearances"].map(count_text)
        return plot_paired_rows_interactive(
            df,
            catalog=ctx.catalog,
            spec=ctx.condition_pairing() or PairingSpec((), "base", "Identity"),
            value_col="elo",
            lo_col="ci_lower" if has_ci else None,
            hi_col="ci_upper" if has_ci else None,
            err_col=None if has_ci else "se_elo",
            identity_col="player_type",
            ref_line=1500,
            ref_label=f"{reference} (1500)",
            tooltip=[("Elo", "tip_elo"), (error_label, "tip_error"),
                     (f"p vs {reference}", "tip_p"), ("Appearances", "tip_n")],
            ascending=False,
            xlabel=(
                "Elo rating (error bars: bootstrap CI)" if has_ci
                else "Elo rating (error bars: +/-1 SE)"
            ),
            title=f"{self.module} ratings",
            provenance_note=self._provenance_text(metadata) or None,
        )

    @staticmethod
    def _provenance_text(metadata: dict) -> str:
        est = metadata.get("strength_estimator")
        model = metadata.get("estimator_model")
        fit = metadata.get("estimator_fit")
        predict = metadata.get("estimator_predict")
        block = metadata.get("adjust_block")
        if not est and not block:
            return ""
        bits = []
        if est:
            model_part = f" ({model}" if model else ""
            if fit or predict:
                fp = "/".join(str(v) for v in (fit, predict) if v)
                model_part += f", {fp}" if model_part else f" ({fp}"
            model_part += ")" if model_part else ""
            bits.append(f"strength estimator: {est}{model_part}")
        if block:
            bits.append(f"adjust block: {block}")
        return "; ".join(bits)
