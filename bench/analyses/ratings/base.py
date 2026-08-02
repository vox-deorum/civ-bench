"""Shared scaffolding for the fitted ``ratings.*`` analyses (BT / PL).

Both Bradley-Terry and Plackett-Luce rate the adjust stage's ``strength`` table
the same way: load it, narrow it (``only_llm`` / ``min_games``, never dropping the
reference), optionally relabel identities for a multi-dimension ``group_by``
(``["player_type","strategy"]`` ⇒ per-strategy Elo via the ``strategy`` grouping,
exactly the legacy ``strategy_ratings`` composite trick), fit the MLE rating, and
optionally attach bootstrap CIs. The subclass only supplies the per-fit calculator.
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
        (including the Vanilla reference that only_llm / min_games later drop — the
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
            fig = self._forest(ratings, ctx, "player_type", metadata)
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
        fig = self._strategy_heatmap(ratings, general, ctx, extra_dims[0], reference, metadata)
        summary = (
            f"Per-{'/'.join(group_by)} ratings: {len(ratings)} composite identities "
            f"across {ratings['strategy'].nunique() if 'strategy' in ratings else len(extra_dims)} group(s)."
        )
        return AnalysisResult(tables={"ratings": ratings},
                              figures={"ratings": fig} if fig is not None else {},
                              summary=summary, metadata=metadata)

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

    def _strategy_heatmap(
        self,
        ratings: pd.DataFrame,
        general: pd.DataFrame,
        ctx: AnalysisContext,
        dim: str,
        reference: str,
        metadata: dict,
    ):
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import norm

        if ratings.empty:
            return None

        configured = list((ctx.config.groupings.get(dim) or {}).get("labels") or [])
        observed = [s for s in ratings["strategy"].dropna().astype(str).unique()]
        strategies = configured + sorted(s for s in observed if s not in configured)
        columns = ["General", *strategies]

        row_order = (
            general.sort_values("elo", ascending=False)["player_type"].astype(str).tolist()
        )
        missing_rows = sorted(set(ratings["player_type"].astype(str)) - set(row_order))
        player_types = row_order + missing_rows

        values = pd.DataFrame(np.nan, index=player_types, columns=columns, dtype=float)
        se = pd.DataFrame(np.nan, index=player_types, columns=columns, dtype=float)
        appearances = pd.DataFrame(np.nan, index=player_types, columns=columns, dtype=float)
        pct = pd.DataFrame(np.nan, index=player_types, columns=columns, dtype=float)

        for row in general.itertuples(index=False):
            pt = str(getattr(row, "player_type"))
            if pt not in values.index:
                continue
            values.loc[pt, "General"] = float(getattr(row, "elo"))
            se.loc[pt, "General"] = float(getattr(row, "se_elo", np.nan))
            appearances.loc[pt, "General"] = float(getattr(row, "appearances", np.nan))

        for row in ratings.itertuples(index=False):
            pt = str(getattr(row, "player_type"))
            strategy = str(getattr(row, "strategy"))
            if pt not in values.index or strategy not in values.columns:
                continue
            values.loc[pt, strategy] = float(getattr(row, "elo"))
            se.loc[pt, strategy] = float(getattr(row, "se_elo", np.nan))
            appearances.loc[pt, strategy] = float(getattr(row, "appearances", np.nan))

        strategy_counts = appearances[strategies]
        row_totals = strategy_counts.sum(axis=1).replace(0, np.nan)
        pct[strategies] = strategy_counts.div(row_totals, axis=0) * 100

        finite = values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            delta = max(float(np.nanmax(np.abs(finite - 1500))), 1.0)
            vmin, vmax = 1500 - delta, 1500 + delta
        else:
            vmin, vmax = 1400, 1600

        fig, ax = plt.subplots(
            figsize=(max(8.5, 1.45 * len(columns) + 4), max(4.0, 0.62 * len(player_types) + 1.5))
        )
        sns.heatmap(
            values,
            annot=False,
            cmap="RdYlGn",
            center=1500,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Elo"},
            ax=ax,
        )

        ref_label = reference if reference in values.index else ctx.catalog.vanilla_label
        ref_elo = values.loc[ref_label] if ref_label in values.index else None
        ref_se = se.loc[ref_label] if ref_label in se.index else None

        for i, pt in enumerate(values.index):
            for j, col in enumerate(values.columns):
                elo_val = values.loc[pt, col]
                if pd.isna(elo_val):
                    continue
                se_val = se.loc[pt, col]
                stars = ""
                if ref_elo is not None and pt != ref_label and pd.notna(ref_elo.get(col)):
                    ref_se_val = ref_se.get(col) if ref_se is not None else np.nan
                    combined = np.sqrt(
                        (se_val if pd.notna(se_val) else 0.0) ** 2
                        + (ref_se_val if pd.notna(ref_se_val) else 0.0) ** 2
                    )
                    if combined > 0:
                        p = 2 * norm.sf(abs((elo_val - ref_elo[col]) / combined))
                        stars = self._sig_stars(p)
                top = f"{elo_val:.0f}"
                if pd.notna(se_val):
                    top += f" +/- {se_val:.0f}"
                top += stars
                ax.text(j + 0.5, i + 0.42, top, ha="center", va="center",
                        fontsize=9, color="black")

                n_val = appearances.loc[pt, col]
                if pd.notna(n_val):
                    pct_val = pct.loc[pt, col]
                    if pd.isna(pct_val):
                        detail = f"(n={int(n_val)})"
                    else:
                        detail = f"(n={int(n_val)}, {pct_val:.0f}%)"
                    ax.text(j + 0.5, i + 0.68, detail, ha="center", va="center",
                            fontsize=7, color="#444444")

        ax.set_title(f"{self.module} strategy ratings", fontsize=12, fontweight="bold")
        ax.set_xlabel(
            f"* p<0.05, ** p<0.01, *** p<0.001 (z-test vs {ref_label})",
            fontsize=8,
            color="#666666",
        )
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        self._add_provenance_note(fig, metadata)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

    @staticmethod
    def _sig_stars(p: float) -> str:
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

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
        it from each rating-filtered resample — so the replicate baseline always
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
            f"{len(ratings)} identities rated; top = {top[identity_col]} "
            f"(Elo {top['elo']:.0f}); spread {rng:.0f} pts."
        )

    def _forest(self, ratings: pd.DataFrame, ctx: AnalysisContext, identity_col: str, metadata: dict):
        import matplotlib.pyplot as plt

        from ...plotting.styles import get_player_color

        df = ratings.sort_values("elo", ascending=True).reset_index(drop=True)
        has_ci = "ci_lower" in df.columns and df["ci_lower"].notna().any()
        pairing = ctx.condition_pairing() if identity_col == "player_type" else None
        if pairing is not None:
            from ...plotting.pairing import plot_paired_rows

            return plot_paired_rows(
                ratings,
                catalog=ctx.catalog,
                spec=pairing,
                value_col="elo",
                lo_col="ci_lower" if has_ci else None,
                hi_col="ci_upper" if has_ci else None,
                err_col=None if has_ci else "se_elo",
                identity_col=identity_col,
                ref_line=1500,
                ascending=False,
                xlabel=(
                    "Elo rating (error bars: bootstrap CI)" if has_ci
                    else "Elo rating (error bars: +/-1 SE)"
                ),
                title=f"{self.module} ratings",
                provenance_note=self._provenance_text(metadata),
            )
        fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(df) + 1.5)))
        for i, row in df.iterrows():
            name = str(row[identity_col])
            # Color by the underlying model: a composite identity (player_type-strategy)
            # has no catalog color of its own, so key off its base player_type.
            color_key = str(row["player_type"]) if "player_type" in df.columns else name
            color = get_player_color(ctx.catalog, color_key)
            x = row["elo"]
            if has_ci and pd.notna(row.get("ci_lower")):
                lo, hi = row["ci_lower"], row["ci_upper"]
                ax.errorbar(x, i, xerr=[[max(0, x - lo)], [max(0, hi - x)]], fmt="o",
                            color=color, markersize=7, capsize=4, elinewidth=1.5, ecolor=color)
            else:
                # +/-1 SE, matching the raw `+/- se_elo` annotation in the strategy
                # heatmap's General column so the generic category reads the same
                # across both figures (the bootstrap branch above stays a true CI).
                se = row.get("se_elo", 0) or 0
                ax.errorbar(x, i, xerr=se, fmt="o", color=color, markersize=7,
                            capsize=4, elinewidth=1.5, ecolor=color)
        ax.axvline(1500, color="gray", linestyle="--", linewidth=1, label="Reference (1500)")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df[identity_col])
        ax.set_xlabel("Elo rating (error bars: bootstrap CI)" if has_ci
                      else "Elo rating (error bars: +/-1 SE)")
        ax.set_title(f"{self.module} ratings", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, axis="x", alpha=0.3)
        self._add_provenance_note(fig, metadata)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        return fig

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

    @classmethod
    def _add_provenance_note(cls, fig, metadata: dict) -> None:
        note = cls._provenance_text(metadata)
        if not note:
            return
        fig.text(0.01, 0.01, note, ha="left", va="bottom",
                 fontsize=8, color="#666666")
