"""``performance.score_ratio`` — OLS of a panel target on identity + civilization.

Ported from ``performance/panel_score_ratio.ipynb``: fit
``score_ratio ~ C(civilization, Sum) + C(player_type, Treatment(ref=Vanilla))``
on ``panel_data`` (via :mod:`bench.stats`), then surface the per-``player_type``
deviation-from-grand-mean effects (with CIs / significance) and the raw
coefficient table. The predictors are config-driven (``params.predictors``); the
target defaults to ``score_ratio``.
"""

from __future__ import annotations

import pandas as pd

from ...plotting.coefficients import deviation_coefficients
from ...stats.regression import fit_regression
from ..base import Analysis, AnalysisContext, AnalysisResult
from ..errors import AnalysisError


class PerformanceScoreRatio(Analysis):
    module = "performance.score_ratio"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        target = self.params.get("target", "score_ratio")
        predictors = list(self.params.get("predictors") or ["player_type", "civilization"])

        panel = ctx.load_table("panel")
        panel = ctx.apply_filter(panel)
        if target not in panel.columns:
            raise AnalysisError(
                f"performance.score_ratio '{self.stage_id}': target '{target}' not in panel."
            )
        panel = panel.dropna(subset=[target]).copy()
        if panel.empty:
            raise AnalysisError(
                f"performance.score_ratio '{self.stage_id}': no rows after filtering."
            )

        vanilla = ctx.catalog.vanilla_label
        formula = f"{target} ~ " + " + ".join(self._term(p, vanilla) for p in predictors)
        result = fit_regression(formula, panel, outcome_col=target)

        tables: dict[str, pd.DataFrame] = {}
        figures = {}
        if "player_type" in predictors:
            effects = deviation_coefficients(result.fit, vanilla)
            tables["player_type_effects"] = effects
            figures["player_type_effects"] = self._forest(effects, target)

        coef = pd.DataFrame({
            "term": result.params.index,
            "coef": result.params.to_numpy(),
            "ci_low": result.conf_int[0].to_numpy(),
            "ci_high": result.conf_int[1].to_numpy(),
            "p_value": result.pvalues.to_numpy(),
        })
        tables["coefficients"] = coef

        summary = (
            f"OLS {target} ~ {' + '.join(predictors)}: {result.summary_line()}."
        )
        return AnalysisResult(
            tables=tables, figures=figures, summary=summary,
            metadata={"formula": formula, "n": result.nobs, "r2": result.rsquared},
        )

    def _forest(self, effects: pd.DataFrame, target: str):
        """Forest plot of the per-player_type deviation effects (with 95% CIs)."""
        import matplotlib.pyplot as plt

        df = effects.sort_values("Effect").reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(9, max(3, 0.42 * len(df) + 1.5)))
        for i, row in df.iterrows():
            color = "darkblue" if row["Sig"] else "gray"
            ax.plot([row["CI_Low"], row["CI_High"]], [i, i], color=color, linewidth=2.2,
                    alpha=0.85, solid_capstyle="round")
            ax.scatter(row["Effect"], i, s=80, color=color, zorder=3,
                       edgecolors="black", linewidth=0.4)
            if row["Sig"]:
                ax.text(row["CI_High"], i, f" {row['Sig']}", va="center", color="darkred", fontsize=10)
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["Name"])
        ax.set_xlabel(f"Effect on {target} (deviation from grand mean)")
        ax.set_title(f"Effect of strategist on {target}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    @staticmethod
    def _term(predictor: str, vanilla: str) -> str:
        if predictor == "player_type":
            return f'C(player_type, Treatment(reference="{vanilla}"))'
        if predictor == "civilization":
            return "C(civilization, Sum)"
        return f"C({predictor})"
