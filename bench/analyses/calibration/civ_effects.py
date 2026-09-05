"""``calibration.civ_effects``: per-civilization adjustment effects.

Visualizes the adjust stage's ``civ_effects.csv`` (``civilization, civ_effect,
n_rows``): a sorted horizontal lollipop of ``civ_effect`` on the **logit scale**,
diverging-coloured by sign and centered at 0 (the Sum-coded effects sum to ~0),
each bar annotated with ``n_rows`` (the panel rows backing that civ's effect).

``civ_effects.csv`` carries no CI (a single OLS point effect), so this is a
lollipop, **not** a forest plot. It renders only when the file is non-empty (i.e.
the civ OLS ran, ``civ_adjust:"ols_logit"``); otherwise an empty result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..base import Analysis, AnalysisContext, AnalysisResult


class CalibrationCivEffects(Analysis):
    module = "calibration.civ_effects"
    friendly_name = "Civilization strength effects"
    description = (
        "Estimates how much civilization choice shifts player strength in "
        "uncontrolled games (ordinary least squares)."
    )
    report_defaults = {"tables": [], "figures": ["civ_effects"]}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        adjust_dir = ctx.adjust_dir(ctx.strength_table_id())
        path = Path(adjust_dir) / "civ_effects.csv"
        if not path.exists():
            return AnalysisResult(
                summary="Civilization strength effects are unavailable because the adjustment stage did not produce them."
            )
        df = pd.read_csv(path)
        if df.empty:
            return AnalysisResult(
                summary="Civilization strength effects are unavailable because civilization adjustment was disabled."
            )

        df = df.sort_values("civ_effect").reset_index(drop=True)
        fig = self._plot(df)
        summary = (
            f"Civilization strength effects range from {df['civ_effect'].min():+.3f} "
            f"to {df['civ_effect'].max():+.3f} across {len(df)} civilizations "
            f"(log-odds scale)."
        )
        return AnalysisResult(
            tables={"civ_effects": df},
            figures={"civ_effects": fig},
            summary=summary,
            metadata={"n_civs": int(len(df))},
        )

    def _plot(self, df: pd.DataFrame):
        import matplotlib.pyplot as plt

        n = len(df)
        fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * n + 1.5)))
        y = range(n)
        colors = ["#2166AC" if v >= 0 else "#B2182B" for v in df["civ_effect"]]
        ax.hlines(y, 0, df["civ_effect"], color=colors, linewidth=1.8, alpha=0.8)
        ax.scatter(df["civ_effect"], y, color=colors, s=45, zorder=3,
                   edgecolors="black", linewidth=0.4)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_yticks(list(y))
        ax.set_yticklabels(df["civilization"])
        ax.set_xlabel("Civilization effect (logit scale)")
        ax.set_title("Per-civilization adjustment effects", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        x_pad = (df["civ_effect"].max() - df["civ_effect"].min() or 1.0) * 0.02
        for i, row in enumerate(df.itertuples()):
            ax.text(row.civ_effect + (x_pad if row.civ_effect >= 0 else -x_pad), i,
                    f"n={int(row.n_rows)}", va="center",
                    ha="left" if row.civ_effect >= 0 else "right",
                    fontsize=7, color="gray")
        fig.tight_layout()
        return fig
