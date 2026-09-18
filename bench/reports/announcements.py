"""Build front-page updates from saved coverage and rating tables."""

from __future__ import annotations

import math
import re

import pandas as pd

from bench.reports.context import ReportBuildContext
from bench.reports.model import Announcement


def _literal(text: str) -> str:
    """Keep configured names literal inside inline Markdown."""
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", str(text))


def build_announcements(ctx: ReportBuildContext) -> list[Announcement]:
    """Use report order to choose the coverage and overall rating sources."""
    coverage = next((
        section for section in ctx.sections
        if section.module == "performance.experiment_completeness"
        and ctx.has_table(section.id, "condition_progress")
    ), None)
    if coverage is None:
        return []
    progress = ctx.load_table(coverage.id, "condition_progress")
    if progress.empty:
        return []

    announcements = []
    latest = _latest_result(ctx, progress)
    if latest is not None:
        announcements.append(latest)
    if ctx.meta.get("condition_completeness_filter"):
        pending = progress[progress["completed_games"] < progress["required_games"]]
        labels = []
        for _, condition in pending.groupby("experiment", sort=True):
            names = " / ".join(
                f"**{_literal(name)}**"
                for name in sorted(condition["player_type"].dropna().unique())
            )
            row = condition.iloc[0]
            labels.append(f"{names} ({int(row['completed_games'])}/{int(row['required_games'])})")
        if labels:
            announcements.append(Announcement(
                kind="testing", title="Currently being tested", text="; ".join(labels),
            ))
    return announcements


def _latest_result(ctx: ReportBuildContext, progress: pd.DataFrame) -> Announcement | None:
    rating_section = next((
        section for section in ctx.sections
        if section.module in {"ratings.bradley_terry", "ratings.plackett_luce"}
        and section.metadata.get("group_by", ["player_type"]) == ["player_type"]
        and ctx.has_table(section.id, "ratings")
    ), None)
    if rating_section is None:
        return None
    ratings = ctx.load_table(rating_section.id, "ratings")
    if not {"player_type", "elo"} <= set(ratings.columns) or "composite_type" in ratings:
        return None
    ratings = ratings[["player_type", "elo"]].copy()
    ratings["elo"] = pd.to_numeric(ratings["elo"], errors="coerce")
    ratings = ratings[ratings["elo"].map(math.isfinite)]
    labels = (ctx.meta.get("rating_labels") or {}).get(rating_section.id, {})
    identities = [labels.get(str(identity), {
        "model": str(identity), "condition": "", "label": "Base", "baseline": False,
    }) for identity in ratings["player_type"]]
    for column, key in (("model", "model"), ("condition", "condition"),
                        ("condition_label", "label"), ("baseline", "baseline")):
        ratings[column] = [identity[key] for identity in identities]
    ratings = ratings[~ratings["baseline"].astype(bool)].copy()
    ratings["rank"] = ratings.groupby("condition")["elo"].rank(
        ascending=False, method="min",
    ).astype(int)
    complete = progress[
        (progress["required_games"] > 0)
        & (progress["completed_games"] == progress["required_games"])
    ].copy()
    complete["completed_at"] = pd.to_datetime(
        complete["completed_at"], format="ISO8601", utc=True, errors="coerce",
    )
    complete = complete.dropna(subset=["completed_at"])
    complete = complete.merge(ratings, on="player_type", how="inner")
    if complete.empty:
        return None
    complete = complete.sort_values(
        ["completed_at", "experiment", "player_type"], ascending=[False, True, True],
        kind="mergesort",
    )
    latest = complete.iloc[0]
    condition = complete[complete["experiment"] == latest["experiment"]]
    descriptions = []
    for model in sorted(condition["model"].unique()):
        scores = ratings[ratings["model"] == model].sort_values("condition", kind="mergesort")
        values = " or ".join(
            f"**{row.elo:.0f} Elo** ({_literal(row.condition_label)}, #{row.rank})"
            for row in scores.itertuples(index=False)
        )
        descriptions.append(f"**{_literal(model)}** scored {values}")
    text = "; ".join(descriptions) + "."
    return Announcement(
        kind="result", title="Latest score", text=text,
        date=latest["completed_at"].date().isoformat(),
    )
