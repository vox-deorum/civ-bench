"""Build the plain-language front page from ``report.intro`` (stage 5).

The prose (headline, body, glossary, project cards, chart headings) comes from
the run spec. Everything with a number in it comes from the run's saved tables,
loaded through :class:`~bench.reports.context.ReportBuildContext`: the rating
table for the leaderboard, the usage table for cost against skill, the grand
strategy table for play styles, and the game log for the fact chips. Catalog
facts (each player type's model, condition, and color) arrive pre-resolved in
``meta["front_catalog"]``, so this module never reads the catalog.

Hover tips are plain text in the shared tooltip format: the first line is the
bold title, ``- `` lines are bullets, and ``**text**`` is bold.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from fnmatch import fnmatchcase

import pandas as pd

from .content import report_summary
from .context import ReportBuildContext
from .game_log import games_url
from .model import Announcement, FrontCard, FrontChart, FrontFact, FrontLink, FrontPage, Section
from .model_tips import model_tip

# Grand strategy -> (plain name, CSS class, what it means).
GOALS = (
    ("Conquest", "Domination", "dom", "Win by capturing every other nation's original capital."),
    ("Culture", "Culture", "culture",
     "Win by spreading your culture until the rest of the world adopts your way of life."),
    ("UnitedNations", "Diplomacy", "diplo", "Win by being voted world leader at the United Nations."),
    ("Spaceship", "Science", "science", "Win by being first to build and launch a spaceship to another star."),
)

# Module -> (card label, icon) for the plain finding cards.
CARD_STYLES = {
    "ratings.bradley_terry": ("Strongest player", "crown"),
    "ratings.plackett_luce": ("Strongest player", "crown"),
    "ratings.outcome_matchups": ("Most wins", "flag"),
    "ratings.matchups": ("Head to head", "flag"),
    "performance.usage_efficiency": ("Best value", "coin"),
    "behavior.commitment": ("Ways to win", "target"),
    "behavior.diplomacy": ("Diplomacy", "hand"),
    "behavior.flavors": ("Changed habits", "dial"),
    "behavior.policies": ("Politics", "scroll"),
    "prediction.evaluate": ("How reliable is this?", "eye"),
}

_TOP_ROWS = 10  # leaderboard rows shown before "Show the rest", and play-style rows
_SHAPES = ("circle", "square", "diamond", "triangle")


class _Names:
    """Plain names, condition labels, and series for the charted player types."""

    def __init__(self, intro: dict, front_catalog: dict, model_tips: dict[str, str]):
        self.identities = front_catalog.get("identities") or {}
        self.catalog_models = front_catalog.get("models") or []
        self.names = intro.get("names") or {}
        self.conditions = intro.get("conditions") or {}
        self.series_rules = intro.get("series") or []
        self.model_tips = model_tips

    def info(self, pt: str) -> dict:
        return self.identities.get(pt) or {"model": pt, "condition": "", "baseline": False, "color": ""}

    def baseline(self, pt: str) -> bool:
        return bool(self.info(pt).get("baseline"))

    def reference(self, pt: str) -> bool:
        """Whether this is the built-in AI, the yardstick for head-to-head odds."""
        return bool(self.info(pt).get("reference"))

    def model(self, pt: str) -> str:
        return str(self.info(pt)["model"])

    def name(self, pt: str) -> str:
        if pt in self.names:
            return self.names[pt]
        model = self.model(pt)
        return self.names.get(model, model)

    def mode(self, pt: str) -> tuple[str, str]:
        """The plain condition label (``every 5 turns``) and its tip."""
        if self.baseline(pt):
            return "", ""
        condition = str(self.info(pt).get("condition") or "")
        entry = self.conditions.get(condition)
        if entry is None:
            return condition, ""
        return entry["label"], entry.get("tip", "")

    def mode_index(self, pt: str) -> int:
        """The condition's position in ``intro.conditions`` (unknown ones last)."""
        condition = str(self.info(pt).get("condition") or "")
        order = list(self.conditions)
        return order.index(condition) if condition in order else len(order)

    def shape(self, pt: str) -> str:
        return _SHAPES[min(self.mode_index(pt), len(_SHAPES) - 1)]

    def title(self, pt: str) -> str:
        mode, _ = self.mode(pt)
        return f"{self.name(pt)} (deciding {mode})" if mode else self.name(pt)

    def plain(self, pt: str) -> str:
        """``**GLM-5.3** (every 5 turns)`` for card sentences."""
        mode, _ = self.mode(pt)
        return f"**{self.name(pt)}** ({mode})" if mode else f"**{self.name(pt)}**"

    def series(self, pt: str) -> str:
        model = self.model(pt)
        for rule in self.series_rules:
            if any(fnmatchcase(model, pattern) for pattern in rule["models"]):
                return rule["name"]
        return "Other"

    def series_color(self, series: str) -> str:
        for rule in self.series_rules:
            if rule["name"] != series:
                continue
            for model, color in self.catalog_models:
                if color and any(fnmatchcase(model, pattern) for pattern in rule["models"]):
                    return color
        return "#8d99a9"

    def color(self, pt: str) -> str:
        return self.series_color(self.series(pt)) if self.series_rules else (
            self.info(pt).get("color") or "#8d99a9"
        )

    def tip(self, pt: str) -> str:
        return model_tip(self.model(pt), self.model_tips)

    def replace_labels(self, text: str, modes: bool = True) -> str:
        """Swap technical identity labels in a summary for plain ones.

        With ``modes=False`` every label becomes just the plain model name, for
        text that already states each condition.
        """
        forms: dict[str, str] = {}
        for pt, info in self.identities.items():
            plain = self.plain(pt) if modes else f"**{self.name(pt)}**"
            forms[pt] = plain
            condition = str(info.get("condition") or "")
            base = pt[: len(pt) - len(condition) - 1] if pt.endswith("-" + condition) else pt
            forms[f"{base} | {condition}"] = plain
            # A bare base (the identity without its condition) names just the model.
            forms.setdefault(base, f"**{self.name(pt)}**")
        for form in sorted(forms, key=len, reverse=True):
            # A label the summary already bolds keeps only the plain form's bold.
            escaped = re.escape(form)
            pattern = rf"\*\*{escaped}\*\*|(?<![\w.-]){escaped}(?![\w.-])"
            text = re.sub(pattern, lambda _match, plain=forms[form]: plain, text)
        return text


def in_sentence(name: str) -> str:
    """``Built-in AI`` -> ``built-in AI`` after "the"; model names keep their case."""
    if len(name) > 1 and name[0].isupper() and name[1].islower():
        return name[0].lower() + name[1:]
    return name


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _beats(elo: float, reference: float) -> float:
    """Chance to beat the reference head-to-head under the Elo model."""
    return 1.0 / (1.0 + 10 ** ((reference - elo) / 400.0))


def _shares(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of turns (0 to 1) each player type holds each grand strategy."""
    rows = frame[frame["metric"].astype(str).str.startswith("grand_strategy_share_")]
    shares = rows.pivot_table(index="player_type", columns="category", values="mean", aggfunc="first")
    shares = shares.reindex(columns=[goal for goal, *_ in GOALS]).fillna(0.0)
    totals = shares.sum(axis=1)
    return shares[totals > 0].div(totals[totals > 0], axis=0)


class _Builder:
    def __init__(self, ctx: ReportBuildContext, intro: dict):
        self.ctx = ctx
        self.intro = intro
        self.names = _Names(intro, ctx.meta.get("front_catalog") or {}, dict(ctx.meta.get("model_tips") or {}))
        self.resolved = {section.id for section in ctx.sections}
        charts = intro.get("charts") or {}
        self.ratings = self._table(charts, "leaderboard", "ratings", {"player_type", "elo"})
        self.usage = self._table(charts, "cost", "usage_vs_rating", {"player_type", "elo", "avg_cost_per_player_game"})
        styles = self._table(charts, "styles", "grand_strategy", {"player_type", "metric", "category", "mean"})
        self.shares = _shares(styles) if styles is not None else None
        if self.ratings is not None:
            self.ratings = self.ratings.assign(elo=_numeric(self.ratings, "elo"))
            self.ratings = self.ratings[self.ratings["elo"].map(math.isfinite)]
            self.ratings = self.ratings.sort_values("elo", ascending=False, kind="mergesort").reset_index(drop=True)
            self.ratings["rank"] = range(1, len(self.ratings) + 1)
        self.games = self._game_links()
        self.cost = {}
        if self.usage is not None:
            cost = _numeric(self.usage, "avg_cost_per_player_game")
            self.cost = {pt: c for pt, c in zip(self.usage["player_type"].astype(str), cost) if math.isfinite(c) and c > 0}

    # ── data access ──────────────────────────────────────────────────────
    def _table(self, charts: dict, kind: str, name: str, columns: set) -> pd.DataFrame | None:
        chart = charts.get(kind)
        if chart is None:
            return None
        stage = chart["stage"]
        where = f"report.intro.charts.{kind}"
        if stage not in self.resolved:
            self.ctx.warnings.append(f"{where}: stage '{stage}' is not in the report; the chart was skipped.")
            return None
        if not self.ctx.has_table(stage, name):
            self.ctx.warnings.append(f"{where}: stage '{stage}' saved no '{name}' table; the chart was skipped.")
            return None
        frame = self.ctx.load_table(stage, name)
        missing = sorted(columns - set(frame.columns))
        if missing:
            self.ctx.warnings.append(f"{where}: table '{name}' lacks columns {missing}; the chart was skipped.")
            return None
        group_by = self.ctx.section(stage).metadata.get("group_by", ["player_type"])
        if "composite_type" in frame or group_by != ["player_type"]:
            self.ctx.warnings.append(f"{where}: stage '{stage}' is not rated per player type; the chart was skipped.")
            return None
        frame = frame.copy()
        frame["player_type"] = frame["player_type"].astype(str)
        return frame

    def _game_links(self) -> dict[str, str]:
        """Each player type's Game Log page, filtered to its strategist and condition.

        The page shows controlled games by default; a type with none of those
        links to all of its games instead.
        """
        game_log = self._game_log()
        if game_log is None:
            return {}
        games, players = game_log
        if not {"game_id", "player_type", "strategist", "condition"} <= set(players.columns):
            return {}
        controlled = set()
        if "controlled" in games:
            flags = games["controlled"].astype(str).str.lower().isin({"true", "1", "1.0"})
            controlled = set(games.loc[flags, "game_id"].astype(str))
        links = {}
        for pt, seats in players.groupby(players["player_type"].astype(str), sort=True):
            first = seats.iloc[0]
            strategist, condition = (str(first[key]) if pd.notna(first[key]) else ""
                                     for key in ("strategist", "condition"))
            if not strategist:
                continue
            any_controlled = seats["game_id"].astype(str).isin(controlled).any()
            links[pt] = games_url(strategist=strategist, condition=condition or None,
                                  controlled=None if any_controlled else "false")
        return links

    @property
    def reference(self) -> tuple[str, float] | None:
        """The built-in AI's name and rating, the yardstick for head-to-head odds."""
        if self.ratings is None:
            return None
        for row in self.ratings.itertuples(index=False):
            if self.names.reference(row.player_type):
                return self.names.name(row.player_type), float(row.elo)
        return None

    def llm(self, pts) -> list[str]:
        return [pt for pt in pts if not self.names.baseline(pt)]

    # ── tip lines ────────────────────────────────────────────────────────
    def favorite(self, pt: str) -> str | None:
        if self.shares is None or pt not in self.shares.index:
            return None
        row = self.shares.loc[pt]
        goal = row.idxmax()
        label = next(plain for key, plain, *_ in GOALS if key == goal)
        return f"- Favorite way to win: **{label}** ({_pct(row[goal])})"

    def cost_line(self, pt: str) -> str | None:
        if pt not in self.cost:
            return None
        return f"- Costs about **${self.cost[pt]:.2f}** per player per game"

    def tip(self, pt: str, lines: list[str | None]) -> str:
        extra = self.names.tip(pt)
        return "\n".join([self.names.title(pt), *[line for line in lines if line], *([f"- {extra}"] if extra else [])])

    # ── charts ───────────────────────────────────────────────────────────
    def leaderboard(self, spec: dict) -> FrontChart | None:
        if self.ratings is None:
            return None
        reference = self.reference
        rows = []
        se = _numeric(self.ratings, "se_elo") if "se_elo" in self.ratings else pd.Series([math.nan] * len(self.ratings))
        for row, err in zip(self.ratings.itertuples(index=False), se):
            pt, elo = row.player_type, float(row.elo)
            half = 1.96 * float(err) if math.isfinite(err) else 0.0
            mode, mode_tip = self.names.mode(pt)
            if self.names.reference(pt):
                lines = [f"- Rating: **{elo:.0f}**, the yardstick for every other rating"]
            else:
                rank = f"±{half:.0f}, #{row.rank}" if half else f"#{row.rank}"
                lines = [f"- Rating: **{elo:.0f}** ({rank})"]
                if reference is not None:
                    lines.append(
                        f"- Beats the {in_sentence(reference[0])} **{_pct(_beats(elo, reference[1]))}** of the time head-to-head"
                    )
            lines += [self.favorite(pt), self.cost_line(pt)]
            base = reference[1] if reference is not None else 1500.0
            rows.append({
                "key": pt, "name": self.names.name(pt), "mode": mode, "mode_tip": mode_tip,
                "mode_index": self.names.mode_index(pt),
                "elo": elo, "low": elo - half, "high": elo + half, "rank": int(row.rank),
                "kind": "ref" if self.names.baseline(pt) else ("up" if elo >= base else "down"),
                "tip": self.tip(pt, lines), "href": self.games.get(pt, ""),
            })
        extra = {
            "reference": reference[1] if reference is not None else 1500.0,
            "reference_name": reference[0] if reference is not None else "",
            "top": _TOP_ROWS,
        }
        return FrontChart("leaderboard", spec["stage"], spec["title"], spec.get("text", ""), rows, extra)

    def cost_chart(self, spec: dict) -> FrontChart | None:
        if self.usage is None:
            return None
        usage = self.usage
        if "is_baseline" in usage:
            usage = usage[~usage["is_baseline"].astype(str).str.lower().isin({"true", "1"})]
        efficiency = _numeric(usage, "efficiency_elo_cost") if "efficiency_elo_cost" in usage else None
        rank = {} if self.ratings is None else dict(zip(self.ratings["player_type"], self.ratings["rank"]))
        rows = []
        for i, (pt, elo) in enumerate(zip(usage["player_type"], _numeric(usage, "elo"))):
            if pt not in self.cost or not math.isfinite(elo):
                continue
            eff = float(efficiency.iloc[i]) if efficiency is not None else math.nan
            lines = [self.cost_line(pt), f"- Rating: **{elo:.0f}**" + (f" (#{rank[pt]})" if pt in rank else "")]
            if math.isfinite(eff):
                side = "above" if eff >= 0 else "below"
                lines.append(f"- **{abs(eff):.0f}** points {side} the typical rating for its price")
            rows.append({
                "key": pt, "name": self.names.name(pt), "series": self.names.series(pt),
                "color": self.names.color(pt), "shape": self.names.shape(pt),
                "cost": self.cost[pt], "elo": float(elo), "efficiency": eff,
                "tip": self.tip(pt, lines),
            })
        if not rows:
            return None
        fit = ((self.ctx.section(spec["stage"]).metadata.get("usage_skill_fits") or {}).get("cost") or {})
        best = {}
        for row in rows:
            best[row["series"]] = max(best.get(row["series"], -math.inf), row["elo"])
        series = [{"name": name, "color": self.names.series_color(name) if self.names.series_rules else ""}
                  for name in sorted(best, key=lambda name: (-best[name], name))]
        if not self.names.series_rules:
            series = []
        shapes = []
        for condition, entry in (self.intro.get("conditions") or {}).items():
            shapes.append({"shape": _SHAPES[min(len(shapes), len(_SHAPES) - 1)], "label": entry["label"]})
        # Labels shown before a series is picked: the top three and the best and worst value.
        by_elo = sorted(rows, key=lambda row: -row["elo"])
        defaults = {row["key"] for row in by_elo[:3]}
        rated = [row for row in rows if math.isfinite(row["efficiency"])]
        if rated:
            defaults |= {max(rated, key=lambda r: r["efficiency"])["key"], min(rated, key=lambda r: r["efficiency"])["key"]}
        for row in rows:
            row["label"] = row["key"] in defaults
        extra = {
            "fit": {"intercept": fit.get("intercept"), "slope": fit.get("slope")}
            if fit.get("intercept") is not None and fit.get("slope") is not None else None,
            "reference": self.reference[1] if self.reference else None,
            "reference_name": self.reference[0] if self.reference else "",
            "series": series, "shapes": shapes,
        }
        return FrontChart("cost", spec["stage"], spec["title"], spec.get("text", ""), rows, extra)

    def styles_chart(self, spec: dict) -> FrontChart | None:
        if self.shares is None or self.shares.empty:
            return None
        order = list(self.ratings["player_type"]) if self.ratings is not None else list(self.shares.index)
        picked = [pt for pt in self.llm(order) if pt in self.shares.index][:_TOP_ROWS]
        refs = [pt for pt in order if self.names.reference(pt) and pt in self.shares.index][:1]
        rows = []
        for pt in picked + refs:
            share = self.shares.loc[pt]
            mode, _ = self.names.mode(pt)
            segments = []
            for key, label, cls, _ in GOALS:
                lines = [
                    f"- **{other}: {_pct(share[k])}**" if k == key else f"- {other}: {_pct(share[k])}"
                    for k, other, *_ in GOALS
                ]
                segments.append({"cls": cls, "label": label, "share": float(share[key]),
                                 "tip": "\n".join([self.names.title(pt), *lines])})
            rows.append({"key": pt, "name": self.names.name(pt), "mode": mode,
                         "reference": pt in refs, "segments": segments, "href": self.games.get(pt, "")})
        extra = {"goals": [{"cls": cls, "label": label, "tip": tip} for _, label, cls, tip in GOALS]}
        return FrontChart("styles", spec["stage"], spec["title"], spec.get("text", ""), rows, extra)

    # ── hero ─────────────────────────────────────────────────────────────
    def fact_values(self) -> dict[str, int]:
        values: dict[str, int] = {}
        source = self.ratings if self.ratings is not None else self.usage
        if source is not None:
            llm = self.llm(source["player_type"])
            values["setups"] = len(llm)
            values["models"] = len({self.names.model(pt) for pt in llm})
            if self.names.series_rules:
                values["makers"] = len({self.names.series(pt) for pt in llm})
        game_log = self._game_log()
        if game_log is not None:
            games, players = game_log
            values["games"] = int(games["game_id"].nunique()) if "game_id" in games else len(games)
            if "turns" in games and games["turns"].notna().any():
                values["turns"] = int(round(float(_numeric(games, "turns").median())))
            if "seed" in games:
                values["starts"] = int(games["seed"].nunique())
            if {"player_type", "game_id"} <= set(players.columns):
                counts = players.groupby(players["player_type"].astype(str))["game_id"].nunique()
                counts = counts[[not self.names.baseline(pt) for pt in counts.index]]
                if self.ratings is not None:
                    counts = counts[counts.index.isin(self.ratings["player_type"])]
                if not counts.empty:
                    values["min_games"] = int(counts.min())
        return values

    def _game_log(self):
        section = next((s for s in self.ctx.sections if s.module == "performance.game_log"), None)
        if section is None or not self.ctx.has_table(section.id, "games"):
            return None
        games = self.ctx.load_table(section.id, "games")
        players = (self.ctx.load_table(section.id, "game_players")
                   if self.ctx.has_table(section.id, "game_players") else pd.DataFrame())
        return games, players

    def facts(self) -> list[FrontFact]:
        values = self.fact_values()
        facts = []
        for entry in self.intro.get("facts") or []:
            if entry["value"] not in values:
                self.ctx.warnings.append(
                    f"report.intro.facts: no data for '{entry['value']}'; the fact was skipped."
                )
                continue
            tip = entry.get("tip", "")
            try:
                tip = tip.format_map(values)
            except KeyError as exc:
                self.ctx.warnings.append(f"report.intro.facts: no data for {exc} in a tip; the tip was dropped.")
                tip = ""
            facts.append(FrontFact(f"{values[entry['value']]:,}", entry["label"], tip))
        return facts

    def leader_type(self) -> str:
        """The highest-rated model setup, or ``""``."""
        if self.ratings is None:
            return ""
        llm = self.llm(self.ratings["player_type"])
        return llm[0] if llm else ""

    def leader(self) -> str:
        pt = self.leader_type()
        if not pt:
            return ""
        best = self.ratings[self.ratings["player_type"] == pt].iloc[0]
        mode, _ = self.names.mode(best.player_type)
        deciding = f", deciding {mode}," if mode else ""
        return f"**{self.names.name(best.player_type)}**{deciding} rated **{best.elo:.0f}**"

    # ── cards ────────────────────────────────────────────────────────────
    def card_text(self, section: Section) -> str:
        builder = {
            "ratings.bradley_terry": self._card_ratings,
            "ratings.plackett_luce": self._card_ratings,
            "ratings.outcome_matchups": self._card_wins,
            "performance.usage_efficiency": self._card_value,
            "behavior.commitment": self._card_goals,
            "prediction.evaluate": self._card_prediction,
        }.get(section.module)
        text = None
        if builder is not None and not section.empty:
            try:
                text = builder(section)
            except (KeyError, ValueError, IndexError):
                text = None
        if text:
            return text
        summary = report_summary(section.summary.strip(), section.metadata)
        return self.names.replace_labels(summary) or "No result summary was produced for this analysis."

    def _load(self, section: Section, name: str) -> pd.DataFrame | None:
        return self.ctx.load_table(section.id, name) if self.ctx.has_table(section.id, name) else None

    def _card_ratings(self, section: Section) -> str | None:
        frame = self._load(section, "ratings")
        if frame is None or "composite_type" in frame or section.metadata.get("group_by", ["player_type"]) != ["player_type"]:
            return None
        frame = frame.assign(elo=_numeric(frame, "elo"), player_type=frame["player_type"].astype(str))
        frame = frame.sort_values("elo", ascending=False, kind="mergesort")
        llm = self.llm(frame["player_type"])
        if not llm:
            return None
        best = frame[frame["player_type"] == llm[0]].iloc[0]
        text = f"{self.names.plain(best.player_type)} is the strongest so far."
        baselines = frame[[self.names.reference(pt) for pt in frame["player_type"]]]
        if not baselines.empty:
            ref = baselines.iloc[0]
            text += (f" It beats the {in_sentence(self.names.name(ref.player_type))} "
                     f"**{_pct(_beats(float(best.elo), float(ref.elo)))}** of the time head-to-head.")
        return text

    def _card_wins(self, section: Section) -> str | None:
        frame = self._load(section, "vs_reference")
        if frame is None:
            return None
        frame = frame.assign(rate=_numeric(frame, "win_rate_vs_ref"), player_type=frame["player_type"].astype(str))
        frame = frame[frame["rate"].notna() & ~frame["player_type"].map(self.names.baseline)]
        if frame.empty:
            return None
        best = frame.sort_values(["rate", "player_type"], ascending=[False, True], kind="mergesort").iloc[0]
        text = f"{self.names.plain(best.player_type)} won **{_pct(best.rate)}** of its games."
        expected = float(best.expected_win_rate) if "expected_win_rate" in frame else math.nan
        if math.isfinite(expected) and expected > 0:
            text += f" With {round(1 / expected)} players, an even chance would be 1 in {round(1 / expected)}."
        return text

    def _card_value(self, section: Section) -> str | None:
        frame = self._load(section, "usage_vs_rating")
        if frame is None or "efficiency_elo_cost" not in frame:
            return None
        frame = frame.assign(eff=_numeric(frame, "efficiency_elo_cost"), player_type=frame["player_type"].astype(str))
        frame = frame[frame["eff"].notna()].sort_values(["eff", "player_type"], ascending=[False, True], kind="mergesort")
        if len(frame) < 2:
            return None
        best, worst = frame.iloc[0], frame.iloc[-1]
        return (f"{self.names.plain(best.player_type)} gets the most skill for its price. "
                f"{self.names.plain(worst.player_type)} gets the least.")

    def _card_goals(self, section: Section) -> str | None:
        frame = self._load(section, "grand_strategy")
        if frame is None:
            return None
        shares = _shares(frame)
        shares = shares[[not self.names.baseline(pt) for pt in shares.index]]
        if shares.empty:
            return None
        parts = []
        for key, label, *_ in GOALS:
            column = shares[key].sort_index(kind="mergesort")
            pt = column.idxmax()
            parts.append(f"{label}: {self.names.plain(pt)}, {_pct(column[pt])}")
        return "Who aims for each kind of win most often. " + "; ".join(parts) + "."

    def _card_prediction(self, section: Section) -> str | None:
        frame = self._load(section, "metrics")
        if frame is None or "roc_auc" not in frame:
            return None
        best = float(_numeric(frame, "roc_auc").max())
        if not math.isfinite(best):
            return None
        return (f"Shown a winner and a loser from the same game, our best predictor picks "
                f"the winner **{_pct(best)}** of the time, well before the game ends.")

    def cards(self, sections: list[Section]) -> list[FrontCard]:
        cards = []
        for section in sections:
            label, icon = CARD_STYLES.get(section.module, (section.title, "dot"))
            cards.append(FrontCard(section=section, label=label, icon=icon, text=self.card_text(section)))
        return cards


def front_page_document(
    ctx: ReportBuildContext,
    overview_sections: list[Section],
    announcements: list[Announcement] | None = None,
) -> FrontPage | None:
    """The front page for ``report.intro``, or ``None`` when it is not configured."""
    intro = ctx.meta.get("intro")
    if not intro:
        return None
    builder = _Builder(ctx, intro)
    charts_spec = intro.get("charts") or {}
    charts = []
    for kind, make in (("leaderboard", builder.leaderboard), ("cost", builder.cost_chart),
                       ("styles", builder.styles_chart)):
        if kind in charts_spec:
            chart = make(charts_spec[kind])
            if chart is not None:
                charts.append(chart)
    projects = intro.get("projects") or {}
    body = [part.strip() for part in re.split(r"\n\s*\n", intro.get("body") or "") if part.strip()]
    return FrontPage(
        headline=intro["headline"],
        eyebrow=intro.get("eyebrow") or "",
        body=body,
        glossary=dict(intro.get("glossary") or {}),
        facts=builder.facts(),
        leader=builder.leader(),
        leader_type=builder.leader_type(),
        charts=charts,
        projects_title=projects.get("title", ""),
        projects_text=projects.get("text") or "",
        projects=[FrontLink(label=p["link"], url=p["url"], text=p["text"], name=p["name"])
                  for p in projects.get("items") or []],
        links=[FrontLink(label=link["label"], url=link["url"]) for link in projects.get("links") or []],
        cards=builder.cards(overview_sections),
        # The latest score names models and lists each condition after its Elo.
        news=[replace(item, text=builder.names.replace_labels(item.text, modes=item.kind != "result"))
              for item in announcements or []],
    )
