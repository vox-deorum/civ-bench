"""Report document builders: assemble produced :class:`Section`s (stage 5).

A builder is a pure function over the :class:`~bench.reports.context.ReportBuildContext`
(run metadata + resolved sections + the full-artifact loader) returning a document
model. It decides structure only (grouping, ordering, headings, intro prose), never
*which* analyses ran (that is the run-spec's job) and never an analysis's own content
(that is the module's job). The default layout groups sections into the five analysis
families (ratings / prediction / calibration / performance / exploratory), preserving
the dependency order the runner resolved, so no analysis hardcodes its place in the
document (invariant 3).

When the resolved sections contain the ``performance.controlled_seed_report``
analysis, the document additionally carries the controlled-seed chapter
(:func:`bench.reports.controlled_seed.controlled_seed_document` builds it from
that section's three persisted tables): the section leaves its module family
and becomes a chapter of its own, parallel to the families, rendered as the
heatmap pages under ``controlled-seed/``.
"""

from __future__ import annotations

from bench.reports.announcements import build_announcements
from bench.reports.game_log import game_log_document
from .context import ReportBuildContext
from bench.reports.content import resolve_footer
from .controlled_seed import (
    CONTROLLED_SEED_DIR,
    CONTROLLED_SEED_OVERVIEW,
    CONTROLLED_SEED_PURPOSE,
    CONTROLLED_SEED_TITLE,
    controlled_seed_document,
)
from .model import Download, FamilyGroup, ReportDocument, Section

# Friendly titles for the known module families; an unknown family falls back to
# a title-cased version of its registry prefix (so a future family renders too).
_FAMILY_TITLES = {
    "ratings": "Ratings",
    "prediction": "Prediction",
    "calibration": "Calibration",
    "performance": "Performance",
    "exploratory": "Exploratory",
}

_FAMILY_PURPOSES = {
    "ratings": (
        "Compare strategists' relative skill across games, using ratings that "
        "summarize how consistently they outperform their opponents."
    ),
    "prediction": (
        "Compare how well win-probability estimators predict game outcomes, "
        "since these estimates underpin the strength scores used to evaluate "
        "strategists."
    ),
    "calibration": (
        "Check whether predicted win probabilities match observed win rates, "
        "and examine how prediction errors, civilizations, and starting "
        "positions affect the evaluation."
    ),
    "performance": (
        "Compare strategists' strength, progress, cost, and token use, with "
        "coverage checks to show which experiments support the results."
    ),
    "exploratory": (
        "Explore game patterns and strategy profiles across the dataset."
    ),
}

# Stable display order for the families; unknown families sort after these, in the
# order they first appear among the sections.
_FAMILY_ORDER = ["ratings", "prediction", "calibration", "performance", "exploratory"]


def family_of(module: str) -> str:
    """The family key of a ``family.module`` registry name (``"ratings"``)."""
    return module.split(".", 1)[0] if "." in module else module


def family_title(key: str) -> str:
    return _FAMILY_TITLES.get(key, key.replace("_", " ").title())


def family_sort_index(key: str) -> int:
    """Canonical position of a family (unknown families sort last). The runner uses
    this to order the *default* (``sections:null``) document; explicit
    ``report.sections`` take priority, followed by the remaining enabled analyses."""
    return _FAMILY_ORDER.index(key) if key in _FAMILY_ORDER else len(_FAMILY_ORDER)


def _group_by_family(sections: list[Section]) -> list[FamilyGroup]:
    """Bucket sections into family groups, preserving the incoming section order
    both within a group and across groups (groups appear in first-appearance order).

    Ordering is therefore the runner's responsibility: the default path hands us
    sections already sorted into canonical family order, while an explicit
    ``report.sections`` list hands us the author's priorities followed by the
    remaining enabled analyses. Either way we keep
    it (invariant 3: the template never reorders behind the config's back)."""
    groups: dict[str, FamilyGroup] = {}
    order: list[str] = []
    for section in sections:
        key = family_of(section.module)
        if key not in groups:
            groups[key] = FamilyGroup(key=key, title=family_title(key))
            order.append(key)
        groups[key].sections.append(section)
    return [groups[k] for k in order]


def _join_titles(titles: list[str]) -> str:
    """Join report titles as a short phrase for a family summary."""
    if len(titles) == 1:
        return titles[0]
    if len(titles) == 2:
        return f"{titles[0]} and {titles[1]}"
    return f"{', '.join(titles[:-1])}, and {titles[-1]}"


def _summarize_family(group: FamilyGroup) -> str:
    """Explain what readers can learn from this family of analyses."""
    if group.key in _FAMILY_PURPOSES:
        return _FAMILY_PURPOSES[group.key]
    titles = [section.title for section in group.sections]
    noun = "result" if len(titles) == 1 else "results"
    return f"This page brings together the {noun} for {_join_titles(titles)}."


def default_template(ctx: ReportBuildContext) -> ReportDocument:
    """The default report layout: family chapters in dependency order, plus the
    controlled-seed chapter when the report carries its analysis."""
    meta = ctx.meta
    sections = ctx.sections

    # The controlled-seed annex document and its section. The section leaves
    # the module families and becomes a chapter of its own. The heatmap link
    # is added after the annex snapshot is taken, so the chapter page's own
    # "Source tables" list never links back to itself, and only when the run
    # renders html (the pages are html-only; a markdown-only report would
    # otherwise carry a dead link).
    controlled = controlled_seed_document(ctx)
    if controlled is not None:
        annex_section = ctx.section(controlled.section_id)
        family_sections = [s for s in sections if s is not annex_section]
    else:
        family_sections = sections

    game_log = game_log_document(ctx)
    if game_log is not None:
        family_sections = [s for s in family_sections if s.id != game_log.section_id]
        if controlled is not None:
            controlled.game_log = game_log
        if "html" in (meta.get("formats") or []):
            download = Download(label="Game Log (HTML)", rel_path="games.html")
            ctx.section(game_log.section_id).downloads.append(download)
            game_log.downloads.append(download)

    groups = _group_by_family(family_sections)
    for group in groups:
        group.summary = _summarize_family(group)
    if controlled is not None:
        groups.append(
            FamilyGroup(
                key=CONTROLLED_SEED_DIR,
                title=CONTROLLED_SEED_TITLE,
                sections=[annex_section],
                summary=CONTROLLED_SEED_PURPOSE,
            )
        )
        section_positions = {section.id: i for i, section in enumerate(sections)}
        groups.sort(key=lambda group: section_positions[group.sections[0].id])
        if "html" in (meta.get("formats") or []):
            annex_section.downloads.append(
                Download(
                    label="Matched-map heatmaps (HTML)",
                    rel_path=CONTROLLED_SEED_OVERVIEW,
                )
            )

    section_by_id = {section.id: section for section in sections}
    overview_sections = [
        section_by_id[section_id]
        for section_id in meta.get("overview_section_ids", [])
        if section_id in section_by_id
        and (game_log is None or section_id != game_log.section_id)
    ]
    return ReportDocument(
        title=meta["title"],
        run_name=meta["run_name"],
        seed=meta["seed"],
        config_path=meta["config_path"],
        output_root=meta["output_root"],
        description=meta.get("description", "") or "",
        footer=resolve_footer(meta.get("footer")),
        benchmark_citation=meta.get("benchmark_citation"),
        groups=groups,
        overview_sections=overview_sections,
        controlled_seed=controlled,
        game_log=game_log,
        announcements=build_announcements(ctx),
    )
