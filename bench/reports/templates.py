"""Report document builders: assemble produced :class:`Section`s (stage 5).

A builder is a pure function over the :class:`~bench.reports.context.ReportBuildContext`
(run metadata + resolved sections + the full-artifact loader) returning a document
model. It decides structure only — grouping, ordering, headings, intro prose — never
*which* analyses ran (that is the run-spec's job) and never an analysis's own content
(that is the module's job). The default layout groups sections into the five analysis
families (ratings / prediction / calibration / performance / exploratory), preserving
the dependency order the runner resolved, so no analysis hardcodes its place in the
document (invariant 3).

When the resolved sections contain the ``performance.controlled_seed_report``
analysis, the document additionally carries the controlled-seed annex
(:func:`bench.reports.controlled_seed.controlled_seed_document` builds it from that
section's three persisted tables) and the section links to its heatmap pages.
"""

from __future__ import annotations

from .context import ReportBuildContext
from .controlled_seed import CONTROLLED_SEED_OVERVIEW, controlled_seed_document
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
    ``report.sections`` keep their authored order instead."""
    return _FAMILY_ORDER.index(key) if key in _FAMILY_ORDER else len(_FAMILY_ORDER)


def _group_by_family(sections: list[Section]) -> list[FamilyGroup]:
    """Bucket sections into family groups, preserving the incoming section order
    both within a group and across groups (groups appear in first-appearance order).

    Ordering is therefore the runner's responsibility: the default path hands us
    sections already sorted into canonical family order, while an explicit
    ``report.sections`` list hands us the author's exact order — either way we keep
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
    """Return one sentence describing the results collected on a family page."""
    titles = [section.title for section in group.sections]
    noun = "result" if len(titles) == 1 else "results"
    return f"This page brings together the {noun} for {_join_titles(titles)}."


def default_template(ctx: ReportBuildContext) -> ReportDocument:
    """The default report layout: family chapters in dependency order, plus the
    controlled-seed annex when the report carries its analysis."""
    meta = ctx.meta
    sections = ctx.sections
    groups = _group_by_family(sections)
    for group in groups:
        group.summary = _summarize_family(group)
    n = len(sections)
    section_noun = "analysis section" if n == 1 else "analysis sections"
    family_noun = "family" if len(groups) == 1 else "families"
    intro = (
        f"Run **{meta['run_name']}** (seed {meta['seed']}) produced {n} "
        f"{section_noun} across {len(groups)} {family_noun} from "
        f"`{meta['config_path']}`; `civ-bench` regenerated every result from "
        f"pipeline artifacts."
    )
    section_by_id = {section.id: section for section in sections}
    overview_sections = [
        section_by_id[section_id]
        for section_id in meta.get("overview_section_ids", [])
        if section_id in section_by_id
    ]
    # The controlled-seed annex rides along when its analysis is resolved. The
    # heatmap link is added after the annex snapshot is taken, so the annex's
    # own "Source tables" list never links back to itself, and only when the
    # run renders html (the pages are html-only; a markdown-only report would
    # otherwise carry a dead link).
    controlled = controlled_seed_document(ctx)
    if controlled is not None and "html" in (meta.get("formats") or []):
        ctx.section(controlled.section_id).downloads.append(
            Download(
                label="Controlled-seed heatmap pages (HTML)",
                rel_path=CONTROLLED_SEED_OVERVIEW,
            )
        )
    return ReportDocument(
        title=meta["title"],
        run_name=meta["run_name"],
        seed=meta["seed"],
        config_path=meta["config_path"],
        output_root=meta["output_root"],
        description=meta.get("description", "") or "",
        groups=groups,
        overview_sections=overview_sections,
        intro=intro,
        controlled_seed=controlled,
    )
