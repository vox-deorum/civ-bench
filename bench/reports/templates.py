"""Report templates: assemble produced :class:`Section`s into a document (stage 5).

A *template* is a pure function over the :class:`~bench.reports.context.ReportBuildContext`
(run metadata + resolved sections + the full-artifact loader) returning a document
model. It decides structure only — grouping, ordering, headings, intro prose — never
*which* analyses ran (that is the run-spec's job) and never an analysis's own content
(that is the module's job). The shipped ``default`` template groups sections into the
five analysis families (ratings / prediction / calibration / performance /
exploratory), preserving the dependency order the runner resolved, so no analysis
hardcodes its place in the document (invariant 3).

The ``controlled_seed`` template builds the dedicated
:class:`~bench.reports.model.ControlledSeedDocument` for the controlled-seed report:
it requires exactly one ``performance.controlled_seed_report`` section and loads that
section's three persisted tables through the build context.

Register a new template by adding it to :data:`TEMPLATES`; the run-spec selects it
by ``report.template`` (default ``"default"``). :data:`TEMPLATE_FORMATS` declares the
formats each template renders and the default used when ``report.formats`` is omitted.
"""

from __future__ import annotations

from typing import Callable

from ..config import schema as S
from .context import ReportBuildContext
from .errors import ReportError
from .model import (
    ControlledSeedDocument,
    FamilyGroup,
    ReportDocument,
    Section,
)

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
    """The default report layout: family chapters in dependency order."""
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
    )


CONTROLLED_SEED_MODULE = "performance.controlled_seed_report"
CONTROLLED_SEED_TABLES = (
    "seed_player_summary",
    "seed_player_probability",
    "seed_player_index",
)


def controlled_seed_template(ctx: ReportBuildContext) -> ControlledSeedDocument:
    """Build the controlled-seed document from its single analysis section.

    The template is strict about its input: exactly one section, and it must be
    an enabled, non-empty ``performance.controlled_seed_report`` section. Any
    other section in the resolved report is incompatible with this template.
    """
    sections = ctx.sections
    if len(sections) != 1 or sections[0].module != CONTROLLED_SEED_MODULE:
        raise ReportError(
            "the controlled_seed template requires exactly one report section, the "
            f"'{CONTROLLED_SEED_MODULE}' analysis (resolved: "
            f"{[f'{s.id} ({s.module})' for s in sections]}). Curate report.sections "
            "to contain only the controlled-seed report stage."
        )
    section = sections[0]
    if section.empty:
        raise ReportError(
            f"report section '{section.id}' produced no artifacts; the controlled_seed "
            "template has nothing to render."
        )
    tables = {
        name: ctx.load_table(section.id, name) for name in CONTROLLED_SEED_TABLES
    }
    return ControlledSeedDocument(
        title=ctx.meta["title"],
        run_name=ctx.meta["run_name"],
        seed=ctx.meta["seed"],
        config_path=ctx.meta["config_path"],
        output_root=ctx.meta["output_root"],
        description=ctx.meta.get("description", "") or "",
        section_id=section.id,
        summary=section.summary,
        metadata=dict(section.metadata or {}),
        summary_table=tables["seed_player_summary"],
        probability_table=tables["seed_player_probability"],
        index_table=tables["seed_player_index"],
        downloads=list(section.downloads),
    )


TEMPLATES: dict[str, Callable[[ReportBuildContext], object]] = {
    "default": default_template,
    "controlled_seed": controlled_seed_template,
}

# Formats each template renders, in default order; `report.formats` for a
# template must be a subset (the first entry is the omitted-formats default).
TEMPLATE_FORMATS: dict[str, list[str]] = dict(S.TEMPLATE_FORMATS)


def template_formats(name: str) -> list[str]:
    """The formats a template supports (unknown templates fall back to md+html)."""
    return list(TEMPLATE_FORMATS.get(name, ["md", "html"]))


def get_template(name: str) -> Callable[[ReportBuildContext], object]:
    if name not in TEMPLATES:
        raise ReportError(
            f"unknown report template '{name}'. Available: {sorted(TEMPLATES)}."
        )
    return TEMPLATES[name]
