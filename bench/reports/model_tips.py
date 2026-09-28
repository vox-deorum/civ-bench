"""Per-model tooltips for the "Strategist | Condition" labels (``report.model_tips``).

The run spec lists ordered ``{models, tip}`` rules. :func:`resolve_model_tips`
maps each catalog strategist id to the tip of its first matching rule, and
:func:`model_tip` finds the tip for a rendered label such as
``GPT-6-Luna-Simple | Every-turn``.
"""

from __future__ import annotations

from fnmatch import fnmatchcase


def resolve_model_tips(rules: list[dict] | None, catalog) -> tuple[dict[str, str], list[str]]:
    """``{model id: tip}`` for every LLM strategist, plus warnings for dead patterns.

    A rule without ``models`` covers every model no earlier rule matched.
    Vanilla and Null are baselines, not models, and never get a tip.
    """
    if not rules:
        return {}, []
    baselines = {catalog.vanilla_label, catalog.null_label}
    ids = [m["id"] for m in catalog.strategist_models() if m["id"] not in baselines]
    tips: dict[str, str] = {}
    warnings = []
    for i, rule in enumerate(rules):
        patterns = rule.get("models")
        if patterns is None:
            tips.update({model: rule["tip"] for model in ids if model not in tips})
            continue
        for pattern in patterns:
            matched = [model for model in ids if fnmatchcase(model, pattern)]
            if not matched:
                warnings.append(
                    f"report.model_tips[{i}].models: '{pattern}' matches no strategist model."
                )
            for model in matched:
                tips.setdefault(model, rule["tip"])
    return tips, warnings


def model_tip(label: str, tips: dict[str, str] | None) -> str:
    """The tip of the longest model id that starts ``label`` at a word boundary."""
    if not tips:
        return ""
    for model in sorted(tips, key=len, reverse=True):
        if label == model or (label.startswith(model) and label[len(model)] in "- |"):
            return tips[model]
    return ""
