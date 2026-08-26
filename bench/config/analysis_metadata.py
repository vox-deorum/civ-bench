"""Import-light discovery of analysis class metadata.

Analysis behavior flags live on the individual ``Analysis`` subclasses. The
config layer still needs a tiny slice of that metadata while building the DAG,
but importing the analysis registry would pull in plotting/stats/R dependencies.
Read the source with ``ast`` instead so dry-runs stay cheap.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path


def analysis_defaults_to_all_estimators(module: str | None) -> bool:
    """Whether an analysis module defaults omitted uses.estimators to all enabled."""
    if module is None:
        return False
    return module in _default_all_estimator_modules()


def analysis_report_defaults(module: str | None) -> dict | None:
    """Return the report-owned inline artifact defaults for an analysis.

    The lookup intentionally reads source with :mod:`ast`. Importing the analysis
    registry would make config loading depend on plotting and statistical
    dependencies. Unknown modules, including custom or older modules without the
    class attribute, return ``None`` so callers can retain their existing behavior.
    """
    if module is None:
        return None
    defaults = _report_defaults()
    value = defaults.get(module)
    return dict(value) if value is not None else None


@lru_cache
def _default_all_estimator_modules() -> frozenset[str]:
    root = Path(__file__).resolve().parents[1] / "analyses"
    modules: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name.startswith("__") or path.name in {"base.py", "registry.py"}:
            continue
        modules.update(_defaults_in_file(path))
    return frozenset(modules)


def _defaults_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        module = None
        default_all = False
        for stmt in node.body:
            name, value = _class_assignment(stmt)
            if name == "module" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                module = value.value
            elif name == "default_all_estimators" and _literal_true(value):
                default_all = True
        if module is not None and default_all:
            modules.add(module)
    return modules


@lru_cache
def _report_defaults() -> dict[str, dict]:
    root = Path(__file__).resolve().parents[1] / "analyses"
    defaults: dict[str, dict] = {}
    for path in root.rglob("*.py"):
        if path.name.startswith("__") or path.name in {"base.py", "registry.py"}:
            continue
        defaults.update(_report_defaults_in_file(path))
    return defaults


def _report_defaults_in_file(path: Path) -> dict[str, dict]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    defaults: dict[str, dict] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        module = None
        report_defaults = None
        for stmt in node.body:
            name, value = _class_assignment(stmt)
            if name == "module" and isinstance(value, ast.Constant) and isinstance(value.value, str):
                module = value.value
            elif name == "report_defaults" and value is not None:
                try:
                    parsed = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError):
                    parsed = None
                if isinstance(parsed, dict):
                    report_defaults = parsed
        if module is not None and report_defaults is not None:
            defaults[module] = report_defaults
    return defaults


def _class_assignment(stmt: ast.stmt) -> tuple[str | None, ast.expr | None]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id, stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    return None, None


def _literal_true(value: ast.expr | None) -> bool:
    return isinstance(value, ast.Constant) and value.value is True
