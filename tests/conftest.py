"""Shared pytest fixtures for the civ-bench test suite.

Tests live at ``civ-bench/tests/`` and run with ``pytest`` from the repo root
(``pip install -e ".[test]"`` first). They exercise the stage-0 scaffold: config
load/validation, the DAG, and the orthodox player_type composition — no stage
execution and no machine-specific data roots.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEV_CONFIG = CONFIGS_DIR / "benchmark.dev.json"
PRETRAINED_TEMPLATE = CONFIGS_DIR / "benchmark.pretrained.template.json"


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS_DIR


@pytest.fixture
def dev_spec() -> dict:
    """The dev run-spec parsed as a plain dict (deep-copied per test)."""
    return copy.deepcopy(json.loads(DEV_CONFIG.read_text(encoding="utf-8")))


@pytest.fixture
def write_spec(tmp_path):
    """Write a spec dict next to the real catalogs so sibling lookup resolves.

    Returns a callable ``write(spec) -> Path``. The file is created inside the
    repo ``configs/`` dir (under a tmp name) so ``models.json`` / ``experiments.json``
    siblings resolve, then removed on teardown.
    """
    created: list[Path] = []

    def _write(spec: dict) -> Path:
        path = CONFIGS_DIR / f"_pytest_{tmp_path.name}.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        created.append(path)
        return path

    yield _write

    for p in created:
        p.unlink(missing_ok=True)
