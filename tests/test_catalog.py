"""Orthodox player_type composition + legacy fallback (benchmark.md §3.3)."""

from __future__ import annotations

import pytest

from bench.catalog import Catalog


@pytest.fixture
def catalog(configs_dir) -> Catalog:
    return Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")


@pytest.mark.parametrize(
    "model,strategist,condition,slot,expected",
    [
        ("claude-sonnet-4-5", "simple-strategist-briefed", "2026-staff-standard", 3, "Sonnet-4.5-Briefed"),
        ("claude-sonnet-4-5", "simple-strategist", "2026-staff-standard", 2, "Sonnet-4.5-Simple"),
        ("openai-compatible/glm-4.7", "simple-strategist-briefed", None, None, "GLM-4.7-Briefed"),
        ("VPAI", "simple-strategist", "2026-staff-standard", 0, "Vanilla"),
        ("VPAI", "null-strategist", "2026-null-ai-standard", 4, "Null"),
    ],
)
def test_compose_orthodox(catalog, model, strategist, condition, slot, expected):
    assert catalog.compose_player_type(model, strategist, condition, slot) == expected


def test_canonicalize_aliases(catalog):
    assert catalog.canonicalize_model_name("openai-compatible/gpt-oss-120b") == "GPT-OSS-120B"


def test_label_suffix_and_override():
    cat = Catalog(
        {
            "player_type_template": "{model}-{variant}{suffix}",
            "vanilla_model_aliases": ["VPAI"],
            "vanilla_label": "Vanilla",
            "null_label": "Null",
            "strategist_variant_map": {"simple-strategist": "Simple", "simple-strategist-briefed": "Briefed"},
            "strategist_models": [{"id": "Sonnet-4.5", "aliases": ["claude-sonnet-4-5"], "color": "#000"}],
            "strategist_variants": {"Simple": {"suffix": "-Simple"}, "Briefed": {"suffix": "-Briefed"}},
        },
        {
            "player_type_labels": {
                "cond_suffix": {"3": "-A", "_default": "-X"},
                "cond_override": "Custom",
            }
        },
    )
    # per-slot suffix beats the condition default
    assert cat.compose_player_type("claude-sonnet-4-5", "simple-strategist-briefed", "cond_suffix", 3) == "Sonnet-4.5-Briefed-A"
    assert cat.compose_player_type("claude-sonnet-4-5", "simple-strategist", "cond_suffix", 2) == "Sonnet-4.5-Simple-X"
    # full override replaces the composed type
    assert cat.compose_player_type("claude-sonnet-4-5", "simple-strategist", "cond_override", 0) == "Custom"
    # baselines skip suffixes so they pool across conditions
    assert cat.compose_player_type("VPAI", "simple-strategist", "cond_suffix", 0) == "Vanilla"


def test_fallback_seat_map(catalog):
    # player_id 3 in 2026-staff-standard is Sonnet-4.5-Briefed (legacy seat map)
    assert catalog.fallback_player_type("2026-staff-standard", 3) == "Sonnet-4.5-Briefed"
    # unknown condition/slot → Player {id}
    assert catalog.fallback_player_type("no-such-condition", 7) == "Player 7"
