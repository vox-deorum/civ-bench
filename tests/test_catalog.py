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


def test_canonicalize_longer_alias_not_shadowed(catalog):
    # "minimax" is an alias of Minimax-M2.5 and is a substring of "minimax-m2.7";
    # the longer/exact alias must win so M2.7 is not misattributed to M2.5.
    assert catalog.canonicalize_model_name("openai-compatible/Minimax-M2.7") == "MiniMax-M2.7"
    assert catalog.canonicalize_model_name("minimax-m2.7") == "MiniMax-M2.7"
    assert catalog.canonicalize_model_name("minimax-m2.5") == "MiniMax-M2.5"
    # A bare "minimax" still resolves to its owning model (substring fallback).
    assert catalog.canonicalize_model_name("minimax") == "MiniMax-M2.5"


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


def test_label_wildcard_condition_key():
    cat = Catalog(
        {
            "player_type_template": "{model}-{variant}{suffix}",
            "vanilla_model_aliases": ["VPAI"],
            "vanilla_label": "Vanilla",
            "null_label": "Null",
            "strategist_variant_map": {"simple-strategist": "Simple", "simple-strategist-briefed": "Briefed"},
            "strategist_models": [{"id": "GPT-OSS-120B", "aliases": ["gpt-oss-120b"], "color": "#000"}],
            "strategist_variants": {"Simple": {"suffix": "-Simple"}, "Briefed": {"suffix": "-Briefed"}},
        },
        {
            "player_type_labels": {
                "*-per-5": "-Per-5",                 # broad glob
                "oss-*-per-5": "-OssPer5",           # narrower glob (more specific)
                "oss-120b-standard-fixed-per-5": "Exact",  # exact key beats every glob
            }
        },
    )
    # exact condition key wins over any pattern (here a full override)
    assert cat.compose_player_type("gpt-oss-120b", "simple-strategist", "oss-120b-standard-fixed-per-5", 2) == "Exact"
    # most-specific pattern wins among matches (oss-*-per-5 over *-per-5)
    assert cat.compose_player_type("gpt-oss-120b", "simple-strategist", "oss-200b-standard-fixed-per-5", 2) == "GPT-OSS-120B-Simple-OssPer5"
    # only the broad glob matches → "-Per-5" suffix on both variants
    assert cat.compose_player_type("gpt-oss-120b", "simple-strategist", "glm-standard-fixed-per-5", 2) == "GPT-OSS-120B-Simple-Per-5"
    assert cat.compose_player_type("gpt-oss-120b", "simple-strategist-briefed", "glm-standard-fixed-per-5", 3) == "GPT-OSS-120B-Briefed-Per-5"
    # no pattern matches → plain composed type
    assert cat.compose_player_type("gpt-oss-120b", "simple-strategist", "2026-staff-standard", 2) == "GPT-OSS-120B-Simple"
    # baselines skip the wildcard suffix so they pool across conditions
    assert cat.compose_player_type("VPAI", "simple-strategist", "glm-standard-fixed-per-5", 0) == "Vanilla"


def test_fallback_seat_map(catalog):
    # player_id 3 in 2026-staff-standard is Sonnet-4.5-Briefed (legacy seat map)
    assert catalog.fallback_player_type("2026-staff-standard", 3) == "Sonnet-4.5-Briefed"
    # unknown condition/slot → Player {id}
    assert catalog.fallback_player_type("no-such-condition", 7) == "Player 7"


def test_player_color_resolution(catalog):
    from bench.plotting.styles import get_player_color

    oss = catalog.strategist_model_colors()["GPT-OSS-120B"]
    # plain variant resolves to the base model color
    assert get_player_color(catalog, "GPT-OSS-120B-Simple") == oss
    # a label suffix (e.g. -Per-5) still resolves to the same base model color
    assert get_player_color(catalog, "GPT-OSS-120B-Simple-Per-5") == oss
    assert get_player_color(catalog, "GPT-OSS-120B-Briefed-Per-5") == oss
    # newly added model gets its own color, suffix-stable
    nemo = catalog.strategist_model_colors()["Nemotron-3-Super"]
    assert nemo == "#76B900"
    assert get_player_color(catalog, "Nemotron-3-Super-Simple") == nemo
    assert get_player_color(catalog, "Nemotron-3-Super-Briefed-Per-5") == nemo
    # an unrecognized full-override label falls through to black
    assert get_player_color(catalog, "Custom-Thing") == "#000000"
