"""Import-light analysis metadata discovery tests."""

from bench.config.analysis_metadata import analysis_report_defaults


def test_report_defaults_are_discovered_without_importing_registry():
    assert analysis_report_defaults("ratings.bradley_terry") == {
        "tables": ["strategy_ratings"],
        "figures": ["ratings", "ratings_*"],
    }
    assert analysis_report_defaults("ratings.matchups") == {
        "tables": ["matchup_heatmap", "strength_winrate_heatmap", "strength_mean_heatmap"],
        "figures": ["matchup", "strength_mean", "strength_winrate"],
    }
    assert analysis_report_defaults("performance.usage_efficiency") == {
        "tables": [],
        "figures": ["usage_vs_rating"],
    }


def test_report_defaults_fall_back_for_unknown_or_legacy_modules():
    assert analysis_report_defaults(None) is None
    assert analysis_report_defaults("custom.module") is None


def test_report_defaults_are_literal_and_include_empty_curations():
    assert analysis_report_defaults("calibration.cell_baseline") == {
        "tables": [],
        "figures": [],
    }


def test_report_defaults_match_every_registered_class():
    from bench.analyses.registry import ANALYSIS_REGISTRY

    for module, cls in ANALYSIS_REGISTRY.items():
        expected = getattr(cls, "report_defaults", None)
        if expected is None:
            continue
        assert analysis_report_defaults(module) == expected, module


def test_report_defaults_reject_non_literals(tmp_path):
    import pytest

    from bench.config.analysis_metadata import _report_defaults_in_file

    path = tmp_path / "broken.py"
    path.write_text(
        'SLUG = "x"\n'
        "class Broken:\n"
        '    module = "custom.broken"\n'
        '    report_defaults = {"tables": [f"{SLUG}_relative"], "figures": []}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Broken.report_defaults"):
        _report_defaults_in_file(path)
