"""Import-light analysis metadata discovery tests."""

from bench.config.analysis_metadata import analysis_report_defaults


def test_report_defaults_are_discovered_without_importing_registry():
    assert analysis_report_defaults("ratings.bradley_terry") == {
        "tables": [],
        "figures": ["ratings"],
    }
    assert analysis_report_defaults("ratings.matchups") == {
        "tables": [],
        "figures": ["matchup", "strength_mean", "strength_winrate"],
    }
    assert analysis_report_defaults("exploratory.model_token_costs") == {
        "tables": [],
        "figures": ["token_costs"],
    }


def test_report_defaults_fall_back_for_unknown_or_legacy_modules():
    assert analysis_report_defaults(None) is None
    assert analysis_report_defaults("custom.module") is None


def test_report_defaults_are_literal_and_include_empty_curations():
    assert analysis_report_defaults("calibration.cell_baseline") == {
        "tables": [],
        "figures": [],
    }
