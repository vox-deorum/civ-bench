"""Synthetic coverage for the report-facing ``performance.game_log`` tables."""

from __future__ import annotations

import pandas as pd
import pytest

from bench.analyses import run_analysis
from bench.analyses.errors import AnalysisError
from bench.catalog import Catalog
from bench.config import load_config


def _games() -> list[dict]:
    return [
        {"game_id": "old", "timestamp": 1_700_000_000_000, "experiment": "old-exp", "seed": -1, "seating_rotation": -1},
        {"game_id": "latest", "timestamp": 1_700_000_200_000, "experiment": "new-exp", "seed": 2, "seating_rotation": 1},
        {"game_id": "failed", "timestamp": 1_700_000_100_000, "experiment": "bad-exp", "seed": 3, "seating_rotation": 0},
    ]


def _panel() -> list[dict]:
    rows: list[dict] = []
    players = {
        "old": [(0, "Vanilla", "Rome", 0), (1, "GPT-OSS-120B-Simple", "Greece", 0)],
        "latest": [(0, "Vanilla", "Rome", 0), (1, "GPT-OSS-120B-Simple-Per-5", "Greece", 1)],
        "failed": [(0, "Vanilla", "Rome", 1), (1, "GPT-OSS-120B-Simple", "Greece", 0)],
    }
    for game_id, seats in players.items():
        for player_id, player_type, civilization, winner in seats:
            rows.append({
                "game_id": game_id, "player_id": player_id, "player_type": player_type,
                "model": "vpai" if player_type == "Vanilla" else "gpt-oss-120b",
                "civilization": civilization, "config_slot": player_id,
                "score": 120 + player_id, "score_rank": player_id + 1,
                "survival_turn": 240, "is_winner": winner, "turn": 240,
                "map_type": "Continents", "map_size": "Standard", "game_speed": "Standard",
                "difficulty": "Emperor", "victory_type": "Science",
                "victory_player_id": 1,
            })
    return rows


def _run(tmp_path, dev_spec, write_spec, *, pairing: bool = True):
    games_path = tmp_path / "games.csv"
    panel_path = tmp_path / "panel.csv"
    tokens_path = tmp_path / "tokens.csv"
    pd.DataFrame(_games()).to_csv(games_path, index=False)
    pd.DataFrame(_panel()).to_csv(panel_path, index=False)
    pd.DataFrame([
        {"experiment": "bad-exp", "game_id": "failed", "player_id": 1,
         "valid_turn_count": 10, "failed_turn_count": 3},
    ]).to_csv(tokens_path, index=False)
    dev_spec["output"] = {"root": str(tmp_path / "out"), "suffix": ""}
    dev_spec["data"]["extract"]["enabled"] = False
    dev_spec["data"]["tables"] = {
        "games": str(games_path), "panel": str(panel_path), "tokens": str(tokens_path),
        "turns": str(tokens_path),
    }
    dev_spec["data"]["filter"] = None
    dev_spec["presentation"] = {
        "condition_pairing": {"enabled": pairing, "suffixes": ["-Per-5"], "base_label": "Every-turn"},
    }
    stage = {
        "id": "game_log", "module": "performance.game_log", "enabled": True,
        "uses": {}, "params": {},
    }
    dev_spec["analyses"] = [stage]
    cfg = load_config(write_spec(dev_spec))
    result = run_analysis(cfg, stage, catalog=Catalog.from_run_config(cfg))
    return result, {name: pd.read_csv(path) for name, path in result.table_paths.items()}


def test_game_log_builds_sorted_report_tables_and_excludes_problem_games(tmp_path, dev_spec, write_spec):
    result, tables = _run(tmp_path, dev_spec, write_spec)

    assert list(tables) == ["games", "game_players"]
    games = tables["games"]
    assert games["game_id"].tolist() == ["latest", "old"]
    assert games["label"].tolist() == ["GPT-OSS-120B-Simple | Per-5", "GPT-OSS-120B-Simple | Every-turn"]
    assert games.loc[0, "date_utc"] == "2023-11-14"
    assert bool(games.loc[0, "controlled"])
    assert games.loc[1, "seed"] == -1
    assert games.loc[0, "winner_player_id"] == 1
    assert games.loc[0, "winner_civilization"] == "Greece"
    old = games.loc[games["game_id"] == "old"].iloc[0]
    assert old["winner_player_id"] == 1
    assert old["winner_player_type"] == "GPT-OSS-120B-Simple"
    assert result.metadata["latest_game_id"] == "latest"
    assert result.metadata["n_games"] == 2
    assert result.metadata["n_controlled"] == 1
    assert result.metadata["experiments"] == ["new-exp", "old-exp"]

    players = tables["game_players"]
    vanilla = players[(players["game_id"] == "latest") & (players["player_id"] == 0)].iloc[0]
    assert bool(vanilla["is_vanilla"])
    assert vanilla["condition"] == "Every-turn"
    assert "failed" not in set(players["game_id"])


def test_game_log_pairing_can_be_disabled(tmp_path, dev_spec, write_spec):
    result, tables = _run(tmp_path, dev_spec, write_spec, pairing=False)

    latest = tables["game_players"].query("game_id == 'latest' and player_id == 1").iloc[0]
    assert latest["strategist"] == "GPT-OSS-120B-Simple-Per-5"
    assert pd.isna(latest["condition"])
    assert tables["games"].loc[0, "label"] == "GPT-OSS-120B-Simple-Per-5"
    assert result.metadata["condition_order"] == []


def test_game_log_rejects_missing_canonical_columns(tmp_path, dev_spec, write_spec):
    games = pd.DataFrame(_games()).drop(columns="timestamp")
    games_path = tmp_path / "games.csv"
    panel_path = tmp_path / "panel.csv"
    tokens_path = tmp_path / "tokens.csv"
    games.to_csv(games_path, index=False)
    pd.DataFrame(_panel()).to_csv(panel_path, index=False)
    pd.DataFrame().to_csv(tokens_path, index=False)
    dev_spec["data"]["extract"]["enabled"] = False
    dev_spec["data"]["tables"] = {
        "games": str(games_path), "panel": str(panel_path), "tokens": str(tokens_path),
        "turns": str(tokens_path),
    }
    stage = {"id": "game_log", "module": "performance.game_log", "enabled": True, "params": {}}
    dev_spec["analyses"] = [stage]
    cfg = load_config(write_spec(dev_spec))
    with pytest.raises(AnalysisError, match="games is missing required column"):
        run_analysis(cfg, stage, catalog=Catalog.from_run_config(cfg))
