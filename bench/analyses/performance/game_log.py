"""``performance.game_log``: one deterministic record for every finished game.

The report consumes these two tables to render its replay-oriented game log.  The
analysis keeps the canonical experiment id only for provenance and locating a
save.  Every reader-facing identity is derived from ``player_type``.
"""

from __future__ import annotations

import pandas as pd

from bench.analyses.base import Analysis, AnalysisContext, AnalysisResult
from bench.analyses.errors import AnalysisError
from bench.plotting.pairing import condition_display, order_conditions, order_strategists


GAMES_COLUMNS = [
    "game_id", "timestamp", "date_utc", "experiment", "label", "seed",
    "seating_rotation", "controlled", "turns", "map_type", "map_size",
    "game_speed", "difficulty", "n_players", "victory_type",
    "winner_player_id", "winner_player_type", "winner_civilization",
    "winner_is_vanilla",
]
GAME_PLAYERS_COLUMNS = [
    "game_id", "player_id", "player_type", "strategist", "condition",
    "is_vanilla", "model", "civilization", "config_slot", "score",
    "score_rank", "survival_turn", "is_winner",
]
_GAME_COLUMNS = ["game_id", "timestamp", "experiment", "seed", "seating_rotation"]
_PANEL_COLUMNS = [
    "game_id", "player_id", "player_type", "model", "civilization",
    "config_slot", "score", "score_rank", "survival_turn", "is_winner",
    "turn", "map_type", "map_size", "game_speed", "difficulty",
    "victory_type", "victory_player_id",
]


def _require_columns(df: pd.DataFrame, columns: list[str], source: str, stage: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise AnalysisError(
            f"performance.game_log '{stage}': {source} is missing required "
            f"column(s) {missing}. Run extract to rebuild the canonical tables."
        )


def _first_non_null(values: pd.Series):
    values = values.dropna()
    return values.iloc[0] if not values.empty else None


def _as_int(values: pd.Series, default: int = -1) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(default).astype(int)


class PerformanceGameLog(Analysis):
    module = "performance.game_log"
    friendly_name = "Game Log"
    description = "Lists finished games and their player-level outcomes for replay links."
    report_defaults = {"tables": [], "figures": []}

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        from bench.data.loading import drop_problem_games

        stage = self.stage_id
        games = drop_problem_games(ctx.load_table("games"), ctx.decision_failure_ids())
        panel = drop_problem_games(ctx.load_table("panel"), ctx.decision_failure_ids())
        _require_columns(games, _GAME_COLUMNS, "games", stage)
        _require_columns(panel, _PANEL_COLUMNS, "panel", stage)

        games = games[_GAME_COLUMNS].copy()
        panel = panel[_PANEL_COLUMNS].copy()
        games["game_id"] = games["game_id"].astype(str)
        panel["game_id"] = panel["game_id"].astype(str)
        panel["player_id"] = _as_int(panel["player_id"])
        if games["game_id"].duplicated().any():
            raise AnalysisError(
                f"performance.game_log '{stage}': games has duplicate game_id rows."
            )
        if panel.duplicated(["game_id", "player_id"]).any():
            raise AnalysisError(
                f"performance.game_log '{stage}': panel has duplicate records per "
                "(game_id, player_id)."
            )

        game_ids = set(games["game_id"])
        panel = panel[panel["game_id"].isin(game_ids)].copy()
        vanilla_label = str(ctx.catalog.vanilla_label)
        panel["player_type"] = panel["player_type"].fillna("").astype(str)
        panel["is_vanilla"] = panel["player_type"].eq(vanilla_label)
        panel["strategist"], panel["condition"] = self._identities(ctx, panel)
        players = self._players_table(panel)
        game_table = self._games_table(games, panel)

        non_vanilla = players.loc[~players["is_vanilla"], "strategist"].astype(str)
        strategist_order = order_strategists(ctx.catalog, non_vanilla.tolist())
        pairing = ctx.condition_pairing()
        if pairing is None:
            condition_order: list[str] = []
        else:
            keys = set()
            for value in panel["player_type"].astype(str):
                _, suffix = ctx.catalog.split_condition_suffix(value, list(pairing.suffixes))
                keys.add("base" if not suffix else suffix)
            condition_order = order_conditions(pairing, keys, vanilla_label)

        latest = game_table.iloc[0]["game_id"] if not game_table.empty else None
        date_values = game_table["date_utc"].dropna().astype(str)
        latest_date = date_values.iloc[0] if not date_values.empty else None
        experiments = sorted(game_table["experiment"].dropna().astype(str).unique())
        summary = (
            f"**{len(game_table)}** games across **{len(experiments)}** experiments"
            + (f"; the latest finished on **{latest_date}**." if latest_date else ".")
        )
        return AnalysisResult(
            tables={"games": game_table, "game_players": players},
            summary=summary,
            metadata={
                "latest_game_id": str(latest) if latest is not None else None,
                "n_games": int(len(game_table)),
                "n_controlled": int(game_table["controlled"].sum()),
                "experiments": experiments,
                "strategist_order": strategist_order,
                "condition_order": condition_order,
                "vanilla_label": vanilla_label,
                "base_label": pairing.base_label if pairing is not None else "",
            },
        )

    def _identities(self, ctx: AnalysisContext, panel: pd.DataFrame) -> tuple[list[str], list[str]]:
        spec = ctx.condition_pairing()
        if spec is None:
            values = panel["player_type"].astype(str).tolist()
            return values, [""] * len(values)
        strategists: list[str] = []
        conditions: list[str] = []
        for player_type in panel["player_type"].astype(str):
            strategist, suffix = ctx.catalog.split_condition_suffix(
                player_type, list(spec.suffixes)
            )
            strategists.append(str(strategist or ""))
            conditions.append(
                condition_display(spec, "base" if not suffix else suffix, ctx.catalog.vanilla_label)
            )
        return strategists, conditions

    def _players_table(self, panel: pd.DataFrame) -> pd.DataFrame:
        players = panel[GAME_PLAYERS_COLUMNS].copy()
        for column in ("config_slot", "score", "score_rank", "survival_turn", "is_winner"):
            players[column] = pd.to_numeric(players[column], errors="coerce")
        players["is_winner"] = players["is_winner"].fillna(0).astype(int).astype(bool)
        players = players.sort_values(["game_id", "player_id"], kind="mergesort")
        return players.reset_index(drop=True)

    def _games_table(self, games: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        work = games.copy()
        work["seed"] = _as_int(work["seed"])
        work["seating_rotation"] = _as_int(work["seating_rotation"])
        work["timestamp"] = pd.to_numeric(work["timestamp"], errors="coerce")
        work["controlled"] = work["seed"].ne(-1)
        dates = pd.to_datetime(work["timestamp"], unit="ms", utc=True, errors="coerce")
        work["date_utc"] = dates.dt.strftime("%Y-%m-%d")

        records: list[dict] = []
        for game in work.itertuples(index=False):
            row = game._asdict()
            seats = panel[panel["game_id"] == row["game_id"]].sort_values(
                "player_id", kind="mergesort"
            )
            winners = seats[seats["is_winner"].fillna(0).astype(float).eq(1)]
            winner = winners.iloc[0] if not winners.empty else None
            winner_id = int(winner["player_id"]) if winner is not None else None
            if winner is None:
                recorded_winner_ids = pd.to_numeric(
                    seats["victory_player_id"], errors="coerce"
                ).dropna()
                if not recorded_winner_ids.empty:
                    winner_id = int(recorded_winner_ids.iloc[0])
                    matching_seats = seats[seats["player_id"] == winner_id]
                    winner = matching_seats.iloc[0] if not matching_seats.empty else None
            labels = []
            for identity in seats.loc[~seats["is_vanilla"], ["strategist", "condition"]].itertuples(index=False):
                label = str(identity.strategist)
                if identity.condition:
                    label = f"{label} | {identity.condition}"
                if label not in labels:
                    labels.append(label)
            turns = pd.to_numeric(seats["turn"], errors="coerce").max() if not seats.empty else None
            records.append({
                "game_id": row["game_id"],
                "timestamp": row["timestamp"],
                "date_utc": row["date_utc"],
                "experiment": row["experiment"],
                "label": " · ".join(labels) if labels else "VPAI",
                "seed": row["seed"],
                "seating_rotation": row["seating_rotation"],
                "controlled": bool(row["controlled"]),
                "turns": int(turns) if pd.notna(turns) else None,
                "map_type": _first_non_null(seats["map_type"]),
                "map_size": _first_non_null(seats["map_size"]),
                "game_speed": _first_non_null(seats["game_speed"]),
                "difficulty": _first_non_null(seats["difficulty"]),
                "n_players": int(len(seats)),
                "victory_type": _first_non_null(seats["victory_type"]),
                "winner_player_id": winner_id,
                "winner_player_type": winner["player_type"] if winner is not None else None,
                "winner_civilization": winner["civilization"] if winner is not None else None,
                "winner_is_vanilla": bool(winner["is_vanilla"]) if winner is not None else False,
            })
        result = pd.DataFrame(records, columns=GAMES_COLUMNS)
        return result.sort_values(
            ["timestamp", "game_id"], ascending=[False, True], kind="mergesort", na_position="last"
        ).reset_index(drop=True)
