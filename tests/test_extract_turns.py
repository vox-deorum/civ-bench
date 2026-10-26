"""Turn extraction: city yields use one row per city and turn.

Each fixture is a tiny synthetic game DB. ``FlavorChanges`` and
``StrategyChanges`` are intentionally absent, exercising the extractor's
missing-schema fallback.
"""

from __future__ import annotations

import sqlite3

from bench.extract.extract_turns import extract_game_turn_data


SUMMARY_COLUMNS = (
    "ID", "Key", "Turn", "Score", "Cities", "Population", "Territory",
    "Technologies", "MilitaryStrength", "MilitaryUnits", "MilitarySupply",
    "Gold", "GoldPerTurn", "HappinessPercentage", "CulturePerTurn",
    "SciencePerTurn", "TourismPerTurn", "FaithPerTurn", "PolicyBranches",
    "Votes", "FoundedReligion", "Relationships", "MajorAlly",
)


def _summary(id_, key, turn, score, **values):
    row = dict.fromkeys(SUMMARY_COLUMNS)
    row.update({"ID": id_, "Key": key, "Turn": turn, "Score": score})
    row.update(values)
    return row


def _make_db(path, *, players, cities, summaries):
    """Write a synthetic game DB with no FlavorChanges or StrategyChanges."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE GameMetadata (Key TEXT, Value TEXT)")
    cur.execute("INSERT INTO GameMetadata VALUES ('gameId', 'game-a')")

    cur.execute(
        "CREATE TABLE PlayerInformations "
        "(Key INTEGER, Civilization TEXT, IsMajor INTEGER)"
    )
    cur.executemany("INSERT INTO PlayerInformations VALUES (?, ?, ?)", players)

    cur.execute(f"CREATE TABLE PlayerSummaries ({', '.join(SUMMARY_COLUMNS)})")
    placeholders = ", ".join("?" for _ in SUMMARY_COLUMNS)
    cur.executemany(
        f"INSERT INTO PlayerSummaries VALUES ({placeholders})",
        [tuple(row[column] for column in SUMMARY_COLUMNS) for row in summaries],
    )

    cur.execute(
        "CREATE TABLE CityInformations "
        "(ID INTEGER, Turn INTEGER, Key INTEGER, Owner TEXT, "
        "ProductionPerTurn INTEGER, FoodPerTurn INTEGER, MajorityReligion TEXT)"
    )
    cur.executemany(
        "INSERT INTO CityInformations VALUES (?, ?, ?, ?, ?, ?, NULL)",
        cities,
    )
    conn.commit()
    conn.close()
    return str(path)


def _by_player_turn(rows):
    return {(row["player_id"], row["turn"]): row for row in rows}


def test_city_yields_use_latest_version_per_city_and_turn(tmp_path):
    db = _make_db(
        tmp_path / "exp" / "game-a_1.db",
        players=[(0, "Rome", 1), (1, "Greece", 1)],
        cities=[
            # Turn 10: the same city appears twice; only the higher ID counts.
            (1, 10, 100, "Rome", 5, 4),
            (2, 10, 100, "Rome", 8, 6),
            # A second city in the same turn, plus the same city again next turn.
            (3, 10, 101, "Rome", 3, 2),
            (4, 20, 100, "Rome", 10, 9),
            (5, 20, 101, "Greece", 4, 3),
        ],
        summaries=[
            _summary(1, 0, 10, 30),
            _summary(2, 0, 20, 40),
            _summary(3, 1, 10, 20),
            _summary(4, 1, 20, 25),
        ],
    )

    rows = _by_player_turn(extract_game_turn_data(db))

    # Rome: turn 10 = 8 + 3 production and 6 + 2 food (not 5 + 8 + 3).
    assert (rows[(0, 10)]["production_per_turn"],
            rows[(0, 10)]["food_per_turn"]) == (11, 8)
    assert (rows[(0, 20)]["production_per_turn"],
            rows[(0, 20)]["food_per_turn"]) == (10, 9)
    # Greece has no city in turn 10, so its yields are zero.
    assert (rows[(1, 10)]["production_per_turn"],
            rows[(1, 10)]["food_per_turn"]) == (0, 0)
    assert (rows[(1, 20)]["production_per_turn"],
            rows[(1, 20)]["food_per_turn"]) == (4, 3)
    assert len(rows) == 4


def test_latest_city_row_decides_owner_before_major_filter(tmp_path):
    db = _make_db(
        tmp_path / "exp" / "game-a_2.db",
        players=[(0, "Rome", 1), (1, "Greece", 1), (2, "Egypt", 0)],
        cities=[
            # Turn 4: city 100 is Roman. Its ID is higher than the turn 5
            # rows so a global latest-row lookup would misplace it.
            (10, 4, 100, "Rome", 9, 7),
            # Captured by a minor during turn 5: the earlier Roman row is stale.
            (1, 5, 100, "Rome", 12, 10),
            (2, 5, 100, "Egypt", 4, 3),
            # Still Roman.
            (3, 5, 101, "Rome", 2, 1),
            # Captured by Greece during the same turn.
            (4, 5, 102, "Rome", 50, 50),
            (5, 5, 102, "Greece", 6, 5),
        ],
        summaries=[
            _summary(6, 0, 4, 25),
            _summary(7, 1, 4, 15),
            _summary(1, 0, 5, 30),
            _summary(2, 1, 5, 20),
        ],
    )

    rows = _by_player_turn(extract_game_turn_data(db))

    # Turn 4 keeps city 100's Roman yields even though the latest turn 5
    # owner is minor Egypt.
    assert (rows[(0, 4)]["production_per_turn"],
            rows[(0, 4)]["food_per_turn"]) == (9, 7)
    # In turn 5, Rome keeps only city 101: 2 production and 1 food.
    assert (rows[(0, 5)]["production_per_turn"],
            rows[(0, 5)]["food_per_turn"]) == (2, 1)
    assert (rows[(1, 5)]["production_per_turn"],
            rows[(1, 5)]["food_per_turn"]) == (6, 5)
    # The minor player has no turn row at all.
    assert 2 not in {player_id for player_id, _ in rows}


def test_duplicate_player_summaries_use_the_highest_id(tmp_path):
    db = _make_db(
        tmp_path / "exp" / "game-a_3.db",
        players=[(0, "Rome", 1)],
        cities=[(1, 7, 100, "Rome", 6, 5)],
        summaries=[
            _summary(1, 0, 7, 10, Cities=1, Gold=5, GoldPerTurn=2),
            _summary(2, 0, 7, 40, Cities=3, Gold=9, GoldPerTurn=4),
        ],
    )

    rows = extract_game_turn_data(db)
    assert len(rows) == 1
    row = rows[0]

    assert row["score"] == 40
    assert row["cities"] == 3
    assert row["gold"] == 9
    assert row["gold_per_turn"] == 4
    assert (row["production_per_turn"], row["food_per_turn"]) == (6, 5)
