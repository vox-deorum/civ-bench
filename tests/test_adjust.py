"""Stage 3 — adjust (strength panel) tests.

Exercise the strength derivation on tiny **synthetic** fixtures (no machine data
roots, per AGENTS.md): the legacy parity math (weighted → relative → enforce →
finite logit), the uncontrolled civ-OLS path, the controlled matched start-cell
baseline (implicit + explicit), the always-written audit trails, the coverage
diagnostics (warn vs. throw), and the ``run_adjust`` file-writing integration.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from bench.catalog import Catalog
from bench.config import load_config
from bench.adjust import AdjustError, run_adjust
from bench.adjust.strength import build_strength_panel
from bench.stats.transforms import inv_logit, logit


# ── synthetic data builders ──────────────────────────────────────────────────
# A "seat" is (player_id, player_type, model, civ, p, is_winner). A game expands
# each seat over three late-game turns (turn_progress 0.4/0.6/0.8) with a constant
# predicted_win_probability `p`, so the weighted average is exactly `p`.
LATE_TURNS = [(4, 0.4), (6, 0.6), (8, 0.8)]
MAX_TURN = 10


def _build_frames(games):
    """games: list of dicts {experiment, game_id, seed, seating_rotation, seats:[...]}."""
    pred_rows, panel_rows, game_rows = [], [], []
    for g in games:
        game_rows.append({
            "game_id": g["game_id"],
            "timestamp": g.get("timestamp", "2026-01-01"),
            "experiment": g["experiment"],
            "seed": g["seed"],
            "seating_rotation": g["seating_rotation"],
        })
        for (pid, ptype, model, civ, p, win) in g["seats"]:
            panel_rows.append({
                "experiment": g["experiment"], "game_id": g["game_id"], "player_id": pid,
                "player_type": ptype, "model": model, "strategist": "simple-strategist",
                "config_slot": 0 if ptype == "Vanilla" else 1, "civilization": civ,
            })
            extra = g.get("extra_turns", [])
            for (turn, tp) in LATE_TURNS + extra:
                pred_rows.append({
                    "experiment": g["experiment"], "game_id": g["game_id"], "player_id": pid,
                    "civilization": civ, "turn": turn, "max_turn": MAX_TURN,
                    "is_winner": int(win and turn == LATE_TURNS[-1][0]),
                    "predicted_win_probability": p, "turn_progress": turn / MAX_TURN,
                })
    return pd.DataFrame(pred_rows), pd.DataFrame(panel_rows), pd.DataFrame(game_rows)


def _write(tmp_path, games):
    pred, panel, game = _build_frames(games)
    pp = tmp_path / "predictions.csv"
    pa = tmp_path / "panel.csv"
    gp = tmp_path / "games.csv"
    pred.to_csv(pp, index=False)
    panel.to_csv(pa, index=False)
    game.to_csv(gp, index=False)
    return str(pp), str(pa), str(gp)


@pytest.fixture
def catalog(configs_dir) -> Catalog:
    return Catalog.from_paths(configs_dir / "models.json", configs_dir / "experiments.json")


def _params(**over):
    base = {
        "turn_progress_min": 0.2, "weight": "turn_progress", "relative_to": "game_leader",
        "enforce_winner": True, "civ_adjust": "ols_logit", "block": "auto",
        "baseline_experiment": None, "post_cell_normalize": "none",
    }
    base.update(over)
    return base


# A standard uncontrolled (seed=-1) fixture: one game, vanilla + LLM seats.
def _uncontrolled_games():
    return [{
        "experiment": "exp-llm", "game_id": "g1", "seed": -1, "seating_rotation": -1,
        "seats": [
            (0, "TestLLM-Simple", "TestLLM", "Rome", 0.5, True),
            (1, "Vanilla", "VPAI", "Egypt", 0.25, False),
            (2, "Vanilla", "VPAI", "Greece", 0.40, False),
        ],
    }]


# ── parity math: weighted → relative → enforce → finite logit ────────────────
def test_weighted_relative_logit_exact(tmp_path, catalog):
    """Hand-checkable values pin the 2dp re-round, strict > filter, weighting, eps."""
    games = [{
        "experiment": "exp-llm", "game_id": "g1", "seed": -1, "seating_rotation": -1,
        # an extra turn at turn_progress exactly 0.2 that MUST be excluded (strict >)
        "extra_turns": [(2, 0.2)],
        "seats": [
            (0, "TestLLM-Simple", "TestLLM", "Rome", 0.5, True),
            (1, "Vanilla", "VPAI", "Egypt", 0.25, False),
        ],
    }]
    pred, panel, game = _build_frames(games)
    # junk value at the excluded turn: if the filter were `>=`, A's average would shift
    pred.loc[(pred.turn == 2), "predicted_win_probability"] = 0.99
    pp, pa, gp = tmp_path / "p.csv", tmp_path / "pa.csv", tmp_path / "g.csv"
    pred.to_csv(pp, index=False); panel.to_csv(pa, index=False); game.to_csv(gp, index=False)
    art = build_strength_panel(str(pp), str(pa), str(gp), _params(civ_adjust="none"), catalog)
    p = art.panel.set_index("player_id")

    # weighted_strength = p (constant over the kept late turns; turn_progress 0.2 excluded)
    assert p.loc[0, "weighted_strength"] == pytest.approx(0.5)
    assert p.loc[1, "weighted_strength"] == pytest.approx(0.25)
    # relative to game leader (max 0.5)
    assert p.loc[0, "relative_strength"] == pytest.approx(1.0)
    assert p.loc[1, "relative_strength"] == pytest.approx(0.5)
    # finite logit via eps=1e-5 clip for the rel==1.0 winner
    assert p.loc[0, "logit_strength"] == pytest.approx(math.log((1 - 1e-5) / 1e-5))
    assert p.loc[1, "logit_strength"] == pytest.approx(0.0, abs=1e-9)
    assert np.isfinite(art.panel["logit_strength"]).all()


def test_absolute_strength_no_leader_division(tmp_path, catalog):
    """relative_to="none" ⇒ strength is the raw P(win); not divided by the game leader."""
    games = [{
        "experiment": "exp-llm", "game_id": "g1", "seed": -1, "seating_rotation": -1,
        "seats": [
            (0, "TestLLM-Simple", "TestLLM", "Rome", 0.5, True),
            (1, "Vanilla", "VPAI", "Egypt", 0.25, False),
        ],
    }]
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(pp, pa, gp, _params(relative_to="none", civ_adjust="none"), catalog)
    p = art.panel.set_index("player_id")
    # relative_strength mirrors the raw weighted_strength (NOT 1.0 / 0.5 leader-relative)
    assert p.loc[0, "relative_strength"] == pytest.approx(0.5)
    assert p.loc[1, "relative_strength"] == pytest.approx(0.25)
    # logit_strength is the logit of the raw P(win)
    assert p.loc[0, "logit_strength"] == pytest.approx(logit(np.array([0.5]))[0])
    assert p.loc[1, "logit_strength"] == pytest.approx(logit(np.array([0.25]))[0])
    assert np.isfinite(art.panel["logit_strength"]).all()


def test_absolute_is_default_when_relative_to_unset(tmp_path, catalog):
    """Omitting relative_to falls to the coded default (None) ⇒ absolute strength."""
    pp, pa, gp = _write(tmp_path, _uncontrolled_games())
    params = {  # deliberately no relative_to key
        "turn_progress_min": 0.2, "weight": "turn_progress", "enforce_winner": True,
        "civ_adjust": "none", "block": "none", "baseline_experiment": None,
        "post_cell_normalize": "none",
    }
    art = build_strength_panel(pp, pa, gp, params, catalog)
    # absolute: relative_strength == raw weighted_strength (no leader normalization)
    assert np.allclose(art.panel["relative_strength"], art.panel["weighted_strength"])


def test_absolute_enforce_winner_bumps_to_top_raw(tmp_path, catalog):
    """In absolute mode enforce_winner still guarantees the winner holds the top raw strength."""
    games = [{
        "experiment": "exp-llm", "game_id": "g1", "seed": -1, "seating_rotation": -1,
        "seats": [
            (0, "TestLLM-Simple", "TestLLM", "Rome", 0.30, True),   # winner, but lower raw P(win)
            (1, "Vanilla", "VPAI", "Egypt", 0.60, False),           # higher raw P(win), did not win
        ],
    }]
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(pp, pa, gp, _params(relative_to="none", civ_adjust="none"), catalog)
    p = art.panel.set_index("player_id")
    # winner bumped above the original max (0.60 + 0.001); strictly the top raw strength
    assert p.loc[0, "weighted_strength"] == pytest.approx(0.601)
    assert p.loc[0, "weighted_strength"] > p.loc[1, "weighted_strength"]
    assert p.loc[0, "relative_strength"] == pytest.approx(p.loc[0, "weighted_strength"])


def test_uniform_weight_is_simple_mean(tmp_path, catalog):
    games = [{
        "experiment": "exp-llm", "game_id": "g1", "seed": -1, "seating_rotation": -1,
        "seats": [
            (0, "TestLLM-Simple", "TestLLM", "Rome", 0.5, True),
            (1, "Vanilla", "VPAI", "Egypt", 0.2, False),
        ],
    }]
    # vary p over turns so weighted != uniform unless we use the right scheme
    pred, panel, game = _build_frames(games)
    pred.loc[(pred.player_id == 1), "predicted_win_probability"] = [0.1, 0.2, 0.3]
    pp, pa, gp = tmp_path / "p.csv", tmp_path / "pa.csv", tmp_path / "g.csv"
    pred.to_csv(pp, index=False); panel.to_csv(pa, index=False); game.to_csv(gp, index=False)
    art = build_strength_panel(str(pp), str(pa), str(gp), _params(weight="uniform", civ_adjust="none"), catalog)
    # uniform mean of 0.1,0.2,0.3 = 0.2; leader 0.5 → relative 0.4
    assert art.panel.set_index("player_id").loc[1, "weighted_strength"] == pytest.approx(0.2)


def test_column_contract_and_winner_enforcement(tmp_path, catalog):
    pp, pa, gp = _write(tmp_path, _uncontrolled_games())
    art = build_strength_panel(pp, pa, gp, _params(), catalog)
    for col in ("experiment", "game_id", "player_id", "player_type", "civilization",
                "relative_strength", "logit_strength", "adjusted_strength",
                "controlled", "seed", "seating_rotation", "config_slot", "adjust_method"):
        assert col in art.panel.columns
    winners = art.panel[art.panel["is_winner"] == 1]
    assert (winners["relative_strength"] == 1.0).all()
    assert np.isfinite(art.panel["logit_strength"]).all()
    assert (~art.panel["controlled"]).all()  # seed=-1 ⇒ uncontrolled


# ── uncontrolled civ-OLS path ────────────────────────────────────────────────
def test_civ_adjust_none_equals_relative(tmp_path, catalog):
    pp, pa, gp = _write(tmp_path, _uncontrolled_games())
    art = build_strength_panel(pp, pa, gp, _params(civ_adjust="none", block="none"), catalog)
    assert np.allclose(art.panel["adjusted_strength"], art.panel["relative_strength"])
    assert art.civ_effects.empty
    assert art.cell_baseline.empty and art.cell_coverage.empty


def test_civ_effects_reconcile_and_sum_zero(tmp_path, catalog):
    # two games so the OLS has civ + player_type variation
    games = _uncontrolled_games() + [{
        "experiment": "exp-llm", "game_id": "g2", "seed": -1, "seating_rotation": -1,
        "seats": [
            (0, "TestLLM-Simple", "TestLLM", "Greece", 0.6, True),
            (1, "Vanilla", "VPAI", "Rome", 0.3, False),
            (2, "Vanilla", "VPAI", "Egypt", 0.45, False),
        ],
    }]
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(pp, pa, gp, _params(block="none"), catalog)
    assert not art.civ_effects.empty
    assert list(art.civ_effects.columns) == ["civilization", "civ_effect", "n_rows"]
    # Sum coding ⇒ civ effects sum to ~0
    assert art.civ_effects["civ_effect"].sum() == pytest.approx(0.0, abs=1e-9)
    # reconcile: adjusted == inv_logit(logit - civ_effect)
    ce = dict(zip(art.civ_effects["civilization"], art.civ_effects["civ_effect"]))
    recon = inv_logit(art.panel["logit_strength"].to_numpy()
                      - art.panel["civilization"].map(ce).fillna(0).to_numpy())
    assert np.allclose(recon, art.panel["adjusted_strength"].to_numpy())


# ── controlled matched start-cell baseline ───────────────────────────────────
# Build a seated rotation design with a fixed strong leader seat (4) so the game
# max is constant 0.9, vanilla seats sit at p=0.45 (relative 0.5 ⇒ logit 0), and
# the LLM seat sits at p = 0.9 * inv_logit(delta) ⇒ logit_strength == delta. Each
# (seed, player_id) cell then has vanilla observations (logit 0) plus, in the one
# rotation where the LLM lands there, an LLM observation (logit delta).
DELTA = 1.0
P_VANILLA = 0.45
P_LEADER = 0.9
P_LLM = P_LEADER * inv_logit(DELTA)
CIV_BY_SEAT = {0: "Rome", 1: "Egypt", 2: "Greece", 3: "Japan", 4: "Spain"}


def _seated_games(experiment, seed, rotations, llm_type="TestLLM-Simple", llm_model="TestLLM"):
    games = []
    for r in rotations:
        seats = []
        for pid in range(5):
            if pid == 4:
                seats.append((4, "Vanilla", "VPAI", CIV_BY_SEAT[4], P_LEADER, False))
            elif pid == r:
                seats.append((pid, llm_type, llm_model, CIV_BY_SEAT[pid], P_LLM, False))
            else:
                seats.append((pid, "Vanilla", "VPAI", CIV_BY_SEAT[pid], P_VANILLA, False))
        games.append({
            "experiment": experiment, "game_id": f"{experiment}-s{seed}-r{r}",
            "seed": seed, "seating_rotation": r, "seats": seats,
        })
    return games


def test_implicit_recovers_baseline_and_uplift(tmp_path, catalog):
    # LLM rotates through seats 0..3 across 4 complete rotations ⇒ every cell has
    # both vanilla observations and (for its own rotation) an LLM observation.
    games = _seated_games("ctrl", seed=1, rotations=[0, 1, 2, 3])
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(pp, pa, gp, _params(block="auto"), catalog)
    p = art.panel
    assert p["controlled"].all()

    # Vanilla cell baseline ≈ 0 (logit 0), so vanilla rows adjust to inv_logit(0)=0.5
    vanilla = p[p["player_type"] == "Vanilla"]
    assert np.allclose(vanilla["adjusted_strength"], 0.5, atol=1e-3)
    # LLM rows recover inv_logit(delta)
    llm = p[p["player_type"] != "Vanilla"]
    assert np.allclose(llm["adjusted_strength"], inv_logit(DELTA), atol=1e-3)

    # cell_baseline trail reconciles: adjusted == inv_logit(logit - cell_baseline)
    bmap = {(r.seed, r.player_id): r.cell_baseline
            for r in art.cell_baseline[art.cell_baseline.pathway == "implicit"].itertuples()}
    diff = p["logit_strength"].to_numpy() - np.array(
        [bmap[(s, pid)] for s, pid in zip(p["seed"], p["player_id"])]
    )
    assert np.allclose(inv_logit(diff), p["adjusted_strength"].to_numpy(), atol=1e-9)

    # cell_logit_advantage persists the EXACT pre-normalization delta (logit_strength
    # - cell_baseline) — not a re-logit of the clipped adjusted_strength.
    assert np.allclose(p["cell_logit_advantage"].to_numpy(), diff, atol=1e-12)
    assert np.allclose(llm["cell_logit_advantage"], DELTA, atol=1e-3)        # LLM uplift
    assert np.allclose(vanilla["cell_logit_advantage"], 0.0, atol=1e-3)      # baseline ≈ 0


def test_implicit_partial_coverage_computes_complete_falls_back_rest(tmp_path, catalog):
    # seed 1 has complete rotations (every cell gets a Vanilla observation); seed 2
    # has only rotation 0, so cell (2, 0) — the LLM's own seat — has NO Vanilla
    # baseline. Implicit must NOT abort: it adjusts every complete cell via the
    # start-cell baseline and falls the incomplete cell back to the civ path (WARN).
    games = (
        _seated_games("ctrl", seed=1, rotations=[0, 1, 2, 3])
        + _seated_games("ctrl", seed=2, rotations=[0])
    )
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(pp, pa, gp, _params(block="auto"), catalog)
    p = art.panel

    # complete seed-1 LLM cells recover the uplift via the start-cell baseline
    llm_s1 = p[(p.player_type != "Vanilla") & (p.seed == 1)]
    assert (llm_s1["adjust_method"] == "cell").all()
    assert np.allclose(llm_s1["adjusted_strength"], inv_logit(DELTA), atol=1e-3)
    assert llm_s1["cell_logit_advantage"].notna().all()  # cell rows carry an advantage

    # the incomplete cell (2, 0) falls back (no own Vanilla baseline) — not 'cell'
    fallback = p[(p.player_type != "Vanilla") & (p.seed == 2) & (p.player_id == 0)]
    assert len(fallback) == 1 and (fallback["adjust_method"] == "civ").all()
    assert fallback["cell_logit_advantage"].isna().all()  # non-cell rows have no advantage

    # report-only: a warning names the incomplete self-coverage, never raises
    assert any("self-coverage incomplete" in w for w in art.warnings)
    # coverage marks the cell as having no baseline (but present rows ⇒ not 'missing')
    cov = art.cell_coverage
    cell20 = cov[(cov.seed == 2) & (cov.player_id == 0)]
    assert not cell20["has_baseline"].any()


def test_min_condition_completeness_drops_incomplete_experiments(tmp_path, catalog):
    # "full" occupies every reference rotation (seed 1 × 0..3 ⇒ completeness 1.0);
    # "partial" occupies only rotations 0..1 (completeness 0.5) and is incomplete.
    full = _seated_games("full", seed=1, rotations=[0, 1, 2, 3])
    partial = _seated_games("partial", seed=1, rotations=[0, 1])
    pp, pa, gp = _write(tmp_path, full + partial)

    # default (null) keeps every condition — unchanged behavior
    kept = build_strength_panel(pp, pa, gp, _params(block="auto"), catalog)
    assert set(kept.panel["experiment"]) == {"full", "partial"}

    # a threshold of 1.0 drops every condition missing any reference slot (as a whole)
    pruned = build_strength_panel(
        pp, pa, gp, _params(block="auto", min_condition_completeness=1.0), catalog
    )
    assert set(pruned.panel["experiment"]) == {"full"}
    assert any("dropped 1 experiment(s)" in w for w in pruned.warnings)
    assert any("partial" in w for w in pruned.warnings)

    # a threshold at or below the partial's completeness keeps it
    kept2 = build_strength_panel(
        pp, pa, gp, _params(block="auto", min_condition_completeness=0.5), catalog
    )
    assert set(kept2.panel["experiment"]) == {"full", "partial"}

    # the dropped condition is absent from the audit trails too (consistent panel)
    assert set(pruned.cell_coverage["experiment"]) == {"full"}


def test_cell_logit_advantage_unaffected_by_post_cell_normalize(tmp_path, catalog):
    # The advantage is captured BEFORE post_cell_normalize, so it stays the exact cell
    # delta even when adjusted_strength is rescaled relative to the game leader.
    games = _seated_games("ctrl", seed=1, rotations=[0, 1, 2, 3])
    pp, pa, gp = _write(tmp_path, games)
    plain = build_strength_panel(pp, pa, gp, _params(block="auto"), catalog).panel
    normed = build_strength_panel(
        pp, pa, gp, _params(block="auto", post_cell_normalize="relative_to_leader"), catalog
    ).panel

    assert not np.allclose(plain["adjusted_strength"], normed["adjusted_strength"])
    assert np.allclose(plain["cell_logit_advantage"], normed["cell_logit_advantage"], atol=1e-12)
    llm = normed[normed["player_type"] != "Vanilla"]
    assert np.allclose(llm["cell_logit_advantage"], DELTA, atol=1e-3)


def test_explicit_missing_baseline_still_raises(tmp_path, catalog):
    # Explicit pathway stays FATAL: the designated baseline experiment only covers
    # seed 1, but the controlled experiment also plays seed 2 ⇒ undefined baseline.
    baseline = [{
        "experiment": "vp-self", "game_id": "vp-s1-r0", "seed": 1, "seating_rotation": 0,
        "seats": [(pid, "Vanilla", "VPAI", CIV_BY_SEAT[pid],
                   P_LEADER if pid == 4 else P_VANILLA, False) for pid in range(5)],
    }]
    ctrl = (
        _seated_games("ctrl", seed=1, rotations=[0, 1])
        + _seated_games("ctrl", seed=2, rotations=[0, 1])
    )
    pp, pa, gp = _write(tmp_path, baseline + ctrl)
    with pytest.raises(AdjustError, match=r"explicit baseline"):
        build_strength_panel(pp, pa, gp, _params(block="auto", baseline_experiment="vp-self"), catalog)


def test_explicit_pathway_recovers_uplift_and_reports_both(tmp_path, catalog):
    # Designated baseline experiment: a pure-VP self-play that seats Vanilla in every
    # position (here a single rotation covering all five seats with vanilla).
    baseline = [{
        "experiment": "vp-self", "game_id": "vp-s1-r0", "seed": 1, "seating_rotation": 0,
        "seats": [(pid, "Vanilla", "VPAI", CIV_BY_SEAT[pid],
                   P_LEADER if pid == 4 else P_VANILLA, False) for pid in range(5)],
    }]
    ctrl = _seated_games("ctrl", seed=1, rotations=[0, 1, 2, 3])
    pp, pa, gp = _write(tmp_path, baseline + ctrl)
    art = build_strength_panel(pp, pa, gp, _params(block="auto", baseline_experiment="vp-self"), catalog)
    llm = art.panel[art.panel["player_type"] != "Vanilla"]
    assert np.allclose(llm["adjusted_strength"], inv_logit(DELTA), atol=1e-3)

    pathways = set(art.cell_baseline["pathway"].unique())
    assert "explicit" in pathways and "implicit" in pathways  # both reported
    # explicit rows carry the designated experiment id
    expl = art.cell_baseline[art.cell_baseline.pathway == "explicit"]
    assert (expl["experiment"] == "vp-self").all()


def test_explicit_pathway_warns_on_missing_implicit_comparison_cell(tmp_path, catalog):
    # The selected explicit baseline covers every seat, so adjustment is defined.
    # The controlled experiment has only rotation 0, so its LLM cell has rows but
    # no own Vanilla row; that gap should be reported for implicit comparison.
    baseline = [{
        "experiment": "vp-self", "game_id": "vp-s1-r0", "seed": 1, "seating_rotation": 0,
        "seats": [(pid, "Vanilla", "VPAI", CIV_BY_SEAT[pid],
                   P_LEADER if pid == 4 else P_VANILLA, False) for pid in range(5)],
    }]
    ctrl = _seated_games("ctrl", seed=1, rotations=[0])
    pp, pa, gp = _write(tmp_path, baseline + ctrl)
    art = build_strength_panel(pp, pa, gp, _params(block="auto", baseline_experiment="vp-self"), catalog)

    assert any("comparison pathway" in w for w in art.warnings)
    ctrl_impl = art.cell_baseline[
        (art.cell_baseline.pathway == "implicit")
        & (art.cell_baseline.experiment == "ctrl")
    ]
    assert (1, 0) not in set(zip(ctrl_impl.seed, ctrl_impl.player_id))


def test_implicit_only_run_has_no_explicit_rows(tmp_path, catalog):
    games = _seated_games("ctrl", seed=1, rotations=[0, 1, 2, 3])
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(pp, pa, gp, _params(block="auto"), catalog)
    assert set(art.cell_baseline["pathway"].unique()) == {"implicit"}
    assert list(art.cell_baseline.columns) == [
        "experiment", "pathway", "seed", "player_id", "civilization",
        "cell_baseline", "n_vanilla", "win_rate", "n_games", "n_models",
        "has_vanilla_baseline", "vanilla_connected",
    ]
    assert art.cell_baseline["win_rate"].between(0.0, 1.0).all()


def test_coverage_truly_missing_cell(tmp_path, catalog):
    # expB only plays seed 2; against the entirety union (seeds 1 and 2) it is
    # missing every seed-1 cell.
    expA = _seated_games("expA", seed=1, rotations=[0, 1, 2, 3])
    expB = _seated_games("expB", seed=2, rotations=[0, 1, 2, 3])
    pp, pa, gp = _write(tmp_path, expA + expB)
    art = build_strength_panel(pp, pa, gp, _params(block="auto"), catalog)
    cov = art.cell_coverage
    miss = cov[(cov.experiment == "expB") & (cov.seed == 1)]
    assert not miss.empty and miss["missing"].all()
    present = cov[(cov.experiment == "expA") & (cov.seed == 1)]
    assert (~present["missing"]).all()
    # report-only: a missing comparison cell warns, never raises
    assert any("coverage" in w for w in art.warnings)


def test_post_cell_normalize_relative_to_leader(tmp_path, catalog):
    games = _seated_games("ctrl", seed=1, rotations=[0, 1, 2, 3])
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(
        pp, pa, gp, _params(block="auto", post_cell_normalize="relative_to_leader"), catalog
    )
    # every game's max adjusted_strength is exactly 1.0 after re-normalization
    gmax = art.panel.groupby("game_id")["adjusted_strength"].max()
    assert np.allclose(gmax.to_numpy(), 1.0)


# ── run_adjust integration (file writing + estimator resolution) ─────────────
def _run_spec(turns_path, pred_path, panel_path, games_path, save_path, params, baseline=None):
    pen = {
        "turn_progress_min": 0.2, "weight": "turn_progress", "relative_to": "game_leader",
        "enforce_winner": True, "civ_adjust": "ols_logit", "block": "none",
        "post_cell_normalize": "none",
    }
    pen.update(params)
    if baseline is not None:
        pen["baseline_experiment"] = baseline
    return {
        "name": "test-adjust", "seed": 42,
        "output": {"root": "reports", "suffix": ""},
        "data": {
            "extract": {"enabled": False},
            "tables": {
                "turns": str(turns_path), "panel": str(panel_path),
                "games": str(games_path), "tokens": "runs/model_token_usage.csv",
            },
        },
        "estimators": [{
            "id": "attention", "model": "attention_mlp", "fit": "pretrained",
            "predict": "in_sample", "enabled": True, "predict_subset": "all",
            "save_predictions": str(pred_path),
            "pretrained": {"model_dir": "pretrained/attention_mlp/"},
        }],
        "adjust": [{
            "id": "strength", "module": "strength", "enabled": True,
            "uses": {"estimators": ["attention"]}, "save": str(save_path),
            "params": pen,
        }],
        "analyses": [{
            "id": "bt", "module": "ratings.bradley_terry", "enabled": True,
            "uses": {"tables": ["strength"]},
            "params": {"group_by": ["player_type"], "ref": "Vanilla"},
        }],
        "report": {"template": "default", "out_dir": "reports/", "formats": ["md"]},
    }


def test_run_adjust_writes_panel_and_trails(tmp_path, write_spec):
    pp, pa, gp = _write(tmp_path, _uncontrolled_games())
    save = tmp_path / "out" / "player_strength_panel.csv"
    spec = _run_spec(tmp_path / "turns.csv", pp, pa, gp, save, {"block": "none"})
    cfg = load_config(write_spec(spec))
    result = run_adjust(cfg, cfg.adjust[0].raw)

    assert result.table_path == str(save)
    assert result.estimator_id == "attention"
    panel = pd.read_csv(save)
    assert result.n_rows == len(panel)
    for name in ("civ_effects.csv", "cell_baseline.csv", "cell_coverage.csv"):
        assert (save.parent / name).exists()
    # uncontrolled run ⇒ cell trails are empty (headers only)
    assert pd.read_csv(save.parent / "cell_baseline.csv").empty


def test_run_adjust_missing_predictions_raises(tmp_path, write_spec):
    pp, pa, gp = _write(tmp_path, _uncontrolled_games())
    save = tmp_path / "out" / "panel.csv"
    spec = _run_spec(tmp_path / "turns.csv", tmp_path / "nope.csv", pa, gp, save, {"block": "none"})
    cfg = load_config(write_spec(spec))
    with pytest.raises(AdjustError, match="predictions not found"):
        run_adjust(cfg, cfg.adjust[0].raw)


# ── config validation: free-form baseline_experiment id ──────────────────────
def test_unknown_baseline_experiment_id_allowed_at_config_load(tmp_path, write_spec):
    pp, pa, gp = _write(tmp_path, _uncontrolled_games())
    spec = _run_spec(tmp_path / "t.csv", pp, pa, gp, tmp_path / "o.csv",
                     {"block": "auto"}, baseline="no-such-experiment")
    cfg = load_config(write_spec(spec))
    assert cfg.adjust[0].raw["params"]["baseline_experiment"] == "no-such-experiment"


# ── problem-game exclusion (WS2): flagged games never reach the fit/panel ─────
def test_build_strength_panel_drops_flagged_games(tmp_path, catalog):
    games = [
        {
            "experiment": "exp-llm", "game_id": "g1", "seed": -1, "seating_rotation": -1,
            "seats": [
                (0, "TestLLM-Simple", "TestLLM", "Rome", 0.5, True),
                (1, "Vanilla", "VPAI", "Egypt", 0.25, False),
            ],
        },
        {
            "experiment": "exp-llm", "game_id": "g_bad", "seed": -1, "seating_rotation": -1,
            "seats": [
                (0, "TestLLM-Simple", "TestLLM", "Spain", 0.6, True),
                (1, "Vanilla", "VPAI", "Greece", 0.30, False),
            ],
        },
    ]
    pp, pa, gp = _write(tmp_path, games)
    art = build_strength_panel(
        pp, pa, gp, _params(civ_adjust="ols_logit"), catalog,
        problem_game_ids={"g_bad"},
    )
    # the flagged game contributes no rows to the emitted panel …
    assert set(art.panel["game_id"]) == {"g1"}
    # … nor to the civ-effects fit (Spain/Greece only appear in the flagged game)
    assert set(art.civ_effects["civilization"]) <= {"Rome", "Egypt"}
