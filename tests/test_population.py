"""Import PokerStars : stats par adversaire (fixture écrite à la main) et composition population_reelle."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from pokerlab.config import PROJECT_ROOT, load_archetypes, load_table
from pokerlab.journal.pokerstars import import_texts, read_files
from pokerlab.journal.population import (
    build_profiles,
    calibrate,
    initial_params,
    merge_opponents,
    player_stats,
    write_population,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pokerstars_adversaires.txt"


@pytest.fixture(scope="module")
def opp_rows() -> pd.DataFrame:
    rows: list = []
    hands, skipped = import_texts(read_files([FIXTURE]), None, 100, "Hero", opponents=rows)
    assert len(hands) == 3 and skipped == 0
    return pd.DataFrame(rows)


def test_counts_per_opponent(opp_rows: pd.DataFrame) -> None:
    """Compteurs attendus, calculés à la main sur les 3 mains de la fixture."""
    st = player_stats(opp_rows).set_index("joueur")
    assert "Hero" not in st.index and set(st.index) == {"Alice", "Bob", "Carl", "Dana", "Eve"}
    expected = {  # mains, vpip, pfr, occasions 3-bet, 3-bet, c-bet (occ., fait), face c-bet (occ., couché), flop vu, abattage, gagné, agr., paies
        "Alice": (3, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1),
        "Bob": (3, 1, 0, 2, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0),
        "Carl": (3, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0),
        "Dana": (3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        "Eve": (3, 2, 1, 2, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1),
    }
    cols = ["f_hands", "f_vpip", "f_pfr", "f_tb_opp", "f_tb", "f_cbet_opp", "f_cbet", "f_fcb_opp", "f_fcb", "f_saw_flop",
            "f_wtsd", "f_wsd", "f_post_aggr", "f_post_calls"]
    for name, exp in expected.items():
        assert tuple(int(st.loc[name, c]) for c in cols) == exp, name
    a = st.loc["Alice"]
    assert a["VPIP"] == pytest.approx(200 / 3) and a["PFR"] == pytest.approx(100 / 3) and a["3bet"] == pytest.approx(100 / 3)
    assert a["AF"] == 1.0 and a["WTSD"] == 50.0 and a["fold_cbet"] == 100.0
    assert st.loc["Eve", "WTSD"] == 100.0 and st.loc["Eve", "fold_cbet"] == 0.0
    assert math.isnan(st.loc["Bob", "AF"]) or math.isinf(st.loc["Bob", "AF"])   # 2 agressions, 0 paie


def test_reimport_does_not_double_count(opp_rows: pd.DataFrame) -> None:
    once = merge_opponents(pd.DataFrame(), opp_rows.to_dict("records"))
    twice = merge_opponents(once, opp_rows.to_dict("records"))
    assert len(twice) == len(once) == 15


def test_profiles_and_initial_params(opp_rows: pd.DataFrame) -> None:
    players = player_stats(opp_rows)
    prof = build_profiles(players, k=2, min_hands=3, seed=1)
    assert 1 <= len(prof) <= 2 and abs(sum(p["part"] for p in prof) - 1) < 1e-9
    assert sum(p["joueurs"] for p in prof) == 5
    only_rest = build_profiles(players, k=2, min_hands=50, seed=1)
    assert len(only_rest) == 1 and only_rest[0]["type"] == "occasionnels"
    q = initial_params({"VPIP": 40.0, "PFR": 10.0, "3bet": 4.0, "AF": 1.5, "WTSD": 35.0, "fold_cbet": 30.0, "cbet": 50.0})
    assert q["vpip"] == 0.4 and q["pfr"] == 0.1 and q["limp"] and 0 < q["fold_to_cbet"] < 0.5


def test_population_composition_usable_in_compare(opp_rows: pd.DataFrame, tmp_path: Path, monkeypatch, lab) -> None:
    """Profils calibrés (mini calibrage), fichiers écrits, composition chargée et jouée comme une autre."""
    from pokerlab.experiments.common import DuplicateTask, run_duplicate_task
    from pokerlab.strategies.registry import reset_archetypes, resolve_spec

    root = tmp_path / "configs"
    shutil.copytree(PROJECT_ROOT / "configs", root)
    monkeypatch.setenv("POKERLAB_CONFIGS", str(root))
    reset_archetypes()
    try:
        prof = build_profiles(player_stats(opp_rows), k=2, min_hands=3, seed=1)
        calibrate(prof, lab, iterations=1, decks=2, log=lambda m: None)
        assert all("params" in p and "mesure" in p for p in prof)
        opp, tab = write_population(prof, root, "fixture", 15)
        arch = load_archetypes(root)
        assert {p["name"] for p in prof} <= set(arch) and "maniac" in arch
        comp = load_table("population_reelle", root)
        assert set(comp["weights"]) == {p["name"] for p in prof}
        reset_archetypes()
        out = run_duplicate_task(DuplicateTask(lab.raw, resolve_spec("joueur"), comp, 0, 1, {"strategy": "joueur"}, record_decisions=False))
        assert out.hands_played == 6
    finally:
        monkeypatch.delenv("POKERLAB_CONFIGS")
        reset_archetypes()
