"""Robustesse : médiane, pire cas, classement par pire cas et règle de la stratégie gagnante."""

from __future__ import annotations

import pandas as pd

from pokerlab.analysis.robustness import robustness_table, winner_verdict


def _cells(rows):
    return pd.DataFrame([{"strategy": s, "composition": c, "bb100": b, "ci": ci, "mains": 1000} for s, c, b, ci in rows])


def test_median_worst_case_and_ranking() -> None:
    cells = _cells([
        ("a", "x", 10, 2), ("a", "y", 4, 2), ("a", "z", 8, 2),
        ("b", "x", 30, 3), ("b", "y", -1, 3), ("b", "z", 20, 3),
    ])
    rob = robustness_table(cells, references=())
    assert list(rob["strategy"]) == ["a", "b"]            # classé par pire cas (4 > -1), pas par moyenne
    a = rob.iloc[0]
    assert a["bb100_median"] == 8 and a["pire_cas"] == 4 and a["pire_composition"] == "y"
    assert a["borne_basse_min"] == 2 and a["gagnante_partout"]
    assert not rob.iloc[1]["gagnante_partout"]
    name, text = winner_verdict(rob)
    assert name == "a" and text.startswith("Gagnante : a")


def test_no_winner_is_said_clearly() -> None:
    cells = _cells([("a", "x", 10, 2), ("a", "y", 1, 2), ("b", "x", 5, 1), ("b", "y", 0.5, 1)])
    name, text = winner_verdict(robustness_table(cells, references=()))
    assert name is None and text.startswith("PAS DE GAGNANTE")


def test_references_and_incomplete_strategies_never_win() -> None:
    cells = _cells([("ref", "x", 50, 1), ("ref", "y", 40, 1), ("a", "x", 10, 1), ("a", "y", 9, 1), ("c", "x", 99, 1)])
    rob = robustness_table(cells, references=("ref",))
    assert not rob.set_index("strategy").loc["c", "gagnante_partout"]   # mesurée sur 1 composition sur 2
    name, _ = winner_verdict(rob)
    assert name == "a"
