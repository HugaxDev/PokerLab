"""Bankroll : ruine détectée main par main, comparaison avec/sans stop-loss à nombre de mains égal.

Aucune main de poker jouée : sessions synthétiques.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pokerlab.experiments.bankroll import session_profiles, simulate_paths_hands


def _sessions(results: list[list[float]]) -> pd.DataFrame:
    rows = []
    for s, res in enumerate(results):
        for h, r in enumerate(res):
            rows.append({"strategy": "x", "session": s, "hand_no": h, "result": r})
    return pd.DataFrame(rows)


def test_mid_session_ruin_is_detected() -> None:
    """Une session qui plonge à -150 bb puis finit à +50 bb ruine une bankroll de 100 bb."""
    df = _sessions([[-150.0, 200.0]])
    r, n, low = session_profiles(df, "x", None)
    assert r.tolist() == [50.0] and n.tolist() == [2] and low.tolist() == [-150.0]
    with_low = simulate_paths_hands(r, n, 100.0, 50, 2, 1.0, 1, grid_step=1, lows=low)
    end_only = simulate_paths_hands(r, n, 100.0, 50, 2, 1.0, 1, grid_step=1)
    assert with_low["p_ruin"] == 1.0
    assert end_only["p_ruin"] == 0.0  # l'ancienne méthode ne voyait pas cette ruine


def test_stop_loss_cuts_session_at_threshold() -> None:
    df = _sessions([[-10.0, -20.0, 30.0, 40.0], [5.0, 5.0, 5.0, 5.0]])
    r, n, low = session_profiles(df, "x", 25.0)
    assert r.tolist() == [-30.0, 20.0] and n.tolist() == [2, 4] and low.tolist() == [-30.0, 0.0]


def test_stop_loss_neutral_on_independent_hands() -> None:
    """Mains indépendantes : à nombre de mains égal, le stop-loss ne change pas la ruine (au bruit près)."""
    rng = np.random.default_rng(3)
    hands = rng.normal(0.5, 25.0, size=(3000, 50))
    df = _sessions(hands.tolist())
    ruin = {}
    for sl in (None, 25.0):
        r, n, low = session_profiles(df, "x", sl)
        ruin[sl] = simulate_paths_hands(r, n, 150.0, 3000, 3000, 1.0, 11, lows=low)["p_ruin"]
    assert abs(ruin[None] - ruin[25.0]) < 0.05, ruin
