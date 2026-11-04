"""Évaluateur (contre PokerKit), équité Monte Carlo, all-in EV, table préflop."""

from __future__ import annotations

import random

import numpy as np
import pytest
from pokerkit import StandardHighHand

from pokerlab.cards.cards import parse_cards, to_pokerkit
from pokerlab.cards.ranges import CLASS_INDEX_BY_NAME, parse_range
from pokerlab.equity.allin import allin_ev_payoffs
from pokerlab.equity.evaluator import evaluate, evaluate_np
from pokerlab.equity.montecarlo import equity_vs_ranges


def test_evaluator_matches_pokerkit() -> None:
    rng = random.Random(11)
    for _ in range(1500):
        cards = rng.sample(range(52), 9)
        h1, h2, board = cards[:2], cards[2:4], cards[4:]
        mine = (evaluate(h1 + board) > evaluate(h2 + board)) - (evaluate(h1 + board) < evaluate(h2 + board))
        p1 = StandardHighHand.from_game(to_pokerkit(h1), to_pokerkit(board))
        p2 = StandardHighHand.from_game(to_pokerkit(h2), to_pokerkit(board))
        theirs = (p1 > p2) - (p1 < p2)
        assert mine == theirs, (h1, h2, board)


def test_numpy_matches_scalar() -> None:
    rng = random.Random(5)
    rows = [rng.sample(range(52), 7) for _ in range(2000)]
    vec = evaluate_np(np.array(rows))
    assert all(int(v) == evaluate(r) for v, r in zip(vec, rows))


def test_aa_vs_kk() -> None:
    eq = equity_vs_ranges(parse_cards("AsAh"), [], [parse_range("KK")], 20000, 3)
    assert 0.80 <= eq <= 0.83


def test_equity_vs_random_and_board() -> None:
    eq = equity_vs_ranges(parse_cards("AsAh"), [], [parse_range("any")], 8000, 4)
    assert 0.83 <= eq <= 0.87
    made = equity_vs_ranges(parse_cards("AsAh"), parse_cards("Ad7c2h"), [parse_range("KK")], 4000, 5)
    assert made > 0.97


def test_allin_ev_conserves_chips() -> None:
    holes = {0: tuple(parse_cards("AsAh")), 2: tuple(parse_cards("KdKc")), 3: tuple(parse_cards("7s6s"))}
    contributions = [3000, 100, 3000, 1500, 0, 0]
    pots = [(1500 * 3 + 100, (0, 2, 3)), (1500 * 2, (0, 2))]
    ev = allin_ev_payoffs(holes, parse_cards("Qs8s2c"), pots, contributions, 2000, 1000, 1)
    assert sum(ev) == pytest.approx(0.0, abs=1e-6)
    assert ev[1] == -100 and ev[4] == 0


def test_preflop_table(lab) -> None:
    from pokerlab.strategies.registry import get_context

    table = get_context(lab).table
    aa, kk, x72 = CLASS_INDEX_BY_NAME["AA"], CLASS_INDEX_BY_NAME["KK"], CLASS_INDEX_BY_NAME["72o"]
    assert table.vs_random[aa] == table.vs_random.max()
    assert table.vs_random[x72] < 0.37
    assert 0.78 <= table.matrix[aa, kk] <= 0.85
    assert table.matrix[aa, kk] + table.matrix[kk, aa] == pytest.approx(1.0)
    assert table.top_fraction(1.0).sum() == 1326
