"""Espérance « all-in adjusted » : gains remplacés par équité x pot quand l'argent est à tapis."""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Sequence

import numpy as np

from pokerlab.equity.evaluator import CARD_BITS, evaluate_packed


def runouts(known: Sequence[int], k: int, exact_max: int, mc_samples: int, seed: int) -> np.ndarray:
    """Énumère (ou échantillonne) les tableaux restants : (R, k)."""
    remaining = np.array([c for c in range(52) if c not in set(known)], dtype=np.int64)
    if k == 0:
        return np.zeros((1, 0), dtype=np.int64)
    if comb(remaining.size, k) <= exact_max:
        return np.array(list(combinations(remaining.tolist(), k)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    keys = rng.random((mc_samples, remaining.size))
    idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
    return remaining[idx]


def allin_ev_payoffs(
    holes: dict[int, tuple[int, int]],
    board: Sequence[int],
    pots: Sequence[tuple[int, tuple[int, ...]]],
    contributions: Sequence[int],
    exact_max: int,
    mc_samples: int,
    seed: int,
) -> list[float]:
    """Gain espéré de chaque joueur (en jetons) une fois l'argent engagé.

    ``holes`` : cartes des joueurs encore en jeu ; ``pots`` : (montant, joueurs éligibles) ;
    ``contributions`` : mise totale de chaque joueur dans la main.
    """
    known = list(board) + [c for h in holes.values() for c in h]
    outs = runouts(known, 5 - len(board), exact_max, mc_samples, seed)
    base = int(CARD_BITS[list(board)].sum()) if board else 0
    board_bits = CARD_BITS[outs].sum(axis=1) + base
    scores = {
        p: evaluate_packed(board_bits + int(CARD_BITS[h[0]] + CARD_BITS[h[1]])) for p, h in holes.items()
    }
    ev = [0.0 - float(c) for c in contributions]
    for amount, eligible in pots:
        live = [p for p in eligible if p in scores]
        if not live:
            continue
        if len(live) == 1:
            ev[live[0]] += amount
            continue
        stack = np.stack([scores[p] for p in live])
        best = stack.max(axis=0)
        winners = stack == best
        share = winners / winners.sum(axis=0)
        for i, p in enumerate(live):
            ev[p] += amount * float(share[i].mean())
    return ev
