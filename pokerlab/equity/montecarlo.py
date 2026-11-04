"""Équité Monte Carlo d'une main contre une ou plusieurs ranges (vectorisé numpy)."""

from __future__ import annotations

from collections import OrderedDict
from typing import Sequence

import numpy as np

from pokerlab.cards.ranges import COMBO_CARDS, N_COMBOS, remove_dead
from pokerlab.equity.evaluator import CARD_BITS, evaluate_packed

_UNIFORM = np.ones(N_COMBOS, dtype=np.float64)


def sample_board(rng: np.random.Generator, blocked: np.ndarray, k: int) -> np.ndarray:
    """Tire ``k`` cartes par ligne parmi les cartes non bloquées (blocked : (N, 52) bool)."""
    if k == 0:
        return np.zeros((blocked.shape[0], 0), dtype=np.int64)
    keys = rng.random(blocked.shape)
    keys[blocked] = 2.0
    return np.argpartition(keys, k - 1, axis=1)[:, :k].astype(np.int64)


def _cumulative(weights: np.ndarray, dead: list[int]) -> np.ndarray:
    clean = remove_dead(np.asarray(weights, dtype=np.float64), dead)
    total = clean.sum()
    if total <= 0:
        clean = remove_dead(_UNIFORM, dead)
        total = clean.sum()
    return np.cumsum(clean / total)


def _draw(cum: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.searchsorted(cum, u, side="right").clip(max=N_COMBOS - 1)


def sample_villains(
    rng: np.random.Generator,
    weights: Sequence[np.ndarray],
    dead: Sequence[int],
    sims: int,
    max_rounds: int = 40,
) -> np.ndarray:
    """Tire les combos des adversaires (sims, n_vilains) sans collision de cartes."""
    dead = list(dead)
    cums = [_cumulative(w, dead) for w in weights]
    n = len(cums)
    out = np.zeros((sims, n), dtype=np.int64)
    for j in range(n):
        out[:, j] = _draw(cums[j], rng.random(sims))
    if n == 1:
        return out
    for _ in range(max_rounds):
        cards = np.sort(COMBO_CARDS[out].reshape(sims, -1), axis=1)
        conflict = (cards[:, 1:] == cards[:, :-1]).any(axis=1)
        if not conflict.any():
            break
        rows = np.flatnonzero(conflict)
        for j in range(n):
            out[rows, j] = _draw(cums[j], rng.random(rows.size))
    return out


def showdown_shares(hero_score: np.ndarray, villain_scores: list[np.ndarray]) -> np.ndarray:
    """Part du pot du héros par simulation (1 gagne, 1/k en cas d'égalité à k)."""
    best = np.max(np.stack(villain_scores), axis=0) if villain_scores else np.zeros_like(hero_score)
    ties = np.zeros(hero_score.shape, dtype=np.int64)
    for score in villain_scores:
        ties += score == hero_score
    return np.where(hero_score > best, 1.0, np.where(hero_score == best, 1.0 / (1 + ties), 0.0))


def equity_vs_ranges(
    hero: Sequence[int],
    board: Sequence[int],
    villain_weights: Sequence[np.ndarray],
    sims: int,
    seed: int,
) -> float:
    """Équité (gain + part des égalités) de ``hero`` contre les ranges données."""
    rng = np.random.default_rng(seed)
    hero = list(hero)
    board = list(board)
    dead = hero + board
    villains = sample_villains(rng, villain_weights, dead, sims)
    vcards = COMBO_CARDS[villains].reshape(sims, -1).astype(np.int64)
    blocked = np.zeros((sims, 52), dtype=bool)
    blocked[:, dead] = True
    np.put_along_axis(blocked, vcards, True, axis=1)
    runout = sample_board(rng, blocked, 5 - len(board))
    base = int(CARD_BITS[board].sum()) if board else 0
    board_bits = CARD_BITS[runout].sum(axis=1) + base
    hero_score = evaluate_packed(board_bits + int(CARD_BITS[hero].sum()))
    villain_scores = [
        evaluate_packed(board_bits + CARD_BITS[vcards[:, 2 * j]] + CARD_BITS[vcards[:, 2 * j + 1]])
        for j in range(villains.shape[1])
    ]
    return float(showdown_shares(hero_score, villain_scores).mean())


class EquityCache:
    """Cache LRU en mémoire (clé déterministe -> valeur)."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.data: OrderedDict[object, float] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: object) -> float | None:
        value = self.data.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return value

    def put(self, key: object, value: float) -> None:
        self.data[key] = value
        if len(self.data) > self.size:
            self.data.popitem(last=False)
