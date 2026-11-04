"""Estimation de la range adverse à partir de ses actions dans la main.

Préflop : percentiles de mains (classement par équité contre une main aléatoire) selon la
ligne jouée (ouverture, 3-bet, paie...). Postflop : à chaque street, une mise garde le haut
de la range, une relance le très haut, une paie retire le bas.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from pokerlab.cards.ranges import COMBO_CARDS, N_COMBOS
from pokerlab.engine.types import Kind, Observation
from pokerlab.equity.evaluator import CARD_BITS, PAIR, POP, STRAIGHT_OUTS, evaluate_packed
from pokerlab.equity.preflop import PreflopTable

DEFAULTS: dict[str, Any] = {
    "looseness": 1.0,
    "open_pct": {"UTG": 0.16, "MP": 0.20, "CO": 0.28, "BTN": 0.42, "SB": 0.38, "BB": 0.30},
    "limp_pct": 0.55,
    "call_pct": 0.32,
    "call_exclude_top": 0.03,
    "three_bet_pct": 0.08,
    "four_bet_pct": 0.035,
    "bb_check_pct": 1.0,
    "keep_after_bet": 0.60,
    "keep_after_raise": 0.35,
    "keep_after_call": 0.80,
}

_COMBO_BITS = CARD_BITS[COMBO_CARDS[:, 0]] + CARD_BITS[COMBO_CARDS[:, 1]]


def combo_strength(board: tuple[int, ...]) -> np.ndarray:
    """Force de chaque combo sur le tableau (score évaluateur, tirages assimilés à une petite paire)."""
    base = int(CARD_BITS[list(board)].sum())
    packed = _COMBO_BITS + base
    score = evaluate_packed(packed)
    if len(board) < 5:
        suits = [(packed >> (16 * s)) & 0x1FFF for s in range(4)]
        board_suits = [(base >> (16 * s)) & 0x1FFF for s in range(4)]
        flush_draw = np.zeros(N_COMBOS, dtype=bool)
        for s in range(4):
            flush_draw |= (POP[suits[s]] == 4) & (POP[board_suits[s]] < 4)
        m_all = suits[0] | suits[1] | suits[2] | suits[3]
        m_board = board_suits[0] | board_suits[1] | board_suits[2] | board_suits[3]
        oesd = POP[STRAIGHT_OUTS[m_all] & ~STRAIGHT_OUTS[m_board]] >= 2
        draw_score = (PAIR << 20) | (5 << 16)
        score = np.where((flush_draw | oesd) & (score < draw_score), draw_score, score)
    return score


def keep_top(weights: np.ndarray, strength: np.ndarray, keep: float, drop_bottom: bool = False) -> np.ndarray:
    """Garde la fraction ``keep`` (en masse) la plus forte, ou retire le bas si ``drop_bottom``."""
    total = weights.sum()
    if total <= 0 or keep >= 1.0:
        return weights
    order = np.argsort(-strength, kind="stable")
    cum = np.cumsum(weights[order]) / total
    before = cum - weights[order] / total
    keep_mask = np.zeros(N_COMBOS, dtype=bool)
    keep_mask[order] = before < keep
    out = np.where(keep_mask, weights, 0.0)
    return out if out.sum() > 0 else weights


class RangeModel:
    """Range estimée de chaque adversaire, mise en cache par signature de la main."""

    def __init__(self, table: PreflopTable, params: dict[str, Any] | None = None, cache_size: int = 20000) -> None:
        self.table = table
        p = dict(DEFAULTS)
        p.update(params or {})
        self.p = p
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self.cache_size = cache_size

    def _pct(self, value: float) -> float:
        return min(1.0, value * float(self.p["looseness"]))

    def preflop_range(self, obs: Observation, villain: int) -> tuple[tuple, np.ndarray]:
        """Signature et poids de la range préflop d'un adversaire."""
        raises_before = 0
        line: list[str] = []
        for a in obs.history:
            if a.street != 0 or a.kind == Kind.POST:
                continue
            if a.seat == villain:
                if a.kind in (Kind.BET, Kind.RAISE):
                    line.append(["open", "3bet", "4bet"][min(raises_before, 2)])
                elif a.kind == Kind.CALL:
                    line.append("call" if raises_before else "limp")
                elif a.kind == Kind.CHECK:
                    line.append("check")
            if a.kind in (Kind.BET, Kind.RAISE):
                raises_before += 1
        pos = obs.positions[villain]
        top = line[-1] if line else "check"
        t = self.table
        if "4bet" in line:
            w = t.top_fraction(self._pct(self.p["four_bet_pct"]))
        elif "3bet" in line:
            w = t.top_fraction(self._pct(self.p["three_bet_pct"]))
        elif top == "call" and "open" in line:
            w = t.band(0.0, self._pct(self.p["three_bet_pct"] * 2.5))
        elif "open" in line:
            w = t.top_fraction(self._pct(self.p["open_pct"].get(pos, 0.3)))
        elif top == "call":
            w = t.band(self._pct(self.p["call_exclude_top"]), self._pct(self.p["call_pct"]))
        elif top == "limp":
            w = t.top_fraction(self._pct(self.p["limp_pct"]))
        else:
            w = t.top_fraction(self.p["bb_check_pct"])
        return (pos, tuple(line)), w

    def villain_range(self, obs: Observation, villain: int) -> tuple[tuple, np.ndarray]:
        """Signature (déterministe) et poids (1326,) de la range estimée de ``villain``."""
        pre_sig, weights = self.preflop_range(obs, villain)
        post: list[tuple[int, str]] = []
        for street in range(1, obs.street + 1):
            kinds = [a.kind for a in obs.history if a.street == street and a.seat == villain]
            if Kind.RAISE in kinds:
                post.append((street, "raise"))
            elif Kind.BET in kinds:
                post.append((street, "bet"))
            elif Kind.CALL in kinds:
                post.append((street, "call"))
        board = obs.board
        sig = (pre_sig, tuple(post), board)
        cached = self._cache.get(sig)
        if cached is not None:
            self._cache.move_to_end(sig)
            return sig, cached
        w = np.array(weights, dtype=np.float64)
        dead = np.isin(COMBO_CARDS, np.asarray(board, dtype=np.int16)).any(axis=1) if board else None
        if dead is not None:
            w[dead] = 0.0
        for street, action in post:
            sub_board = board[: {1: 3, 2: 4, 3: 5}[street]]
            strength = combo_strength(sub_board)
            if action == "raise":
                w = keep_top(w, strength, self.p["keep_after_raise"])
            elif action == "bet":
                w = keep_top(w, strength, self.p["keep_after_bet"])
            else:
                w = keep_top(w, strength, self.p["keep_after_call"])
        self._cache[sig] = w
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sig, w
