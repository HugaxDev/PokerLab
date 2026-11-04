"""Évaluateur de mains 5 à 7 cartes (scalaire et vectorisé numpy).

Sert uniquement aux calculs d'équité et à la lecture des mains par les robots.
Le showdown réel des parties reste calculé par PokerKit ; un test vérifie que cet
évaluateur classe les mains exactement comme PokerKit.

Score : entier croissant avec la force. Catégorie = score >> 20
(0 hauteur, 1 paire, 2 double paire, 3 brelan, 4 quinte, 5 couleur, 6 full,
7 carré, 8 quinte flush).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "hauteur",
    "paire",
    "double paire",
    "brelan",
    "quinte",
    "couleur",
    "full",
    "carré",
    "quinte flush",
)

_SIZE = 1 << 13


def _build_tables() -> tuple[np.ndarray, ...]:
    pop = np.zeros(_SIZE, dtype=np.int64)
    hb = np.zeros(_SIZE, dtype=np.int64)
    straight = np.full(_SIZE, -1, dtype=np.int64)
    top2 = np.zeros(_SIZE, dtype=np.int64)
    top3 = np.zeros(_SIZE, dtype=np.int64)
    top5 = np.zeros(_SIZE, dtype=np.int64)
    windows = []
    for top in range(12, 3, -1):
        windows.append((top, sum(1 << r for r in range(top - 4, top + 1))))
    windows.append((3, (1 << 12) | 0b1111))
    for m in range(_SIZE):
        bits = [r for r in range(12, -1, -1) if m >> r & 1]
        pop[m] = len(bits)
        hb[m] = bits[0] if bits else 0
        for k, table in ((2, top2), (3, top3), (5, top5)):
            value = 0
            for i, r in enumerate(bits[:k]):
                value |= r << (4 * (k - 1 - i))
            table[m] = value
        for top, window in windows:
            if m & window == window:
                straight[m] = top
                break
    outs = np.zeros(_SIZE, dtype=np.int64)
    for m in range(_SIZE):
        if straight[m] >= 0:
            continue
        acc = 0
        for r in range(13):
            if not m >> r & 1 and straight[m | (1 << r)] >= 0:
                acc |= 1 << r
        outs[m] = acc
    return pop, hb, straight, top2, top3, top5, outs


POP, HB, STRAIGHT_TOP, TOP2, TOP3, TOP5, STRAIGHT_OUTS = _build_tables()
_POP = POP.tolist()
_HB = HB.tolist()
_ST = STRAIGHT_TOP.tolist()
_T2 = TOP2.tolist()
_T3 = TOP3.tolist()
_T5 = TOP5.tolist()
_OUTS = STRAIGHT_OUTS.tolist()
CARD_BITS = np.array([1 << ((c & 3) * 16 + (c >> 2)) for c in range(52)] + [0], dtype=np.int64)


def suit_masks(cards: Iterable[int]) -> list[int]:
    masks = [0, 0, 0, 0]
    for c in cards:
        masks[c & 3] |= 1 << (c >> 2)
    return masks


def evaluate(cards: Sequence[int]) -> int:
    """Score d'une main de 1 à 7 cartes (entiers 0..51)."""
    a, b, c, d = suit_masks(cards)
    m1 = a | b | c | d
    m2 = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)
    m3 = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d)
    m4 = a & b & c & d
    fm = 0
    for s in (a, b, c, d):
        if _POP[s] >= 5:
            fm = s
            break
    if fm:
        sf = _ST[fm]
        if sf >= 0:
            return (STRAIGHT_FLUSH << 20) | (sf << 16)
    if m4:
        q = _HB[m4]
        return (QUADS << 20) | (q << 16) | (_HB[m1 & ~(1 << q)] << 12)
    if m3:
        t = _HB[m3]
        rest = m2 & ~(1 << t)
        if rest:
            return (FULL_HOUSE << 20) | (t << 16) | (_HB[rest] << 12)
    if fm:
        return (FLUSH << 20) | _T5[fm]
    st = _ST[m1]
    if st >= 0:
        return (STRAIGHT << 20) | (st << 16)
    if m3:
        t = _HB[m3]
        return (TRIPS << 20) | (t << 16) | (_T2[m1 & ~(1 << t)] << 8)
    if m2:
        p = _HB[m2]
        rest2 = m2 & ~(1 << p)
        if rest2:
            p2 = _HB[rest2]
            kick = _HB[m1 & ~(1 << p) & ~(1 << p2)]
            return (TWO_PAIR << 20) | (p << 16) | (p2 << 12) | (kick << 8)
        return (PAIR << 20) | (p << 16) | (_T3[m1 & ~(1 << p)] << 4)
    return _T5[m1]


def category(score: int) -> int:
    return score >> 20


def straight_out_ranks(rank_mask: int) -> int:
    """Masque des rangs qui compléteraient une quinte (0 si déjà quinte)."""
    return _OUTS[rank_mask]


def popcount(mask: int) -> int:
    return _POP[mask & (_SIZE - 1)]


def highbit(mask: int) -> int:
    return _HB[mask & (_SIZE - 1)]


def evaluate_np(cards: np.ndarray) -> np.ndarray:
    """Évaluation vectorisée : ``cards`` (N, K) entiers, -1 = carte absente."""
    cards = np.asarray(cards, dtype=np.int64)
    packed = CARD_BITS[np.where(cards >= 0, cards, 52)].sum(axis=1)
    return evaluate_packed(packed)


def pack_cards(cards: np.ndarray) -> np.ndarray:
    """Encode chaque ligne de cartes en un entier 64 bits (16 bits par couleur)."""
    cards = np.asarray(cards, dtype=np.int64)
    return CARD_BITS[np.where(cards >= 0, cards, 52)].sum(axis=1)


def evaluate_packed(packed: np.ndarray) -> np.ndarray:
    """Évalue des mains déjà encodées par :func:`pack_cards` (cartes distinctes)."""
    a = packed & 0x1FFF
    b = (packed >> 16) & 0x1FFF
    c = (packed >> 32) & 0x1FFF
    d = (packed >> 48) & 0x1FFF
    m1 = a | b | c | d
    m2 = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)
    m3 = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d)
    m4 = a & b & c & d
    one = np.int64(1)

    score = TOP5[m1]
    p = HB[m2]
    not_p = ~np.left_shift(one, p)
    score = np.where(m2 != 0, (PAIR << 20) | (p << 16) | (TOP3[m1 & not_p] << 4), score)
    rest2 = m2 & not_p
    p2 = HB[rest2]
    kick = HB[m1 & not_p & ~np.left_shift(one, p2)]
    score = np.where(rest2 != 0, (TWO_PAIR << 20) | (p << 16) | (p2 << 12) | (kick << 8), score)
    t = HB[m3]
    not_t = ~np.left_shift(one, t)
    score = np.where(m3 != 0, (TRIPS << 20) | (t << 16) | (TOP2[m1 & not_t] << 8), score)
    st = STRAIGHT_TOP[m1]
    score = np.where(st >= 0, (STRAIGHT << 20) | (st << 16), score)
    fm = np.where(POP[a] >= 5, a, np.where(POP[b] >= 5, b, np.where(POP[c] >= 5, c, np.where(POP[d] >= 5, d, 0))))
    score = np.where(fm != 0, (FLUSH << 20) | TOP5[fm], score)
    rest3 = m2 & not_t
    score = np.where((m3 != 0) & (rest3 != 0), (FULL_HOUSE << 20) | (t << 16) | (HB[rest3] << 12), score)
    q = HB[m4]
    score = np.where(m4 != 0, (QUADS << 20) | (q << 16) | (HB[m1 & ~np.left_shift(one, q)] << 12), score)
    sf = STRAIGHT_TOP[fm]
    score = np.where((fm != 0) & (sf >= 0), (STRAIGHT_FLUSH << 20) | (sf << 16), score)
    return score
