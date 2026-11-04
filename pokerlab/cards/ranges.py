"""Parseur de ranges ("99+, AKs, ATs+, KQo, T9s-54s"), combinaisons et grille 13x13.

Indexation :
- combo : index 0..1325 d'une paire de cartes non ordonnée (c1 > c2) ;
- classe : index 0..168 sur la grille 13x13. Ligne/colonne 0 = As. Au-dessus de la
  diagonale (ligne < colonne) : assorties ; en dessous : dépareillées ; diagonale : paires.
"""

from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from pokerlab.cards.cards import RANK_INDEX, RANKS, parse_cards, rank_of, suit_of

N_COMBOS = 1326
N_CLASSES = 169

# Combinaisons : (haute, basse) avec haute > basse.
COMBOS: list[tuple[int, int]] = [(a, b) for a in range(52) for b in range(a)]
COMBO_INDEX = np.full((52, 52), -1, dtype=np.int32)
for _i, (_a, _b) in enumerate(COMBOS):
    COMBO_INDEX[_a, _b] = _i
    COMBO_INDEX[_b, _a] = _i
COMBO_CARDS = np.array(COMBOS, dtype=np.int16)


def grid_pos(r1: int, r2: int, suited: bool) -> tuple[int, int]:
    """Position (ligne, colonne) dans la grille pour deux rangs (0 = 2, 12 = As)."""
    hi, lo = max(r1, r2), min(r1, r2)
    row_hi, row_lo = 12 - hi, 12 - lo
    if hi == lo:
        return row_hi, row_hi
    return (row_hi, row_lo) if suited else (row_lo, row_hi)


def class_index(r1: int, r2: int, suited: bool) -> int:
    row, col = grid_pos(r1, r2, suited)
    return row * 13 + col


def class_name(index: int) -> str:
    row, col = divmod(index, 13)
    hi, lo = RANKS[12 - min(row, col)], RANKS[12 - max(row, col)]
    if row == col:
        return hi + lo
    return hi + lo + ("s" if row < col else "o")


CLASS_NAMES = [class_name(i) for i in range(N_CLASSES)]
CLASS_INDEX_BY_NAME = {name: i for i, name in enumerate(CLASS_NAMES)}


def combo_class(c1: int, c2: int) -> int:
    return class_index(rank_of(c1), rank_of(c2), suit_of(c1) == suit_of(c2) and c1 != c2)


COMBO_CLASS = np.array([combo_class(a, b) for a, b in COMBOS], dtype=np.int16)
CLASS_COMBO_COUNT = np.bincount(COMBO_CLASS, minlength=N_CLASSES).astype(np.int32)
CLASS_COMBOS: list[np.ndarray] = [np.flatnonzero(COMBO_CLASS == k) for k in range(N_CLASSES)]


def hole_class(cards: tuple[int, int] | list[int]) -> int:
    return combo_class(cards[0], cards[1])


def class_is_pair(index: int) -> bool:
    row, col = divmod(index, 13)
    return row == col


def class_is_suited(index: int) -> bool:
    row, col = divmod(index, 13)
    return row < col


def class_ranks(index: int) -> tuple[int, int]:
    """Rangs (haut, bas) d'une classe (0 = 2, 12 = As)."""
    row, col = divmod(index, 13)
    return 12 - min(row, col), 12 - max(row, col)


_TOKEN = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)(\+?)$")


def _classes_for(hi: int, lo: int, kind: str) -> list[int]:
    if hi == lo:
        return [class_index(hi, lo, False)]
    if kind == "s":
        return [class_index(hi, lo, True)]
    if kind == "o":
        return [class_index(hi, lo, False)]
    return [class_index(hi, lo, True), class_index(hi, lo, False)]


def _parse_simple(token: str) -> tuple[int, int, str, bool]:
    match = _TOKEN.match(token)
    if not match:
        raise ValueError(f"Élément de range invalide : {token!r}")
    a, b, kind, plus = match.groups()
    r1, r2 = RANK_INDEX[a], RANK_INDEX[b]
    hi, lo = max(r1, r2), min(r1, r2)
    if hi == lo and kind:
        raise ValueError(f"Une paire ne peut pas être assortie/dépareillée : {token!r}")
    return hi, lo, kind, bool(plus)


def _expand_token(token: str) -> dict[int, float]:
    """Développe un élément en {index de combo: poids}."""
    weight = 1.0
    if ":" in token:
        token, w = token.split(":", 1)
        weight = float(w)
    token = token.strip()
    out: dict[int, float] = {}
    upper = token.upper()
    if upper in {"ANY", "RANDOM", "100%", "*"}:
        return {i: weight for i in range(N_COMBOS)}
    if len(token) == 4 and token[1] in "cdhsCDHS" and token[3] in "cdhsCDHS":
        cards = parse_cards(token)
        if cards[0] == cards[1]:
            raise ValueError(f"Combinaison invalide : {token!r}")
        return {int(COMBO_INDEX[cards[0], cards[1]]): weight}
    norm = token[:2].upper() + token[2:].lower()
    classes: list[int] = []
    if "-" in norm:
        left, right = norm.split("-", 1)
        right = right[:2].upper() + right[2:].lower()
        h1, l1, k1, _ = _parse_simple(left)
        h2, l2, k2, _ = _parse_simple(right)
        if k1 != k2:
            raise ValueError(f"Intervalle incohérent : {token!r}")
        if h1 == l1 and h2 == l2:  # paires : 22-55
            for r in range(min(h1, h2), max(h1, h2) + 1):
                classes += _classes_for(r, r, "")
        elif h1 == h2:  # même carte haute : A2s-A5s
            for lo in range(min(l1, l2), max(l1, l2) + 1):
                classes += _classes_for(h1, lo, k1)
        elif h1 - l1 == h2 - l2:  # même écart : T9s-54s
            gap = h1 - l1
            for hi in range(min(h1, h2), max(h1, h2) + 1):
                classes += _classes_for(hi, hi - gap, k1)
        else:
            raise ValueError(f"Intervalle non reconnu : {token!r}")
    else:
        hi, lo, kind, plus = _parse_simple(norm)
        if hi == lo:
            top = 12 if plus else hi
            for r in range(hi, top + 1):
                classes += _classes_for(r, r, "")
        else:
            top = hi - 1 if plus else lo
            for k in range(lo, top + 1):
                classes += _classes_for(hi, k, kind)
    for cls in classes:
        for combo in CLASS_COMBOS[cls]:
            out[int(combo)] = weight
    return out


@lru_cache(maxsize=4096)
def parse_range(text: str) -> np.ndarray:
    """Parse une range texte -> vecteur de poids (1326,) en lecture seule."""
    weights = np.zeros(N_COMBOS, dtype=np.float64)
    for raw in text.replace(";", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        for combo, w in _expand_token(token).items():
            weights[combo] = max(weights[combo], w)
    weights.setflags(write=False)
    return weights


@lru_cache(maxsize=4096)
def range_classes(text: str) -> frozenset[int]:
    """Classes (0..168) présentes dans une range (au moins un combo de poids > 0)."""
    weights = parse_range(text)
    return frozenset(int(c) for c in np.unique(COMBO_CLASS[weights > 0]))


def combo_count(text: str) -> float:
    return float(parse_range(text).sum())


def class_weights(combo_weights: np.ndarray) -> np.ndarray:
    """Poids par classe (somme des combos)."""
    return np.bincount(COMBO_CLASS, weights=combo_weights, minlength=N_CLASSES)


def remove_dead(weights: np.ndarray, dead: list[int] | tuple[int, ...]) -> np.ndarray:
    """Met à zéro les combos contenant une carte morte."""
    if not dead:
        return weights
    out = np.array(weights, dtype=np.float64, copy=True)
    dead_arr = np.asarray(dead, dtype=np.int16)
    mask = np.isin(COMBO_CARDS[:, 0], dead_arr) | np.isin(COMBO_CARDS[:, 1], dead_arr)
    out[mask] = 0.0
    return out


def grid_matrix(values_by_class: np.ndarray) -> np.ndarray:
    """Vecteur (169,) -> matrice 13x13 (ligne/colonne 0 = As)."""
    return np.asarray(values_by_class, dtype=float).reshape(13, 13)


GRID_LABELS = [RANKS[12 - i] for i in range(13)]
