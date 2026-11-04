"""Force de la main par rapport au tableau : niveau de main faite et tirages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pokerlab.equity.evaluator import (
    FLUSH,
    PAIR,
    STRAIGHT,
    TRIPS,
    TWO_PAIR,
    evaluate,
    popcount,
    straight_out_ranks,
)

MADE_LEVELS = (
    "nothing",
    "weak_pair",
    "middle_pair",
    "top_pair",
    "overpair",
    "two_pair",
    "trips",
    "straight",
    "flush",
    "full_house",
)
MADE_LABELS = (
    "rien",
    "petite paire",
    "paire moyenne",
    "top paire",
    "overpair",
    "deux paires",
    "brelan",
    "quinte",
    "couleur",
    "full ou mieux",
)
LEVEL = {name: i for i, name in enumerate(MADE_LEVELS)}
LEVEL_ALIASES = {"pair": LEVEL["weak_pair"], "any_pair": LEVEL["weak_pair"], "set": LEVEL["trips"]}


def level_of(name: str | int) -> int:
    """Convertit un nom de niveau (YAML) en entier."""
    if isinstance(name, int):
        return name
    if name in LEVEL:
        return LEVEL[name]
    if name in LEVEL_ALIASES:
        return LEVEL_ALIASES[name]
    raise ValueError(f"Niveau de main inconnu : {name!r} (attendu : {', '.join(MADE_LEVELS)})")


@dataclass(frozen=True, slots=True)
class HandRead:
    """Lecture d'une main postflop."""

    level: int
    category: int
    flush_draw: bool
    straight_draw: int
    outs: int
    kicker: int

    @property
    def strong_draw(self) -> bool:
        """Tirage couleur ou quinte ouverte (bilatérale ou double ventrale)."""
        return self.flush_draw or self.straight_draw >= 2

    @property
    def any_draw(self) -> bool:
        return self.flush_draw or self.straight_draw >= 1

    @property
    def label(self) -> str:
        base = MADE_LABELS[self.level]
        if self.level == 0 and self.any_draw:
            return "tirage couleur" if self.flush_draw else ("tirage quinte" if self.straight_draw >= 2 else "ventrale")
        return base


def _pocket_level(rank: int, board_ranks: list[int]) -> int:
    if rank > board_ranks[0]:
        return LEVEL["overpair"]
    if len(board_ranks) > 1 and rank > board_ranks[1]:
        return LEVEL["middle_pair"]
    return LEVEL["weak_pair"]


def made_level(hole: Sequence[int], board: Sequence[int], score: int | None = None) -> tuple[int, int, int]:
    """(niveau, catégorie évaluateur, kicker) d'une main sur un tableau de 3 à 5 cartes."""
    cards = list(hole) + list(board)
    score = evaluate(cards) if score is None else score
    cat = score >> 20
    h1, h2 = hole[0] >> 2, hole[1] >> 2
    board_ranks = sorted({c >> 2 for c in board}, reverse=True)
    if cat >= STRAIGHT:
        board_score = evaluate(board)
        if cat > board_score >> 20 or score > board_score:
            if cat == STRAIGHT:
                return LEVEL["straight"], cat, -1
            if cat == FLUSH:
                return LEVEL["flush"], cat, -1
            return LEVEL["full_house"], cat, -1
        cat = board_score >> 20
    if cat == TRIPS:
        t = (score >> 16) & 15
        return (LEVEL["trips"] if t in (h1, h2) else LEVEL["nothing"]), cat, -1
    if cat in (PAIR, TWO_PAIR) or cat >= STRAIGHT:
        if h1 == h2:
            return _pocket_level(h1, board_ranks), cat, -1
        matched = [r for r in (h1, h2) if r in board_ranks]
        if len(matched) == 2:
            return LEVEL["two_pair"], cat, -1
        if len(matched) == 1:
            r = matched[0]
            kicker = h2 if r == h1 else h1
            idx = board_ranks.index(r)
            if idx == 0:
                return LEVEL["top_pair"], cat, kicker
            return (LEVEL["middle_pair"] if idx == 1 else LEVEL["weak_pair"]), cat, kicker
    return LEVEL["nothing"], cat, -1


def read_hand(hole: Sequence[int], board: Sequence[int]) -> HandRead:
    """Niveau de main faite + tirages (tirages seulement au flop et au turn)."""
    level, cat, kicker = made_level(hole, board)
    flush_draw = False
    straight_draw = 0
    if len(board) in (3, 4):
        cards = list(hole) + list(board)
        if cat < FLUSH:
            for s in range(4):
                if sum(1 for c in cards if c & 3 == s) == 4 and any(c & 3 == s for c in hole):
                    flush_draw = True
                    break
        if cat < STRAIGHT:
            m_all = 0
            for c in cards:
                m_all |= 1 << (c >> 2)
            m_board = 0
            for c in board:
                m_board |= 1 << (c >> 2)
            outs_ranks = straight_out_ranks(m_all) & ~straight_out_ranks(m_board)
            n = popcount(outs_ranks)
            straight_draw = 2 if n >= 2 else (1 if n == 1 else 0)
    outs = 0
    if flush_draw:
        outs += 9
    if straight_draw == 2:
        outs += 8 if not flush_draw else 6
    elif straight_draw == 1:
        outs += 4 if not flush_draw else 3
    return HandRead(level, cat, flush_draw, straight_draw, outs, kicker)


def draw_equity(outs: int, board_len: int, two_cards: bool = False) -> float:
    """Probabilité de toucher (règle des outs) avec une ou deux cartes à venir."""
    if outs <= 0 or board_len >= 5:
        return 0.0
    unseen = 52 - 2 - board_len
    one = outs / unseen
    if two_cards and board_len == 3:
        return 1.0 - (1.0 - one) * (1.0 - outs / (unseen - 1))
    return one
