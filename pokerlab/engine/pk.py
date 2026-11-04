"""Adaptateur PokerKit : création d'état NLHE et mémoïsation de l'évaluation des mains.

Les règles (actions légales, pots, side pots, showdown) restent celles de PokerKit.
``MemoHighHand`` ne change pas l'évaluation : il mémorise le résultat de
``StandardHighHand.from_game`` (même ensemble de 7 cartes = même main), car PokerKit
réévalue plusieurs fois les mêmes mains pendant un showdown.
"""

from __future__ import annotations

from pokerkit import Automation, Card, Mode, NoLimitTexasHoldem, StandardHighHand, State

AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
)

_MEMO_LIMIT = 200_000


class MemoHighHand(StandardHighHand):
    """``StandardHighHand`` avec cache du meilleur 5 cartes et de l'entrée de lookup."""

    _memo: dict = {}

    @classmethod
    def from_game(cls, hole_cards, board_cards=()):  # type: ignore[override]
        cards = tuple(Card.clean(hole_cards)) + tuple(Card.clean(board_cards))
        key = frozenset(cards)
        hand = cls._memo.get(key)
        if hand is None:
            if len(cls._memo) > _MEMO_LIMIT:
                cls._memo.clear()
            hand = super().from_game(cards, ())
            cls._memo[key] = hand
        return hand

    @property
    def entry(self):  # type: ignore[override]
        cached = self.__dict__.get("_entry")
        if cached is None:
            cached = self.lookup.get_entry(self.cards)
            self.__dict__["_entry"] = cached
        return cached


class MemoNoLimitTexasHoldem(NoLimitTexasHoldem):
    """NLHE PokerKit utilisant ``MemoHighHand``."""

    hand_types = (MemoHighHand,)


def create_state(small_blind: int, big_blind: int, stacks: list[int], memoize: bool = True) -> State:
    """Crée un état NLHE (mode tournoi PokerKit = un seul déroulé en cas de tapis)."""
    game = MemoNoLimitTexasHoldem if memoize else NoLimitTexasHoldem
    return game.create_state(
        AUTOMATIONS,
        True,
        0,
        (small_blind, big_blind),
        big_blind,
        tuple(stacks),
        len(stacks),
        mode=Mode.TOURNAMENT,
    )
