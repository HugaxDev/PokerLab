"""Encodage des cartes : entier 0..51 = rang * 4 + couleur (rang 0 = 2, 12 = As)."""

from __future__ import annotations

from typing import Iterable, Sequence

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}
SUIT_INDEX = {s: i for i, s in enumerate(SUITS)}
SUIT_SYMBOLS = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}

N_CARDS = 52
CARD_STR = [RANKS[c >> 2] + SUITS[c & 3] for c in range(N_CARDS)]


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def make_card(rank: int, suit: int) -> int:
    return (rank << 2) | suit


def parse_card(text: str) -> int:
    """'As' -> entier. Accepte aussi '10s'."""
    text = text.strip()
    if text.startswith("10"):
        text = "T" + text[2:]
    if len(text) != 2:
        raise ValueError(f"Carte invalide : {text!r}")
    rank, suit = text[0].upper(), text[1].lower()
    if rank not in RANK_INDEX or suit not in SUIT_INDEX:
        raise ValueError(f"Carte invalide : {text!r}")
    return make_card(RANK_INDEX[rank], SUIT_INDEX[suit])


def parse_cards(text: str) -> list[int]:
    """'AsKd 7h' ou 'AsKd7h' -> liste d'entiers."""
    compact = text.replace(" ", "").replace(",", "")
    if len(compact) % 2:
        raise ValueError(f"Suite de cartes invalide : {text!r}")
    return [parse_card(compact[i:i + 2]) for i in range(0, len(compact), 2)]


def cards_str(cards: Iterable[int], sep: str = "") -> str:
    return sep.join(CARD_STR[c] for c in cards)


_PK_CARDS: list | None = None
_PK_INDEX: dict | None = None


def pokerkit_cards() -> list:
    """Table entier -> ``pokerkit.Card`` (construite une seule fois)."""
    global _PK_CARDS, _PK_INDEX
    if _PK_CARDS is None:
        from pokerkit import Card

        _PK_CARDS = [next(iter(Card.parse(CARD_STR[c]))) for c in range(N_CARDS)]
        _PK_INDEX = {card: i for i, card in enumerate(_PK_CARDS)}
    return _PK_CARDS


def from_pokerkit(card: object) -> int:
    pokerkit_cards()
    assert _PK_INDEX is not None
    return _PK_INDEX[card]


def to_pokerkit(cards: Sequence[int]) -> list:
    table = pokerkit_cards()
    return [table[c] for c in cards]
