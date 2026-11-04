"""Lecture du tableau : sec / mouillé, couleurs, connexions, paires."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

BOARD_LABELS = ("préflop", "sec", "deux couleurs", "connecté", "apparié", "monotone")


@dataclass(frozen=True, slots=True)
class BoardTexture:
    """Caractéristiques d'un tableau (flop, turn ou river)."""

    n_cards: int
    max_suit: int
    paired: bool
    trips: bool
    connected: bool
    paired_wet: bool
    dry: bool
    label: str


def _connected(ranks: list[int], max_span: int) -> bool:
    """Vrai si 3 rangs distincts tiennent dans une fenêtre de ``max_span`` (As haut ou bas)."""
    distinct = sorted(set(ranks))
    variants = [distinct]
    if 12 in distinct:
        variants.append(sorted([-1 if r == 12 else r for r in distinct]))
    for values in variants:
        for i in range(len(values) - 2):
            if values[i + 2] - values[i] <= max_span:
                return True
    return False


def board_texture(board: Sequence[int], connected_max_span: int = 4, paired_wet_max_gap: int = 2) -> BoardTexture:
    """Analyse le tableau. Définition « sec » (règle bluff) :

    pas deux cartes de même couleur, pas trois rangs distincts dans un écart
    ≤ ``connected_max_span``, pas de paire au tableau « mouillée » (paire dont une autre
    carte est à ≤ ``paired_wet_max_gap`` rangs).
    """
    if not board:
        return BoardTexture(0, 0, False, False, False, False, False, "préflop")
    ranks = [c >> 2 for c in board]
    suits = [c & 3 for c in board]
    max_suit = max(suits.count(s) for s in range(4))
    counts = {r: ranks.count(r) for r in set(ranks)}
    paired = any(v >= 2 for v in counts.values())
    trips = any(v >= 3 for v in counts.values())
    connected = _connected(ranks, connected_max_span)
    paired_wet = False
    if paired:
        pair_ranks = [r for r, v in counts.items() if v >= 2]
        others = [r for r in counts if r not in pair_ranks]
        paired_wet = any(abs(p - o) <= paired_wet_max_gap for p in pair_ranks for o in others)
    dry = max_suit < 2 and not connected and not paired_wet
    if max_suit >= 3:
        label = "monotone"
    elif paired:
        label = "apparié"
    elif connected:
        label = "connecté"
    elif max_suit == 2:
        label = "deux couleurs"
    else:
        label = "sec"
    return BoardTexture(len(board), max_suit, paired, trips, connected, paired_wet, dry, label)
