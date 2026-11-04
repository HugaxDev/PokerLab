"""Catégories de mains préflop (groupes lisibles pour la décomposition des gains)."""

from __future__ import annotations

from pokerlab.cards.ranges import CLASS_NAMES, N_CLASSES, class_is_pair, class_is_suited, class_ranks

PREFLOP_GROUPS = (
    "paire haute (TT+)",
    "paire moyenne (66-99)",
    "petite paire (22-55)",
    "As fort (AT+)",
    "As faible (A2-A9)",
    "figures (deux cartes ≥ T)",
    "assorties connectées/à un trou",
    "roi assorti",
    "autres",
)


def preflop_group(index: int) -> str:
    """Groupe préflop d'une classe 0..168."""
    hi, lo = class_ranks(index)
    if class_is_pair(index):
        if hi >= 8:
            return PREFLOP_GROUPS[0]
        if hi >= 4:
            return PREFLOP_GROUPS[1]
        return PREFLOP_GROUPS[2]
    if hi == 12:
        return PREFLOP_GROUPS[3] if lo >= 8 else PREFLOP_GROUPS[4]
    if lo >= 8:
        return PREFLOP_GROUPS[5]
    if class_is_suited(index) and hi - lo <= 2:
        return PREFLOP_GROUPS[6]
    if class_is_suited(index) and hi == 11:
        return PREFLOP_GROUPS[7]
    return PREFLOP_GROUPS[8]


CLASS_GROUP = [preflop_group(i) for i in range(N_CLASSES)]

__all__ = ["PREFLOP_GROUPS", "preflop_group", "CLASS_GROUP", "CLASS_NAMES"]
