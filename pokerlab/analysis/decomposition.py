"""Décomposition des gains : position, street, catégorie de main, showdown, ligne jouée."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pokerlab.cards.categories import CLASS_GROUP, PREFLOP_GROUPS
from pokerlab.cards.ranges import N_CLASSES
from pokerlab.engine.flags import LINE_LABELS, STREET_END_LABELS
from pokerlab.engine.types import position_names


def by_position(hands: pd.DataFrame, n_players: int = 6, column: str = "result_ev") -> pd.DataFrame:
    """bb/100 et mains par position (ordre BTN, CO, MP, UTG, SB, BB)."""
    order = list(reversed(position_names(n_players)[2:])) + ["SB", "BB"]
    g = hands.groupby("position")[column].agg(["mean", "count", "sum"])
    g = g.reindex([p for p in order if p in g.index])
    g["bb100"] = g["mean"] * 100
    return g.reset_index()


def by_street(hands: pd.DataFrame, column: str = "result_ev") -> pd.DataFrame:
    """Gains selon la street où la main s'est terminée pour le joueur."""
    g = hands.groupby("street_end")[column].agg(["sum", "count", "mean"]).reindex(range(5), fill_value=0)
    g["street"] = [STREET_END_LABELS[i] for i in g.index]
    g["bb_per_100_total"] = g["sum"] / max(len(hands), 1) * 100
    return g.reset_index(drop=True)


def by_class(hands: pd.DataFrame, column: str = "result_ev") -> tuple[np.ndarray, np.ndarray]:
    """(bb moyen par main, nombre de mains) pour les 169 classes."""
    sums = np.bincount(hands["hand_class"].to_numpy(dtype=int), weights=hands[column].to_numpy(dtype=float), minlength=N_CLASSES)
    counts = np.bincount(hands["hand_class"].to_numpy(dtype=int), minlength=N_CLASSES)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return means, counts


def by_group(hands: pd.DataFrame, column: str = "result_ev") -> pd.DataFrame:
    groups = pd.Series([CLASS_GROUP[c] for c in hands["hand_class"].to_numpy(dtype=int)], index=hands.index)
    g = hands.groupby(groups)[column].agg(["mean", "count", "sum"]).reindex(PREFLOP_GROUPS).fillna(0)
    g["bb_per_100_total"] = g["sum"] / max(len(hands), 1) * 100
    return g.reset_index(names="groupe")


def by_line(hands: pd.DataFrame, column: str = "result_ev") -> pd.DataFrame:
    """Value bet / bluff / semi-bluff / paie / check-abandon / préflop."""
    g = hands.groupby("line")[column].agg(["sum", "count", "mean"]).reindex(range(len(LINE_LABELS)), fill_value=0)
    g["ligne"] = list(LINE_LABELS)
    g["bb_per_100_total"] = g["sum"] / max(len(hands), 1) * 100
    return g.reset_index(drop=True)


def showdown_split(hands: pd.DataFrame, column: str = "result") -> tuple[np.ndarray, np.ndarray]:
    """Cumuls des gains avec showdown (ligne bleue) et sans showdown (ligne rouge)."""
    values = hands[column].to_numpy(dtype=float)
    sd = hands["showdown"].to_numpy(dtype=bool)
    return np.cumsum(np.where(sd, values, 0.0)), np.cumsum(np.where(sd, 0.0, values))
