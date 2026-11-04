"""Détecteur de fuites : situations récurrentes où continuer la main a une EV négative.

Pour chaque décision active (paie, mise, relance), on mesure le gain all-in adjusted
*à partir de cette décision* (résultat final + mise déjà investie). Si la moyenne d'une
situation est négative et que l'IC à 95 % reste sous 0, abandonner la main à ce moment
aurait coûté moins cher : c'est une fuite.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from pokerlab.cards.categories import CLASS_GROUP
from pokerlab.engine.types import KIND_LABELS, STREET_NAMES, Kind
from pokerlab.features.hand import MADE_LABELS

FACING_LABELS = ("rien à payer", "face à une mise", "face à une relance", "face à un tapis", "face à une sur-relance")
PREFLOP_FACING_LABELS = ("rien à payer", "sans relance devant", "face à une relance", "face à un tapis", "face à une sur-relance")


def facing_label(street: int, code: int) -> str:
    """Libellé de ce qu'il y a à payer (préflop : les blinds seules = pas de relance devant)."""
    labels = PREFLOP_FACING_LABELS if street == 0 else FACING_LABELS
    return labels[min(int(code), len(labels) - 1)]


DRAW_LABELS = ("", " + tirage faible", " + tirage")


def situation_labels(dec: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les libellés de situation (street, position, main, tableau, face à, action)."""
    if "hand_label" in dec.columns and "net_after" in dec.columns:
        return dec
    out = dec.copy()
    out["street_label"] = [STREET_NAMES[s] for s in out["street"].to_numpy(dtype=int)]
    hand_labels = []
    for street, cls, level, draw in zip(out["street"], out["hand_class"], out["level"], out["draw"]):
        if street == 0:
            hand_labels.append(CLASS_GROUP[int(cls)])
        else:
            hand_labels.append(MADE_LABELS[int(level)] + DRAW_LABELS[int(draw)])
    out["hand_label"] = hand_labels
    out["facing_label"] = [facing_label(int(st), int(f)) for st, f in zip(out["street"], out["facing"])]
    out["action_label"] = [KIND_LABELS[Kind(int(k))] for k in out["kind"]]
    out["net_after"] = out["result_ev"] + out["invested_bb"]
    return out


GROUP_KEYS = ("street_label", "position", "hand_label", "board_label", "facing_label", "action_label")


def find_leaks(dec: pd.DataFrame, min_count: int = 30, confidence_z: float = 1.96, top: int = 20) -> pd.DataFrame:
    """Situations à EV négative (actions paie/mise/relance), classées par perte totale."""
    if dec.empty:
        return pd.DataFrame()
    d = situation_labels(dec)
    d = d[d["kind"].isin([int(Kind.CALL), int(Kind.BET), int(Kind.RAISE)])]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby(list(GROUP_KEYS))["net_after"].agg(["count", "mean", "std", "sum"]).reset_index()
    g = g[g["count"] >= min_count]
    if g.empty:
        return g
    g["ci_half"] = confidence_z * g["std"].fillna(0) / np.sqrt(g["count"])
    g["upper"] = g["mean"] + g["ci_half"]
    g["significant"] = g["upper"] < 0
    leaks = g[g["mean"] < 0].sort_values("sum").head(top).reset_index(drop=True)
    return leaks


def leak_examples(dec: pd.DataFrame, leak: pd.Series, k: int = 3) -> pd.DataFrame:
    """Les ``k`` décisions les plus coûteuses de la situation (clés de rejeu)."""
    d = situation_labels(dec)
    mask = np.ones(len(d), dtype=bool)
    for key in GROUP_KEYS:
        mask &= (d[key] == leak[key]).to_numpy()
    return d[mask].sort_values("net_after").head(k)


def leak_title(row: pd.Series) -> str:
    return f"{row['street_label']} · {row['position']} · {row['hand_label']} · {row['board_label']} · {row['facing_label']} → {row['action_label']}"


def format_ci(mean: float, half: float) -> str:
    if math.isinf(half) or math.isnan(half):
        return f"{mean:+.2f}"
    return f"{mean:+.2f} ± {half:.2f}"
