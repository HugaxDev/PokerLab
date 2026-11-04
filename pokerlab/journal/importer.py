"""Import du journal de mains réel (CSV) et comparaison avec la stratégie simulée.

Schéma CSV (une ligne par main, séparateur virgule ou point-virgule) :
``phase, main, cartes, assorties, position, mise_totale, board, action_resume, tapis_apres, estime, hors_regle``
Les lignes commençant par ``#`` sont ignorées. Montants dans la monnaie de la table (ex. 0.37).
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.cards.cards import RANK_INDEX
from pokerlab.cards.categories import CLASS_GROUP
from pokerlab.cards.ranges import CLASS_NAMES, class_index

COLUMNS = ("phase", "main", "cartes", "assorties", "position", "mise_totale", "board", "action_resume", "tapis_apres", "estime", "hors_regle")
YES = {"oui", "o", "yes", "y", "true", "vrai", "1", "x", "s"}
POSITION_ALIASES = {"BU": "BTN", "BOUTON": "BTN", "D": "BTN", "HJ": "MP", "LJ": "UTG", "UTG+1": "MP", "PB": "SB", "GB": "BB"}
RAISE_WORDS = ("relance", "raise", "ouvre", "open", "3bet", "3-bet", "4bet", "4-bet", "tapis", "all-in", "allin", "shove", "mise", "bet")
CALL_WORDS = ("paie", "paye", "call", "suit", "limp", "complète", "complete")
REBUY = re.compile(r"recave\s*([0-9]+(?:[.,][0-9]+)?)", re.IGNORECASE)


def _yes(value: str) -> bool:
    return str(value).strip().lower() in YES


def _num(value: str) -> float:
    text = str(value).strip().replace("€", "").replace("$", "").replace(" ", "").replace(",", ".")
    return float(text) if text else float("nan")


def parse_hole(cards: str, suited_flag: str) -> tuple[int, int, bool]:
    """'AK' + assorties, ou cartes exactes 'AsKd'. Renvoie (rang haut, rang bas, assorties)."""
    text = cards.strip().upper().replace("10", "T").replace(" ", "")
    ranks = [ch for ch in text if ch in RANK_INDEX]
    suits = [ch.lower() for ch in text if ch.lower() in "cdhs" and ch not in RANK_INDEX]
    if len(ranks) != 2:
        raise ValueError(f"Cartes illisibles : {cards!r}")
    r1, r2 = RANK_INDEX[ranks[0]], RANK_INDEX[ranks[1]]
    suited = (len(suits) == 2 and suits[0] == suits[1]) if len(suits) == 2 else _yes(suited_flag)
    if r1 == r2:
        suited = False
    return max(r1, r2), min(r1, r2), suited


FACING_WORDS = ("sur relance", "sur une relance", "face à", "face a ", "contre", "3bet", "3-bet", "4bet", "4-bet",
                "paie", "paye", "payé", "call", "suit")
FOLD_WORDS = ("couch", "fold")


def raw_preflop(summary: str) -> str:
    text = summary.lower()
    for sep in ("|", "flop", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return text


def preflop_situation(summary: str) -> str:
    """'vs_raise' si le résumé préflop indique une relance devant (paie, 3bet, « sur relance »...), sinon 'unopened'."""
    raw = raw_preflop(summary)
    return "vs_raise" if any(w in raw for w in FACING_WORDS) else "unopened"


def preflop_played(summary: str) -> str:
    """Action préflop jouée : 'relance', 'paie', 'couché' ou 'check'."""
    seg = preflop_segment(summary)
    if any(w in seg for w in RAISE_WORDS):
        return "relance"
    if any(w in seg for w in CALL_WORDS):
        return "paie"
    if any(w in seg for w in FOLD_WORDS):
        return "couché"
    return "check"


CONTEXT = re.compile(r"\b(sur|face (à|a)|contre|après|apres)\s+(une?\s+)?\S+")


def preflop_segment(summary: str) -> str:
    """Partie préflop du résumé, sans les mentions de l'action adverse (« couché sur relance »)."""
    text = summary.lower()
    for sep in ("|", "flop", ";"):
        if sep in text:
            text = text.split(sep, 1)[0]
    return CONTEXT.sub(" ", text)


@dataclass
class Journal:
    """Journal importé (une ligne par main) avec résultats en bb."""

    hands: pd.DataFrame
    phases: dict[str, dict[str, Any]]


def load_journal(csv_path: str | Path, phases: dict[str, dict[str, Any]]) -> Journal:
    """Lit le CSV, calcule le résultat de chaque main (variation de tapis, recaves déduites)."""
    raw = Path(csv_path).read_text(encoding="utf-8-sig")
    lines = [ln for ln in raw.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    delimiter = ";" if lines and lines[0].count(";") > lines[0].count(",") else ","
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter)
    missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le journal : {', '.join(missing)}")
    rows = []
    for rec in reader:
        phase = str(rec["phase"]).strip()
        meta = phases.get(phase, {})
        bb = float(meta.get("big_blind", 0.02))
        hi, lo, suited = parse_hole(rec["cartes"], rec["assorties"])
        cls = class_index(hi, lo, suited)
        pos = rec["position"].strip().upper()
        pos = POSITION_ALIASES.get(pos, pos)
        pre = preflop_segment(rec["action_resume"])
        raised = any(w in pre for w in RAISE_WORDS)
        called = any(w in pre for w in CALL_WORDS)
        invested = _num(rec["mise_totale"])
        blind = {"SB": float(meta.get("small_blind", bb / 2)), "BB": bb}.get(pos, 0.0)
        vpip = raised or called or (not np.isnan(invested) and invested > blind + 1e-9 and "couch" not in pre and "fold" not in pre)
        rebuy = sum(_num(m) for m in REBUY.findall(rec["action_resume"]))
        rows.append({
            "phase": phase,
            "strategie": meta.get("strategie"),
            "main": int(_num(rec["main"])),
            "classe": CLASS_NAMES[cls],
            "hand_class": cls,
            "groupe": CLASS_GROUP[cls],
            "position": pos,
            "mise_totale": invested,
            "board": rec["board"].strip(),
            "action_resume": rec["action_resume"].strip(),
            "tapis_apres": _num(rec["tapis_apres"]),
            "recave": rebuy,
            "estime": _yes(rec["estime"]),
            "hors_regle": _yes(rec["hors_regle"]),
            "vpip": bool(vpip),
            "pfr": bool(raised),
            "situation": preflop_situation(rec["action_resume"]),
            "joue": preflop_played(rec["action_resume"]),
            "bb_value": bb,
        })
    df = pd.DataFrame(rows).sort_values(["phase", "main"]).reset_index(drop=True)
    results = []
    for phase, g in df.groupby("phase", sort=False):
        meta = phases.get(phase, {})
        start = float(meta.get("tapis_depart", 100 * float(meta.get("big_blind", 0.02))))
        prev = start
        for _, row in g.iterrows():
            delta = row["tapis_apres"] - prev - row["recave"]
            results.append(delta / row["bb_value"])
            prev = row["tapis_apres"]
    df["result_bb"] = results
    return Journal(df, phases)


PLAYED_CODE = {"relance": 2, "paie": 1, "check": 1, "couché": 0}
CODE_LABEL = {2: "relance", 1: "paie / check", 0: "couché"}


def flag_off_sheet(df: pd.DataFrame, charts: dict[str, dict]) -> pd.DataFrame:
    """Ajoute 'fiche' (action prévue) et 'hors_fiche' en comparant l'action préflop à la fiche de la phase.

    ``charts`` : stratégie -> grilles préflop (voir ``pokerlab.fiche``). Les mains sans fiche
    (phase sans stratégie, UTG face à une relance) restent non évaluées (``hors_fiche`` = None)."""
    out = df.copy()
    expected, off = [], []
    for _, row in out.iterrows():
        chart = charts.get(row.get("strategie") or "")
        codes = chart.get((row["position"], row["situation"])) if chart else None
        if codes is None:
            expected.append("non évaluable")
            off.append(None)
            continue
        want = int(codes[int(row["hand_class"])])
        expected.append(CODE_LABEL[want])
        off.append(PLAYED_CODE[row["joue"]] != want)
    out["fiche"] = expected
    out["hors_fiche"] = off
    return out


def journal_stats(df: pd.DataFrame) -> pd.DataFrame:
    """VPIP, PFR, résultats et mains hors règle par phase."""
    rows = []
    for phase, g in df.groupby("phase", sort=False):
        rows.append({
            "phase": phase,
            "mains": len(g),
            "VPIP %": 100 * g["vpip"].mean(),
            "PFR %": 100 * g["pfr"].mean(),
            "résultat (bb)": g["result_bb"].sum(),
            "bb/100": 100 * g["result_bb"].mean(),
            "mains hors règle": int(g["hors_regle"].sum()),
            "résultat hors règle (bb)": g.loc[g["hors_regle"], "result_bb"].sum(),
            "montants estimés": int(g["estime"].sum()),
            "hors fiche": int(g["hors_fiche"].fillna(False).astype(bool).sum()) if "hors_fiche" in g else 0,
        })
    return pd.DataFrame(rows)


def window_distribution(sim_hands: pd.DataFrame, strategy: str, n: int) -> np.ndarray:
    """Distribution des résultats sur ``n`` mains consécutives simulées (sessions)."""
    h = sim_hands[sim_hands["strategy"] == strategy].sort_values(["session", "hand_no"])
    values = h["result"].to_numpy(dtype=float)
    if values.size < n or n <= 0:
        return np.array([])
    usable = values[: values.size // n * n].reshape(-1, n)
    return usable.sum(axis=1)


def percentile_of(value: float, dist: np.ndarray) -> float:
    if dist.size == 0:
        return float("nan")
    return 100.0 * float((dist < value).mean() + 0.5 * (dist == value).mean())
