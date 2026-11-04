"""Résultats par case (stratégie x composition) et score de robustesse entre compositions.

Score de robustesse = la pire borne basse de l'IC 95 % du bb/100 sur toutes les compositions
mesurées : le gain qu'on peut garantir quelle que soit la table. Plus il est haut, plus la
stratégie est robuste. On publie aussi le rang moyen et la composition la plus défavorable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pokerlab.analysis.stats import summarize


def cell_summaries(hands: pd.DataFrame, confidence: float = 0.95) -> list[dict[str, Any]]:
    """bb/100 (EV et réel), IC, écart type et mains nécessaires pour chaque case d'un compare."""
    rows = []
    df = hands.copy()
    df["strategy"] = df["strategy"].astype(str)
    df["composition"] = df["composition"].astype(str)
    for (strategy, composition), h in df.groupby(["strategy", "composition"], sort=True):
        per = h.groupby("deck")[["result_ev", "result"]].mean()
        hpu = len(h) / max(len(per), 1)
        ev = summarize(per["result_ev"].to_numpy(), h["result_ev"].to_numpy(), hpu, confidence)
        real = summarize(per["result"].to_numpy(), h["result"].to_numpy(), hpu, confidence)
        rows.append({
            "strategy": strategy,
            "composition": composition,
            "bb100": ev.bb100,
            "ci": ev.ci_half,
            "bb100_reel": real.bb100,
            "ci_reel": real.ci_half,
            "sd_main_bb100": real.sd_hand_bb100,
            "mains": int(len(h)),
            "mains_necessaires_duplicate": ev.hands_needed,
            "mains_necessaires_reelles": _raw_hands_needed(h["result"].to_numpy(), confidence),
            "moyenne_main": float(h["result"].mean()),
            "ecart_type_main": float(h["result"].std(ddof=1)),
            "stats": _stats(h),
        })
    return rows


def _stats(h: pd.DataFrame) -> dict[str, float]:
    """VPIP, PFR, 3-bet, AF, WTSD mesurés (si les colonnes de compteurs sont présentes)."""
    if "f_hands" not in h.columns:
        return {}
    from pokerlab.analysis.player_stats import stats_from_hands

    st = stats_from_hands(h)
    return {k: float(st[k]) for k in ("VPIP", "PFR", "3bet", "AF", "WTSD", "W$SD", "cbet", "fold_cbet")}


def _raw_hands_needed(values, confidence: float) -> float:
    """Mains réelles (sans duplicate ni EV) pour que l'IC exclue 0 : (z σ / μ)²."""
    from pokerlab.analysis.stats import z_value

    mu = float(values.mean())
    sd = float(values.std(ddof=1))
    if mu == 0:
        return float("inf")
    return (z_value(confidence) * sd / abs(mu)) ** 2


def load_compare_cells(root: str | Path, exclude: Iterable[Path] = ()) -> pd.DataFrame:
    """Toutes les cases des compare trouvés sous ``root`` (resume.json avec « resultats »).

    Si une même case apparaît dans plusieurs campagnes, on garde celle qui a le plus de mains."""
    root = Path(root)
    skip = [Path(e).resolve() for e in exclude]
    rows: list[dict[str, Any]] = []
    for resume in sorted(root.glob("**/resume.json")):
        if any(e in resume.resolve().parents for e in skip):
            continue
        try:
            meta = json.loads(resume.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("experience") != "compare" or "resultats" not in meta:
            continue
        for row in meta["resultats"]:
            rows.append({**row, "source": str(resume.parent)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("mains", ascending=False)
    return df.drop_duplicates(["strategy", "composition"]).reset_index(drop=True)


def robustness_table(cells: pd.DataFrame, references: Iterable[str] | None = None) -> pd.DataFrame:
    """Robustesse par stratégie sur toutes les compositions mesurées, classée par pire cas.

    bb/100 médian et pire cas (plus faible bb/100 sur les compositions), plus faible borne basse
    d'IC 95 %, et « gagnante partout » : borne basse > 0 sur toutes les compositions mesurées."""
    if cells.empty:
        return pd.DataFrame()
    refs = set(reference_names() if references is None else references)
    df = cells.copy()
    df["borne_basse"] = df["bb100"] - df["ci"]
    df["rang"] = df.groupby("composition")["bb100"].rank(ascending=False, method="min")
    n_total = int(df["composition"].nunique())
    out = []
    for strategy, g in df.groupby("strategy"):
        worst = g.loc[g["bb100"].idxmin()]
        complete = int(g["composition"].nunique()) == n_total
        low = float(g["borne_basse"].min())
        out.append({
            "strategy": strategy,
            "compositions": int(g["composition"].nunique()),
            "bb100_median": float(g["bb100"].median()),
            "pire_cas": float(worst["bb100"]),
            "pire_cas_ci": float(worst["ci"]),
            "pire_composition": str(worst["composition"]),
            "borne_basse_min": low,
            "gagnante_partout": bool(complete and low > 0),
            "reference": strategy in refs,
            "rang_moyen": float(g["rang"].mean()),
            "bb100_moyen": float(g["bb100"].mean()),
        })
    return pd.DataFrame(out).sort_values("pire_cas", ascending=False).reset_index(drop=True)


def reference_names() -> set[str]:
    """Stratégies de référence ou témoins (``reference: true``) : jamais déclarées gagnantes."""
    from pokerlab.config import list_strategies
    from pokerlab.strategies.registry import resolve_spec

    return {n for n in list_strategies() if resolve_spec(n).get("reference")}


def winner_verdict(rob: pd.DataFrame, labels: dict[str, str] | None = None) -> tuple[str | None, str]:
    """Gagnante = stratégie (hors références) dont la borne basse d'IC est > 0 sur toutes les compositions ;
    s'il y en a plusieurs, la meilleure en pire cas. Sinon, phrase explicite : pas de gagnante."""
    if rob.empty:
        return None, "Aucun résultat : pas de verdict."
    labels = labels or {}
    lab = lambda s: labels.get(s, s)  # noqa: E731
    n = int(rob["compositions"].max())
    cand = rob[~rob["reference"]]
    ok = cand[cand["gagnante_partout"]]
    if not ok.empty:
        w = ok.iloc[0]
        return str(w["strategy"]), (f"Gagnante : {lab(w['strategy'])} — borne basse de l'IC 95 % > 0 sur les {n} composition(s) ; "
                                    f"pire cas {w['pire_cas']:+.1f} ± {w['pire_cas_ci']:.1f} bb/100 ({w['pire_composition']}), médiane {w['bb100_median']:+.1f}.")
    if cand.empty:
        return None, "PAS DE GAGNANTE : aucune stratégie candidate (seulement des références)."
    b = cand.iloc[0]
    return None, (f"PAS DE GAGNANTE : aucune stratégie n'a une borne basse d'IC 95 % > 0 sur les {n} composition(s). "
                  f"Meilleure en pire cas : {lab(b['strategy'])}, {b['pire_cas']:+.1f} ± {b['pire_cas_ci']:.1f} bb/100 sur {b['pire_composition']} "
                  f"(borne basse la plus faible {b['borne_basse_min']:+.1f}).")
