"""Population adverse réelle : stats par joueur (historiques PokerStars) et composition « population_reelle ».

1. À l'import, chaque adversaire de chaque main donne une ligne de compteurs (VPIP, PFR, 3-bet, AF,
   WTSD, fold to c-bet...) : ``data/reel/adversaires.parquet``.
2. ``python -m pokerlab population`` regroupe les joueurs assez observés en quelques profils (k-moyennes
   pondérées par le nombre de mains, graine fixe), plus un profil « occasionnels » pour les autres,
   traduit chaque profil en paramètres d'archétype, puis les calibre par simulation : on joue la
   population, on mesure ses stats, on corrige les paramètres, jusqu'à coller aux stats réelles.
3. Écrit ``configs/opponents/population_reelle.yaml`` (archétypes) et ``configs/tables/population_reelle.yaml``
   (composition, poids = part des mains observées) : utilisable dans compare comme les autres.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from pokerlab.analysis.player_stats import stats_from_flags
from pokerlab.config import LabConfig, configs_root
from pokerlab.engine.flags import FLAG_NAMES, seat_flags
from pokerlab.opponents.archetype import PARAM_DEFAULTS

STAT_KEYS = ("VPIP", "PFR", "3bet", "AF", "WTSD", "fold_cbet")
FLAG_COLS = ["f_" + k for k in FLAG_NAMES]
# paramètre d'archétype -> (stat visée, exposant de correction)
CALIBRATION = (("vpip", "VPIP", 1.0), ("pfr", "PFR", 1.0), ("three_bet", "3bet", 1.0), ("aggression", "AF", 0.5),
               ("call_weak", "WTSD", 1.0), ("fold_to_cbet", "fold_cbet", 1.0))
BOUNDS = {"vpip": (0.03, 0.95), "pfr": (0.01, 0.9), "three_bet": (0.003, 0.5), "aggression": (0.05, 0.95),
          "call_weak": (0.02, 0.98), "fold_to_cbet": (0.02, 0.98)}


# ------------------------------------------------------------------------------ import
def opponent_rows(hand_id: str, date: Any, hr: Any, hero_index: int) -> list[dict[str, Any]]:
    """Une ligne de compteurs par adversaire de la main (le héros est exclu)."""
    rows = []
    for i, name in enumerate(hr.names):
        if i == hero_index:
            continue
        row: dict[str, Any] = {"hand_id": str(hand_id), "joueur": str(name), "date": date}
        row.update({c: int(v) for c, v in zip(FLAG_COLS, seat_flags(hr, i))})
        rows.append(row)
    return rows


def merge_opponents(old: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Ajoute les lignes, sans doublon (même main, même joueur) : réimporter un fichier ne compte pas deux fois."""
    new = pd.DataFrame(new_rows)
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    if merged.empty:
        return merged
    return merged.drop_duplicates(["hand_id", "joueur"], keep="last").reset_index(drop=True)


def player_stats(rows: pd.DataFrame) -> pd.DataFrame:
    """Stats par joueur adverse : mains, VPIP, PFR, 3-bet, AF, WTSD, fold to c-bet (et compteurs bruts)."""
    if rows.empty:
        return pd.DataFrame(columns=["joueur", "mains", *STAT_KEYS])
    sums = rows.groupby("joueur")[FLAG_COLS].sum()
    out = []
    for name, f in sums.iterrows():
        st = stats_from_flags([float(f[c]) for c in FLAG_COLS])
        out.append({"joueur": name, "mains": int(f["f_hands"]), **{k: st[k] for k in STAT_KEYS}, "cbet": st["cbet"],
                    **{c: int(f[c]) for c in FLAG_COLS}})
    return pd.DataFrame(out).sort_values(["mains", "joueur"], ascending=[False, True]).reset_index(drop=True)


# --------------------------------------------------------------------------- profils
def _pooled(df: pd.DataFrame) -> dict[str, float]:
    st = stats_from_flags([float(df[c].sum()) for c in FLAG_COLS])
    return {k: st[k] for k in (*STAT_KEYS, "cbet")}


def _features(players: pd.DataFrame, pooled: dict[str, float]) -> np.ndarray:
    x = players[list(STAT_KEYS)].astype(float).copy()
    x["AF"] = x["AF"].clip(upper=6.0)
    for k in STAT_KEYS:
        fill = pooled[k] if math.isfinite(pooled[k]) else 0.0
        x[k] = x[k].fillna(min(fill, 6.0) if k == "AF" else fill).replace([np.inf, -np.inf], 6.0 if k == "AF" else fill)
    return x.to_numpy()


def kmeans(x: np.ndarray, w: np.ndarray, k: int, seed: int, iters: int = 100) -> np.ndarray:
    """k-moyennes pondérées, initialisation k-means++ à graine fixe ; renvoie l'étiquette de chaque point."""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[int(rng.choice(n, p=w / w.sum()))]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centers], axis=0) * w
        centers.append(x[int(rng.choice(n, p=d2 / d2.sum()))] if d2.sum() > 0 else x[int(rng.integers(n))])
    c = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        new = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(axis=2), axis=1)
        if _ and np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                c[j] = np.average(x[m], axis=0, weights=w[m])
    return labels


def build_profiles(players: pd.DataFrame, k: int = 5, min_hands: int = 50, seed: int = 1) -> list[dict[str, Any]]:
    """Profils de la population : joueurs observés sur ``min_hands`` mains ou plus regroupés en ``k`` profils,
    les autres réunis dans un profil « occasionnels ». Chaque profil : stats cibles (mains mises en commun),
    part des mains, nombre de joueurs, dispersion du VPIP entre joueurs."""
    if players.empty:
        return []
    total = float(players["mains"].sum())
    pooled = _pooled(players)
    regs = players[players["mains"] >= min_hands].reset_index(drop=True)
    rest = players[players["mains"] < min_hands]
    groups: list[tuple[str, pd.DataFrame]] = []
    if len(regs):
        kk = max(1, min(k, len(regs)))
        x = _features(regs, pooled)
        scale = x.std(axis=0)
        scale[scale == 0] = 1.0
        labels = kmeans(x / scale, regs["mains"].to_numpy(dtype=float), kk, seed) if kk > 1 else np.zeros(len(regs), dtype=int)
        for j in sorted(set(labels), key=lambda j: -regs.loc[labels == j, "mains"].sum()):
            groups.append(("reguliers", regs[labels == j]))
    if len(rest):
        groups.append(("occasionnels", rest))
    out = []
    for idx, (kind, g) in enumerate(groups, 1):
        target = _pooled(g)
        vp = g["VPIP"].astype(float)
        spread = float(np.average(np.abs(vp - vp.mean()), weights=g["mains"])) / max(float(vp.mean()), 1.0) if len(g) > 1 else 0.15
        out.append({
            "name": f"pop_{idx}",
            "type": kind,
            "joueurs": int(len(g)),
            "mains": int(g["mains"].sum()),
            "part": float(g["mains"].sum() / total),
            "cibles": {k2: _round(v) for k2, v in target.items()},
            "ecart": float(np.clip(spread, 0.05, 0.4)),
        })
    return out


def _round(v: float, d: int = 2) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), d)


def initial_params(t: dict[str, float | None]) -> dict[str, Any]:
    """Première traduction stats -> paramètres d'archétype (corrigée ensuite par calibrage)."""
    def g(key: str, default: float) -> float:
        v = t.get(key)
        return default if v is None or not math.isfinite(v) else float(v)

    vpip = np.clip(g("VPIP", 25) / 100, 0.03, 0.95)
    pfr = np.clip(g("PFR", 15) / 100, 0.01, vpip)
    af = g("AF", 2.0)
    wtsd = g("WTSD", 28)
    return {
        "vpip": round(float(vpip), 4),
        "pfr": round(float(pfr), 4),
        "three_bet": round(float(np.clip(g("3bet", 6) / 100, 0.003, 0.5)), 4),
        "limp": bool(vpip - pfr > 0.15),
        "aggression": round(float(np.clip(af / (af + 1.5), 0.1, 0.9)), 4),
        "bluff": round(float(np.clip(0.04 * af, 0.01, 0.35)), 4),
        "cbet": round(float(np.clip(g("cbet", 60) / 100, 0.1, 0.95)), 4),
        "fold_to_cbet": round(float(np.clip(g("fold_cbet", 45) / 100 / 0.75, 0.05, 0.95)), 4),
        "call_weak": round(float(np.clip(0.45 * wtsd / 28, 0.05, 0.95)), 4),
    }


def archetype_spec(profile: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    c = profile["cibles"]
    label = (f"Population réelle {profile['name'][4:]} ({'occasionnels' if profile['type'] == 'occasionnels' else 'réguliers'} ; "
             f"VPIP {c.get('VPIP') or 0:.0f} / PFR {c.get('PFR') or 0:.0f}, {profile['joueurs']} joueur(s))")
    spec = {"name": profile["name"], "label": label, "type": "archetype", **params,
            "humain": {"tirage": {"enabled": True, "ecart": round(profile["ecart"], 3)}, "mixte": {"enabled": True}}}
    return spec


# ------------------------------------------------------------------------- calibrage
def measure(lab: LabConfig, specs: dict[str, dict[str, Any]], decks: int, offset: int, hero: str = "equity_bot") -> dict[str, dict[str, float]]:
    """Joue la population (poids égaux, pour mesurer chaque profil) et renvoie les stats mesurées par profil."""
    from pokerlab.experiments.common import DuplicateTask, run_duplicate_task
    from pokerlab.strategies.registry import register_archetypes, resolve_spec

    register_archetypes(specs)
    comp = {"name": "population_calibrage", "mode": "sample", "session_decks": 10, "weights": {n: 1.0 for n in specs}}
    out = run_duplicate_task(DuplicateTask(lab.raw, resolve_spec(hero), comp, offset, offset + decks, {"strategy": hero},
                                           record_decisions=False, deck_space="calibrage"))
    return {n: stats_from_flags(out.flags_by_name[n]) for n in specs if n in out.flags_by_name}


def calibrate(profiles: list[dict[str, Any]], lab: LabConfig, iterations: int = 4, decks: int = 300,
              log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Ajuste les paramètres de chaque profil pour que ses stats simulées collent aux stats réelles."""
    params = {p["name"]: initial_params(p["cibles"]) for p in profiles}
    measured: dict[str, dict[str, float]] = {}
    for it in range(iterations + 1):
        specs = {p["name"]: archetype_spec(p, params[p["name"]]) for p in profiles}
        measured = measure(lab, specs, decks, it * decks)
        err = []
        for p in profiles:
            m, t = measured.get(p["name"], {}), p["cibles"]
            err += [abs(m[s] - t[s]) for _, s, _ in CALIBRATION if s != "AF" and t.get(s) is not None and math.isfinite(m.get(s, float("nan")))]
        log(f"  calibrage {it}/{iterations} : écart moyen {np.mean(err) if err else float('nan'):.1f} points de % sur VPIP, PFR, 3-bet, WTSD, fold to c-bet")
        if it == iterations:
            break
        for p in profiles:
            m, t, q = measured.get(p["name"], {}), p["cibles"], params[p["name"]]
            for key, stat, expo in CALIBRATION:
                tv, mv = t.get(stat), m.get(stat, float("nan"))
                if tv is None or not math.isfinite(mv) or mv <= 0 or tv <= 0:
                    continue
                lo, hi = BOUNDS[key]
                q[key] = round(float(np.clip(q[key] * (tv / mv) ** (0.8 * expo), lo, hi)), 4)
            q["pfr"] = min(q["pfr"], q["vpip"])
    for p in profiles:
        p["params"] = params[p["name"]]
        p["mesure"] = {k: _round(measured.get(p["name"], {}).get(k, float("nan"))) for k in STAT_KEYS}
    return profiles


# --------------------------------------------------------------------------- écriture
def write_population(profiles: list[dict[str, Any]], root: Path | None = None, source: str = "", n_hands: int = 0) -> tuple[Path, Path]:
    """Écrit les archétypes et la composition ``population_reelle`` (fichiers générés, à régénérer après import)."""
    root = root or configs_root()
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    head = (f"# GÉNÉRÉ par python -m pokerlab population le {stamp} à partir de {n_hands} mains d'adversaires ({source}).\n"
            "# Ne pas éditer à la main : relance la commande après de nouveaux imports.\n"
            "# cibles = stats réelles du profil ; mesure = stats obtenues en simulation après calibrage.\n")
    arch: dict[str, Any] = {}
    for p in profiles:
        spec = archetype_spec(p, p.get("params") or initial_params(p["cibles"]))
        spec.pop("type", None)
        spec.pop("name", None)
        spec["cibles"] = p["cibles"]
        spec["mesure"] = p.get("mesure", {})
        spec["observes"] = {"joueurs": p["joueurs"], "mains": p["mains"]}
        arch[p["name"]] = spec
    opp = root / "opponents" / "population_reelle.yaml"
    opp.parent.mkdir(parents=True, exist_ok=True)
    opp.write_text(head + yaml.safe_dump(arch, allow_unicode=True, sort_keys=False), encoding="utf-8")
    table = {
        "name": "population_reelle",
        "label": "Population réelle",
        "description": f"Adversaires calibrés sur tes historiques PokerStars ({n_hands} mains d'adversaires, {len(profiles)} profils).",
        "mode": "sample",
        "session_decks": 10,
        "weights": {p["name"]: round(p["part"], 4) for p in profiles},
    }
    tab = root / "tables" / "population_reelle.yaml"
    tab.write_text(head + yaml.safe_dump(table, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return opp, tab


def generate(lab: LabConfig, rows: pd.DataFrame, k: int = 5, min_hands: int = 50, iterations: int = 4, decks: int = 300,
             root: Path | None = None, source: str = "data/reel", log: Callable[[str], None] = print) -> dict[str, Any]:
    """Stats par joueur -> profils -> calibrage -> fichiers de composition."""
    players = player_stats(rows)
    profiles = build_profiles(players, k, min_hands, lab.seed)
    if not profiles:
        raise ValueError("Aucun adversaire importé : importe d'abord des historiques PokerStars (python -m pokerlab import).")
    log(f"population : {len(players)} adversaires, {int(players['mains'].sum())} mains d'adversaires, {len(profiles)} profils")
    if iterations >= 0:
        calibrate(profiles, lab, iterations, decks, log)
    paths = write_population(profiles, root, source, int(players["mains"].sum()))
    from pokerlab.strategies.registry import reset_archetypes

    reset_archetypes()
    return {"players": players, "profiles": profiles, "paths": paths}


__all__ = ["opponent_rows", "merge_opponents", "player_stats", "build_profiles", "initial_params", "calibrate", "write_population",
           "generate", "STAT_KEYS", "PARAM_DEFAULTS"]
