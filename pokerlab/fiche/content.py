"""Contenu de la fiche : tailles de mise, règles postflop en clair, blocs issus d'une campagne."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.analysis.leaks import AUDIT_GROUP_KEYS, situation_labels
from pokerlab.config import deep_merge
from pokerlab.features.hand import MADE_LABELS, level_of
from pokerlab.analysis.stats import summarize
from pokerlab.strategies.rule_engine import POSTFLOP_DEFAULTS

FRACTIONS = {0.25: "1/4", 0.33: "1/3", 0.333: "1/3", 0.5: "1/2", 0.66: "2/3", 0.667: "2/3", 0.75: "3/4", 1.0: "le pot"}
LEVEL_PHRASES = {
    "nothing": "n'importe quelle main",
    "weak_pair": "une paire (même petite)",
    "middle_pair": "une paire moyenne ou mieux",
    "top_pair": "top paire ou mieux",
    "overpair": "une overpair ou mieux",
    "two_pair": "deux paires ou mieux",
    "trips": "un brelan ou mieux",
    "straight": "une quinte ou mieux",
    "flush": "une couleur ou mieux",
    "full_house": "un full ou mieux",
}
DRAW_PHRASES = {"strong": "un tirage couleur ou quinte ouverte", "any": "un tirage (même ventral)"}


def fraction_label(value: float) -> str:
    for key, label in FRACTIONS.items():
        if abs(value - key) < 0.01:
            return label + (" du pot" if label != "le pot" else "")
    return f"{value:.0%} du pot".replace("%", " %")


FORCE_PHRASES = {
    "top_paire_bon_kicker": "une top paire avec un bon kicker (valet ou plus)",
    "overpair": "une overpair",
    "overpair_qq": "une overpair QQ+",
    "deux_paires": "deux paires",
    "brelan": "un brelan",
}


def level_phrase(name: str | int) -> str:
    if isinstance(name, int):
        from pokerlab.features.hand import MADE_LEVELS

        name = MADE_LEVELS[name]
    name = {"pair": "weak_pair", "any_pair": "weak_pair", "set": "trips"}.get(name, name)
    return LEVEL_PHRASES.get(name, name)


def money(bb_count: float, big_blind: float) -> str:
    return f"{bb_count * big_blind:.2f}".replace(".", ",")


# --------------------------------------------------------------------------- tailles
@dataclass
class SizeRow:
    label: str
    bb: float | None
    detail: str


def size_rows(cfg: dict[str, Any], reference_raise_bb: float, reference_pot_bb: float) -> list[SizeRow]:
    """Tailles de mise de la stratégie, exprimées en bb (montants en jetons calculés au rendu)."""
    kind = cfg.get("type", "rules")
    rows: list[SizeRow] = []
    r = reference_raise_bb
    pot = reference_pot_bb
    if kind == "rules":
        pre = cfg.get("preflop", {})
        un = pre.get("unopened", {})
        if un.get("raise_size_bb") is not None:
            extra = float(un.get("per_limper_bb", 1.0))
            rows.append(SizeRow("Ouverture (personne n'a relancé)", float(un["raise_size_bb"]), f"+{extra:g} bb par joueur qui a juste payé (limper)"))
        vo = pre.get("vs_open", {})
        if vo.get("raise_range"):
            mult = float(vo.get("raise_multiplier", 3.0))
            rows.append(SizeRow(f"3-bet face à une relance à {r:g} bb", mult * r, f"{mult:g} fois la relance"))
        for tier in vo.get("call", []) or []:
            rng = tier["range"] if isinstance(tier["range"], str) else ""
            detail = f"avec {rng}" if rng and len(rng) <= 28 else "mains : voir les grilles « face à une relance »"
            rows.append(SizeRow(f"Paie une relance jusqu'à {float(tier['max_bb']):g} bb", float(tier["max_bb"]), detail))
        v3 = pre.get("vs_3bet", {})
        if v3.get("raise_range"):
            rows.append(SizeRow("Face à un 3-bet (après avoir ouvert)", None, "tapis" if v3.get("raise_allin") else f"{float(v3.get('raise_multiplier', 2.5)):g} fois le 3-bet"))
        post = deep_merge(POSTFLOP_DEFAULTS, cfg.get("postflop", {}))
        if post["value_bet"]["enabled"]:
            for t in post["value_bet"].get("tiers", []):
                f = float(t["size_pot"])
                rows.append(SizeRow(f"Mise de value ({level_phrase(t['min_made'])}), pot de {pot:g} bb", f * pot, fraction_label(f)))
        for key, label in (("draw_bet", "Mise avec tirage"), ("cbet", "Continuation bet"), ("bluff", "Bluff")):
            if post[key]["enabled"]:
                f = float(post[key]["size_pot"])
                rows.append(SizeRow(f"{label}, pot de {pot:g} bb", f * pot, fraction_label(f)))
        if post["raise_vs_bet"]["enabled"]:
            m = float(post["raise_vs_bet"]["multiplier"])
            bet = round(pot / 2, 2)
            rows.append(SizeRow(f"Relance face à une mise de {bet:g} bb", m * bet, f"{m:g} fois la mise"))
    elif kind == "equity":
        pre = cfg.get("preflop", {})
        post = cfg.get("postflop", {})
        rows.append(SizeRow("Ouverture", float(pre.get("open_size_bb", 2.5)), "+1 bb par limper"))
        mult = float(pre.get("three_bet_multiplier", 3.0))
        rows.append(SizeRow(f"3-bet face à une relance à {r:g} bb", mult * r, f"{mult:g} fois la relance"))
        f = float(post.get("bet_size_pot", 0.66))
        rows.append(SizeRow(f"Mise de value, pot de {pot:g} bb", f * pot, fraction_label(f)))
    return rows


# ------------------------------------------------------------------- règles postflop
def postflop_rules(cfg: dict[str, Any], max_lines: int = 10) -> list[str]:
    """Règles postflop en langage simple (10 lignes maximum)."""
    kind = cfg.get("type", "rules")
    if kind == "equity":
        return [
            "Calcule ton équité contre les mains probables de l'adversaire (d'après ses actions).",
            "Personne n'a misé : mise 2/3 du pot si ton équité dépasse nettement ta part équitable.",
            "Avec un bon tirage et un seul adversaire : semi-bluff à 1/2 pot.",
            "Face à une mise : paie si ton équité bat la cote du pot, relance si elle est très haute.",
            "Sinon : couche-toi.",
        ]
    if kind == "random":
        return ["Joue au hasard parmi les actions légales : c'est le témoin qui doit perdre."]
    if kind != "rules":
        return ["Adversaire modélisé : voir ses paramètres dans configs/opponents/archetypes.yaml."]
    p = deep_merge(POSTFLOP_DEFAULTS, cfg.get("postflop", {}))
    lines: list[str] = []
    if p["value_bet"]["enabled"]:
        tiers = sorted(p["value_bet"].get("tiers", []), key=lambda t: -level_of(t["min_made"]))
        parts = [f"{fraction_label(float(t['size_pot']))} avec {level_phrase(t['min_made'])}" for t in tiers]
        lines.append("Personne n'a misé : mise " + " ; ".join(parts) + ".")
    if p["draw_bet"]["enabled"]:
        lines.append(f"Avec {DRAW_PHRASES.get(p['draw_bet']['draws'], 'un tirage')} : mise {fraction_label(float(p['draw_bet']['size_pot']))} (flop et turn).")
    if p["cbet"]["enabled"]:
        lines.append(f"Tu as relancé avant le flop et il reste un seul adversaire : mise {fraction_label(float(p['cbet']['size_pot']))} au flop, même sans main (une fois).")
    if p["bluff"]["enabled"]:
        b = p["bluff"]
        cond = []
        if b.get("require_preflop_raiser"):
            cond.append("tu as relancé avant le flop")
        cond.append("un seul adversaire" if int(b.get("max_opponents", 1)) == 1 else f"{b['max_opponents']} adversaires max")
        if b.get("require_dry"):
            cond.append("flop sec (3 couleurs différentes, cartes éloignées, pas de paire collée)")
        lines.append(f"Bluff : une seule mise de {fraction_label(float(b['size_pot']))} au flop si " + ", ".join(cond) + ".")
        lines.append("Bluff payé ou relancé et toujours rien : arrête (check, puis couche-toi face à une mise).")
    lines.append("Sinon : check.")
    if p["raise_vs_bet"]["enabled"]:
        lines.append(f"Face à une mise : relance à {float(p['raise_vs_bet']['multiplier']):g} fois la mise avec {level_phrase(p['raise_vs_bet']['min_made'])}.")
    g = p["guardrail"]
    seuil = g.get("seuil")
    tapis_seul = seuil is not None and str(seuil).lower() == "tapis"
    frac = float(g["trigger_fraction"]) if seuil is None else (None if tapis_seul else float(seuil))
    size = f" d'au plus {fraction_label(frac)}" if g["enabled"] and frac is not None else ""
    lines.append(f"Face à une mise{size} : paie avec {level_phrase(p['call_vs_bet']['min_made'])}, sinon couche-toi.")
    cd = p["call_draws"]
    if cd["enabled"]:
        if cd.get("use_pot_odds", True):
            lines.append(f"Avec {DRAW_PHRASES.get(cd['draws'], 'un tirage')} : paie seulement si la mise est petite (au plus 1/4 à 1/3 du pot, cote du pot).")
        else:
            lines.append(f"Avec {DRAW_PHRASES.get(cd['draws'], 'un tirage')} : paie les petites mises.")
    if g["enabled"]:
        trigger = "Face à un tapis" if frac is None else f"Face à un tapis ou une mise de plus de {fraction_label(frac)}"
        if g.get("force_min"):
            text = f"{trigger} : ne suis qu'avec {FORCE_PHRASES.get(str(g['force_min']).lower(), g['force_min'])} ou mieux"
        else:
            text = f"{trigger} : ne suis qu'avec {level_phrase(g['min_made'])}"
            if g.get("allow_overpair_from"):
                text += f", ou une overpair {str(g['allow_overpair_from']).upper()[0] * 2}+"
        if g.get("equity_check"):
            text += " (sauf si ton équité contre sa range estimée bat la cote)"
        lines.append(text + ".")
    if p.get("never_bet_without_pair"):
        lines.append("Jamais de mise ni de relance sans au moins une paire.")
    return lines[:max_lines]


# ------------------------------------------------------------------------ campagne
@dataclass
class Campaign:
    """Résultats d'une campagne utiles à la fiche (compare et audit)."""

    root: Path
    compare_dir: Path | None = None
    audit_dir: Path | None = None
    notes: list[str] = field(default_factory=list)


def _read_table(folder: Path, name: str) -> pd.DataFrame | None:
    pq, csv = folder / f"{name}.parquet", folder / f"{name}.csv"
    try:
        if pq.exists():
            return pd.read_parquet(pq)
        if csv.exists():
            return pd.read_csv(csv)
    except (ValueError, OSError, ImportError):
        return None
    return None


def find_campaign(root: str | Path, strategy: str) -> Campaign:
    """Cherche dans ``root`` (et ses sous-dossiers) un compare et un audit couvrant ``strategy``."""
    root = Path(root)
    camp = Campaign(root)
    if not root.exists():
        camp.notes.append(f"Dossier de campagne introuvable : {root}")
        return camp
    candidates = [root] + sorted(p for p in root.iterdir() if p.is_dir()) + sorted(p for p in root.glob("*/*") if p.is_dir())
    for folder in candidates:
        resume = folder / "resume.json"
        if not resume.exists():
            continue
        try:
            meta = json.loads(resume.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        exp = meta.get("experience") or folder.name
        cfg = meta.get("config") or {}
        if exp == "compare" and camp.compare_dir is None:
            names = [s if isinstance(s, str) else s.get("name") for s in cfg.get("strategies", [])]
            if not names or strategy in names:
                camp.compare_dir = folder
        elif exp == "audit" and camp.audit_dir is None:
            if cfg.get("strategy", strategy) == strategy:
                camp.audit_dir = folder
    return camp


def leak_prohibitions(camp: Campaign, strategy: str, top: int, min_count: int, z: float = 1.96,
                      keys: tuple[str, ...] | list[str] = AUDIT_GROUP_KEYS) -> tuple[list[dict[str, Any]], str]:
    """Les fuites les plus chères de l'audit, formulées comme des interdits."""
    if camp.audit_dir is None:
        return [], "aucun audit de cette stratégie dans la campagne"
    df = _read_table(camp.audit_dir, "decisions")
    if df is None or df.empty or "loss_cv" not in df.columns:
        return [], f"audit sans décisions exploitables ({camp.audit_dir})"
    if "strategy" in df.columns:
        df = df[df["strategy"].astype(str) == strategy]
    if df.empty:
        return [], "l'audit ne concerne pas cette stratégie"
    if "result_ev" not in df.columns:
        df = df.assign(result_ev=0.0)
    d = situation_labels(df)
    g = d.groupby(list(keys)).agg(count=("loss_cv", "size"), mean=("loss_cv", "mean"), std=("loss_cv", "std"), total=("loss_cv", "sum")).reset_index()
    g = g[g["mean"] > 0].sort_values("total", ascending=False).head(top)
    best = d.groupby(list(keys))["best_action"].agg(lambda s: s.value_counts().index[0])
    out = []
    for _, row in g.iterrows():
        key = tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]
        half = z * (row["std"] if row["std"] == row["std"] else 0.0) / np.sqrt(row["count"])
        out.append({
            "texte": prohibition_text(row, str(best.loc[key])),
            "count": int(row["count"]),
            "mean": float(row["mean"]),
            "total": float(row["total"]),
            "fiable": bool(row["count"] >= min_count and row["mean"] - half > 0),
        })
    return out, f"{len(df)} décisions auditées ({camp.audit_dir})"


VERBS = {"se couche": "Ne te couche pas", "check": "Ne checke pas", "paie": "Ne paie pas", "mise": "Ne mise pas", "relance": "Ne relance pas"}
STREET_AT = {"préflop": "avant le flop", "flop": "au flop", "turn": "au turn", "river": "à la river"}
BEST_TEXT = {"se couche": "couche-toi", "paie": "paie", "check": "checke", "tapis": "fais tapis"}


def prohibition_text(row: pd.Series, best_action: str) -> str:
    verb = VERBS.get(row["action_label"], f"N'{row['action_label']} pas")
    if str(best_action).split(" ")[0] == row["action_label"] and row["action_label"] in ("mise", "relance"):
        verb += " à ta taille habituelle"
    facing = "" if row["facing_label"] == "rien à payer" else f" {row['facing_label']}"
    board = "" if row["street_label"] == "préflop" or "board_label" not in row else f", tableau {row['board_label']}"
    where = f" en {row['position']}" if "position" in row else ""
    best = BEST_TEXT.get(best_action, re.sub(r"(\d)\.(\d)", r"\1,\2", best_action))
    return f"{verb} avec {row['hand_label']}{facing} {STREET_AT.get(row['street_label'], row['street_label'])}{where}{board} : {best}."


@dataclass
class Expected:
    composition: str
    hands: int
    bb100_ev: float
    ci_half: float
    bb100_real: float
    sd100: float
    ranges: list[dict[str, float]]
    reliable: bool


def expected_result(camp: Campaign, strategy: str, horizons: list[int], resamples: int, min_reliable: int,
                    composition: str | None = None, seed: int = 0, confidence: float = 0.95) -> tuple[Expected | None, str]:
    """bb/100, écart type et fourchettes 5e-95e centile sur ``horizons`` mains (résultats réels)."""
    if camp.compare_dir is None:
        return None, "aucun compare dans la campagne"
    df = _read_table(camp.compare_dir, "mains")
    if df is None or df.empty or "strategy" not in df.columns:
        return None, f"compare sans mains exploitables ({camp.compare_dir})"
    df = df[df["strategy"].astype(str) == strategy]
    if df.empty:
        return None, "le compare ne contient pas cette stratégie"
    if composition is None:
        composition = str(df["composition"].astype(str).value_counts().index[0])
    df = df[df["composition"].astype(str) == composition].sort_values(["deck", "rot"])
    if df.empty:
        return None, f"aucune main pour la composition {composition}"
    per = df.groupby("deck")[["result_ev", "result"]].mean()
    hpu = len(df) / max(len(per), 1)
    ev = summarize(per["result_ev"].to_numpy(), df["result_ev"].to_numpy(), hpu, confidence)
    real = summarize(per["result"].to_numpy(), df["result"].to_numpy(), hpu, confidence)
    values = df["result"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    ranges = []
    for n in horizons:
        sums = values[rng.integers(0, values.size, size=(resamples, n))].sum(axis=1)
        ranges.append({"mains": n, "moyenne": float(values.mean() * n), "p5": float(np.quantile(sums, 0.05)), "p95": float(np.quantile(sums, 0.95))})
    return Expected(composition, len(df), ev.bb100, ev.ci_half, real.bb100, real.sd_hand_bb100, ranges, len(df) >= min_reliable), \
        f"{len(df)} mains, composition {composition} ({camp.compare_dir})"


__all__ = ["MADE_LABELS", "size_rows", "postflop_rules", "find_campaign", "leak_prohibitions", "expected_result", "money"]
