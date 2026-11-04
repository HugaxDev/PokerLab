"""Données du tableau de bord : mains réelles, référence simulée, campagnes (calculs testables)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from pokerlab.analysis.leaks import situation_labels
from pokerlab.analysis.player_stats import stats_from_hands
from pokerlab.analysis.robustness import load_compare_cells, robustness_table
from pokerlab.analysis.stats import z_value
from pokerlab.cards.cards import CARD_STR
from pokerlab.cards.ranges import CLASS_NAMES

STAT_KEYS = ("VPIP", "PFR", "3bet", "AF", "WTSD")


def money_label(currency: str) -> str:
    return currency if currency else "jetons"


@dataclass
class Reference:
    """Résultats simulés de la stratégie suivie (compare) : base du cône et des comparaisons."""

    strategy: str
    composition: str
    mean: float
    sd: float
    bb100: float
    ci: float
    stats: dict[str, float]
    sample: np.ndarray | None
    source: str

    @property
    def hands_needed(self) -> float:
        """Mains réelles pour distinguer le taux de gain de zéro à 95 % : (z σ / μ)²."""
        if self.mean <= 0:
            return math.inf
        return (z_value(0.95) * self.sd / self.mean) ** 2


def load_reference(root: Path, strategy: str, composition: str, exclude: Iterable[Path] = ()) -> Reference | None:
    """Cherche la case (stratégie, composition) dans les compare de ``root`` ; échantillon de mains si dispo."""
    cells = load_compare_cells(root, exclude)
    if cells.empty:
        return None
    sel = cells[(cells["strategy"] == strategy) & (cells["composition"] == composition)]
    if sel.empty:
        sel = cells[cells["strategy"] == strategy]
        if sel.empty:
            return None
    row = sel.iloc[0]
    sample = None
    pq = Path(row["source"]) / "mains.parquet"
    if pq.exists():
        try:
            df = pd.read_parquet(pq, columns=["strategy", "composition", "result"])
            df = df[(df["strategy"].astype(str) == strategy) & (df["composition"].astype(str) == str(row["composition"]))]
            sample = df["result"].to_numpy(dtype=float) if len(df) else None
        except (ValueError, OSError):
            sample = None
    mean = float(row.get("moyenne_main", row["bb100_reel"] / 100))
    sd = float(row.get("ecart_type_main", row["sd_main_bb100"] / 10))
    stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
    return Reference(strategy, str(row["composition"]), mean, sd, float(row["bb100"]), float(row["ci"]), stats or {}, sample, str(row["source"]))


def expected_cone(ref: Reference, n_max: int, draws: int, seed: int, points: int = 160) -> dict[str, list[float]]:
    """Percentiles (5, 25, 50, 75, 95) des gains cumulés attendus après n mains.

    Simulation (tirage de mains simulées avec remise) si l'échantillon existe et que le calcul
    reste raisonnable, sinon approximation normale (théorème central limite)."""
    n_max = max(1, int(n_max))
    xs = np.unique(np.linspace(1, n_max, min(points, n_max)).astype(int))
    qs = (5, 25, 50, 75, 95)
    out: dict[str, list[float]] = {"x": xs.tolist()}
    if ref.sample is not None and ref.sample.size > 100 and n_max * draws <= 30_000_000:
        rng = np.random.default_rng(seed)
        acc = np.zeros((draws, xs.size))
        total = np.zeros(draws)
        prev = 0
        for j, n in enumerate(xs):
            step = n - prev
            total += ref.sample[rng.integers(0, ref.sample.size, size=(draws, step))].sum(axis=1)
            acc[:, j] = total
            prev = n
        for q in qs:
            out[f"p{q}"] = np.percentile(acc, q, axis=0).tolist()
        out["methode"] = "simulation"  # type: ignore[assignment]
    else:
        from scipy.stats import norm

        for q in qs:
            out[f"p{q}"] = (ref.mean * xs + norm.ppf(q / 100) * ref.sd * np.sqrt(xs)).tolist()
        out["methode"] = "approximation normale"  # type: ignore[assignment]
    return out


def verdict_vs_cone(total: float, cone: dict[str, list[float]]) -> tuple[str, str]:
    """(message, niveau) selon la position du cumul réel dans le cône à la dernière main."""
    if not cone.get("x"):
        return "Pas assez de mains.", "ok"
    if total < cone["p5"][-1]:
        return "Sous le 5e centile : à analyser (malchance rare ou jeu différent de la fiche).", "warn"
    if total > cone["p95"][-1]:
        return "Au-dessus du 95e centile : très chanceux, ne pas en conclure trop vite.", "ok"
    return "Dans la variance normale de la stratégie.", "ok"


@dataclass
class Played:
    """Mes mains réelles (ou l'exemple) et leurs agrégats."""

    hands: pd.DataFrame
    example: bool
    currency: str = ""
    decisions: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n(self) -> int:
        return int(len(self.hands))

    def bb100(self) -> tuple[float, float]:
        r = self.hands["result"].to_numpy(dtype=float)
        if r.size < 2:
            return float(r.mean() * 100) if r.size else 0.0, math.inf
        return float(r.mean() * 100), float(z_value(0.95) * r.std(ddof=1) / math.sqrt(r.size) * 100)

    def respect_rate(self) -> float:
        total = float(self.hands["decisions"].sum())
        off = float(self.hands["decisions_hors_fiche"].sum())
        return 100.0 * (total - off) / total if total else float("nan")

    def result_money(self) -> float:
        return float(self.hands["result_cur"].sum())

    def stats(self) -> dict[str, float]:
        st = stats_from_hands(self.hands)
        return {k: float(st[k]) for k in STAT_KEYS}

    def sessions(self) -> pd.DataFrame:
        g = self.hands.groupby("session", sort=True)
        out = pd.DataFrame({
            "session": g.size().index,
            "début": g["date"].min().dt.strftime("%d/%m %H:%M").to_numpy(),
            "mains": g.size().to_numpy(),
            "bb": g["result"].sum().to_numpy(),
            money_label(self.currency): g["result_cur"].sum().to_numpy(),
            "hors fiche": g["hors_fiche"].sum().astype(int).to_numpy(),
        })
        return out.sort_values("session", ascending=False).reset_index(drop=True)

    def by_position(self) -> pd.DataFrame:
        order = ["BTN", "CO", "MP", "UTG", "SB", "BB"]
        g = self.hands.groupby("position")["result"].agg(["count", "mean", "sum"])
        g = g.reindex([p for p in order if p in g.index] + [p for p in g.index if p not in order])
        g["bb100"] = g["mean"] * 100
        return g.reset_index()

    def by_class(self) -> tuple[np.ndarray, np.ndarray]:
        from pokerlab.analysis.decomposition import by_class

        return by_class(self.hands, "result")

    def losing_situations(self, top: int = 5) -> list[dict[str, Any]]:
        """Situations (street, position, main, action) qui ont le plus coûté à partir de la décision."""
        d = self.decisions
        if d.empty:
            return []
        lab = situation_labels(d)
        keys = ["street_label", "position", "hand_label", "action_label"]
        g = lab.groupby(keys).agg(count=("net_after", "size"), total=("net_after", "sum"), mean=("net_after", "mean")).reset_index()
        g = g[g["total"] < 0].sort_values("total").head(top)
        out = []
        for _, row in g.iterrows():
            mask = np.ones(len(lab), dtype=bool)
            for k in keys:
                mask &= (lab[k] == row[k]).to_numpy()
            ex = lab[mask].sort_values("net_after").head(3)["hand_id"].astype(str).tolist()
            out.append({"situation": " · ".join(str(row[k]) for k in keys[:-1]) + " → " + str(row["action_label"]),
                        "count": int(row["count"]), "total": float(row["total"]), "mean": float(row["mean"]), "hands": ex})
        return out


def played_from_hands(hands: pd.DataFrame, example: bool) -> Played:
    """Prépare les mains importées (dates, décisions à plat)."""
    if hands.empty:
        return Played(hands, example)
    h = hands.copy()
    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h = h.sort_values(["date", "hand_id"]).reset_index(drop=True)
    rows = []
    for _, r in h.iterrows():
        for dec in json.loads(r["decisions_json"]):
            rows.append({**dec, "hand_id": str(r["hand_id"]), "result_ev": float(r["result"]), "facing": dec.get("facing", 0)})
    dec = pd.DataFrame(rows)
    currency = str(h["currency"].iloc[0]) if "currency" in h.columns else ""
    return Played(h, example, currency, dec)


def hands_payload(p: Played, limit_replays: int = 5000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Liste des mains (pour filtres et recherche) et données de replay."""
    out, replays = [], {}
    for idx, r in p.hands.iterrows():
        cards = [CARD_STR[int(r["c1"])], CARD_STR[int(r["c2"])]]
        board = [CARD_STR[int(r[f"b{i}"])] for i in range(5) if int(r[f"b{i}"]) >= 0]
        hid = str(r["hand_id"])
        out.append({
            "id": hid, "date": str(r["date"])[:16] if pd.notna(r["date"]) else "", "session": int(r["session"]),
            "position": str(r["position"]), "cards": "".join(cards), "cards_list": cards, "classe": CLASS_NAMES[int(r["hand_class"])],
            "board": "".join(board), "board_list": board, "result": float(r["result"]), "hors_fiche": bool(r["hors_fiche"]),
        })
        if idx >= len(p.hands) - limit_replays:
            replays[hid] = json.loads(r["replay"])
    return out, replays


@dataclass
class Campaign:
    name: str
    experience: str
    folder: Path
    date: datetime
    meta: dict[str, Any]

    @property
    def report(self) -> Path:
        return self.folder / "rapport.html"


def list_campaigns(root: Path, exclude: Iterable[Path] = ()) -> list[Campaign]:
    """Campagnes trouvées sous ``root`` (resume.json avec un type d'expérience)."""
    skip = [Path(e).resolve() for e in exclude]
    out = []
    for resume in sorted(root.glob("**/resume.json")):
        if any(e in resume.resolve().parents for e in skip):
            continue
        try:
            meta = json.loads(resume.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        exp = meta.get("experience")
        if not exp:
            continue
        rel = resume.parent.relative_to(root) if resume.parent != root else Path(root.name)
        out.append(Campaign(str(rel).replace("/", " / "), exp, resume.parent, datetime.fromtimestamp(resume.stat().st_mtime), meta))
    return sorted(out, key=lambda c: c.date, reverse=True)


def campaign_params(c: Campaign) -> str:
    """Résumé lisible des paramètres d'une campagne."""
    cfg = c.meta.get("config") or {}
    parts = []
    for key in ("strategies", "strategy", "compositions", "composition", "decks", "n_decisions", "rollouts", "n_sessions",
                "hands_per_session", "capital", "stop_loss_bb"):
        if key in cfg:
            v = cfg[key]
            v = ", ".join(map(str, v)) if isinstance(v, list) else v
            parts.append(f"{key} = {v}")
    auto = cfg.get("autostop") or {}
    if auto.get("enabled", False):
        parts.append(f"arrêt auto ± {auto.get('ci_halfwidth_bb100')} bb/100, max {auto.get('max_hands_per_strategy', auto.get('max_hands'))} mains")
    if int(c.meta.get("mains_jouees", 0) or 0) > 0:
        parts.append(f"{int(c.meta['mains_jouees']):,} mains jouées".replace(",", " "))
    if "duree_s" in c.meta:
        parts.append(f"{float(c.meta['duree_s']) / 60:.1f} min")
    return " · ".join(parts)


def lab_tables(root: Path, exclude: Iterable[Path] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cases de tous les compare et table de robustesse."""
    cells = load_compare_cells(root, exclude)
    return cells, robustness_table(cells)
