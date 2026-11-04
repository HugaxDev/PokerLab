"""Expérience « bankroll » : risque de ruine et trajectoires, avec et sans stop-loss de session.

Les trajectoires enchaînent des sessions tirées (avec remise) parmi les sessions simulées.
Approximation documentée : quand la bankroll passe sous 100 bb, la perte d'une session
est plafonnée à la bankroll restante (on ne peut pas perdre plus que ce qu'on a).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.config import PROJECT_ROOT
from pokerlab.experiments import sessions as sessions_exp
from pokerlab.experiments.base import ExperimentContext, save_outputs
from pokerlab.report import charts, sections
from pokerlab.report.html import Report
from pokerlab.rng import derive_seed


def session_arrays(hands: pd.DataFrame, name: str, stop_loss_bb: float | None) -> np.ndarray:
    """Résultat réel de chaque session, tronqué au premier passage sous -stop_loss si demandé."""
    out = []
    for _, h in hands[hands["strategy"] == name].groupby("session", sort=True):
        cum = np.cumsum(h.sort_values("hand_no")["result"].to_numpy(dtype=float))
        if stop_loss_bb is not None:
            hit = np.flatnonzero(cum <= -stop_loss_bb)
            out.append(float(cum[hit[0]]) if hit.size else float(cum[-1]))
        else:
            out.append(float(cum[-1]))
    return np.asarray(out, dtype=float)


def simulate_paths(results: np.ndarray, capital_bb: float, n_traj: int, max_sessions: int, ruin_below_bb: float, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws = results[rng.integers(0, results.size, size=(n_traj, max_sessions))]
    bank = np.full(n_traj, capital_bb, dtype=float)
    ruined_at = np.full(n_traj, -1, dtype=int)
    paths = np.empty((n_traj, max_sessions + 1))
    paths[:, 0] = bank
    for t in range(max_sessions):
        alive = ruined_at < 0
        delta = np.maximum(draws[:, t], -bank)
        bank = np.where(alive, bank + delta, bank)
        newly = alive & (bank < ruin_below_bb)
        ruined_at[newly] = t + 1
        paths[:, t + 1] = bank
    ruin_curve = np.array([(ruined_at > 0) & (ruined_at <= t) for t in range(max_sessions + 1)]).mean(axis=1)
    bands = {f"p{q}": np.quantile(paths, q / 100, axis=0) for q in (5, 25, 50, 75, 95)}
    return {"paths": paths, "ruin_curve": ruin_curve, "bands": bands, "p_ruin": float((ruined_at > 0).mean()), "final": bank}


def analytic_ror(mu_hand: float, sd_hand: float, capital_bb: float) -> float:
    """Formule classique exp(-2 μ B / σ²) (μ > 0), avec μ, σ par main."""
    if mu_hand <= 0 or sd_hand <= 0:
        return 1.0
    return float(math.exp(-2 * mu_hand * capital_bb / sd_hand ** 2))


def stop_loss_verdict(rows: list[dict[str, Any]], stop: float, max_sessions: int) -> str:
    """Conclusion chiffrée sur l'effet du stop-loss (et non une règle générale)."""
    if not rows:
        return "Pas de données de bankroll."
    without = [r[f"ruine en {max_sessions} sessions (sans stop) %"] for r in rows]
    with_sl = [r[f"ruine en {max_sessions} sessions (stop {stop:.0f} bb) %"] for r in rows]
    worse = sum(b > a for a, b in zip(without, with_sl))
    if worse == len(rows):
        return (f"Le stop-loss de {stop:.0f} bb augmente le risque de ruine ici : il coupe aussi des sessions gagnantes "
                "en moyenne et ne plafonne pas la perte (une seule main peut coûter plus que le seuil).")
    if worse == 0:
        return f"Le stop-loss de {stop:.0f} bb réduit le risque de ruine pour toutes les stratégies simulées."
    return f"Le stop-loss de {stop:.0f} bb réduit la ruine pour certaines stratégies et l'augmente pour d'autres (voir le tableau)."


def run(ctx: ExperimentContext, source_hands: pd.DataFrame | None = None, report: Report | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    specs = cfg.get("strategies", [])
    names, colors, labels = sections.colors_labels(specs)
    game = ctx.lab.game
    capital = float(cfg.get("capital", 3.5))
    bb_value = float(cfg.get("currency_per_bb", game.currency_per_bb))
    capital_bb = capital / bb_value
    stop = float(cfg.get("stop_loss_bb", 25))
    n_traj = int(cfg.get("n_trajectories", 5000))
    max_sessions = int(cfg.get("max_sessions", 200))
    ruin_below = float(cfg.get("ruin_below_bb", 1.0))
    if source_hands is None and cfg.get("sessions_dir"):
        folder = Path(cfg["sessions_dir"])
        folder = folder if folder.is_absolute() else PROJECT_ROOT / folder
        pq = folder / "mains.parquet"
        source_hands = pd.read_parquet(pq) if pq.exists() else pd.read_csv(folder / "mains.csv")
        missing = [n for n in names if not (source_hands["strategy"].astype(str) == n).any()]
        if missing:
            raise ValueError(f"Pas de sessions pour {', '.join(missing)} dans {folder}")
        ctx.log(f"bankroll : sessions lues dans {folder}")
    if source_hands is None:
        sub = dict(cfg.get("sessions", {}))
        n_sessions = int(sub.get("n_sessions", 2000))
        per = int(sub.get("hands_per_session", 50))
        ctx.log(f"bankroll : simulation de {n_sessions} sessions de {per} mains par stratégie")
        source_hands = sessions_exp.simulate(ctx, specs, sub.get("composition", cfg.get("composition", "micro_play_money")), n_sessions, per)
    own = report is None
    report = report or Report("PokerLab — bankroll et risque de ruine", f"Capital {capital:.2f} {game.currency_symbol} = {capital_bb:.0f} bb")
    sec = report.section(f"Bankroll : {capital:.2f} {game.currency_symbol} ({capital_bb:.0f} bb), stop-loss {stop:.0f} bb")
    rows = []
    verdict = []
    for name in names:
        if not (source_hands["strategy"] == name).any():
            continue
        h = source_hands[source_hands["strategy"] == name]
        mu, sd = float(h["result"].mean()), float(h["result"].std(ddof=1))
        res = {}
        for label, sl in (("sans stop-loss", None), (f"stop-loss {stop:.0f} bb", stop)):
            arr = session_arrays(source_hands, name, sl)
            res[label] = simulate_paths(arr, capital_bb, n_traj, max_sessions, ruin_below, derive_seed(ctx.lab.seed, "bankroll", name, label))
        a, b = res["sans stop-loss"], res[f"stop-loss {stop:.0f} bb"]
        rows.append({
            "stratégie": labels[name],
            "bb/100 réel": mu * 100,
            f"ruine en {max_sessions} sessions (sans stop) %": 100 * a["p_ruin"],
            f"ruine en {max_sessions} sessions (stop {stop:.0f} bb) %": 100 * b["p_ruin"],
            "ruine théorique (horizon infini) %": 100 * analytic_ror(mu, sd, capital_bb),
            "bankroll finale médiane sans stop (bb)": float(np.median(a["final"])),
            "bankroll finale médiane avec stop (bb)": float(np.median(b["final"])),
        })
        x = list(range(max_sessions + 1))
        step = max(1, max_sessions // 100)
        xs = x[::step]
        bands = {k: v[::step].tolist() for k, v in a["bands"].items()}
        samples = [p[::step].tolist() for p in a["paths"][:12]]
        sec.figure(charts.fan_chart(xs, bands, colors[name], f"{labels[name]} : trajectoires sans stop-loss", samples))
        verdict.append(f"{labels[name]} : ruine {100 * a['p_ruin']:.0f} % sans stop-loss, {100 * b['p_ruin']:.0f} % avec stop-loss {stop:.0f} bb "
                       f"({max_sessions} sessions).")
    sec.table(pd.DataFrame(rows), "Ruine = bankroll sous " + f"{ruin_below:g} bb. Trajectoires : sessions simulées tirées avec remise. "
              "Ruine théorique : exp(−2μB/σ²), sans plafond de perte ni stop-loss.")
    if own:
        report.verdict = verdict[:4] + [stop_loss_verdict(rows, stop, max_sessions)]
        paths = save_outputs(ctx, report, None, None, (), {"capital_bb": capital_bb, "lignes": rows})
    else:
        paths = {}
    return {"report": report, "paths": paths, "rows": rows}
