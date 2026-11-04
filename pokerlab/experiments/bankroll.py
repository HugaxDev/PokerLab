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
    return session_results(hands, name, stop_loss_bb)[0]


def session_results(hands: pd.DataFrame, name: str, stop_loss_bb: float | None) -> tuple[np.ndarray, np.ndarray]:
    """(résultat, mains jouées) de chaque session ; avec stop-loss, la session s'arrête à la main
    où le cumul passe sous -stop_loss (cette main comprise)."""
    res, length, _ = session_profiles(hands, name, stop_loss_bb)
    return res, length


def session_profiles(hands: pd.DataFrame, name: str, stop_loss_bb: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(résultat, mains jouées, creux) de chaque session. Le creux (cumul le plus bas, <= 0) sert à
    détecter une ruine au milieu d'une session, main par main, et pas seulement en fin de session."""
    res, length, low = [], [], []
    sub = hands[hands["strategy"].astype(str) == name]
    for _, h in sub.groupby("session", sort=True):
        cum = np.cumsum(h.sort_values("hand_no")["result"].to_numpy(dtype=float))
        hit = np.flatnonzero(cum <= -stop_loss_bb) if stop_loss_bb is not None else np.array([], dtype=int)
        end = int(hit[0]) if hit.size else cum.size - 1
        res.append(float(cum[end]))
        length.append(end + 1)
        low.append(min(0.0, float(cum[: end + 1].min())))
    return np.asarray(res, dtype=float), np.asarray(length, dtype=np.int64), np.asarray(low, dtype=float)


def stop_loss_facts(hands: pd.DataFrame, name: str, stop_loss_bb: float) -> dict[str, float]:
    """Chiffres qui expliquent l'effet du stop-loss (sur les sessions simulées)."""
    full, n_full = session_results(hands, name, None)
    cut, n_cut = session_results(hands, name, stop_loss_bb)
    stopped = n_cut < n_full
    trigger = cut[stopped & (cut <= -stop_loss_bb)]
    return {
        "sessions coupées %": 100 * float(stopped.mean()),
        "mains par session sans stop": float(n_full.mean()),
        "mains par session avec stop": float(n_cut.mean()),
        "bb/100 sans stop": 100 * float(full.sum() / n_full.sum()),
        "bb/100 avec stop": 100 * float(cut.sum() / n_cut.sum()),
        "perte moyenne au déclenchement (bb)": float(trigger.mean()) if trigger.size else float("nan"),
        "déclenchements à -100 bb ou pire %": 100 * float((trigger <= -100).mean()) if trigger.size else float("nan"),
    }


def simulate_paths_hands(results: np.ndarray, lengths: np.ndarray, capital_bb: float, n_traj: int, horizon_hands: int,
                         ruin_below_bb: float, seed: int, grid_step: int = 250, lows: np.ndarray | None = None) -> dict[str, Any]:
    """Trajectoires à nombre de mains égal : on enchaîne des sessions tirées avec remise jusqu'à
    ``horizon_hands`` mains (la dernière session peut dépasser l'horizon de quelques mains).
    Avec ``lows`` (creux de chaque session), la ruine est détectée dès que la bankroll passe sous le
    seuil pendant la session, pas seulement à la fin."""
    rng = np.random.default_rng(seed)
    bank = np.full(n_traj, capital_bb, dtype=float)
    done = np.zeros(n_traj, dtype=np.int64)
    ruined = np.zeros(n_traj, dtype=bool)
    grid = np.arange(0, horizon_hands + 1, grid_step)
    values = np.empty((n_traj, grid.size))
    values[:, 0] = capital_bb
    next_cp = np.ones(n_traj, dtype=np.int64)
    while True:
        rows = np.flatnonzero(~ruined & (done < horizon_hands))
        if rows.size == 0:
            break
        idx = rng.integers(0, results.size, size=rows.size)
        if lows is not None:
            broke = bank[rows] + lows[idx] < ruin_below_bb
            bank[rows] = np.where(broke, np.maximum(bank[rows] + lows[idx], 0.0), bank[rows] + results[idx])
        else:
            bank[rows] += np.maximum(results[idx], -bank[rows])
        done[rows] += lengths[idx]
        ruined[rows[bank[rows] < ruin_below_bb]] = True
        while True:
            hit = rows[(next_cp[rows] < grid.size) & (grid[np.minimum(next_cp[rows], grid.size - 1)] <= done[rows])]
            if hit.size == 0:
                break
            values[hit, next_cp[hit]] = bank[hit]
            next_cp[hit] += 1
    for t in range(n_traj):
        values[t, next_cp[t]:] = bank[t]
    bands = {f"p{q}": np.quantile(values, q / 100, axis=0) for q in (5, 25, 50, 75, 95)}
    return {"grid": grid, "values": values, "bands": bands, "p_ruin": float(ruined.mean()), "final": bank,
            "hands": float(done.mean())}


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


def survivors_median(final: np.ndarray, ruin_below_bb: float) -> float:
    """Médiane des bankrolls finales des trajectoires non ruinées (la médiane globale bascule quand
    la ruine avoisine 50 %)."""
    alive = final[final >= ruin_below_bb]
    return float(np.median(alive)) if alive.size else float("nan")


def analytic_ror(mu_hand: float, sd_hand: float, capital_bb: float) -> float:
    """Formule classique exp(-2 μ B / σ²) (μ > 0), avec μ, σ par main."""
    if mu_hand <= 0 or sd_hand <= 0:
        return 1.0
    return float(math.exp(-2 * mu_hand * capital_bb / sd_hand ** 2))


def stop_loss_verdict(summary: dict[str, dict[str, float]], stop: float) -> str:
    """Conclusion chiffrée, à nombre de mains égal, sur l'effet du stop-loss."""
    if not summary:
        return "Pas de données de bankroll."
    parts = []
    for name, f in summary.items():
        diff = 100 * (f["ruine_avec_stop"] - f["ruine_sans_stop"])
        if abs(diff) < 1.5:
            effect = f"ne change presque pas la ruine ({diff:+.1f} pt)"
        elif diff > 0:
            effect = f"augmente la ruine de {diff:.1f} pt"
        else:
            effect = f"réduit la ruine de {-diff:.1f} pt"
        parts.append(
            f"{name} : à mains égales, le stop-loss {stop:.0f} bb {effect} ; il coupe {f['sessions coupées %']:.0f} % des sessions, "
            f"la perte au déclenchement vaut {f['perte moyenne au déclenchement (bb)']:.0f} bb en moyenne "
            f"({f['déclenchements à -100 bb ou pire %']:.0f} % à -100 bb ou pire) et l'EV par main passe de "
            f"{f['bb/100 sans stop']:+.1f} à {f['bb/100 avec stop']:+.1f} bb/100."
        )
    return " ".join(parts)


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
    facts_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    verdict = []
    for name in names:
        if not (source_hands["strategy"] == name).any():
            continue
        h = source_hands[source_hands["strategy"] == name]
        mu, sd = float(h["result"].mean()), float(h["result"].std(ddof=1))
        per_session = int(round(h.groupby("session").size().mean()))
        horizon = int(cfg.get("horizon_hands", max_sessions * per_session))
        res: dict[str, dict[str, Any]] = {}
        for label, sl in (("sans stop-loss", None), (f"stop-loss {stop:.0f} bb", stop)):
            r, n, low = session_profiles(source_hands, name, sl)
            seed = derive_seed(ctx.lab.seed, "bankroll", name, label)
            res[label] = simulate_paths_hands(r, n, capital_bb, n_traj, horizon, ruin_below, seed, lows=low)
        a, b = res["sans stop-loss"], res[f"stop-loss {stop:.0f} bb"]
        facts = stop_loss_facts(source_hands, name, stop)
        rows.append({
            "stratégie": labels[name],
            "bb/100 réel": mu * 100,
            f"ruine en {horizon} mains (sans stop) %": 100 * a["p_ruin"],
            f"ruine en {horizon} mains (stop {stop:.0f} bb) %": 100 * b["p_ruin"],
            "ruine théorique (horizon infini) %": 100 * analytic_ror(mu, sd, capital_bb),
            f"bankroll médiane des non ruinés, sans stop (bb)": survivors_median(a["final"], ruin_below),
            f"bankroll médiane des non ruinés, avec stop (bb)": survivors_median(b["final"], ruin_below),
        })
        facts_rows.append({"stratégie": labels[name], **facts})
        step = max(1, a["grid"].size // 100)
        xs = a["grid"][::step].tolist()
        bands = {k: v[::step].tolist() for k, v in a["bands"].items()}
        samples = [p[::step].tolist() for p in a["values"][:12]]
        fig = charts.fan_chart(xs, bands, colors[name], f"{labels[name]} : trajectoires sans stop-loss ({horizon} mains)", samples)
        fig["layout"]["xaxis"]["title"] = {"text": "mains"}
        sec.figure(fig)
        verdict.append(f"{labels[name]} : à {horizon} mains, ruine {100 * a['p_ruin']:.0f} % sans stop-loss, "
                       f"{100 * b['p_ruin']:.0f} % avec stop-loss {stop:.0f} bb.")
        summary[name] = {"horizon_mains": horizon, "ruine_sans_stop": a["p_ruin"], "ruine_avec_stop": b["p_ruin"], **facts}
    sec.table(pd.DataFrame(rows), "Ruine = bankroll sous " + f"{ruin_below:g} bb à n'importe quelle main (creux de session compris). "
              "Comparaison à nombre de MAINS égal : on enchaîne des sessions simulées (tirées avec remise) jusqu'à l'horizon ; "
              "avec stop-loss il faut donc plus de sessions. Ruine théorique : exp(−2μB/σ²), horizon infini.")
    sec.html("<h3>Pourquoi le stop-loss fait ce qu'il fait</h3>")
    sec.table(pd.DataFrame(facts_rows), "Sessions coupées : le cumul est passé sous le seuil. La perte au déclenchement dépasse "
              "souvent le seuil : une seule main peut coûter tout un tapis. bb/100 avec stop : EV par main des mains effectivement jouées.")
    if own:
        report.verdict = verdict[:3] + [stop_loss_verdict(summary, stop)]
        paths = save_outputs(ctx, report, None, None, (), {"capital_bb": capital_bb, "lignes": rows, "stop_loss": summary})
    else:
        paths = {}
    return {"report": report, "paths": paths, "rows": rows}
