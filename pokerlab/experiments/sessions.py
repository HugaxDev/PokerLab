"""Expérience « sessions » : N sessions de 50 mains par stratégie (tapis persistants, recave).

Répond à : quelle est la dispersion d'une session de 50 mains ? quelle probabilité de
finir en perte alors que la stratégie est gagnante ? pire session ? drawdown médian ?
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.analysis.stats import max_drawdown
from pokerlab.engine.hand_runner import COUNTERS
from pokerlab.engine.table import Table, pk_order
from pokerlab.experiments.base import ExperimentContext, composition_dict, save_outputs
from pokerlab.experiments.common import TaskResult, lab_from_raw, record_seat, run_tasks, split_range
from pokerlab.opponents.compositions import Composition
from pokerlab.report import charts, sections
from pokerlab.report.html import Report
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_player, build_strategy, resolve_spec


@dataclass
class SessionTask:
    lab_raw: dict[str, Any]
    hero: dict[str, Any]
    composition: dict[str, Any]
    session_start: int
    session_end: int
    hands_per_session: int
    stack_mode: str = "persistent"


def run_session_task(task: SessionTask) -> TaskResult:
    lab = lab_from_raw(task.lab_raw)
    game = replace(lab.game, stack_mode=task.stack_mode)
    n = game.players
    comp = Composition.from_dict(task.composition)
    hero = build_strategy(task.hero, lab)
    out = TaskResult()
    c0 = COUNTERS["hands"]
    for s in range(task.session_start, task.session_end):
        names = comp.opponents(lab.seed, s, n - 1)
        opponents = [build_player(comp.spec_for(x), lab) for x in names]
        for j, opp in enumerate(opponents):   # archétypes « humains » : paramètres tirés et état propres à la session
            if hasattr(opp, "begin_session"):
                opp.begin_session(derive_seed(lab.seed, "humain", comp.name, s, j))
        seats = [hero] + opponents
        button = random.Random(derive_seed(lab.seed, "session_button", s)).randrange(n)
        table = Table(seats, game, button=button)
        for h in range(task.hands_per_session):
            seed = derive_seed(lab.seed, "session", s, h)
            upcoming = pk_order(table.button, n)
            hero_pk = upcoming.index(0)
            for j, opp in enumerate(opponents):
                if hasattr(opp, "begin_hand"):
                    opp.begin_hand(upcoming.index(j + 1), hero_pk)
            hr, order = table.play(shuffled_deck(seed), seed, s * 10_000 + h, track_seats=[0])
            pk_seat = order.index(0)
            for j, opp in enumerate(opponents):
                if hasattr(opp, "end_hand"):
                    opp.end_hand(hr, order.index(j + 1), pk_seat, game.chips_per_bb)
            keys = {"strategy": hero.name, "composition": comp.name, "session": s, "hand_no": h, "deck": s, "rot": h,
                    "hand_id": s * 10_000 + h, "lineup": "|".join(names), "rebuys_bb": table.rebuys[0] / game.chips_per_bb}
            record_seat(out, hr, pk_seat, lab, keys, record_decisions=False)
    out.hands_played = COUNTERS["hands"] - c0
    return out.compact()


def simulate(ctx: ExperimentContext, specs: list[Any], composition: str, n_sessions: int, hands_per_session: int) -> pd.DataFrame:
    """Simule les sessions de chaque stratégie (mêmes graines de sessions pour toutes)."""
    cdict = composition_dict(composition)
    stack_mode = ctx.cfg.get("stack_mode", "persistent")
    tasks = []
    for spec in specs:
        cfg = resolve_spec(spec)
        for a, b in split_range(0, n_sessions, max(1, ctx.workers * 2)):
            tasks.append(SessionTask(ctx.lab.raw, cfg, cdict, a, b, hands_per_session, stack_mode))
    outs = run_tasks(run_session_task, tasks, ctx.workers, ctx.lab.mp_start_method)
    merged = TaskResult()
    for o in outs:
        merged.merge(o)
    ctx.hands_played += merged.hands_played
    return merged.hands.frame()


def session_table(hands: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par session : résultat réel, EV, drawdown intra-session."""
    rows = []
    for (name, s), h in hands.groupby(["strategy", "session"], sort=True):
        cum = np.cumsum(h.sort_values("hand_no")["result"].to_numpy(dtype=float))
        dd, _ = max_drawdown(cum)
        rows.append({"strategy": name, "session": s, "result": float(cum[-1]) if cum.size else 0.0,
                     "result_ev": float(h["result_ev"].sum()), "drawdown": dd, "hands": len(h)})
    return pd.DataFrame(rows)


def distribution_summary(sess: pd.DataFrame, hands: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in sess.groupby("strategy", sort=False):
        r = g["result"].to_numpy(dtype=float)
        winrate = hands[hands["strategy"] == name]["result_ev"].mean() * 100
        rows.append({
            "strategy": name,
            "sessions": len(g),
            "bb/100 (EV, toutes sessions)": winrate,
            "moyenne session (bb)": r.mean(),
            "écart type session (bb)": r.std(ddof=1) if r.size > 1 else float("nan"),
            "P(session en perte) %": 100 * float((r < 0).mean()),
            "5e centile (bb)": float(np.quantile(r, 0.05)),
            "médiane (bb)": float(np.median(r)),
            "95e centile (bb)": float(np.quantile(r, 0.95)),
            "pire session (bb)": float(r.min()),
            "meilleure session (bb)": float(r.max()),
            "drawdown médian (bb)": float(np.median(g["drawdown"])),
        })
    return pd.DataFrame(rows)


def run(ctx: ExperimentContext, report: Report | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    specs = cfg.get("strategies", [])
    comp = cfg.get("composition", "micro_play_money")
    n_sessions = int(cfg.get("n_sessions", 10_000))
    per = int(cfg.get("hands_per_session", 50))
    ctx.log(f"sessions : {len(specs)} stratégies x {n_sessions} sessions x {per} mains ({comp})")
    hands = simulate(ctx, specs, comp, n_sessions, per)
    sess = session_table(hands)
    own_report = report is None
    report = report or Report("PokerLab — sessions de 50 mains", f"{n_sessions} sessions par stratégie, composition {comp}")
    add_sessions_sections(report, ctx, specs, hands, sess, per, verdict=own_report)
    paths = save_outputs(ctx, report, hands, None, (), {"sessions": n_sessions, "mains_par_session": per}) if own_report else {}
    return {"report": report, "paths": paths, "hands": hands, "sessions": sess}


def add_sessions_sections(report: Report, ctx: ExperimentContext, specs, hands, sess, per: int, verdict: bool = True) -> None:
    names, colors, labels = sections.colors_labels(specs)
    summ = distribution_summary(sess, hands)
    sec = report.section(f"Sessions de {per} mains : dispersion")
    sec.figure(charts.boxes({n: sess[sess["strategy"] == n]["result"].tolist() for n in names if (sess["strategy"] == n).any()},
                            colors, labels, f"Résultat d'une session de {per} mains (réel)", "bb"),
               "Boîte : quartiles ; trait : médiane ; pointillé : moyenne.")
    table = summ.copy()
    table["strategy"] = [labels.get(x, x) for x in table["strategy"]]
    sec.table(table.rename(columns={"strategy": "stratégie"}), "« P(session en perte) » : probabilité qu'une session finisse négative, même si la stratégie gagne à long terme.")
    for n in names:
        sub = sess[sess["strategy"] == n]
        if len(sub):
            sec.figure(charts.histogram(sub["result"].tolist(), colors[n], f"{labels[n]} : distribution des sessions", "bb"))
    if verdict:
        lines = []
        for _, row in summ.iterrows():
            if row["bb/100 (EV, toutes sessions)"] > 0:
                lines.append(f"{labels.get(row['strategy'], row['strategy'])} gagne {row['bb/100 (EV, toutes sessions)']:+.1f} bb/100 mais finit "
                             f"{row['P(session en perte) %']:.0f} % des sessions de {per} mains en perte.")
        worst = summ.sort_values("pire session (bb)").iloc[0] if len(summ) else None
        if worst is not None:
            lines.append(f"Pire session observée : {worst['pire session (bb)']:.0f} bb ({labels.get(worst['strategy'], worst['strategy'])}).")
        lines.append(f"Une session de {per} mains ne suffit pas à juger une stratégie : l'écart type d'une session dépasse largement son espérance.")
        report.verdict = lines[:5]
