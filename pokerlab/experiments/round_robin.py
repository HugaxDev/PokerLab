"""Expérience « round_robin » : les stratégies les unes contre les autres à la même table.

Avec 8 stratégies et 6 sièges, chaque table est un sous-ensemble de 6 stratégies. Chaque
deck est joué en 6 rotations (chaque stratégie occupe chaque position avec les mêmes
cartes). Les comparaisons A/B sont appariées sur les (table, deck) où A et B sont assis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.engine.hand_runner import COUNTERS, play_hand
from pokerlab.engine.history import format_hand
from pokerlab.experiments.base import ExperimentContext, apply_holm, pairwise_tests, save_outputs
from pokerlab.experiments.common import TaskResult, deck_seed, lab_from_raw, record_seat, run_tasks, split_range
from pokerlab.report import charts, sections
from pokerlab.report.html import Report
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy, resolve_spec


@dataclass
class TableTask:
    lab_raw: dict[str, Any]
    players: list[dict[str, Any]]
    table_id: int
    deck_start: int
    deck_end: int
    record_decisions: bool = True


def run_table_task(task: TableTask) -> TaskResult:
    lab = lab_from_raw(task.lab_raw)
    n = lab.game.players
    built = [build_strategy(p, lab) for p in task.players]
    out = TaskResult()
    c0, k0 = COUNTERS["hands"], COUNTERS["corrections"]
    for d in range(task.deck_start, task.deck_end):
        seed = deck_seed(lab, d)
        deck = shuffled_deck(seed)
        for r in range(n):
            seated = [built[(i - r) % n] for i in range(n)]
            hr = play_hand(seated, lab.game, deck, [lab.game.starting_stack] * n, d * n + r, seed, track=set(range(n)))
            for i in range(n):
                keys = {"table": task.table_id, "deck": d, "rot": r, "hand_id": d * n + r, "strategy": seated[i].name,
                        "composition": f"table {task.table_id}", "lineup": "|".join(p.name for p in seated)}
                record_seat(out, hr, i, lab, keys, task.record_decisions)
    out.hands_played = COUNTERS["hands"] - c0
    out.corrections = COUNTERS["corrections"] - k0
    return out.compact()


def choose_tables(names: list[str], n: int, mode: str | int, seed: int) -> list[list[str]]:
    """Tous les sous-ensembles de ``n`` stratégies, ou un tirage reproductible."""
    if len(names) <= n:
        return [names]
    subsets = [list(c) for c in combinations(names, n)]
    if mode == "all":
        return subsets
    k = int(mode)
    rng = random.Random(derive_seed(seed, "round_robin_tables"))
    picked = rng.sample(subsets, min(k, len(subsets)))
    return picked


def run(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    specs = cfg.get("strategies", [])
    cfgs = [resolve_spec(s) for s in specs]
    names = [c["name"] for c in cfgs]
    by_name = dict(zip(names, cfgs))
    n = ctx.lab.game.players
    tables = choose_tables(names, n, cfg.get("tables", "all"), ctx.lab.seed)
    decks = int(cfg.get("decks_per_table", 50))
    ctx.log(f"round_robin : {len(tables)} tables x {decks} decks x {n} rotations")
    tasks = []
    for t_id, table in enumerate(tables):
        for a, b in split_range(0, decks, max(1, ctx.workers)):
            tasks.append(TableTask(ctx.lab.raw, [by_name[x] for x in table], t_id, a, b, cfg.get("record_decisions", True)))
    outs = run_tasks(run_table_task, tasks, ctx.workers, ctx.lab.mp_start_method)
    merged = TaskResult()
    for o in outs:
        merged.merge(o)
    ctx.hands_played += merged.hands_played
    hands = merged.hands.frame()
    decisions = merged.decisions.frame()
    report = build_report(ctx, specs, names, tables, hands, decisions)
    histories = []
    for _, row in hands.drop_duplicates(["table", "deck", "rot"]).head(int(cfg.get("history_samples", 10))).iterrows():
        histories.append(replay_text(ctx, by_name, tables, row))
    paths = save_outputs(ctx, report, hands, decisions, histories, {"tables": tables, "corrections_actions": merged.corrections})
    return {"report": report, "paths": paths, "hands": hands, "decisions": decisions}


def replay_text(ctx: ExperimentContext, by_name: dict[str, dict], tables: list[list[str]], row: pd.Series) -> str:
    lab = ctx.lab
    n = lab.game.players
    table = tables[int(row["table"])]
    built = [build_strategy(by_name[x], lab) for x in table]
    r, d = int(row["rot"]), int(row["deck"])
    seated = [built[(i - r) % n] for i in range(n)]
    seed = deck_seed(lab, d)
    hr = play_hand(seated, lab.game, shuffled_deck(seed), [lab.game.starting_stack] * n, d * n + r, seed)
    return format_hand(hr, lab.game, f"round robin table {int(row['table'])}", [s.name for s in seated])


def build_report(ctx: ExperimentContext, specs, names, tables, hands, decisions) -> Report:
    _, colors, labels = sections.colors_labels(specs)
    conf = ctx.confidence
    report = Report("PokerLab — round robin", "Les stratégies s'affrontent à la même table (duplicate par rotation)")
    report.meta = {"tables": len(tables), "mains": f"{hands[['table', 'deck', 'rot']].drop_duplicates().shape[0]:,}".replace(",", " ")}
    # Unité appariée : (table, deck)
    hands = hands.copy()
    hands["unit"] = hands["table"].astype("int64") * 1_000_000 + hands["deck"].astype("int64")
    summaries = {}
    for nme in names:
        h = hands[hands["strategy"] == nme].rename(columns={"deck": "deck_raw"}).rename(columns={"unit": "deck"})
        if len(h):
            summaries[nme] = sections.summarize_strategy(h, nme, conf)
    sec = report.section("Classement")
    sections.comparison_table(sec, summaries, colors, labels, "Chaque stratégie est mesurée sur toutes les tables où elle est assise.")
    sec.figure(sections.winrate_figure(summaries, colors, labels))
    dm = {nme: hands[hands["strategy"] == nme].groupby("unit")["result_ev"].mean() for nme in names}
    tests = apply_holm(pairwise_tests(dm, method=ctx.cfg.get("test", "t"), confidence=conf, seed=ctx.lab.seed, labels=labels))
    mat = np.full((len(names), len(names)), np.nan)
    idx = {labels[x]: i for i, x in enumerate(names)}
    for t in tests:
        mat[idx[t.a], idx[t.b]] = t.diff_bb100
        mat[idx[t.b], idx[t.a]] = -t.diff_bb100
    sig = report.section("Face-à-face (différences appariées)")
    sig.figure(charts.matrix_heatmap(mat, [labels[x] for x in names], [labels[x] for x in names],
                                     "Ligne − colonne (bb/100, mêmes tables et mêmes decks)"))
    sig.table(pd.DataFrame({"verdict": [t.verdict() for t in tests], "unités appariées": [t.n_units for t in tests]}), digits=0)
    ranked = sorted(summaries, key=lambda x: summaries[x].ev.bb100, reverse=True)
    report.verdict = ["Classement round robin : " + " > ".join(labels[x] for x in ranked) + "."]
    if len(ranked) >= 2:
        top = [t for t in tests if {t.a, t.b} == {labels[ranked[0]], labels[ranked[1]]}]
        if top:
            report.verdict.append(top[0].verdict() + ".")
    sig_count = sum(t.significant for t in tests)
    report.verdict.append(f"{sig_count} comparaisons significatives sur {len(tests)} (Holm).")
    report.verdict.append(f"Meilleur : {labels[ranked[0]]} ({summaries[ranked[0]].ev.bb100:+.1f} ± {summaries[ranked[0]].ev.ci_half:.1f} bb/100)."
                          if ranked else "Pas de données.")
    report.verdict.append(f"{len(tables)} tables de {ctx.lab.game.players} stratégies.")
    st = report.section("Statistiques mesurées")
    sections.add_player_stats(st, summaries, labels)
    return report
