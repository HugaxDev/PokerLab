"""Expérience « audit » : audit contrefactuel des décisions par rollouts.

Pour chaque décision échantillonnée, on rejoue la main à l'identique jusqu'à la décision,
on impose chaque action alternative légale, puis on termine la main R fois (cartes à venir
retirées au hasard, aléa des adversaires resalé). Les mêmes R tirages servent à toutes les
actions (comparaison appariée). L'EV perdue est estimée par validation croisée (choix de la
meilleure action sur une moitié des rollouts, mesure sur l'autre) pour ne pas la surestimer.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.engine.hand_runner import COUNTERS, play_hand
from pokerlab.engine.history import format_hand
from pokerlab.cards.ranges import COMBO_CARDS
from pokerlab.engine.types import Action, ActionRecord, Kind, Observation
from pokerlab.equity.montecarlo import sample_villains
from pokerlab.strategies.range_model import RangeModel
from pokerlab.experiments.base import ExperimentContext, composition_dict, save_outputs
from pokerlab.experiments.common import (
    TaskResult,
    deck_seed,
    lab_from_raw,
    lineup_unit,
    replay_duplicate_hand,
    run_tasks,
    seating,
    split_range,
)
from pokerlab.experiments.duplicate import Autostop, Cell, collect, run_cells
from pokerlab.analysis.leaks import AUDIT_GROUP_KEYS, leak_title, situation_labels
from pokerlab.analysis.stats import z_value
from pokerlab.opponents.compositions import Composition
from pokerlab.report import charts, sections
from pokerlab.report.html import Report, esc
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy, get_context, resolve_spec

BOARD_DEALT = {0: 0, 1: 3, 2: 4, 3: 5}


class _Capture:
    """Enveloppe qui mémorise l'observation d'une décision donnée."""

    def __init__(self, inner: Any, index: int) -> None:
        self.inner = inner
        self.name = inner.name
        self.index = index
        self.obs: Observation | None = None

    def decide(self, obs: Observation) -> Action:
        if obs.decision_index == self.index:
            self.obs = obs
        return self.inner.decide(obs)


def alternatives(obs: Observation, sizes: list[float], include_allin: bool) -> list[tuple[str, Action]]:
    """Actions alternatives légales : se coucher, suivre/checker, quelques tailles de mise."""
    out: list[tuple[str, Action]] = []
    legal = obs.legal
    if legal.can_fold:
        out.append(("se couche", Action.fold()))
    out.append(("check" if obs.to_call == 0 else "paie", Action.check_call()))
    if legal.can_raise:
        targets: list[tuple[str, int]] = []
        max_bet = max(obs.bets)
        if obs.street == 0:
            targets += [("relance min", legal.min_raise_to), ("relance 3x", 3 * max_bet)]
        else:
            for f in sizes:
                if obs.to_call == 0:
                    targets.append((f"mise {f:g} pot", obs.bets[obs.seat] + int(round(f * obs.pot))))
                else:
                    targets.append((f"relance {f:g} pot", max_bet + int(round(f * (obs.pot + obs.to_call)))))
        if include_allin:
            targets.append(("tapis", legal.max_raise_to))
        seen: set[int] = set()
        for label, t in targets:
            t = min(max(int(t), legal.min_raise_to), legal.max_raise_to)
            if t not in seen:
                seen.add(t)
                out.append((label, Action.raise_to(t)))
    return out


@dataclass
class AuditTask:
    lab_raw: dict[str, Any]
    hero: dict[str, Any]
    composition: dict[str, Any]
    decisions: list[dict[str, Any]]
    rollouts: int
    sizes: list[float]
    include_allin: bool
    range_model: dict[str, Any] | None = None


def recorded_action(rec: ActionRecord) -> Action:
    """Action de l'historique rejouable à l'identique (montant total visé pour une mise)."""
    if rec.kind == Kind.FOLD:
        return Action.fold()
    if rec.kind in (Kind.BET, Kind.RAISE):
        return Action.raise_to(rec.to)
    return Action.check_call()


def forced_prefix(history: list[ActionRecord], idx: int) -> dict[int, Action]:
    """Impose toutes les décisions déjà prises avant la décision auditée (index global < idx)."""
    out: dict[int, Action] = {}
    k = 0
    for rec in history:
        if rec.kind == Kind.POST:
            continue
        if k >= idx:
            break
        out[k] = recorded_action(rec)
        k += 1
    return out


def posterior_deck(deck: list[int], obs: Observation, n: int, range_model: RangeModel, seed: int) -> list[int]:
    """Paquet de rollout avec l'information du joueur : ses cartes et le tableau vu restent ;
    les adversaires encore en jeu reçoivent des mains tirées dans leur range estimée d'après
    leurs actions ; le reste (joueurs couchés, cartes à venir) est tiré au hasard."""
    rng = np.random.default_rng(seed)
    dealt = BOARD_DEALT[obs.street]
    board = list(deck[2 * n: 2 * n + dealt])
    dead = list(obs.hole) + board
    active = obs.active_opponents
    out = list(deck)
    used = set(dead)
    if active:
        weights = [range_model.villain_range(obs, v)[1] for v in active]
        combos = sample_villains(rng, weights, dead, 1)[0]
        for v, c in zip(active, combos):
            a, b = (int(x) for x in COMBO_CARDS[int(c)])
            out[2 * v], out[2 * v + 1] = a, b
            used.update((a, b))
    rest = [c for c in range(52) if c not in used]
    rng.shuffle(rest)
    it = iter(rest)
    for i in range(n):
        if i != obs.seat and i not in active:
            out[2 * i], out[2 * i + 1] = next(it), next(it)
    for pos in range(2 * n + dealt, 2 * n + 5):
        out[pos] = next(it)
    out[2 * n + 5:] = list(it)
    return out


def run_audit_task(task: AuditTask) -> TaskResult:
    lab = lab_from_raw(task.lab_raw)
    n = lab.game.players
    comp = Composition.from_dict(task.composition)
    hero = build_strategy(task.hero, lab)
    out = TaskResult()
    c0 = COUNTERS["hands"]
    stacks = [lab.game.starting_stack] * n
    model = RangeModel(get_context(lab).table, task.range_model)
    for d in task.decisions:
        deck_i, rot, idx = int(d["deck"]), int(d["rot"]), int(d["idx"])
        names = comp.opponents(lab.seed, lineup_unit(deck_i, "deck"), n - 1)
        opponents = [build_strategy(resolve_spec(x), lab) for x in names]
        seed = deck_seed(lab, deck_i)
        deck = shuffled_deck(seed)
        cap = _Capture(hero, idx)
        original = play_hand(seating(cap, opponents, rot, n), lab.game, deck, stacks, deck_i * n + rot, seed)
        obs = cap.obs
        if obs is None:
            continue
        alts = alternatives(obs, task.sizes, task.include_allin)
        taken_kind, taken_to = int(d["kind"]), int(round(d["to_bb"] * lab.game.chips_per_bb))
        taken = 0
        for j, (_, a) in enumerate(alts):
            if (taken_kind == 0 and a.kind == "fold") or (taken_kind in (1, 2) and a.kind == "call"):
                taken = j
        if taken_kind in (3, 4):
            match = [j for j, (_, a) in enumerate(alts) if a.kind == "raise" and a.amount == taken_to]
            if match:
                taken = match[0]
            else:
                alts.append(("action prise", Action.raise_to(taken_to)))
                taken = len(alts) - 1
        players = seating(hero, opponents, rot, n)
        prefix = forced_prefix(original.history, idx)
        ev = np.zeros((len(alts), task.rollouts))
        for r in range(task.rollouts):
            rdeck = posterior_deck(deck, obs, n, model, derive_seed(seed, "rollout", idx, r))
            for j, (_, a) in enumerate(alts):
                forced = dict(prefix)
                forced[idx] = a
                hr = play_hand(players, lab.game, rdeck, stacks, deck_i * n + rot, seed, forced=forced, fork_after=idx, fork_salt=r)
                ev[j, r] = hr.ev_payoffs[rot] / lab.game.chips_per_bb
        means = ev.mean(axis=1)
        even, odd = ev[:, 0::2], ev[:, 1::2]
        if odd.shape[1]:
            b1 = int(np.argmax(even.mean(axis=1)))
            b2 = int(np.argmax(odd.mean(axis=1)))
            loss = 0.5 * ((odd[b1].mean() - odd[taken].mean()) + (even[b2].mean() - even[taken].mean()))
        else:
            loss = float(means.max() - means[taken])
        best = int(np.argmax(means))
        row = dict(d)
        row.update(
            ev_taken=float(means[taken]),
            ev_best=float(means[best]),
            best_action=alts[best][0],
            taken_action=alts[taken][0],
            loss_cv=float(loss),
            loss_naive=float(means[best] - means[taken]),
            alternatives=json.dumps({lbl: round(float(m), 3) for (lbl, _), m in zip(alts, means)}, ensure_ascii=False),
        )
        out.hands.add(**row)
    out.hands_played = COUNTERS["hands"] - c0
    return out.compact()


def sample_decisions(dec: pd.DataFrame, n: int, seed: int, table_pct: np.ndarray | None, nontrivial_pct: float) -> pd.DataFrame:
    """Échantillon reproductible ; exclut les décisions sans alternative et les abandons préflop triviaux."""
    d = dec[(dec["can_fold"]) | (dec["can_raise"])].copy()
    if table_pct is not None and nontrivial_pct < 1.0:
        pct = table_pct[d["hand_class"].to_numpy(dtype=int)]
        trivial = (d["street"] == 0) & (d["kind"] == 0) & (pct > nontrivial_pct)
        d = d[~trivial]
    if len(d) <= n:
        return d.reset_index(drop=True)
    return d.sample(n=n, random_state=seed % (2 ** 32)).sort_values(["deck", "rot", "idx"]).reset_index(drop=True)


def run(ctx: ExperimentContext, report: Report | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    hero = resolve_spec(cfg["strategy"])
    comp = cfg.get("composition", "micro_play_money")
    cdict = composition_dict(comp)
    pool_decks = int(cfg.get("pool_decks", 400))
    ctx.log(f"audit : {hero['name']} contre {comp}, réserve de {pool_decks} decks")
    cells = [Cell(hero["name"], hero, cdict, {"strategy": hero["name"], "composition": comp})]
    max_minutes = float(cfg.get("max_minutes", 120))
    stop_file = ctx.out_dir / "STOP"
    auto = Autostop(enabled=False, decks=pool_decks, block_decks=int(cfg.get("block_decks", 2000)), max_minutes=max_minutes, stop_file=str(stop_file))
    results, progress = run_cells(cells, ctx.lab, auto, ctx.workers, record_decisions=True, log=ctx.log)
    ctx.partial.extend(sorted({p.stopped for p in progress.values() if "partiels" in p.stopped}))
    _, decisions, _, played, _ = collect(results)
    ctx.hands_played += played
    table = get_context(ctx.lab).table
    n_dec = int(cfg.get("n_decisions", 2000))
    sample = sample_decisions(decisions, n_dec, derive_seed(ctx.lab.seed, "audit"), table.class_percentile, float(cfg.get("nontrivial_preflop_pct", 0.5)))
    rollouts = int(cfg.get("rollouts", 16))
    ctx.log(f"audit : {len(sample)} décisions x {rollouts} rollouts par action")
    chunk = int(cfg.get("decisions_par_tache", 25))
    tasks = [AuditTask(ctx.lab.raw, hero, cdict, sample.iloc[a:b].to_dict("records"), rollouts, list(cfg.get("bet_sizes_pot", [0.5, 1.0])),
                       bool(cfg.get("include_allin", True)), cfg.get("range_model", {"looseness": 1.3}))
             for a, b in split_range(0, len(sample), max(1, ctx.workers * 4, -(-len(sample) // chunk)))]
    merged = TaskResult()
    batch = max(1, ctx.workers * 4)
    t_start, done = time.perf_counter(), 0
    for k in range(0, len(tasks), batch):
        if ctx.partial:
            break
        for o in run_tasks(run_audit_task, tasks[k:k + batch], ctx.workers, ctx.lab.mp_start_method):
            merged.merge(o)
        done += sum(len(t.decisions) for t in tasks[k:k + batch])
        elapsed = time.perf_counter() - t_start
        ctx.log(f"  audit : {done}/{len(sample)} décisions en {elapsed / 60:.1f} min")
        if k + batch < len(tasks):
            if (time.perf_counter() - ctx.started) > max_minutes * 60:
                ctx.partial.append(f"durée maximale atteinte ({max_minutes:g} min) : {done} décisions auditées sur {len(sample)}")
            elif stop_file.exists():
                ctx.partial.append(f"arrêt demandé (fichier STOP) : {done} décisions auditées sur {len(sample)}")
            if ctx.partial:
                ctx.log(f"  ARRÊT : {ctx.partial[-1]}")
    ctx.hands_played += merged.hands_played
    audited = merged.hands.frame()
    own = report is None
    report = report or Report(f"PokerLab — audit contrefactuel de {hero.get('label', hero['name'])}", f"{len(audited)} décisions, {rollouts} rollouts par action")
    leaks = add_audit_section(report, ctx, hero, cdict, audited)
    if own:
        total = audited["loss_cv"].sum() if len(audited) else 0.0
        report.verdict = [f"EV perdue moyenne : {audited['loss_cv'].mean():.3f} bb par décision auditée ({len(audited)} décisions)." if len(audited) else "Aucune décision auditée.",
                          f"Perte totale estimée sur l'échantillon : {total:.1f} bb."]
        if leaks is not None and len(leaks):
            for _, row in leaks.head(3).iterrows():
                report.verdict.append(f"Fuite : {leak_title(row, list(cfg.get('group_by', AUDIT_GROUP_KEYS)))} — {row['mean']:.2f} bb par décision ({int(row['count'])} cas).")
        paths = save_outputs(ctx, report, None, audited, (), {"decisions_auditees": len(audited)})
    else:
        paths = {}
    return {"report": report, "paths": paths, "audited": audited}


def add_audit_section(report: Report, ctx: ExperimentContext, hero: dict[str, Any], cdict: dict[str, Any], audited: pd.DataFrame):
    sec = report.section(f"Audit contrefactuel — {hero.get('label', hero['name'])}")
    if audited.empty:
        sec.paragraph("Aucune décision auditée.")
        return None
    d = situation_labels(audited.assign(result_ev=0.0))
    keys = list(ctx.cfg.get("group_by", AUDIT_GROUP_KEYS))
    g = d.groupby(keys)["loss_cv"].agg(["count", "mean", "std", "sum"]).reset_index()
    g["ci_half"] = z_value(ctx.confidence) * g["std"].fillna(0) / np.sqrt(g["count"])
    g = g.sort_values("sum", ascending=False).reset_index(drop=True)
    top = g.head(int(ctx.cfg.get("top_leaks", 15)))
    color = sections.colors_labels([hero["name"]])[1][hero["name"]]
    sec.figure(charts.bars_with_ci([leak_title(r, keys) for _, r in top.iterrows()], top["mean"].tolist(), top["ci_half"].tolist(), color,
                                   "EV perdue par décision, situations les plus coûteuses (classées par perte totale)", "bb"),
               "EV perdue = EV de la meilleure alternative − EV de l'action prise (validation croisée sur les rollouts).")
    sec.table(pd.DataFrame({"situation": [leak_title(r, keys) for _, r in top.iterrows()], "décisions": top["count"].astype(int),
                            "EV perdue / décision (bb)": top["mean"], "IC ±": top["ci_half"], "perte totale (bb)": top["sum"]}), digits=3)
    by_street = d.groupby("street_label")["loss_cv"].agg(["count", "mean", "sum"]).reset_index()
    sec.table(by_street.rename(columns={"street_label": "street", "count": "décisions", "mean": "EV perdue moy. (bb)", "sum": "total (bb)"}),
              "Par street.", digits=3)
    k = int(ctx.analysis.get("leak_examples", 3))
    lab = ctx.lab
    for _, leak in top.head(8).iterrows():
        mask = np.ones(len(d), dtype=bool)
        for key in keys:
            mask &= (d[key] == leak[key]).to_numpy()
        ex = d[mask].sort_values("loss_cv", ascending=False).head(k)
        blocks = []
        for _, row in ex.iterrows():
            hr, labels = replay_duplicate_hand(lab, hero, cdict, int(row["deck"]), int(row["rot"]))
            alts = json.loads(row["alternatives"])
            alt_txt = " · ".join(f"{k2} : {v:+.2f} bb" for k2, v in alts.items())
            blocks.append(f"<p>Décision n° {int(row['idx'])} ({esc(row['street_label'])}) : prise = {esc(row['taken_action'])}, "
                          f"meilleure = {esc(row['best_action'])}, EV perdue ≈ {row['loss_cv']:.2f} bb</p><p class='legend-note'>{esc(alt_txt)}</p>"
                          f"<pre>{esc(format_hand(hr, lab.game, cdict['name'], labels, hero=int(row['rot'])))}</pre>")
        sec.details(f"Exemples : {leak_title(leak, keys)}", "".join(blocks))
    return top
