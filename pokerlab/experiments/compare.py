"""Expérience « compare » : chaque stratégie contre chaque composition, en duplicate."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pokerlab.engine.history import format_hand
from pokerlab.analysis.robustness import cell_summaries, robustness_table, winner_verdict
from pokerlab.experiments.base import ExperimentContext, apply_holm, composition_dict, pairwise_tests, save_outputs
from pokerlab.experiments.common import replay_duplicate_hand
from pokerlab.experiments.duplicate import Autostop, Cell, collect, run_cells
from pokerlab.report import charts, sections
from pokerlab.report.html import Report
from pokerlab.strategies.registry import archetypes, resolve_spec


def build_cells(strategies: list[Any], compositions: list[str], decision_decks: int | None = None) -> list[Cell]:
    """Cellules stratégie x composition ; décisions enregistrées pour la composition principale seulement."""
    cells = []
    for k, comp in enumerate(compositions):
        cdict = composition_dict(comp)
        for spec in strategies:
            cfg = resolve_spec(spec)
            cells.append(Cell(f"{cfg['name']}@{comp}", cfg, cdict, {"strategy": cfg["name"], "composition": comp},
                              record_decisions=k == 0, decision_decks=decision_decks))
    return cells


def make_replayer(ctx: ExperimentContext, cells: list[Cell]):
    by_key = {(c.keys["strategy"], c.keys["composition"]): c for c in cells}

    def replay(name: str, row: pd.Series) -> str:
        cell = by_key[(name, row["composition"])]
        hr, labels = replay_duplicate_hand(ctx.lab, cell.hero, cell.composition, int(row["deck"]), int(row["rot"]), cell.deck_space)
        return format_hand(hr, ctx.lab.game, row["composition"], labels, hero=int(row["rot"]))

    return replay


def run(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    strategies = cfg.get("strategies", [])
    compositions = cfg.get("compositions", ["micro_play_money"])
    auto = Autostop.from_dict(cfg.get("autostop"), cfg.get("decks"))
    auto.stop_file = str(ctx.out_dir / "STOP")
    ctx.log(f"compare : {len(strategies)} stratégies x {len(compositions)} compositions, workers={ctx.workers}")
    cells = build_cells(strategies, compositions, cfg.get("decision_decks", 10_000))
    results, progress = run_cells(cells, ctx.lab, auto, ctx.workers, record_decisions=cfg.get("record_decisions", True), log=ctx.log)
    hands, decisions, flags, played, corrections = collect(results)
    ctx.hands_played += played
    report = build_report(ctx, cells, hands, decisions, flags, progress, strategies, compositions)
    histories = sample_histories(ctx, cells, hands, int(cfg.get("history_samples", 20)))
    summary = {"cellules": {k: {"decks": p.decks, "arret": p.stopped, "bb100_metric": p.bb100()} for k, p in progress.items()},
               "resultats": cell_summaries(hands, ctx.confidence),
               "compteurs_joueurs": {k: [int(x) for x in v] for k, v in flags.items()},
               "corrections_actions": corrections}
    paths = save_outputs(ctx, report, hands, decisions, histories, summary)
    return {"report": report, "paths": paths, "hands": hands, "decisions": decisions, "flags": flags, "cells": cells}


def rebuild_report(ctx: ExperimentContext) -> dict[str, Any]:
    """Régénère rapport.html d'un compare déjà joué (mains.parquet, decisions.parquet, resume.json), sans re-simuler."""
    import json

    from pokerlab.experiments.duplicate import CellProgress

    folder = ctx.out_dir
    meta = json.loads((folder / "resume.json").read_text(encoding="utf-8"))
    cfg = meta.get("config") or ctx.cfg
    ctx.cfg = cfg
    strategies = cfg.get("strategies", [])
    compositions = cfg.get("compositions", ["micro_play_money"])
    hands = pd.read_parquet(folder / "mains.parquet")
    for col in ("strategy", "composition"):
        hands[col] = hands[col].astype(str)
    dec_path = folder / "decisions.parquet"
    decisions = pd.read_parquet(dec_path) if dec_path.exists() else pd.DataFrame()
    if not decisions.empty:
        for col in ("strategy", "composition"):
            decisions[col] = decisions[col].astype(str)
    progress = {k: CellProgress(decks=int(v.get("decks", 0)), stopped=str(v.get("arret", ""))) for k, v in (meta.get("cellules") or {}).items()}
    flags = {k: list(v) for k, v in (meta.get("compteurs_joueurs") or {}).items()}
    ctx.hands_played = int(meta.get("mains_jouees", len(hands)))
    cells = build_cells(strategies, compositions, cfg.get("decision_decks", 10_000))
    report = build_report(ctx, cells, hands, decisions, flags, progress, strategies, compositions)
    if not flags:
        report.meta["note"] = "rapport régénéré depuis les données sauvegardées ; stats mesurées des archétypes non sauvegardées pour cette campagne"
    path = report.save(folder / "rapport.html")
    return {"report": report, "paths": {"rapport": path}, "hands": hands}


def sample_histories(ctx: ExperimentContext, cells: list[Cell], hands: pd.DataFrame, k: int) -> list[str]:
    if hands.empty or k <= 0:
        return []
    replay = make_replayer(ctx, cells)
    out = []
    for _, row in hands.head(k).iterrows():
        out.append(replay(row["strategy"], row))
    return out


def comparison_block(report: Report, title: str, hands: pd.DataFrame, names: list[str], colors, labels, confidence: float):
    sec = report.section(title)
    summaries = {n: sections.summarize_strategy(hands[hands["strategy"] == n], n, confidence) for n in names if (hands["strategy"] == n).any()}
    sections.comparison_table(sec, summaries, colors, labels, "Meilleure valeur de chaque ligne colorée avec la couleur de sa stratégie. "
                              "σ effectif : écart type équivalent après duplicate et all-in EV.")
    return sec, summaries


def build_report(ctx, cells, hands, decisions, flags, progress, strategies, compositions) -> Report:
    names, colors, labels = sections.colors_labels(strategies)
    conf = ctx.confidence
    an = ctx.analysis
    report = Report("PokerLab — comparaison des stratégies", "Duplicate + all-in EV, mêmes decks pour toutes les stratégies")
    report.meta = {"mains jouées": f"{ctx.hands_played:,}".replace(",", " "), "graine": ctx.lab.seed, "compositions": ", ".join(compositions)}
    partial = [k for k, p in progress.items() if "partiels" in p.stopped]
    if partial:
        report.meta["ATTENTION"] = f"campagne interrompue, {len(partial)} cellule(s) avec résultats partiels"
    main_comp = compositions[0]
    main = hands[hands["composition"] == main_comp]
    sec, summaries = comparison_block(report, f"Tableau comparatif — {main_comp}", main, names, colors, labels, conf)
    # Matrice stratégie x composition
    matrix = np.full((len(names), len(compositions)), np.nan)
    halfs = np.full_like(matrix, np.nan)
    for i, n in enumerate(names):
        for j, comp in enumerate(compositions):
            h = hands[(hands["strategy"] == n) & (hands["composition"] == comp)]
            if len(h):
                s = sections.summarize_strategy(h, n, conf)
                matrix[i, j], halfs[i, j] = s.ev.bb100, s.ev.ci_half
    text = np.vectorize(lambda v, h: "" if not np.isfinite(v) else f"{v:+.1f} ± {h:.1f}")(matrix, halfs)
    sec.figure(charts.matrix_heatmap(matrix, [labels[n] for n in names], compositions, "Matrice stratégie × composition (bb/100 all-in adjusted)", text=text))
    mtable = pd.DataFrame({"stratégie": [labels[n] for n in names]})
    for j, comp in enumerate(compositions):
        mtable[comp] = [f"{matrix[i, j]:+.1f} ± {halfs[i, j]:.1f}" if np.isfinite(matrix[i, j]) else "–" for i in range(len(names))]
    sec.table(mtable, "Table de lecture de la matrice (IC 95 %).")
    rob = robustness_table(pd.DataFrame(cell_summaries(hands, conf)))
    if len(compositions) > 1:
        rob_view = pd.DataFrame({
            "stratégie": [labels.get(x, x) + (" (réf.)" if r else "") for x, r in zip(rob["strategy"], rob["reference"])],
            "pire cas (bb/100)": [f"{v:+.1f} ± {c:.1f}" for v, c in zip(rob["pire_cas"], rob["pire_cas_ci"])],
            "pire composition": rob["pire_composition"],
            "médiane (bb/100)": rob["bb100_median"],
            "borne basse min": rob["borne_basse_min"],
            "gagnante partout": ["oui" if g else "non" for g in rob["gagnante_partout"]],
        })
        sec.table(rob_view, "Robustesse sur les compositions de la campagne, classée par pire cas (plus faible bb/100). "
                  "« Gagnante partout » : borne basse de l'IC 95 % > 0 sur toutes les compositions.")
    # Tests appariés
    tests = []
    for comp in compositions:
        sub = hands[hands["composition"] == comp]
        dm = {n: sub[sub["strategy"] == n].groupby("deck")["result_ev"].mean() for n in names if (sub["strategy"] == n).any()}
        for t in pairwise_tests(dm, method=ctx.cfg.get("test", "t"), confidence=conf, block=int(an.get("bootstrap_block", 50)),
                                resamples=int(an.get("bootstrap_resamples", 2000)), seed=ctx.lab.seed, labels=labels):
            tests.append((comp, t))
    adjusted = apply_holm([t for _, t in tests])
    tests = [(c, t) for (c, _), t in zip(tests, adjusted)]
    sig = report.section("Significativité (tests appariés, correction de Holm)")
    main_tests = [t for c, t in tests if c == main_comp]
    if main_tests:
        lab_names = [labels[n] for n in names]
        pmat = np.full((len(names), len(names)), np.nan)
        idx = {labels[n]: i for i, n in enumerate(names)}
        for t in main_tests:
            pmat[idx[t.a], idx[t.b]] = pmat[idx[t.b], idx[t.a]] = t.p_holm
        sig.figure(charts.sequential_heatmap(pmat, lab_names, lab_names, f"p-valeurs Holm — {main_comp}", "p Holm"),
                   "Plus foncé = différence plus significative. Seuil : 0,05.")
    sig.table(pd.DataFrame({
        "composition": [c for c, _ in tests],
        "verdict": [t.verdict() for _, t in tests],
        "p brut": [t.p_value for _, t in tests],
        "decks appariés": [t.n_units for _, t in tests],
    }), "Différence = moyenne par deck (6 rotations) A − B, all-in adjusted.", digits=4)
    # Verdict
    report.verdict = verdict_lines(names, labels, hands, compositions, tests, main_comp, summaries, progress, ctx, rob)
    # Graphiques
    g = report.section(f"Graphiques — {main_comp}")
    g.figure(sections.winrate_figure(summaries, colors, labels), "Barres pleines : all-in adjusted avec IC 95 % ; claires : résultat réel.")
    by_name = {n: main[main["strategy"] == n].sort_values(["deck", "rot"]) for n in names if (main["strategy"] == n).any()}
    max_pts = int(an.get("max_curve_points", 3000))
    sections.add_curves(g, by_name, colors, labels, max_pts)
    sections.add_decomposition(g, by_name, colors, labels, ctx.lab.game.players)
    sections.add_showdown_lines(g, by_name, labels, max_pts)
    st = report.section("Statistiques mesurées")
    sections.add_player_stats(st, summaries, labels)
    st.html("<h3>Archétypes adverses : annoncé vs mesuré</h3>")
    if flags:
        sections.add_archetype_stats(st, flags, archetypes(), exclude=names)
    else:
        st.paragraph("Compteurs des adversaires non sauvegardés pour cette campagne (rapport régénéré).")
    lk = report.section("Fuites détectées")
    lk.paragraph("Situations récurrentes où continuer la main (payer, miser, relancer) perd de l'argent en moyenne à partir de la décision.")
    dec_main = decisions[decisions["composition"] == main_comp] if not decisions.empty else decisions
    dec_by = {n: dec_main[dec_main["strategy"] == n] for n in names} if not dec_main.empty else {}
    sections.add_leaks(lk, dec_by, labels, make_replayer(ctx, cells), int(an.get("leak_min_count", 30)), int(an.get("leak_examples", 3)),
                       confidence=conf)
    meth = report.section("Méthode et arrêt")
    meth.table(pd.DataFrame({"cellule": list(progress), "mains": [p.decks * ctx.lab.game.players for p in progress.values()],
                             "arrêt": [p.stopped for p in progress.values()]}), "Chaque deck est rejoué 6 fois (une par siège).", digits=0)
    return report


def verdict_lines(names, labels, hands, compositions, tests, main_comp, summaries, progress, ctx, rob=None) -> list[str]:
    """Une stratégie n'est déclarée gagnante que si sa borne basse d'IC est > 0 sur toutes les compositions."""
    lines = []
    if rob is None:
        rob = robustness_table(pd.DataFrame(cell_summaries(hands, ctx.confidence)))
    winner, text = winner_verdict(rob, labels)
    lines.append(text)
    if winner is not None and len(rob) > 1:
        others = rob[(~rob["reference"]) & (rob["strategy"] != winner)]
        if not others.empty:
            second = others.iloc[0]["strategy"]
            match = [t for c, t in tests if c == main_comp and {t.a, t.b} == {labels[winner], labels[second]}]
            if match:
                lines.append(match[0].verdict() + f" — sur {main_comp}.")
    per_comp = []
    for comp in compositions:
        sub = hands[hands["composition"] == comp]
        if sub.empty:
            continue
        m = sub.groupby("strategy").apply(lambda h: h.groupby("deck")["result_ev"].mean().mean() * 100, include_groups=False)
        per_comp.append(f"{comp} → {labels.get(m.idxmax(), m.idxmax())} ({m.max():+.1f})")
    if per_comp:
        lines.append("Meilleur bb/100 par composition (sans condition d'IC) : " + " ; ".join(per_comp) + ".")
    n_hands = sum(p.decks for p in progress.values()) * ctx.lab.game.players
    halfs = [s.ev.ci_half for s in summaries.values() if np.isfinite(s.ev.ci_half)]
    n_txt = f"{n_hands:,}".replace(",", " ")
    mean_half = float(np.mean(halfs)) if halfs else float("nan")
    lines.append(f"Précision : {n_txt} mains au total, IC 95 % moyen ± {mean_half:.1f} bb/100 sur {main_comp}.")
    return lines[:5]
