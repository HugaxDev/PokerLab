"""Rapport du journal réel : vraies stats et position dans la distribution simulée."""

from __future__ import annotations

from typing import Any

import pandas as pd

from pokerlab.analysis.player_stats import stats_from_hands
from pokerlab.config import load_yaml
from pokerlab.experiments import sessions as sessions_exp
from pokerlab.experiments.base import ExperimentContext, save_outputs
from pokerlab.journal.importer import flag_off_sheet, journal_stats, load_journal, percentile_of, window_distribution
from pokerlab.report import charts, sections
from pokerlab.report.html import Report


def run(ctx: ExperimentContext, sim_hands: pd.DataFrame | None = None, report: Report | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    phases_cfg = load_yaml(cfg["phases"]).get("phases", {}) if isinstance(cfg.get("phases"), str) else cfg.get("phases", {})
    phases = {str(k): v for k, v in phases_cfg.items()}
    journal = load_journal(cfg["csv"], phases)
    from pokerlab.fiche.build import chart_for, fiche_config

    raise_bb = float(fiche_config().get("reference_raise_bb", 3.0))
    sheets = {st: chart_for(st, ctx.lab, raise_bb) for st in {v.get("strategie") for v in phases.values() if v.get("strategie")}}
    df = flag_off_sheet(journal.hands, sheets)
    strategies = sorted({v.get("strategie") for v in phases.values() if v.get("strategie")})
    if sim_hands is None:
        sub = cfg.get("simulation", {})
        n_sessions = int(sub.get("n_sessions", 2000))
        per = int(sub.get("hands_per_session", 50))
        ctx.log(f"journal : simulation de {n_sessions} sessions de {per} mains pour {', '.join(strategies)}")
        sim_hands = sessions_exp.simulate(ctx, strategies, sub.get("composition", "micro_play_money"), n_sessions, per)
    _, colors, labels = sections.colors_labels(strategies) if strategies else ([], {}, {})
    own = report is None
    report = report or Report("PokerLab — mon journal réel", "Comparaison de mes mains jouées avec la stratégie simulée de chaque phase")
    sec = report.section("Journal réel : mes vraies stats")
    stats = journal_stats(df)
    sec.table(stats, "VPIP/PFR calculés depuis « action_resume » (mots-clés) et « mise_totale ».")
    by_pos = df.groupby("position")["result_bb"].agg(["count", "sum", "mean"]).reset_index()
    sec.table(by_pos.rename(columns={"count": "mains", "sum": "résultat (bb)", "mean": "bb/main"}), "Résultats par position.")
    by_grp = df.groupby("groupe")["result_bb"].agg(["count", "sum"]).reset_index()
    sec.table(by_grp.rename(columns={"count": "mains", "sum": "résultat (bb)"}), "Résultats par catégorie de main.")
    off = df[df["hors_regle"]]
    if len(off):
        sec.table(off[["phase", "main", "classe", "position", "action_resume", "result_bb"]], "Mains déclarées hors règle (colonne hors_regle).")
    hf = df[df["hors_fiche"].fillna(False).astype(bool)]
    fiche = report.section("Mains jouées hors de la fiche")
    evaluated = int(df["hors_fiche"].notna().sum())
    fiche.paragraph(f"Action préflop comparée à la fiche de la stratégie de la phase (grilles « personne n'a relancé » / « face à une relance à "
                    f"{raise_bb:g} bb ») : {len(hf)} main(s) hors fiche sur {evaluated} évaluée(s).")
    if len(hf):
        table = hf[["phase", "main", "classe", "position", "situation", "joue", "fiche", "hors_regle", "result_bb"]].copy()
        table["situation"] = table["situation"].map({"unopened": "personne n'a relancé", "vs_raise": "face à une relance"})
        table["hors_regle"] = table["hors_regle"].map({True: "oui", False: "non"})
        fiche.table(table.rename(columns={"joue": "joué", "hors_regle": "déclarée hors règle", "result_bb": "résultat (bb)"}),
                    "Situation déduite du résumé préflop : « paie », « 3bet » ou « couché sur relance » = face à une relance.")
    cmp = report.section("Mon résultat est-il normal pour cette stratégie ?")
    rows = []
    verdict = []
    for phase, g in df.groupby("phase", sort=False):
        strat = phases.get(phase, {}).get("strategie")
        if not strat:
            continue
        n = len(g)
        mine = float(g["result_bb"].sum())
        dist = window_distribution(sim_hands, strat, n)
        pct = percentile_of(mine, dist)
        sim_stats = stats_from_hands(sim_hands[sim_hands["strategy"] == strat]) if (sim_hands["strategy"] == strat).any() else {}
        status = "sous le 5e centile" if pct < 5 else ("au-dessus du 95e centile" if pct > 95 else "dans la distribution normale")
        rows.append({"phase": phase, "stratégie": labels.get(strat, strat), "mains": n, "mon résultat (bb)": mine,
                     "centile simulé": pct, "verdict": status, "mon VPIP %": 100 * g["vpip"].mean(),
                     "VPIP simulé %": sim_stats.get("VPIP", float("nan")), "mon PFR %": 100 * g["pfr"].mean(),
                     "PFR simulé %": sim_stats.get("PFR", float("nan"))})
        if dist.size:
            cmp.figure(charts.histogram(dist.tolist(), colors.get(strat, "#2a78d6"), f"Phase {phase} ({labels.get(strat, strat)}) : résultats simulés sur des blocs de {n} mains", "bb", marker_x=mine))
        n_off = int(g["hors_fiche"].fillna(False).astype(bool).sum())
        verdict.append(f"Phase {phase} ({labels.get(strat, strat)}) : {mine:+.1f} bb sur {n} mains → {pct:.0f}e centile, {status} ; "
                       f"{n_off} main(s) hors fiche.")
    cmp.table(pd.DataFrame(rows), "Centile : part des blocs simulés de même longueur qui font moins bien que vous.")
    if own:
        report.verdict = verdict[:4] + ["Sous le 5e centile : soit malchance rare, soit jeu réel différent de la règle (voir mains hors règle)."]
        paths = save_outputs(ctx, report, None, None, (), {"phases": rows})
    else:
        paths = {}
    return {"report": report, "paths": paths, "journal": df, "comparaison": rows}
