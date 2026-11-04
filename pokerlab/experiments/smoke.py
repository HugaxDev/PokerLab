"""Commande ``smoke`` : toutes les expériences en miniature (≤ 2 000 mains) + rapport de démonstration."""

from __future__ import annotations

from typing import Any

from pokerlab.config import PROJECT_ROOT
from pokerlab.experiments import audit, bankroll, compare, round_robin, sessions, variants
from pokerlab.experiments.base import ExperimentContext, save_outputs
from pokerlab.journal import report as journal_report
from pokerlab.report.html import Report


def run(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    budget = int(cfg.get("max_hands", 2000))
    combined = Report("PokerLab — rapport de démonstration (smoke)", "Toutes les expériences en miniature : chiffres NON significatifs, démonstration du pipeline")
    outputs: dict[str, Any] = {}
    played: dict[str, int] = {}

    def sub(name: str) -> ExperimentContext:
        c = ExperimentContext(name, ctx.lab, cfg[name], ctx.out_dir / name, ctx.workers, ctx.log)
        c.out_dir.mkdir(parents=True, exist_ok=True)
        return c

    steps = [
        ("compare", lambda c: compare.run(c)),
        ("round_robin", lambda c: round_robin.run(c)),
        ("sessions", lambda c: sessions.run(c)),
        ("bankroll", lambda c: bankroll.run(c, source_hands=outputs["sessions"]["hands"])),
        ("sweep", lambda c: variants.run_sweep(c)),
        ("ablation", lambda c: variants.run_ablation(c)),
        ("optimize", lambda c: variants.run_optimize(c)),
        ("audit", lambda c: audit.run(c)),
        ("journal", lambda c: journal_report.run(c, sim_hands=outputs["sessions"]["hands"])),
    ]
    for name, fn in steps:
        if name not in cfg:
            continue
        c = sub(name)
        if name == "journal":
            c.cfg = dict(c.cfg)
            c.cfg["csv"] = str(PROJECT_ROOT / c.cfg["csv"])
            c.cfg["phases"] = str(PROJECT_ROOT / c.cfg["phases"])
        ctx.log(f"--- smoke : {name}")
        outputs[name] = fn(c)
        played[name] = c.hands_played
        ctx.hands_played += c.hands_played
        combined.absorb(outputs[name]["report"], prefix=f"[{name}] ")
    fiche_path = None
    if "fiche" in cfg:
        from pokerlab.fiche.build import build_fiche

        name = cfg["fiche"].get("strategy", "joueur")
        res = build_fiche(name, ctx.lab, ctx.out_dir, ctx.out_dir / f"fiche_{name}.html")
        fiche_path = res["path"]
        played["fiche"] = 0
        ctx.log(f"--- smoke : fiche {name} -> {fiche_path} (interdits : {res['filled']['interdits']}, résultat : {res['filled']['resultat']})")
    total = ctx.hands_played
    within = total <= budget
    combined.verdict = [
        (outputs["compare"]["report"].verdict[0] if "compare" in outputs and outputs["compare"]["report"].verdict else "compare non lancé"),
        "Échantillon minuscule : ces chiffres servent à vérifier le pipeline, pas à conclure.",
        f"Mains jouées : {total} au total (budget {budget}) — " + ", ".join(f"{k} {v}" for k, v in played.items()) + ".",
        "Toutes les expériences ont tourné et produit leurs sections, exports et historiques." if within
        else f"ATTENTION : budget dépassé ({total} > {budget}).",
        "Pour un vrai verdict : python -m pokerlab compare (arrêt automatique à ± 3 bb/100).",
    ]
    combined.meta = {"mains jouées": total, "graine": ctx.lab.seed, "workers": ctx.workers}
    hands = outputs.get("compare", {}).get("hands")
    paths = save_outputs(ctx, combined, hands, outputs.get("compare", {}).get("decisions"),
                         (), {"mains_par_experience": played, "budget": budget, "dans_le_budget": within})
    if fiche_path is not None:
        paths["fiche"] = fiche_path
    if "site" in cfg:
        paths["site"] = smoke_site(ctx, dict(cfg["site"] or {}))
    return {"report": combined, "paths": paths, "played": played, "total": total, "within_budget": within, "outputs": outputs}


def smoke_site(ctx: ExperimentContext, opts: dict[str, Any]):
    """Tableau de bord sur les résultats du smoke et les données d'exemple, puis vérifications (liens, Chromium)."""
    from pokerlab.site.build import build_site, site_config
    from pokerlab.site.check import full_check

    cfg = site_config()
    cfg.update({"campagnes": str(ctx.out_dir), "sortie": str(ctx.out_dir / "site"), "exclure": [],
                "donnees_reelles": str(ctx.out_dir / "site" / "aucune_main_reelle"), "fiche_campagne": None})
    cfg.update(opts)
    site = build_site(cfg, lab=ctx.lab)
    res = full_check(site)
    ctx.log(f"--- smoke : site -> {site / 'index.html'} (liens cassés : {len(res['liens_casses'])}, références externes : "
            f"{len(res['references_externes'])}, Chromium : {res['chromium']}, problèmes : {len(res['problemes_chromium'])})")
    for msg in res["liens_casses"] + res["references_externes"] + res["problemes_chromium"]:
        ctx.log(f"    ATTENTION {msg}")
    if not res["ok"]:
        raise RuntimeError("site du smoke : vérifications en échec (voir ATTENTION ci-dessus)")
    return site / "index.html"
