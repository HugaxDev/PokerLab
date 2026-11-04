"""Expériences « sweep », « ablation » et « optimize » : variantes d'une stratégie sur les mêmes decks."""

from __future__ import annotations

import itertools
import random
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.config import get_path, set_path
from pokerlab.experiments.base import ExperimentContext, apply_holm, composition_dict, pairwise_tests, save_outputs
from pokerlab.experiments.duplicate import Autostop, Cell, collect, run_cells
from pokerlab.report import charts, sections
from pokerlab.report.html import Report
from pokerlab.rng import derive_seed
from pokerlab.strategies.registry import resolve_spec

# Alias lisibles pour les paramètres usuels (chemins dans la config de stratégie).
ALIASES = {
    "taille_ouverture": "preflop.unopened.raise_size_bb",
    "seuil_paie": "preflop.vs_open.call[0].max_bb",
    "frequence_bluff": "postflop.bluff.frequency",
    "taille_bluff": "postflop.bluff.size_pot",
    "taille_cbet": "postflop.cbet.size_pot",
    "frequence_cbet": "postflop.cbet.frequency",
    "taille_value": "postflop.value_bet.tiers[0].size_pot",
    "garde_fou_seuil": "postflop.guardrail.seuil",
    "garde_fou_force": "postflop.guardrail.force_min",
}


def resolve_path(path: str) -> str:
    return ALIASES.get(path, path)


def variant(base: dict[str, Any], changes: dict[str, Any], suffix: str) -> dict[str, Any]:
    cfg = dict(base)
    for path, value in changes.items():
        cfg = set_path(cfg, resolve_path(path), value)
    cfg["name"] = f"{base['name']}[{suffix}]"
    cfg["label"] = f"{base.get('label', base['name'])} [{suffix}]"
    cfg.pop("color", None)
    return cfg


def _evaluate(ctx: ExperimentContext, configs: list[dict[str, Any]], composition: str, decks: int, space: str = "deck") -> tuple[pd.DataFrame, pd.DataFrame]:
    cdict = composition_dict(composition)
    cells = [Cell(c["name"], c, cdict, {"strategy": c["name"], "composition": composition}, deck_space=space) for c in configs]
    auto = Autostop(enabled=False, decks=decks, block_decks=int(ctx.cfg.get("block_decks", 2000)),
                    max_minutes=float(ctx.cfg.get("max_minutes", 120)), stop_file=str(ctx.out_dir / "STOP"))
    results, progress = run_cells(cells, ctx.lab, auto, ctx.workers, record_decisions=False, log=ctx.log)
    partial = sorted({p.stopped for p in progress.values() if "partiels" in p.stopped})
    ctx.partial.extend(r for r in partial if r not in ctx.partial)
    hands, decisions, _, played, _ = collect(results)
    ctx.hands_played += played
    return hands, decisions


def _bb100(hands: pd.DataFrame, name: str, conf: float):
    return sections.summarize_strategy(hands[hands["strategy"] == name], name, conf)


# ----------------------------------------------------------------------------- sweep
def _sweep_configs(cfg: dict[str, Any]):
    base = resolve_spec(cfg["strategy"])
    params = cfg["params"][:2]
    grids = [p["values"] for p in params]
    configs = []
    combos = list(itertools.product(*grids))
    for combo in combos:
        changes = {p["path"]: v for p, v in zip(params, combo)}
        suffix = ", ".join(f"{p.get('label', p['path'])}={v}" for p, v in zip(params, combo))
        configs.append(variant(base, changes, suffix))
    return base, params, grids, combos, configs


def run_sweep(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    base, params, grids, combos, configs = _sweep_configs(cfg)
    comp = cfg.get("composition", "micro_play_money")
    ctx.log(f"sweep : {len(configs)} variantes de {base['name']} ({comp})")
    hands, _ = _evaluate(ctx, configs, comp, int(cfg.get("decks", 200)))
    return _sweep_outputs(ctx, base, params, grids, combos, configs, comp, hands)


def rebuild_sweep_report(ctx: ExperimentContext) -> dict[str, Any]:
    """Régénère rapport.html et resume.json d'un sweep déjà joué (mains.parquet du dossier), sans re-simuler."""
    import json

    meta = json.loads((ctx.out_dir / "resume.json").read_text(encoding="utf-8"))
    ctx.cfg = meta.get("config") or ctx.cfg
    ctx.hands_played = int(meta.get("mains_jouees", 0))
    ctx.partial = list(meta.get("partiel") or [])
    base, params, grids, combos, configs = _sweep_configs(ctx.cfg)
    hands = pd.read_parquet(ctx.out_dir / "mains.parquet")
    for col in ("strategy", "composition"):
        if col in hands.columns:
            hands[col] = hands[col].astype(str)
    out = _sweep_outputs(ctx, base, params, grids, combos, configs, ctx.cfg.get("composition", "micro_play_money"), hands, write_hands=False)
    if "duree_s" in meta:
        data = json.loads((ctx.out_dir / "resume.json").read_text(encoding="utf-8"))
        data["duree_s"] = meta["duree_s"]
        (ctx.out_dir / "resume.json").write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return out


def _sweep_outputs(ctx, base, params, grids, combos, configs, comp, hands, write_hands: bool = True) -> dict[str, Any]:
    cfg = ctx.cfg
    conf = ctx.confidence
    summ = {c["name"]: _bb100(hands, c["name"], conf) for c in configs}
    report = Report(f"PokerLab — sweep de {base.get('label', base['name'])}", f"Composition {comp}, mêmes decks pour chaque variante")
    sec = report.section("Balayage des paramètres")
    color = sections.colors_labels([cfg["strategy"]])[1][base["name"]]
    if len(params) == 1:
        xs = grids[0]
        sec.figure(charts.line_with_ci(xs, [summ[c["name"]].ev.bb100 for c in configs], [summ[c["name"]].ev.ci_half for c in configs], color,
                                       f"bb/100 selon {params[0].get('label', params[0]['path'])}", str(params[0].get("label", params[0]["path"]))))
    else:
        z = np.array([summ[c["name"]].ev.bb100 for c in configs]).reshape(len(grids[0]), len(grids[1]))
        ci = np.array([summ[c["name"]].ev.ci_half for c in configs]).reshape(len(grids[0]), len(grids[1]))
        text = np.array([[f"{z[i, j]:+.1f}<br>± {ci[i, j]:.1f}" for j in range(z.shape[1])] for i in range(z.shape[0])], dtype=object)
        sec.figure(charts.matrix_heatmap(z, [str(v) for v in grids[0]], [str(v) for v in grids[1]],
                                         f"bb/100 : {params[0].get('label', params[0]['path'])} (lignes) × {params[1].get('label', params[1]['path'])} (colonnes)",
                                         text=text),
                   "Chaque case : bb/100 all-in adjusted ± demi-largeur de l'IC 95 % (mêmes decks pour toutes les cases).")
    table = pd.DataFrame({
        "variante": [c["label"] for c in configs],
        "bb/100 EV": [summ[c["name"]].ev.bb100 for c in configs],
        "IC ±": [summ[c["name"]].ev.ci_half for c in configs],
        "bb/100 réel": [summ[c["name"]].real.bb100 for c in configs],
        "mains": [summ[c["name"]].ev.hands for c in configs],
    })
    sec.table(table, "Toutes les variantes jouent exactement les mêmes decks (différences appariées).")
    best = max(configs, key=lambda c: summ[c["name"]].ev.bb100)
    worst = min(configs, key=lambda c: summ[c["name"]].ev.bb100)
    dm = {c["name"]: hands[hands["strategy"] == c["name"]].groupby("deck")["result_ev"].mean() for c in (best, worst)}
    test = apply_holm(pairwise_tests(dm, labels={best["name"]: best["label"], worst["name"]: worst["label"]}))
    current_line = ""
    try:   # la config actuelle de la stratégie fait-elle partie de la grille ? Écart apparié avec la meilleure case.
        cur_vals = tuple(get_path(base, resolve_path(p["path"])) for p in params)
        current = next((c for combo, c in zip(combos, configs) if tuple(str(v) for v in combo) == tuple(str(v) for v in cur_vals)), None)
    except (KeyError, IndexError, TypeError):
        current = None
    if current is not None and current is not best:
        dmc = {c["name"]: hands[hands["strategy"] == c["name"]].groupby("deck")["result_ev"].mean() for c in (best, current)}
        tc = apply_holm(pairwise_tests(dmc, labels={best["name"]: best["label"], current["name"]: current["label"]}))
        n_cmp = max(1, len(configs) - 1)
        current_line = (f"Réglage actuel {current['label']} : {summ[current['name']].ev.bb100:+.1f} bb/100. " + (tc[0].verdict() if tc else "")
                        + f" (meilleure case choisie parmi {len(configs)} : gain probablement surestimé, à confirmer sur d'autres decks ; "
                        f"seuil de Bonferroni p < {0.05 / n_cmp:.4f}).")
    elif current is best:
        current_line = f"Le réglage actuel ({current['label']}) est la meilleure case de la grille."
    report.verdict = [
        f"Meilleure variante : {best['label']} ({summ[best['name']].ev.bb100:+.1f} ± {summ[best['name']].ev.ci_half:.1f} bb/100).",
        current_line or f"Moins bonne : {worst['label']} ({summ[worst['name']].ev.bb100:+.1f} bb/100).",
        test[0].verdict() + "." if test else "",
        f"{len(configs)} variantes, {int(cfg.get('decks', 200))} decks chacune (x6 rotations).",
        "Un écart non significatif veut dire : le paramètre compte peu à cet échantillon, pas qu'il est sans effet.",
    ]
    grid = {
        "params": [{"path": p["path"], "label": str(p.get("label", p["path"])), "values": list(p["values"])} for p in params],
        "points": [{"values": list(combo), "bb100": summ[c["name"]].ev.bb100, "ci": summ[c["name"]].ev.ci_half,
                    "label": c["label"]} for combo, c in zip(combos, configs)],
    }
    paths = save_outputs(ctx, report, hands if write_hands else None, None, (), {"strategie": base["name"], "composition": comp,
                                                                                 "variantes": table.to_dict("records"), "grille": grid})
    return {"report": report, "paths": paths, "table": table, "hands": hands}


# -------------------------------------------------------------------------- ablation
def run_ablation(ctx: ExperimentContext, report: Report | None = None) -> dict[str, Any]:
    cfg = ctx.cfg
    base = resolve_spec(cfg["strategy"])
    base_cfg = dict(base)
    base_cfg["name"] = f"{base['name']}[complet]"
    configs = [base_cfg]
    for ab in cfg.get("ablations", []):
        configs.append(variant(base, ab.get("patch", {}), "sans " + ab["name"]))
    comp = cfg.get("composition", "micro_play_money")
    ctx.log(f"ablation : {len(configs) - 1} règles retirées de {base['name']} ({comp})")
    hands, _ = _evaluate(ctx, configs, comp, int(cfg.get("decks", 200)))
    dm = {c["name"]: hands[hands["strategy"] == c["name"]].groupby("deck")["result_ev"].mean() for c in configs}
    labels = {c["name"]: c.get("label", c["name"]) for c in configs}
    pairs = [(base_cfg["name"], c["name"]) for c in configs[1:]]
    tests = apply_holm(pairwise_tests(dm, pairs=pairs, confidence=ctx.confidence, seed=ctx.lab.seed, labels=labels))
    own = report is None
    report = report or Report(f"PokerLab — ablation de {base.get('label', base['name'])}", f"Composition {comp}")
    sec = report.section(f"Ablation : ce que rapporte chaque règle de {base.get('label', base['name'])}")
    color = sections.colors_labels([cfg["strategy"]])[1][base["name"]]
    names = [ab["name"] for ab in cfg.get("ablations", [])]
    sec.figure(charts.bars_with_ci(names, [t.diff_bb100 for t in tests], [t.ci_half for t in tests], color,
                                   "Gain apporté par la règle (stratégie complète − stratégie sans la règle)", "bb/100"),
               "Positif : la règle rapporte de l'argent. Négatif : elle en coûte.")
    sec.table(pd.DataFrame({"règle retirée": names, "apport (bb/100)": [t.diff_bb100 for t in tests], "IC ±": [t.ci_half for t in tests],
                            "p Holm": [t.p_holm for t in tests], "verdict": ["significatif" if t.significant else "pas encore significatif" for t in tests]}),
              digits=3)
    if own:
        lines = [f"{n} : {'rapporte' if t.diff_bb100 > 0 else 'coûte'} {abs(t.diff_bb100):.1f} ± {t.ci_half:.1f} bb/100 "
                 f"({'significatif' if t.significant else 'pas encore significatif'})." for n, t in zip(names, tests)]
        report.verdict = lines[:5]
        paths = save_outputs(ctx, report, hands, None, (), {
            "strategie": base["name"], "composition": comp, "ablations": [t.verdict() for t in tests],
            "resultats": [{"regle": n, "apport_bb100": t.diff_bb100, "ci": t.ci_half, "p_holm": t.p_holm, "significatif": t.significant}
                          for n, t in zip(names, tests)],
            "base_bb100": _bb100(hands, base_cfg["name"], ctx.confidence).ev.bb100,
        })
    else:
        paths = {}
    return {"report": report, "paths": paths, "tests": tests, "hands": hands}


# -------------------------------------------------------------------------- optimize
def _sample(space: list[dict[str, Any]], rng: random.Random) -> dict[str, Any]:
    out = {}
    for p in space:
        kind = p.get("type", "float")
        if kind == "choice":
            out[p["path"]] = rng.choice(p["values"])
        elif kind == "int":
            out[p["path"]] = rng.randint(int(p["low"]), int(p["high"]))
        else:
            step = p.get("step")
            v = rng.uniform(float(p["low"]), float(p["high"]))
            out[p["path"]] = round(round(v / step) * step, 6) if step else round(v, 4)
    return out


def _current(base: dict[str, Any], space: list[dict[str, Any]]) -> dict[str, Any]:
    """Valeurs actuelles de la config de base (paramètres absents : valeur ``default`` de l'espace)."""
    out = {}
    for p in space:
        try:
            out[p["path"]] = get_path(base, resolve_path(p["path"]))
        except (KeyError, IndexError, TypeError):
            if "default" in p:
                out[p["path"]] = p["default"]
    return out


def run_optimize(ctx: ExperimentContext, report: Report | None = None) -> dict[str, Any]:
    """Recherche aléatoire reproductible sur graines d'entraînement, validation sur graines jamais vues."""
    cfg = ctx.cfg
    base = resolve_spec(cfg["strategy"])
    space = cfg["space"]
    rng = random.Random(derive_seed(ctx.lab.seed, "optimize", int(cfg.get("seed", 0))))
    n_trials = int(cfg.get("n_trials", 30))
    trials = [_current(base, space)]
    trials += [_sample(space, rng) for _ in range(n_trials)]
    configs = []
    for i, t in enumerate(trials):
        suffix = "base" if i == 0 else f"essai {i}"
        configs.append(variant(base, t, suffix))
    comp = cfg.get("composition", "micro_play_money")
    train_decks = int(cfg.get("train_decks", 300))
    valid_decks = int(cfg.get("validation_decks", 300))
    ctx.log(f"optimize : {n_trials} essais sur {train_decks} decks d'entraînement ({comp})")
    train_hands, _ = _evaluate(ctx, configs, comp, train_decks, space="train")
    conf = ctx.confidence
    train = {c["name"]: _bb100(train_hands, c["name"], conf) for c in configs}
    top_k = int(cfg.get("top_k", 5))
    ranked = sorted(configs[1:], key=lambda c: train[c["name"]].ev.bb100, reverse=True)[:top_k]
    finalists = [configs[0]] + ranked
    ctx.log(f"optimize : validation de {len(finalists)} finalistes sur {valid_decks} decks jamais vus")
    valid_hands, _ = _evaluate(ctx, finalists, comp, valid_decks, space="validation")
    valid = {c["name"]: _bb100(valid_hands, c["name"], conf) for c in finalists}
    own = report is None
    report = report or Report(f"PokerLab — optimisation de {base.get('label', base['name'])}", f"Composition {comp}")
    sec = report.section("Optimisation : entraînement vs validation")
    color = sections.colors_labels([cfg["strategy"]])[1][base["name"]]
    points = [{"label": c["label"], "train": train[c["name"]].ev.bb100, "validation": valid[c["name"]].ev.bb100} for c in finalists]
    sec.figure(charts.train_validation(points, color, "Score d'entraînement vs validation"),
               "Sous la diagonale : le score d'entraînement surestime la vraie valeur (sur-ajustement).")
    rows = []
    for c, t in zip(configs, trials):
        rows.append({"candidat": c["label"], **{k: v for k, v in t.items()}, "entraînement bb/100": train[c["name"]].ev.bb100,
                     "validation bb/100": valid[c["name"]].ev.bb100 if c["name"] in valid else float("nan"),
                     "écart": (train[c["name"]].ev.bb100 - valid[c["name"]].ev.bb100) if c["name"] in valid else float("nan")})
    table = pd.DataFrame(rows).sort_values("entraînement bb/100", ascending=False)
    sec.table(table, "Les graines de validation ne servent jamais à choisir les candidats.")
    best_valid = max(finalists, key=lambda c: valid[c["name"]].ev.bb100)
    dm = {c["name"]: valid_hands[valid_hands["strategy"] == c["name"]].groupby("deck")["result_ev"].mean() for c in (best_valid, configs[0])}
    test = pairwise_tests(dm, pairs=[(best_valid["name"], configs[0]["name"])], labels={best_valid["name"]: best_valid["label"], configs[0]["name"]: configs[0]["label"]}) if best_valid is not configs[0] else []
    gap = float(np.nanmean([train[c["name"]].ev.bb100 - valid[c["name"]].ev.bb100 for c in ranked])) if ranked else float("nan")
    if own:
        report.verdict = [
            f"Meilleur candidat en validation : {best_valid['label']} ({valid[best_valid['name']].ev.bb100:+.1f} ± {valid[best_valid['name']].ev.ci_half:.1f} bb/100).",
            f"Écart moyen entraînement − validation des finalistes : {gap:+.1f} bb/100 (mesure du sur-ajustement).",
            (test[0].verdict() + " (validation).") if test else "La config de base reste la meilleure en validation.",
            f"{n_trials} essais aléatoires reproductibles, {train_decks} decks d'entraînement, {valid_decks} decks de validation.",
        ]
        paths = save_outputs(ctx, report, None, None, (), {"candidats": table.to_dict("records")})
    else:
        paths = {}
    return {"report": report, "paths": paths, "table": table}


# ---------------------------------------------------------------------- confirmation
def run_confirmation(ctx: ExperimentContext) -> dict[str, Any]:
    """Confirmation hors échantillon : réglage actuel contre des candidats, sur des graines jamais utilisées,
    dans plusieurs compositions. Écart apparié deck par deck, correction de Holm sur tous les tests.
    Un candidat n'est retenu que s'il bat significativement le réglage actuel dans TOUTES les compositions."""
    cfg = ctx.cfg
    base = resolve_spec(cfg["strategy"])
    space = str(cfg.get("deck_space", "confirmation"))
    comps = list(cfg.get("compositions", ["humains_simules"]))
    if cfg.get("reference_patch"):   # référence figée (ex. ancien réglage), sinon la config actuelle
        base = {**variant(base, cfg["reference_patch"], "ref"), "name": base["name"], "label": base.get("label", base["name"])}
    ref_cfg = dict(base)
    ref_cfg["name"] = f"{base['name']}[actuel]"
    ref_cfg["label"] = f"{base.get('label', base['name'])} [{cfg.get('reference_label', 'actuel')}]"
    ref_cfg.pop("color", None)
    cands = [variant(base, c["patch"], c["label"]) for c in cfg["candidates"]]
    variants_ = [ref_cfg] + cands
    decks = int(cfg.get("decks", 1000))
    ctx.log(f"confirmation : {len(cands)} candidats contre le réglage actuel de {base['name']}, {len(comps)} compositions, "
            f"{decks} decks par case, graines « {space} » jamais utilisées")
    cells = [Cell(f"{v['name']}@{comp}", v, composition_dict(comp), {"strategy": v["name"], "composition": comp}, deck_space=space)
             for comp in comps for v in variants_]
    auto = Autostop(enabled=False, decks=decks, block_decks=int(cfg.get("block_decks", 2000)),
                    max_minutes=float(cfg.get("max_minutes", 120)), stop_file=str(ctx.out_dir / "STOP"))
    results, progress = run_cells(cells, ctx.lab, auto, ctx.workers, record_decisions=False, log=ctx.log)
    ctx.partial.extend(sorted({p.stopped for p in progress.values() if "partiels" in p.stopped}))
    hands, _, _, played, _ = collect(results)
    ctx.hands_played += played
    for col in ("strategy", "composition"):
        hands[col] = hands[col].astype(str)
    conf = ctx.confidence
    labels = {v["name"]: v["label"] for v in variants_}
    tests, meta = [], []
    summ: dict[tuple[str, str], Any] = {}
    for comp in comps:
        sub = hands[hands["composition"] == comp]
        for v in variants_:
            summ[(v["name"], comp)] = sections.summarize_strategy(sub[sub["strategy"] == v["name"]], v["name"], conf)
        dm = {v["name"]: sub[sub["strategy"] == v["name"]].groupby("deck")["result_ev"].mean() for v in variants_}
        for c in cands:
            tests += pairwise_tests(dm, pairs=[(c["name"], ref_cfg["name"])], confidence=conf, labels=labels)
            meta.append((c, comp))
    tests = apply_holm(tests)
    rows, retained = [], []
    for c in cands:
        mine = [(t, comp) for t, (cc, comp) in zip(tests, meta) if cc is c]
        ok = all(t.diff_bb100 > 0 and t.p_holm < 0.05 for t, _ in mine)
        if ok:
            retained.append(c)
        for t, comp in mine:
            rows.append({"candidat": c["label"], "composition": comp, "écart vs actuel (bb/100)": t.diff_bb100, "IC ±": t.ci_half,
                         "p Holm": t.p_holm, "bat l'actuel": "oui" if (t.diff_bb100 > 0 and t.p_holm < 0.05) else "non",
                         "decks appariés": t.n_units})
    table = pd.DataFrame(rows)
    report = Report(f"PokerLab — confirmation hors échantillon ({base.get('label', base['name'])})",
                    f"Graines jamais utilisées (« {space} »), {decks} decks par case, compositions : {', '.join(comps)}")
    sec = report.section("Écart apparié avec le réglage actuel")
    for comp in comps:
        sub_rows = [r for r in rows if r["composition"] == comp]
        sec.figure(charts.bars_with_ci([r["candidat"] for r in sub_rows], [r["écart vs actuel (bb/100)"] for r in sub_rows],
                                       [r["IC ±"] for r in sub_rows], sections.colors_labels([cfg["strategy"]])[1][base["name"]],
                                       f"{comp} : candidat − réglage actuel", "bb/100"),
                   "Différence moyenne par deck (6 rotations, mêmes cartes et mêmes adversaires), IC 95 %.")
    sec.table(table, "Correction de Holm sur les " + str(len(tests)) + " comparaisons. Retenu = bat l'actuel (p Holm < 0,05) dans toutes les compositions.", digits=2)
    lvl = report.section("Niveaux absolus")
    lvl.table(pd.DataFrame([{"variante": labels[v["name"]], "composition": comp, "bb/100": summ[(v["name"], comp)].ev.bb100,
                             "IC ±": summ[(v["name"], comp)].ev.ci_half, "mains": summ[(v["name"], comp)].ev.hands}
                            for comp in comps for v in variants_]), "bb/100 all-in adjusted.")
    best = max(retained, key=lambda c: min(t.diff_bb100 for t, (cc, _) in zip(tests, meta) if cc is c)) if retained else None
    report.verdict = ([f"Retenu : {best['label']} (bat le réglage actuel dans les {len(comps)} compositions)."] if best else
                      [f"Aucun candidat retenu : aucun ne bat significativement le réglage actuel dans les {len(comps)} compositions. Garder l'actuel."])
    report.verdict += [f"{c['label']} : " + " ; ".join(f"{comp} {t.diff_bb100:+.1f} ± {t.ci_half:.1f} (p Holm {t.p_holm:.2g})"
                                                         for t, (cc, comp) in zip(tests, meta) if cc is c) for c in cands]
    summary = {"strategie": base["name"], "compositions": comps, "deck_space": space,
               "resultats": [{**{k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}} for r in rows],
               "retenus": [c["label"] for c in retained], "choisi": best["label"] if best else None,
               "patch_choisi": next((cc["patch"] for cc in cfg["candidates"] if best and cc["label"] == best["label"].split("[")[-1].rstrip("]")), None)}
    paths = save_outputs(ctx, report, None, None, (), summary)
    return {"report": report, "paths": paths, "table": table, "retained": retained, "hands": hands}
