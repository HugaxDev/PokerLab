"""Interface en ligne de commande : ``python -m pokerlab <commande>``."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from pokerlab.config import PROJECT_ROOT, LabConfig, list_strategies, load_experiment, set_path

COMMANDS = ("compare", "round_robin", "sessions", "bankroll", "sweep", "ablation", "optimize", "audit", "journal", "smoke", "bench")


def _overrides(pairs: list[str]) -> list[tuple[str, Any]]:
    out = []
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set attend chemin=valeur, reçu : {pair}")
        key, value = pair.split("=", 1)
        out.append((key.strip(), yaml.safe_load(value)))
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pokerlab", description="PokerLab : laboratoire local de simulation de poker NLHE 6-max.")
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name.replace("_", "-") if name == "round_robin" else name, help=f"expérience {name}", aliases=["round_robin"] if name == "round_robin" else [])
        sp.add_argument("--config", help="fichier YAML ou nom (défaut : configs/experiments/<commande>.yaml)")
        sp.add_argument("--out", help="dossier de sortie (défaut : out/<commande>)")
        sp.add_argument("--workers", type=int, help="nombre de processus (défaut : config, 0 = tous les cœurs)")
        sp.add_argument("--seed", type=int, help="graine globale")
        sp.add_argument("--set", action="append", default=[], metavar="CHEMIN=VALEUR", help="surcharge de la config d'expérience")
        sp.add_argument("--game-set", action="append", default=[], metavar="CHEMIN=VALEUR", help="surcharge de configs/game.yaml")
        if name == "journal":
            sp.add_argument("--csv", help="journal CSV")
            sp.add_argument("--phases", help="YAML des phases")
    pre = sub.add_parser("precompute", help="précalcule la table d'équité préflop 169x169 (cache disque)")
    pre.add_argument("--sims", type=int, help="simulations par confrontation")
    rp = sub.add_parser("replay", help="affiche l'historique d'une main de campagne duplicate")
    rp.add_argument("--strategy", required=True)
    rp.add_argument("--composition", default="micro_play_money")
    rp.add_argument("--deck", type=int, required=True)
    rp.add_argument("--rot", type=int, default=0)
    rp.add_argument("--seed", type=int)
    sub.add_parser("strategies", help="liste les stratégies disponibles")
    im = sub.add_parser("import", help="importe des historiques PokerStars (.txt ou dossiers)")
    im.add_argument("chemins", nargs="+")
    im.add_argument("--strategie", help="stratégie suivie (défaut : configs/site.yaml)")
    im.add_argument("--hero", help="ton pseudo PokerStars")
    im.add_argument("--dest", help="dossier de stockage (défaut : data/reel)")
    sub.add_parser("site", help="génère le tableau de bord statique dans out/site")
    sv = sub.add_parser("serve", help="génère et sert le tableau de bord sur http://localhost:8765 (régénération automatique)")
    sv.add_argument("--port", type=int)
    rb = sub.add_parser("robustesse", help="score de robustesse sur toutes les compositions mesurées (compare)")
    rb.add_argument("--campaign", default="out", help="dossier contenant une ou plusieurs campagnes compare")
    fi = sub.add_parser("fiche", help="fiche HTML d'une stratégie (grilles, tailles, règles, interdits, résultat attendu)")
    fi.add_argument("--strategy", required=True)
    fi.add_argument("--campaign", help="dossier de campagne contenant compare et audit (ex. out ou out/smoke)")
    fi.add_argument("--composition", help="composition pour le résultat attendu (défaut : la plus jouée)")
    fi.add_argument("--out", help="fichier HTML de sortie (défaut : out/fiche_<stratégie>.html)")
    fi.add_argument("--config", help="config de fiche (défaut : configs/fiche.yaml)")
    return p


def _lab(args: argparse.Namespace) -> LabConfig:
    overrides: dict[str, Any] = {}
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = args.seed
    lab = LabConfig.load(overrides=overrides or None)
    raw = lab.raw
    for key, value in _overrides(getattr(args, "game_set", [])):
        raw = set_path(raw, key, value)
    if getattr(args, "workers", None) is not None:
        raw = dict(raw, workers=args.workers)
    return LabConfig.load(overrides=raw) if raw is not lab.raw else lab


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cmd = args.command.replace("-", "_")
    if cmd == "strategies":
        print("\n".join(list_strategies()))
        return 0
    if cmd == "precompute":
        from pokerlab.config import EquityConfig
        from pokerlab.equity.preflop import build_table, save_table

        lab = LabConfig.load()
        eq = lab.equity if args.sims is None else EquityConfig(**{**lab.raw.get("equity", {}), "preflop_sims_per_matchup": args.sims})
        table = build_table(eq, verbose=True)
        save_table(table, eq.table_path(), eq.seed)
        print(f"Table écrite : {eq.table_path()} ({eq.preflop_sims_per_matchup} simulations par confrontation)")
        return 0
    if cmd == "import":
        from pokerlab.config import load_yaml, configs_root
        from pokerlab.journal.store import import_paths

        site_cfg = load_yaml(configs_root() / "site.yaml")
        lab = LabConfig.load()
        res = import_paths(args.chemins, lab, args.strategie or site_cfg.get("strategie_suivie", "joueur"),
                           args.dest or site_cfg.get("donnees_reelles", "data/reel"), args.hero or site_cfg.get("hero"),
                           float(site_cfg.get("pause_session_minutes", 30)))
        print(f"{res['lues']} mains importées, {res['ignorees']} ignorées, {res['total']} au total dans {res['dossier']}")
        return 0
    if cmd in ("site", "serve"):
        from pokerlab.site.build import build_site
        from pokerlab.site.serve import serve

        if cmd == "site":
            out = build_site()
            print(f"Site généré : {out / 'index.html'}")
            return 0
        return serve(args.port)
    if cmd == "robustesse":
        from pokerlab.analysis.robustness import load_compare_cells, robustness_table

        cells = load_compare_cells(args.campaign)
        if cells.empty:
            print("Aucun compare avec résultats par case : lance python -m pokerlab compare")
            return 1
        table = robustness_table(cells)
        print(f"Compositions : {', '.join(sorted(cells['composition'].unique()))}")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
        return 0
    if cmd == "fiche":
        from pokerlab.fiche.build import build_fiche, fiche_config
        from pokerlab.strategies.registry import get_context

        lab = LabConfig.load()
        get_context(lab)
        res = build_fiche(args.strategy, lab, args.campaign, args.out, args.composition, fiche_config(args.config))
        print(f"Fiche écrite : {res['path']}")
        if args.campaign:
            for key, ok in res["filled"].items():
                print(f"  {key:10s} {'rempli' if ok else 'non disponible dans la campagne'}")
        return 0
    if cmd == "replay":
        from pokerlab.config import load_table
        from pokerlab.engine.history import format_hand
        from pokerlab.experiments.common import replay_duplicate_hand
        from pokerlab.strategies.registry import resolve_spec

        lab = _lab(args)
        hr, labels = replay_duplicate_hand(lab, resolve_spec(args.strategy), load_table(args.composition), args.deck, args.rot)
        print(format_hand(hr, lab.game, args.composition, labels, hero=args.rot))
        return 0
    lab = _lab(args)
    cfg = load_experiment(args.config or cmd)
    for key, value in _overrides(args.set):
        cfg = set_path(cfg, key, value)
    if cmd == "journal":
        if args.csv:
            cfg["csv"] = args.csv
        if args.phases:
            cfg["phases"] = args.phases
        for key in ("csv", "phases"):
            if isinstance(cfg.get(key), str) and not Path(cfg[key]).is_absolute() and not Path(cfg[key]).exists():
                cfg[key] = str(PROJECT_ROOT / cfg[key])
    from pokerlab.experiments.base import ExperimentContext
    from pokerlab.strategies.registry import get_context

    ctx = ExperimentContext.create(cmd, lab, cfg, args.out, args.workers)
    t0 = time.perf_counter()
    get_context(lab)
    if cmd == "compare":
        from pokerlab.experiments import compare as mod

        res = mod.run(ctx)
    elif cmd == "round_robin":
        from pokerlab.experiments import round_robin as mod

        res = mod.run(ctx)
    elif cmd == "sessions":
        from pokerlab.experiments import sessions as mod

        res = mod.run(ctx)
    elif cmd == "bankroll":
        from pokerlab.experiments import bankroll as mod

        res = mod.run(ctx)
    elif cmd in ("sweep", "ablation", "optimize"):
        from pokerlab.experiments import variants as mod

        res = {"sweep": mod.run_sweep, "ablation": mod.run_ablation, "optimize": mod.run_optimize}[cmd](ctx)
    elif cmd == "audit":
        from pokerlab.experiments import audit as mod

        res = mod.run(ctx)
    elif cmd == "journal":
        from pokerlab.journal import report as mod

        res = mod.run(ctx)
    elif cmd == "smoke":
        from pokerlab.experiments import smoke as mod

        res = mod.run(ctx)
    elif cmd == "bench":
        from pokerlab.experiments import bench as mod

        mod.run(ctx)
        print(f"Résultat : {ctx.out_dir / 'bench.json'}")
        return 0
    else:  # pragma: no cover
        raise SystemExit(f"Commande inconnue : {cmd}")
    report = res.get("report")
    if report is not None:
        print("\nVerdict :")
        for line in report.verdict:
            print(f"  - {line}")
    for key, path in (res.get("paths") or {}).items():
        print(f"{key:12s} {path}")
    print(f"{ctx.hands_played} mains jouées en {time.perf_counter() - t0:.1f} s")
    if cmd == "smoke" and not res.get("within_budget", True):
        print("ERREUR : budget de mains du smoke dépassé", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
