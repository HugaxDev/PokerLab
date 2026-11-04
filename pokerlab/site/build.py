"""Génération du site statique (``python -m pokerlab site``)."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pokerlab.config import PROJECT_ROOT, LabConfig, configs_root, list_strategies, load_yaml
from pokerlab.journal.store import import_paths, load_hands, resolve
from pokerlab.report import theme
from pokerlab.site import pages
from pokerlab.site.assets import MAINS_JS, NAV_CSS, SITE_CSS, site_js
from pokerlab.site.data import expected_cone, hands_payload, lab_tables, list_campaigns, load_reference, played_from_hands
from pokerlab.strategies.registry import resolve_spec


def site_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or configs_root() / "site.yaml")


def strategy_palette() -> tuple[dict[str, str], dict[str, str]]:
    """Couleur fixe et libellé de chaque stratégie (les mêmes partout dans le site et les rapports)."""
    names = list_strategies()
    specs = {n: resolve_spec(n) for n in names}
    colors = theme.strategy_colors(names, {n: specs[n].get("color") for n in names})
    labels = {n: str(specs[n].get("label", n)) for n in names}
    return colors, labels


def _write(path: Path, text: str) -> None:
    """Écriture atomique : le serveur ne sert jamais un fichier à moitié écrit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _plotly_asset(dest: Path) -> None:
    """plotly.js embarqué une seule fois (copié depuis le paquet Python, pas de CDN)."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return
    from plotly.offline import get_plotlyjs

    _write(dest, get_plotlyjs())


def _js_data(path: Path, hands: list[dict[str, Any]], replays: dict[str, Any]) -> None:
    payload = ("window.HANDS = " + json.dumps(hands, ensure_ascii=False, separators=(",", ":")) + ";\n"
               "window.REPLAYS = " + json.dumps(replays, ensure_ascii=False, separators=(",", ":")) + ";\n")
    _write(path, payload.replace("</", "<\\/"))


def ensure_example(cfg: dict[str, Any], lab: LabConfig) -> Path | None:
    """Importe le fichier d'exemple dans son dossier s'il n'y est pas encore."""
    dest = resolve(cfg.get("exemple", "data/exemple"))
    if (dest / "mains.parquet").exists() and (dest / "adversaires.parquet").exists():
        return dest
    for old in ("mains.parquet", "adversaires.parquet", "imports.json"):   # format d'une version précédente : réimport
        (dest / old).unlink(missing_ok=True)
    src = resolve(cfg.get("exemple_source", "journal/exemple_pokerstars.txt"))
    if not src.exists():
        return None
    import_paths([src], lab, str(cfg.get("strategie_suivie", "joueur")), dest, cfg.get("hero"),
                 float(cfg.get("pause_session_minutes", 30)), source="exemple")
    return dest


def report_links(campaigns, site_dir: Path) -> dict[str, str]:
    """Lien relatif vers le rapport détaillé de chaque campagne (copié si hors de la racine servie)."""
    root = site_dir.parent.resolve()
    links: dict[str, str] = {}
    for c in campaigns:
        rep = c.report.resolve()
        if not rep.exists():
            continue
        if root in rep.parents:
            links[str(c.folder)] = os.path.relpath(rep, site_dir.resolve()).replace(os.sep, "/")
        else:
            dest = site_dir / "rapports" / (c.name.replace(" / ", "_") + ".html")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(rep, dest)
            links[str(c.folder)] = f"rapports/{dest.name}"
    return links


def build_site(cfg: dict[str, Any] | None = None, out: str | Path | None = None, campaigns_root: str | Path | None = None,
               real: str | Path | None = None, lab: LabConfig | None = None) -> Path:
    """Génère index (Mon jeu), exemple, labo, mains, fiche et les fichiers partagés ; renvoie le dossier."""
    cfg = dict(cfg or site_config())
    lab = lab or LabConfig.load()
    site = resolve(out or cfg.get("sortie", "out/site"))
    root = resolve(campaigns_root or cfg.get("campagnes", "out"))
    real_dir = resolve(real or cfg.get("donnees_reelles", "data/reel"))
    strategy = str(cfg.get("strategie_suivie", "joueur"))
    composition = str(cfg.get("composition_reference", "micro_play_money"))
    version = datetime.now().isoformat(timespec="milliseconds")
    colors, labels = strategy_palette()
    color = colors.get(strategy, theme.CATEGORICAL_LIGHT[0])
    label = labels.get(strategy, strategy)
    site.mkdir(parents=True, exist_ok=True)

    _plotly_asset(site / "assets" / "plotly.js")
    _write(site / "assets" / "site.css", SITE_CSS)
    _write(site / "assets" / "site.js", site_js())
    _write(site / "assets" / "mains.js", MAINS_JS)

    exclude = [site] + [root / e for e in cfg.get("exclure", [])]
    ref = load_reference(root, strategy, composition, exclude)
    real_hands = load_hands(real_dir)
    ex_dir = ensure_example(cfg, lab)
    ex_hands = load_hands(ex_dir) if ex_dir else real_hands.iloc[0:0]
    has_real = not real_hands.empty
    has_example = not ex_hands.empty

    from pokerlab.config import configs_root
    from pokerlab.journal.population import player_stats
    from pokerlab.journal.store import load_opponents

    has_population = (configs_root() / "tables" / "population_reelle.yaml").exists()

    def jeu_page(played, name: str, active: str, folder) -> None:
        cone = expected_cone(ref, max(played.n, 1), int(cfg.get("cone_tirages", 2000)), lab.seed) if ref else None
        opp = pages.opponents_section(player_stats(load_opponents(folder)), has_population) if folder else ""
        body = pages.mon_jeu(played, ref, cone, color, label, cfg, opp)
        _write(site / name, pages.page("Mon jeu", active, body, version, plotly=True))

    if has_real:
        played = played_from_hands(real_hands, example=False)
        jeu_page(played, "index.html", "index.html", real_dir)
    else:
        _write(site / "index.html", pages.page("Mon jeu", "index.html", pages.empty_state(cfg, has_example), version))
    if has_example:
        ex_played = played_from_hands(ex_hands, example=True)
        jeu_page(ex_played, "exemple.html", "index.html", ex_dir)
    else:
        (site / "exemple.html").unlink(missing_ok=True)

    shown = played if has_real else (ex_played if has_example else None)
    if shown is not None:
        hands, replays = hands_payload(shown)
        sessions = sorted({h["session"] for h in hands})
        positions = [p for p in ("BTN", "CO", "MP", "UTG", "SB", "BB") if any(h["position"] == p for h in hands)]
    else:
        hands, replays, sessions, positions = [], {}, [], []
    _js_data(site / "data" / "mains.js", hands, replays)
    body = pages.mains(not has_real and has_example, sessions, positions)
    if shown is None:
        body += '<div class="banner">Aucune main : importe tes historiques PokerStars (voir <a href="index.html">Mon jeu</a>).</div>'
    _write(site / "mains.html", pages.page("Mains", "mains.html", body, version, scripts=("data/mains.js", "assets/mains.js")))

    campaigns = list_campaigns(root, exclude)
    cells, rob = lab_tables(root, exclude)
    sweeps = [c for c in campaigns if c.experience == "sweep"]
    ablations = [c for c in campaigns if c.experience == "ablation"]
    links = report_links(campaigns, site)
    body = pages.labo(cells, rob, colors, labels, campaigns, links, sweeps, ablations)
    _write(site / "labo.html", pages.page("Labo", "labo.html", body, version, plotly=True))

    from pokerlab.fiche.build import build_fiche

    fiche_campaign = cfg.get("fiche_campagne") or (root if campaigns else None)
    nav = pages.nav("fiche.html")
    head = f"<style>{NAV_CSS}main>button.theme{{display:none}}</style>"
    tmp = site / "fiche.html.tmp"
    build_fiche(strategy, lab, fiche_campaign, tmp, cfg.get("fiche_composition") or composition, nav_html=nav, head_extra=head)
    os.replace(tmp, site / "fiche.html")

    _write(site / "version.json", json.dumps({"version": version}))
    return site


def default_site_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or site_config()
    return resolve(cfg.get("sortie", "out/site"))


__all__ = ["build_site", "site_config", "default_site_dir", "PROJECT_ROOT"]
