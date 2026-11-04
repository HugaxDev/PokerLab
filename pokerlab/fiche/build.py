"""Commande ``fiche`` : assemble et écrit la fiche HTML d'une stratégie."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pokerlab.analysis.stats import z_value
from pokerlab.config import PROJECT_ROOT, LabConfig, configs_root, list_strategies, load_yaml
from pokerlab.fiche.content import expected_result, find_campaign, leak_prohibitions, postflop_rules, size_rows
from pokerlab.fiche.preflop import preflop_chart
from pokerlab.fiche.render import expected_section, leaks_section, render, todo_block
from pokerlab.strategies.registry import build_strategy, resolve_spec

_CHARTS: dict[tuple[str, float], dict] = {}


def _display_path(path: str | Path) -> str:
    """Chemin lisible : relatif au projet quand c'est possible."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fiche_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or configs_root() / "fiche.yaml")


def chart_for(strategy: str, lab: LabConfig, raise_bb: float) -> dict:
    """Grilles préflop (avec cache) : utilisées par la fiche et par le journal."""
    key = (strategy, raise_bb)
    if key not in _CHARTS:
        _CHARTS[key] = preflop_chart(build_strategy(resolve_spec(strategy), lab), lab.game, raise_bb)
    return _CHARTS[key]


def build_fiche(strategy: str, lab: LabConfig, campaign: str | Path | None = None, out: str | Path | None = None,
                composition: str | None = None, cfg: dict[str, Any] | None = None, nav_html: str = "",
                head_extra: str = "") -> dict[str, Any]:
    """Écrit la fiche et renvoie son chemin et ce qui a pu être rempli."""
    cfg = cfg or fiche_config()
    spec = resolve_spec(strategy)
    name = spec["name"]
    raise_bb = float(cfg.get("reference_raise_bb", 3.0))
    chart = chart_for(name, lab, raise_bb)
    sizes = size_rows(spec, raise_bb, float(cfg.get("reference_pot_bb", 6.5)))
    rules = postflop_rules(spec)
    filled = {"interdits": False, "resultat": False}
    if campaign is None:
        why = "aucune campagne fournie"
        leaks_html = todo_block(name, why)
        expected_html = todo_block(name, why)
    else:
        camp = find_campaign(campaign, name)
        conf = float(lab.raw.get("analysis", {}).get("confidence", 0.95))
        leaks, src = leak_prohibitions(camp, name, int(cfg.get("top_leaks", 5)), int(cfg.get("leak_min_count", 30)), z_value(conf))
        leaks_html = leaks_section(leaks, src, name)
        filled["interdits"] = bool(leaks)
        exp, src2 = expected_result(camp, name, [int(h) for h in cfg.get("horizons", [50, 200, 1000])], int(cfg.get("bootstrap_resamples", 4000)),
                                    int(cfg.get("min_hands_reliable", 10000)), composition, lab.seed, conf)
        expected_html = expected_section(exp, src2, name)
        filled["resultat"] = exp is not None
    source = f"configs/strategies/{name}.yaml" if name in list_strategies() else "configs/opponents/archetypes.yaml"
    meta = {"label": spec.get("label", name), "description": spec.get("description", ""), "color": spec.get("color"),
            "source": source, "campaign": _display_path(campaign) if campaign else ""}
    html = render(meta, chart, sizes, list(cfg.get("stakes", [])), rules, leaks_html, expected_html, raise_bb, nav_html, head_extra)
    path = Path(out) if out else PROJECT_ROOT / "out" / f"fiche_{name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return {"path": path, "filled": filled, "rules": rules, "chart": chart}
