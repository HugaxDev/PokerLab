"""Construction des stratégies (et archétypes) à partir de leur nom ou de leur config."""

from __future__ import annotations

import json
from typing import Any

from pokerlab.config import LabConfig, list_strategies, load_archetypes, load_strategy_config
from pokerlab.equity.preflop import get_preflop_table
from pokerlab.strategies.base import Strategy, StrategyContext

_CONTEXT: StrategyContext | None = None
_CONTEXT_KEY: str | None = None
_BUILT: dict[str, Strategy] = {}
_ARCHETYPES: dict[str, dict[str, Any]] | None = None


def archetypes() -> dict[str, dict[str, Any]]:
    global _ARCHETYPES
    if _ARCHETYPES is None:
        _ARCHETYPES = load_archetypes()
    return _ARCHETYPES


def resolve_spec(spec: str | dict[str, Any]) -> dict[str, Any]:
    """Nom -> config complète. Un dict est renvoyé tel quel (variante de sweep, etc.)."""
    if isinstance(spec, dict):
        return spec
    if spec in list_strategies():
        return load_strategy_config(spec)
    arch = archetypes()
    if spec in arch:
        cfg = dict(arch[spec])
        cfg.setdefault("type", "archetype")
        return cfg
    raise KeyError(f"Stratégie ou archétype inconnu : {spec}")


def get_context(lab: LabConfig) -> StrategyContext:
    """Contexte partagé du processus (table préflop chargée une fois)."""
    global _CONTEXT, _CONTEXT_KEY
    key = json.dumps([lab.raw.get("game", {}), lab.raw.get("equity", {})], sort_keys=True, default=str)
    if _CONTEXT is None or _CONTEXT_KEY != key:
        _CONTEXT = StrategyContext(lab.game, lab.equity, get_preflop_table(lab.equity))
        _CONTEXT_KEY = key
        _BUILT.clear()
    return _CONTEXT


def build_strategy(spec: str | dict[str, Any], lab: LabConfig) -> Strategy:
    """Construit (avec cache par processus) la stratégie décrite par ``spec``."""
    cfg = resolve_spec(spec)
    ctx = get_context(lab)
    key = json.dumps(cfg, sort_keys=True, default=str)
    built = _BUILT.get(key)
    if built is not None:
        return built
    kind = cfg.get("type", "rules")
    if kind == "rules":
        from pokerlab.strategies.rule_engine import RuleStrategy

        built = RuleStrategy(cfg, ctx)
    elif kind == "equity":
        from pokerlab.strategies.equity_bot import EquityBot

        built = EquityBot(cfg, ctx)
    elif kind == "random":
        from pokerlab.strategies.random_legal import RandomLegal

        built = RandomLegal(cfg, ctx)
    elif kind == "archetype":
        from pokerlab.opponents.archetype import Archetype

        built = Archetype(cfg, ctx)
    else:
        raise ValueError(f"Type de stratégie inconnu : {kind}")
    _BUILT[key] = built
    return built


def strategy_meta(spec: str | dict[str, Any]) -> dict[str, Any]:
    """Nom, libellé, couleur d'une stratégie (pour le rapport)."""
    cfg = resolve_spec(spec)
    return {"name": cfg["name"], "label": cfg.get("label", cfg["name"]), "color": cfg.get("color")}
