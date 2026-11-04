"""Chargement des configurations YAML (jeu, stratégies, adversaires, tables, expériences)."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def configs_root() -> Path:
    """Dossier des configurations (variable POKERLAB_CONFIGS, sinon ./configs du projet)."""
    env = os.environ.get("POKERLAB_CONFIGS")
    return Path(env).resolve() if env else PROJECT_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Lit un fichier YAML et renvoie un dict (vide si fichier vide)."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Fusion récursive : ``override`` l'emporte, les listes sont remplacées."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _split_path(path: str) -> list[str | int]:
    parts: list[str | int] = []
    for token in path.split("."):
        while "[" in token:
            head, rest = token.split("[", 1)
            if head:
                parts.append(head)
            index, token = rest.split("]", 1)
            parts.append(int(index))
        if token:
            parts.append(token)
    return parts


def get_path(data: dict[str, Any], path: str) -> Any:
    """Lit une valeur par chemin pointé (``preflop.vs_open.call[0].max_bb``)."""
    node: Any = data
    for part in _split_path(path):
        node = node[part]
    return node


def set_path(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    """Renvoie une copie de ``data`` avec la valeur au chemin pointé remplacée."""
    result = copy.deepcopy(data)
    parts = _split_path(path)
    node: Any = result
    for part in parts[:-1]:
        if isinstance(part, str) and part not in node:
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return result


@dataclass(frozen=True)
class AllInEVConfig:
    """Réglages du calcul d'espérance quand l'argent est à tapis."""

    exact_max_runouts: int = 2000
    mc_samples: int = 3000


@dataclass(frozen=True)
class RebuyConfig:
    """Recave en mode tapis persistants."""

    below_bb: float = 100.0
    to_bb: float = 100.0


@dataclass(frozen=True)
class GameConfig:
    """Paramètres de la partie. Unité interne : la grosse blind (bb)."""

    players: int = 6
    chips_per_bb: int = 100
    small_blind_bb: float = 0.5
    starting_stack_bb: float = 100.0
    currency_per_bb: float = 0.02
    currency_symbol: str = "€"
    stack_mode: str = "reset"
    rebuy: RebuyConfig = field(default_factory=RebuyConfig)
    allin_ev: AllInEVConfig = field(default_factory=AllInEVConfig)
    memoize_hand_eval: bool = True
    shove_commit_fraction: float = 0.7

    @property
    def big_blind(self) -> int:
        return self.chips_per_bb

    @property
    def small_blind(self) -> int:
        return int(round(self.small_blind_bb * self.chips_per_bb))

    @property
    def starting_stack(self) -> int:
        return int(round(self.starting_stack_bb * self.chips_per_bb))

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "GameConfig":
        data = dict(data)
        rebuy = RebuyConfig(**data.pop("rebuy", {}))
        allin = AllInEVConfig(**data.pop("allin_ev", {}))
        return GameConfig(rebuy=rebuy, allin_ev=allin, **data)


@dataclass(frozen=True)
class EquityConfig:
    """Réglages de l'équité Monte Carlo."""

    preflop_table_path: str = "cache/preflop_169.npz"
    preflop_sims_per_matchup: int = 400
    postflop_sims: int = 300
    cache_size: int = 50000
    seed: int = 7

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "EquityConfig":
        return EquityConfig(**data)

    def table_path(self) -> Path:
        path = Path(self.preflop_table_path)
        return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class LabConfig:
    """Configuration globale chargée depuis ``configs/game.yaml``."""

    game: GameConfig
    equity: EquityConfig
    seed: int
    workers: int
    mp_start_method: str
    raw: dict[str, Any]

    @staticmethod
    def load(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> "LabConfig":
        raw = load_yaml(path or configs_root() / "game.yaml")
        if overrides:
            raw = deep_merge(raw, overrides)
        return LabConfig(
            game=GameConfig.from_dict(raw.get("game", {})),
            equity=EquityConfig.from_dict(raw.get("equity", {})),
            seed=int(raw.get("seed", 1)),
            workers=int(raw.get("workers", 0)),
            mp_start_method=str(raw.get("mp_start_method", "forkserver")),
            raw=raw,
        )


def load_strategy_config(name: str, root: Path | None = None) -> dict[str, Any]:
    """Charge la config YAML d'une stratégie, en résolvant ``extends`` (héritage)."""
    root = root or configs_root()
    path = root / "strategies" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Stratégie inconnue : {name} ({path})")
    data = load_yaml(path)
    parent = data.pop("extends", None)
    if parent:
        base = load_strategy_config(parent, root)
        for key in ("name", "label", "color", "description"):
            base.pop(key, None)
        data = deep_merge(base, data)
    data.setdefault("name", name)
    return data


def list_strategies(root: Path | None = None) -> list[str]:
    root = root or configs_root()
    return sorted(p.stem for p in (root / "strategies").glob("*.yaml"))


def load_archetypes(root: Path | None = None) -> dict[str, dict[str, Any]]:
    """Charge les archétypes adverses (``configs/opponents/archetypes.yaml``)."""
    root = root or configs_root()
    data = load_yaml(root / "opponents" / "archetypes.yaml")
    defaults = data.pop("defaults", {})
    result: dict[str, dict[str, Any]] = {}
    for name, params in data.items():
        merged = deep_merge(defaults, params or {})
        merged.setdefault("name", name)
        result[name] = merged
    return result


def load_table(name: str, root: Path | None = None) -> dict[str, Any]:
    """Charge une composition de table (``configs/tables/<nom>.yaml``)."""
    root = root or configs_root()
    data = load_yaml(root / "tables" / f"{name}.yaml")
    data.setdefault("name", name)
    return data


def load_experiment(name_or_path: str, root: Path | None = None) -> dict[str, Any]:
    """Charge une config d'expérience par nom ou chemin."""
    root = root or configs_root()
    path = Path(name_or_path)
    if not path.suffix:
        path = root / "experiments" / f"{name_or_path}.yaml"
    return load_yaml(path)
