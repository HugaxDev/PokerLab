"""Protocole de stratégie et contexte partagé (équité, ranges)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from pokerlab.config import EquityConfig, GameConfig
from pokerlab.engine.types import Action, Observation
from pokerlab.equity.montecarlo import EquityCache, equity_vs_ranges
from pokerlab.equity.preflop import PreflopTable
from pokerlab.rng import stable_hash


@runtime_checkable
class Strategy(Protocol):
    """Interface unique : un joueur (règles, robot d'équité, futur LLM...) décide d'une action.

    ``decide`` ne reçoit que l'observation (ce que voit le joueur) et renvoie une ``Action``.
    Le moteur ne connaît rien d'autre de la stratégie.
    """

    name: str

    def decide(self, observation: Observation) -> Action: ...


@dataclass
class StrategyContext:
    """Ressources partagées par les stratégies d'un même processus."""

    game: GameConfig
    equity: EquityConfig
    table: PreflopTable
    cache: EquityCache = field(init=False)

    def __post_init__(self) -> None:
        self.cache = EquityCache(self.equity.cache_size)

    def estimate_equity(self, hero: tuple[int, int], board: tuple[int, ...], ranges: list[np.ndarray], key: Any, sims: int) -> float:
        """Équité MC mise en cache ; la graine dépend de la clé (résultat indépendant de l'ordre)."""
        full_key = ("eq", tuple(sorted(hero)), tuple(board), key, sims)
        value = self.cache.get(full_key)
        if value is None:
            value = equity_vs_ranges(hero, board, ranges, sims, stable_hash(full_key) ^ self.equity.seed)
            self.cache.put(full_key, value)
        return value


def decision_rng(obs: Observation, salt: int = 0) -> random.Random:
    """Générateur aléatoire propre à la décision (reproductible)."""
    return random.Random(obs.decision_seed ^ salt)


def pot_bet(obs: Observation, fraction: float) -> int:
    """Montant total visé pour une mise de ``fraction`` x pot (street sans mise)."""
    return obs.bets[obs.seat] + int(round(fraction * obs.pot))


def make_raise(obs: Observation, target: float, shove_fraction: float, tag: str = "") -> Action:
    """Relance/mise vers ``target`` (jetons), bornée aux montants légaux ; tapis si elle engage
    plus de ``shove_fraction`` du tapis. Sans relance possible : paie/check."""
    legal = obs.legal
    if not legal.can_raise:
        return Action.check_call(tag)
    amount = min(max(int(round(target)), legal.min_raise_to), legal.max_raise_to)
    if amount >= shove_fraction * legal.max_raise_to:
        amount = legal.max_raise_to
    return Action.raise_to(amount, tag)


def check_or_fold(obs: Observation, tag: str = "") -> Action:
    return Action.check_call(tag) if obs.to_call == 0 else Action.fold(tag)
