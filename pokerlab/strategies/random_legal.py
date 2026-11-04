"""Robot témoin : actions légales tirées au hasard (contrôle du moteur et des stats)."""

from __future__ import annotations

from typing import Any

from pokerlab.engine.types import Action, Observation
from pokerlab.strategies.base import decision_rng


class RandomLegal:
    """Choisit uniformément (selon des poids) parmi se coucher / suivre / relancer."""

    def __init__(self, config: dict[str, Any], context: object | None = None) -> None:
        del context
        self.name: str = config["name"]
        self.label: str = config.get("label", self.name)
        weights = config.get("weights", {})
        self.w_fold = float(weights.get("fold", 1.0))
        self.w_call = float(weights.get("call", 1.0))
        self.w_raise = float(weights.get("raise", 1.0))
        self.min_raise_share = float(config.get("min_raise_share", 0.5))

    def decide(self, observation: Observation) -> Action:
        rng = decision_rng(observation)
        legal = observation.legal
        options: list[tuple[str, float]] = [("call", self.w_call)]
        if legal.can_fold:
            options.append(("fold", self.w_fold))
        if legal.can_raise:
            options.append(("raise", self.w_raise))
        total = sum(w for _, w in options)
        u = rng.random() * total
        choice = options[-1][0]
        for name, w in options:
            if u < w:
                choice = name
                break
            u -= w
        if choice == "fold":
            return Action.fold("random")
        if choice == "raise":
            if rng.random() < self.min_raise_share:
                amount = legal.min_raise_to
            else:
                amount = rng.randint(legal.min_raise_to, legal.max_raise_to)
            return Action.raise_to(amount, "random")
        return Action.check_call("random")
