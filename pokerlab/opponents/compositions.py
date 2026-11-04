"""Compositions de table : liste fixe d'adversaires ou tirage pondéré (graine fixe) par deck."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from pokerlab.config import load_table
from pokerlab.rng import derive_seed


@dataclass(frozen=True)
class Composition:
    """Composition de table (5 adversaires en 6-max)."""

    name: str
    label: str
    mode: str
    seats: tuple[str, ...]
    weights: tuple[tuple[str, float], ...]
    description: str
    session_decks: int = 1   # adversaires (et leur état : tilt, observations) conservés sur ce nombre de decks consécutifs

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Composition":
        weights = data.get("weights") or {}
        return Composition(
            name=data["name"],
            label=data.get("label", data["name"]),
            mode=data.get("mode", "sample" if weights else "fixed"),
            seats=tuple(data.get("seats") or ()),
            weights=tuple((k, float(v)) for k, v in weights.items()),
            description=data.get("description", ""),
            session_decks=max(1, int(data.get("session_decks", 1))),
        )

    @staticmethod
    def load(name: str) -> "Composition":
        return Composition.from_dict(load_table(name))

    def opponents(self, base_seed: int, unit: int, n: int) -> list[str]:
        """Adversaires pour l'unité (deck ou session) ``unit`` : identiques pour toutes les stratégies."""
        if self.mode == "fixed":
            seats = list(self.seats)
            if len(seats) < n:
                raise ValueError(f"Composition {self.name} : {len(seats)} sièges pour {n} adversaires")
            return seats[:n]
        rng = random.Random(derive_seed(base_seed, "lineup", self.name, unit))
        names = [k for k, _ in self.weights]
        probs = [w for _, w in self.weights]
        return rng.choices(names, weights=probs, k=n)

    def expected_share(self) -> dict[str, float]:
        if self.mode == "fixed":
            out: dict[str, float] = {}
            for s in self.seats:
                out[s] = out.get(s, 0.0) + 1.0 / len(self.seats)
            return out
        total = sum(w for _, w in self.weights)
        return {k: w / total for k, w in self.weights}
