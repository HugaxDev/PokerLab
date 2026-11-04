"""Table physique : sièges, bouton tournant, tapis persistants et recave."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from pokerlab.config import GameConfig
from pokerlab.engine.hand_runner import HandResult, StrategyLike, play_hand


def pk_order(button: int, n: int) -> list[int]:
    """Sièges physiques dans l'ordre PokerKit (0 = SB, ..., dernier = bouton)."""
    return [(button + 1 + i) % n for i in range(n)]


@dataclass
class Table:
    """Table de ``n`` sièges physiques. En mode ``reset`` chaque main repart à 100 bb."""

    players: Sequence[StrategyLike]
    cfg: GameConfig
    button: int = 0
    stacks: list[int] = field(default_factory=list)
    rebuys: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.players)
        if not self.stacks:
            self.stacks = [self.cfg.starting_stack] * n
        if not self.rebuys:
            self.rebuys = [0] * n

    @property
    def n(self) -> int:
        return len(self.players)

    def apply_rebuys(self) -> None:
        """Recave : un tapis sous ``rebuy.below_bb`` est remonté à ``rebuy.to_bb``."""
        below = self.cfg.rebuy.below_bb * self.cfg.chips_per_bb
        target = int(round(self.cfg.rebuy.to_bb * self.cfg.chips_per_bb))
        for s in range(self.n):
            if self.stacks[s] < below and self.stacks[s] < target:
                self.rebuys[s] += target - self.stacks[s]
                self.stacks[s] = target

    def play(
        self,
        deck: Sequence[int],
        seed: int,
        hand_id: int,
        track_seats: Sequence[int] = (),
    ) -> tuple[HandResult, list[int]]:
        """Joue une main ; renvoie le résultat et l'ordre siège PokerKit -> siège physique."""
        if self.cfg.stack_mode == "reset":
            self.stacks = [self.cfg.starting_stack] * self.n
        else:
            self.apply_rebuys()
        order = pk_order(self.button, self.n)
        players = [self.players[s] for s in order]
        stacks = [self.stacks[s] for s in order]
        track = frozenset(i for i, s in enumerate(order) if s in set(track_seats))
        hr = play_hand(players, self.cfg, deck, stacks, hand_id, seed, track=track)
        for i, s in enumerate(order):
            self.stacks[s] += hr.payoffs[i]
        self.button = (self.button + 1) % self.n
        return hr, order
