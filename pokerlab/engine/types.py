"""Types partagés : actions, historique, observation (tout ce qu'un joueur voit)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Kind(IntEnum):
    """Type d'action enregistrée dans l'historique."""

    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    POST = 5


KIND_LABELS = {
    Kind.FOLD: "se couche",
    Kind.CHECK: "check",
    Kind.CALL: "paie",
    Kind.BET: "mise",
    Kind.RAISE: "relance",
    Kind.POST: "blind",
}

STREET_NAMES = ("préflop", "flop", "turn", "river")


def position_names(n_players: int) -> tuple[str, ...]:
    """Noms de position par index PokerKit (0 = petite blind, dernier = bouton)."""
    if n_players == 2:
        return ("SB", "BB")
    middle = {3: [], 4: ["CO"], 5: ["UTG", "CO"], 6: ["UTG", "MP", "CO"]}.get(n_players)
    if middle is None:
        middle = ["UTG"] + [f"MP{i}" for i in range(1, n_players - 4)] + ["CO"]
    return tuple(["SB", "BB"] + middle + ["BTN"])


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Une action de la main (montants en jetons)."""

    street: int
    seat: int
    kind: Kind
    amount: int
    to: int
    all_in: bool
    pot_before: int


@dataclass(frozen=True, slots=True)
class Action:
    """Décision d'une stratégie. ``raise`` couvre mise et relance ; ``amount`` = total visé sur la street."""

    kind: str
    amount: int = 0
    tag: str = ""

    @staticmethod
    def fold(tag: str = "") -> "Action":
        return Action("fold", 0, tag)

    @staticmethod
    def check_call(tag: str = "") -> "Action":
        return Action("call", 0, tag)

    @staticmethod
    def raise_to(amount: int, tag: str = "") -> "Action":
        return Action("raise", int(amount), tag)


@dataclass(frozen=True, slots=True)
class Legal:
    """Actions légales au moment de la décision."""

    can_fold: bool
    call_amount: int
    can_raise: bool
    min_raise_to: int
    max_raise_to: int


@dataclass(frozen=True)
class Observation:
    """Tout ce que voit le joueur qui doit agir (montants en jetons)."""

    hand_id: int
    seat: int
    position: str
    positions: tuple[str, ...]
    hole: tuple[int, int]
    board: tuple[int, ...]
    street: int
    stacks: tuple[int, ...]
    bets: tuple[int, ...]
    contributions: tuple[int, ...]
    starting_stacks: tuple[int, ...]
    folded: tuple[bool, ...]
    pot: int
    legal: Legal
    history: tuple[ActionRecord, ...]
    big_blind: int
    small_blind: int
    decision_seed: int
    decision_index: int = -1
    cache: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def n_players(self) -> int:
        return len(self.stacks)

    @property
    def to_call(self) -> int:
        return self.legal.call_amount

    @property
    def stack(self) -> int:
        return self.stacks[self.seat]

    @property
    def n_active_opponents(self) -> int:
        return sum(1 for i, f in enumerate(self.folded) if not f and i != self.seat)

    @property
    def active_opponents(self) -> list[int]:
        return [i for i, f in enumerate(self.folded) if not f and i != self.seat]

    def in_bb(self, chips: float) -> float:
        return chips / self.big_blind

    def chips(self, bb: float) -> int:
        return int(round(bb * self.big_blind))
