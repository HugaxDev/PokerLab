"""Fixtures partagées. Budget : la suite entière joue moins de 2 000 mains."""

from __future__ import annotations

import warnings

import pytest

from pokerlab.config import LabConfig
from pokerlab.engine.types import ActionRecord, Kind, Legal, Observation, position_names


@pytest.fixture(scope="session")
def lab() -> LabConfig:
    warnings.simplefilter("error", UserWarning)
    lab = LabConfig.load(overrides={"workers": 1})
    from pokerlab.strategies.registry import get_context

    get_context(lab)
    return lab


def make_obs(
    hole: tuple[int, int],
    board: tuple[int, ...] = (),
    seat: int = 5,
    history: tuple[ActionRecord, ...] = (),
    to_call: int = 0,
    pot: int = 150,
    bets: tuple[int, ...] | None = None,
    folded: tuple[bool, ...] | None = None,
    stacks: tuple[int, ...] | None = None,
    can_raise: bool = True,
    seed: int = 1,
) -> Observation:
    """Construit une observation cohérente pour tester une règle isolée (6 joueurs, bb = 100)."""
    n = 6
    positions = position_names(n)
    bets = bets if bets is not None else tuple([0] * n)
    stacks = stacks if stacks is not None else tuple([10000 - (pot // n)] * n)
    folded = folded if folded is not None else tuple([False] * n)
    street = {0: 0, 3: 1, 4: 2, 5: 3}[len(board)]
    max_to = stacks[seat] + bets[seat]
    min_to = min(max(max(bets) * 2, 100), max_to) if can_raise else 0
    legal = Legal(can_fold=to_call > 0, call_amount=to_call, can_raise=can_raise, min_raise_to=min_to, max_raise_to=max_to if can_raise else 0)
    return Observation(
        hand_id=0,
        seat=seat,
        position=positions[seat],
        positions=positions,
        hole=hole,
        board=tuple(board),
        street=street,
        stacks=stacks,
        bets=bets,
        contributions=tuple(10000 - s for s in stacks),
        starting_stacks=tuple([10000] * n),
        folded=folded,
        pot=pot,
        legal=legal,
        history=history,
        big_blind=100,
        small_blind=50,
        decision_seed=seed,
    )


def rec(street: int, seat: int, kind: Kind, amount: int = 0, to: int = 0, all_in: bool = False) -> ActionRecord:
    return ActionRecord(street, seat, kind, amount, to, all_in, 0)


BLINDS = (rec(0, 0, Kind.POST, 50, 50), rec(0, 1, Kind.POST, 100, 100))


def pytest_terminal_summary(terminalreporter) -> None:  # pragma: no cover - affichage
    from pokerlab.engine.hand_runner import COUNTERS

    terminalreporter.write_line(f"Mains jouées par la suite de tests (processus principal) : {COUNTERS['hands']}")
