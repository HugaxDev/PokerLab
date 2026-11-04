"""Moteur : conservation des jetons (side pots compris), légalité, mémoïsation, recave."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from pokerlab.engine.hand_runner import play_hand
from pokerlab.engine.table import Table
from pokerlab.engine.types import Action, Kind, Observation
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy


class Checked:
    """Enveloppe qui mémorise les actions légales proposées à chaque décision."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.seen: list[tuple[Observation, Action]] = []

    def decide(self, obs: Observation) -> Action:
        action = self.inner.decide(obs)
        self.seen.append((obs, action))
        return action


def _check_history(hr, wrapped: list[Checked]) -> int:
    """Vérifie que chaque action jouée était légale au moment de la décision."""
    legal_by_idx = {}
    for w in wrapped:
        for obs, _ in w.seen:
            if obs.hand_id == hr.hand_id and obs.decision_index >= 0:
                legal_by_idx[obs.decision_index] = obs.legal
    count = 0
    for a in hr.history:
        if a.kind == Kind.POST:
            continue
        legal = legal_by_idx[count]
        if a.kind == Kind.FOLD:
            assert legal.can_fold
        elif a.kind == Kind.CHECK:
            assert legal.call_amount == 0
        elif a.kind == Kind.CALL:
            assert legal.call_amount > 0 and a.amount == legal.call_amount
        else:
            assert legal.can_raise and legal.min_raise_to <= a.to <= legal.max_raise_to
        count += 1
    return count


def test_random_decisions_are_legal_and_chips_conserved(lab) -> None:
    """10 000 décisions aléatoires : uniquement des actions légales, jetons conservés."""
    rnd = build_strategy("random_legal", lab)
    rng = random.Random(3)
    decisions = 0
    hand = 0
    while decisions < 10_000:
        players = [Checked(rnd) for _ in range(6)]
        stacks = [rng.randint(500, 20_000) for _ in range(6)]  # tapis inégaux : side pots
        seed = derive_seed(99, hand)
        hr = play_hand(players, lab.game, shuffled_deck(seed), stacks, hand, seed)
        assert hr.corrections == 0
        assert sum(hr.payoffs) == 0
        assert sum(hr.ev_payoffs) == pytest.approx(0.0, abs=1e-6)
        assert all(s + p >= 0 for s, p in zip(stacks, hr.payoffs))
        decisions += _check_history(hr, players)
        for w in players:
            w.seen.clear()
        hand += 1
    assert decisions >= 10_000


def test_illegal_requests_are_corrected(lab) -> None:
    """Se coucher sans mise à payer devient check (compté) ; une relance démesurée est bornée au tapis."""

    class Bad:
        name = "bad"

        def decide(self, obs: Observation) -> Action:
            return Action.fold() if obs.to_call == 0 else Action("raise", 10**9)

    seed = derive_seed(5, 0)
    stacks = [10000, 4000, 10000, 2500, 10000, 10000]
    hr = play_hand([Bad()] * 6, lab.game, shuffled_deck(seed), stacks, 0, seed)
    assert sum(hr.payoffs) == 0
    assert hr.corrections >= 1
    for a in hr.history:
        assert a.kind != Kind.FOLD
        assert a.amount <= stacks[a.seat]


def test_memoized_evaluation_matches_pokerkit(lab) -> None:
    rnd = build_strategy("random_legal", lab)
    plain = replace(lab.game, memoize_hand_eval=False)
    for h in range(20):
        seed = derive_seed(7, h)
        deck = shuffled_deck(seed)
        a = play_hand([rnd] * 6, lab.game, deck, [10000] * 6, h, seed)
        b = play_hand([rnd] * 6, plain, deck, [10000] * 6, h, seed)
        assert a.payoffs == b.payoffs and a.history == b.history


def test_persistent_table_conserves_chips_with_rebuys(lab) -> None:
    game = replace(lab.game, stack_mode="persistent")
    players = [build_strategy(n, lab) for n in ("random_legal", "maniac", "calling_station", "joueur", "nit_trappeur", "serre")]
    table = Table(players, game, button=2)
    initial = sum(table.stacks)
    for h in range(30):
        seed = derive_seed(11, h)
        table.play(shuffled_deck(seed), seed, h)
        assert sum(table.stacks) == initial + sum(table.rebuys)
        assert all(s >= 0 for s in table.stacks)
    assert table.button == (2 + 30) % 6
