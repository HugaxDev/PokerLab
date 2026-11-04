"""Écriture de mains simulées au format texte PokerStars (jetons fictifs), pour les données d'exemple.

Les montants sont écrits en jetons entiers (format « Hold'em No Limit (50/100) » des tables en
jetons fictifs) : la relecture par :mod:`pokerlab.journal.pokerstars` est donc exacte.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from pokerlab.cards.cards import CARD_STR
from pokerlab.config import LabConfig
from pokerlab.engine.hand_runner import HandResult
from pokerlab.engine.history import uncalled_returns
from pokerlab.engine.table import Table
from pokerlab.engine.types import Action, Kind, Observation
from pokerlab.opponents.compositions import Composition
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy, resolve_spec


def _cards(cards: Sequence[int]) -> str:
    return "[" + " ".join(CARD_STR[c] for c in cards) + "]"


def to_pokerstars(hr: HandResult, hand_no: int, when: datetime, table: str, seat_names: Sequence[str],
                  order: Sequence[int], hero_physical: int, button_physical: int) -> str:
    """Texte PokerStars d'une main du moteur. ``order`` : siège PokerKit -> siège physique."""
    n = len(order)
    name = [seat_names[order[i]] for i in range(n)]
    sb, bb = None, None
    for a in hr.history:
        if a.kind == Kind.POST:
            if sb is None:
                sb = a.amount
            elif bb is None:
                bb = a.amount
    lines = [
        f"PokerStars Hand #{hand_no}:  Hold'em No Limit ({sb}/{bb}) - {when:%Y/%m/%d %H:%M:%S} CET",
        f"Table '{table}' {n}-max Seat #{button_physical + 1} is the button",
    ]
    for phys in sorted(order):
        i = list(order).index(phys)
        lines.append(f"Seat {phys + 1}: {name[i]} ({hr.starting_stacks[i]} in chips)")
    street = -1
    max_bet = 0
    cuts = {1: 3, 2: 4, 3: 5}
    hero_pk = list(order).index(hero_physical)
    posts = 0
    for a in hr.history:
        if a.kind == Kind.POST:
            what = "small blind" if posts == 0 else "big blind"
            posts += 1
            lines.append(f"{name[a.seat]}: posts {what} {a.amount}")
            max_bet = max(max_bet, a.to)
            continue
        if a.street != street:
            street = a.street
            if street == 0:
                lines.append("*** HOLE CARDS ***")
                lines.append(f"Dealt to {name[hero_pk]} {_cards(hr.holes[hero_pk])}")
            else:
                prev = hr.board[: cuts[street] - (3 if street == 1 else 1)]
                new = hr.board[len(prev):cuts[street]]
                title = {1: "FLOP", 2: "TURN", 3: "RIVER"}[street]
                lines.append(f"*** {title} *** {_cards(prev)} {_cards(new)}" if prev else f"*** {title} *** {_cards(new)}")
                max_bet = 0
        tail = " and is all-in" if a.all_in else ""
        if a.kind == Kind.FOLD:
            lines.append(f"{name[a.seat]}: folds")
        elif a.kind == Kind.CHECK:
            lines.append(f"{name[a.seat]}: checks")
        elif a.kind == Kind.CALL:
            lines.append(f"{name[a.seat]}: calls {a.amount}{tail}")
        elif a.kind == Kind.BET:
            lines.append(f"{name[a.seat]}: bets {a.amount}{tail}")
            max_bet = a.to
        elif a.kind == Kind.RAISE:
            lines.append(f"{name[a.seat]}: raises {a.to - max_bet} to {a.to}{tail}")
            max_bet = a.to
    dealt = {a.street for a in hr.history}
    for s in (1, 2, 3):
        if s not in dealt and len(hr.board) >= cuts[s]:
            prev = hr.board[: cuts[s] - (3 if s == 1 else 1)]
            new = hr.board[len(prev):cuts[s]]
            title = {1: "FLOP", 2: "TURN", 3: "RIVER"}[s]
            lines.append(f"*** {title} *** {_cards(prev)} {_cards(new)}" if prev else f"*** {title} *** {_cards(new)}")
    returns = uncalled_returns(hr)
    for seat, amount in returns.items():
        if amount > 0:
            lines.append(f"Uncalled bet ({amount}) returned to {name[seat]}")
    contributed: dict[int, int] = defaultdict(int)
    for a in hr.history:
        contributed[a.seat] += a.amount
    for seat, amount in returns.items():
        contributed[seat] -= amount
    if hr.showdown:
        lines.append("*** SHOW DOWN ***")
        for i in range(n):
            if not hr.folded[i]:
                lines.append(f"{name[i]}: shows {_cards(hr.holes[i])}")
    total = sum(contributed.values())
    for i in range(n):
        won = hr.payoffs[i] + contributed[i]
        if won > 0:
            lines.append(f"{name[i]} collected {won} from pot")
    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {total} | Rake 0")
    if hr.board:
        lines.append(f"Board {_cards(hr.board)}")
    return "\n".join(lines) + "\n"


class Deviant:
    """Joueur qui suit une stratégie mais s'en écarte parfois (données d'exemple réalistes)."""

    def __init__(self, inner: Any, rate: float, salt: int) -> None:
        self.inner = inner
        self.name = inner.name
        self.rate = rate
        self.salt = salt

    def decide(self, obs: Observation) -> Action:
        rng = random.Random(obs.decision_seed ^ self.salt)
        if rng.random() < self.rate:
            options = ["call"] + (["fold"] if obs.legal.can_fold else []) + (["raise"] if obs.legal.can_raise else [])
            pick = rng.choice(options)
            if pick == "fold":
                return Action.fold("écart")
            if pick == "raise":
                return Action.raise_to(obs.legal.min_raise_to, "écart")
            return Action.check_call("écart")
        return self.inner.decide(obs)


def generate_example(lab: LabConfig, path: Path, strategy: str = "joueur", composition: str = "micro_play_money",
                     sessions: int = 3, hands_per_session: int = 40, deviation: float = 0.08, start: datetime | None = None) -> int:
    """Écrit un fichier d'historiques PokerStars INVENTÉ (mains simulées) ; renvoie le nombre de mains."""
    comp = Composition.load(composition)
    hero = Deviant(build_strategy(resolve_spec(strategy), lab), deviation, 0xE7)
    start = start or datetime(2026, 10, 20, 21, 0, 0)
    texts = ["# EXEMPLE INVENTÉ : mains simulées par PokerLab au format PokerStars (jetons fictifs). Ce ne sont PAS de vraies mains.\n"]
    hand_no = 250_000_000_000
    count = 0
    for s in range(sessions):
        names_opp = comp.opponents(lab.seed, derive_seed("exemple", s), lab.game.players - 1)
        players = [hero] + [build_strategy(resolve_spec(x), lab) for x in names_opp]
        seat_names = ["Hero"] + [f"Joueur{s}{k}" for k in range(1, lab.game.players)]
        table = Table(players, replace(lab.game, stack_mode="persistent"), button=s % lab.game.players)
        when = start + timedelta(days=s)
        for h in range(hands_per_session):
            seed = derive_seed(lab.seed, "exemple_ps", s, h)
            button = table.button
            hr, order = table.play(shuffled_deck(seed), seed, h)
            hand_no += 1
            texts.append(to_pokerstars(hr, hand_no, when, f"Exemple {s + 1}", seat_names, order, 0, button))
            when += timedelta(seconds=55)
            count += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(texts), encoding="utf-8")
    return count
