"""Historique de main au format texte proche de PokerStars (pour relire/rejouer une main)."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from pokerlab.cards.cards import CARD_STR
from pokerlab.config import GameConfig
from pokerlab.engine.hand_runner import HandResult
from pokerlab.engine.types import Kind

STREET_TITLES = ("*** HOLE CARDS ***", "*** FLOP ***", "*** TURN ***", "*** RIVER ***")


def _money(chips: float, cfg: GameConfig) -> str:
    value = chips / cfg.chips_per_bb * cfg.currency_per_bb
    return f"{cfg.currency_symbol}{value:.2f}"


def _cards(cards: Sequence[int]) -> str:
    return "[" + " ".join(CARD_STR[c] for c in cards) + "]"


def uncalled_returns(hr: HandResult) -> dict[int, int]:
    """Mises non suivies rendues (règle PokerKit : au-delà de la 2e mise la plus haute de la street)."""
    returns: dict[int, int] = defaultdict(int)
    for street in range(4):
        totals: dict[int, int] = defaultdict(int)
        for a in hr.history:
            if a.street == street:
                totals[a.seat] = max(totals[a.seat], a.to)
        if len(totals) < 1:
            continue
        ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
        top_seat, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if top > second:
            returns[top_seat] += top - second
    return dict(returns)


def format_hand(
    hr: HandResult,
    cfg: GameConfig,
    table_name: str = "PokerLab",
    labels: Sequence[str] | None = None,
    hero: int | None = None,
) -> str:
    """Texte lisible de la main. ``labels`` : nom affiché par siège PokerKit."""
    n = len(hr.positions)
    names = [f"{hr.positions[i]}-{(labels or hr.names)[i]}" for i in range(n)]
    lines = [
        f"PokerLab Hand #{hr.hand_id}: Hold'em No Limit ({_money(cfg.small_blind, cfg)}/{_money(cfg.big_blind, cfg)}) - graine {hr.seed}",
        f"Table '{table_name}' {n}-max Seat #{n} is the button",
    ]
    for i in range(n):
        lines.append(f"Seat {i + 1}: {names[i]} ({_money(hr.starting_stacks[i], cfg)} in chips)")
    street = -1
    max_bet = 0
    board_cut = {1: 3, 2: 4, 3: 5}
    for a in hr.history:
        if a.kind == Kind.POST:
            what = "small blind" if a.seat == 0 else "big blind"
            lines.append(f"{names[a.seat]}: posts {what} {_money(a.amount, cfg)}")
            max_bet = max(max_bet, a.to)
            continue
        if a.street != street:
            street = a.street
            if street == 0:
                lines.append(STREET_TITLES[0])
                shown = [hero] if hero is not None else list(range(n))
                for i in shown:
                    lines.append(f"Dealt to {names[i]} {_cards(hr.holes[i])}")
            else:
                cut = board_cut[street]
                prev = hr.board[: cut - (3 if street == 1 else 1)]
                new = hr.board[len(prev):cut]
                head = f"{STREET_TITLES[street]} {_cards(prev)} {_cards(new)}" if prev else f"{STREET_TITLES[street]} {_cards(new)}"
                lines.append(head)
                max_bet = 0
        tail = " and is all-in" if a.all_in else ""
        if a.kind == Kind.FOLD:
            lines.append(f"{names[a.seat]}: folds")
        elif a.kind == Kind.CHECK:
            lines.append(f"{names[a.seat]}: checks")
        elif a.kind == Kind.CALL:
            lines.append(f"{names[a.seat]}: calls {_money(a.amount, cfg)}{tail}")
        elif a.kind == Kind.BET:
            lines.append(f"{names[a.seat]}: bets {_money(a.amount, cfg)}{tail}")
            max_bet = a.to
        elif a.kind == Kind.RAISE:
            lines.append(f"{names[a.seat]}: raises {_money(a.to - max_bet, cfg)} to {_money(a.to, cfg)}{tail}")
            max_bet = a.to
    # Cartes restantes distribuées après un tapis.
    dealt_streets = {a.street for a in hr.history}
    for s in (1, 2, 3):
        if s not in dealt_streets and len(hr.board) >= board_cut[s]:
            cut = board_cut[s]
            prev = hr.board[: cut - (3 if s == 1 else 1)]
            new = hr.board[len(prev):cut]
            lines.append(f"{STREET_TITLES[s]} {_cards(prev)} {_cards(new)}" if prev else f"{STREET_TITLES[s]} {_cards(new)}")
    returns = uncalled_returns(hr)
    for seat, amount in returns.items():
        if amount > 0:
            lines.append(f"Uncalled bet ({_money(amount, cfg)}) returned to {names[seat]}")
    contributed = defaultdict(int)
    for a in hr.history:
        contributed[a.seat] += a.amount
    for seat, amount in returns.items():
        contributed[seat] -= amount
    if hr.showdown:
        lines.append("*** SHOW DOWN ***")
        for i in range(n):
            if not hr.folded[i]:
                lines.append(f"{names[i]}: shows {_cards(hr.holes[i])}")
    total = sum(contributed.values())
    for i in range(n):
        won = hr.payoffs[i] + contributed[i]
        if won > 0:
            lines.append(f"{names[i]} collected {_money(won, cfg)} from pot")
    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {_money(total, cfg)} | Rake {_money(0, cfg)}")
    if hr.board:
        lines.append(f"Board {_cards(hr.board)}")
    for i in range(n):
        res_bb = hr.payoffs[i] / cfg.chips_per_bb
        ev_bb = hr.ev_payoffs[i] / cfg.chips_per_bb + 0.0
        extra = f" | all-in EV {ev_bb:+.2f} bb" if hr.allin_adjusted else ""
        status = "folded" if hr.folded[i] else ("showed " + _cards(hr.holes[i]) if hr.showdown else "won uncontested")
        lines.append(f"Seat {i + 1}: {names[i]} {status} | résultat {res_bb:+.2f} bb{extra}")
    return "\n".join(lines) + "\n"
