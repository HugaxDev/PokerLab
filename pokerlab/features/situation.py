"""Lecture de la situation à partir de l'observation : spot préflop, cote, SPR, position."""

from __future__ import annotations

from pokerlab.engine.types import ActionRecord, Kind, Observation

AGGRESSIVE = (Kind.BET, Kind.RAISE)


def street_actions(history: tuple[ActionRecord, ...], street: int) -> list[ActionRecord]:
    return [a for a in history if a.street == street and a.kind != Kind.POST]


def preflop_raises(history: tuple[ActionRecord, ...]) -> list[ActionRecord]:
    return [a for a in history if a.street == 0 and a.kind in AGGRESSIVE]


def preflop_aggressor(history: tuple[ActionRecord, ...]) -> int | None:
    raises = preflop_raises(history)
    return raises[-1].seat if raises else None


def limpers(history: tuple[ActionRecord, ...]) -> int:
    """Nombre de joueurs qui ont payé la grosse blind avant toute relance."""
    count = 0
    for a in history:
        if a.street != 0 or a.kind == Kind.POST:
            continue
        if a.kind in AGGRESSIVE:
            break
        if a.kind == Kind.CALL:
            count += 1
    return count


def preflop_spot(obs: Observation) -> str:
    """'unopened' | 'vs_open' | 'vs_3bet' (le joueur a ouvert) | 'vs_4bet' (au-delà)."""
    raises = preflop_raises(obs.history)
    if not raises:
        return "unopened"
    if len(raises) == 1:
        return "vs_open"
    if len(raises) == 2 and raises[0].seat == obs.seat:
        return "vs_3bet"
    return "vs_4bet"


def raise_to_bb(obs: Observation) -> float:
    """Montant total de la relance la plus haute de la street, en bb."""
    return max(obs.bets) / obs.big_blind


def pot_odds(obs: Observation) -> float:
    """Équité minimale pour payer : à payer / (pot + à payer)."""
    if obs.to_call <= 0:
        return 0.0
    return obs.to_call / (obs.pot + obs.to_call)


def facing_fraction(obs: Observation) -> float:
    """Taille de la mise adverse rapportée au pot avant cette mise."""
    if obs.to_call <= 0:
        return 0.0
    return obs.to_call / max(obs.pot - obs.to_call, 1)


def facing_all_in(obs: Observation) -> bool:
    """Vrai si payer met le joueur à tapis ou si la dernière mise adverse est un tapis."""
    if obs.to_call <= 0:
        return False
    if obs.to_call >= obs.stack:
        return True
    acts = street_actions(obs.history, obs.street)
    for a in reversed(acts):
        if a.kind in AGGRESSIVE:
            return a.all_in
    return False


def effective_stack(obs: Observation) -> int:
    mine = obs.stacks[obs.seat] + obs.bets[obs.seat]
    others = [obs.stacks[i] + obs.bets[i] for i in obs.active_opponents]
    return min(mine, max(others)) if others else mine


def spr(obs: Observation) -> float:
    return effective_stack(obs) / max(obs.pot, 1)


def no_bet_yet(obs: Observation) -> bool:
    return not any(a.kind in AGGRESSIVE for a in street_actions(obs.history, obs.street))


def bets_on_street(obs: Observation, seat: int | None = None) -> int:
    return sum(
        1 for a in street_actions(obs.history, obs.street) if a.kind in AGGRESSIVE and (seat is None or a.seat == seat)
    )


def hero_was_aggressor_on(obs: Observation, street: int) -> bool:
    return any(a.seat == obs.seat and a.kind in AGGRESSIVE for a in street_actions(obs.history, street))


def in_position(obs: Observation) -> bool:
    """Vrai si le joueur agit en dernier après le flop parmi les joueurs actifs."""
    active = [i for i, f in enumerate(obs.folded) if not f]
    return obs.seat == max(active)
