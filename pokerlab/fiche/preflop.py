"""Grilles préflop 13x13 d'une stratégie, calculées en interrogeant la stratégie elle-même.

Pour chaque position et chaque classe de main, on construit l'observation type
(« personne n'a relancé » ou « face à une relance de référence ») et on demande sa
décision à la stratégie : la fiche affiche donc exactement ce que joue le moteur.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pokerlab.cards.ranges import CLASS_COMBOS, COMBOS, N_CLASSES
from pokerlab.config import GameConfig
from pokerlab.engine.types import ActionRecord, Kind, Legal, Observation, position_names
from pokerlab.rng import derive_seed

FOLD, CALL, RAISE = 0, 1, 2
POSITIONS = ("UTG", "MP", "CO", "BTN", "SB", "BB")
SITUATIONS = ("unopened", "vs_raise")
SITUATION_LABELS = {"unopened": "personne n'a relancé", "vs_raise": "face à une relance"}
ACTING_ORDER = (2, 3, 4, 5, 0, 1)  # index PokerKit : UTG, MP, CO, BTN, SB, BB


def raiser_for(position: str) -> str | None:
    """Siège qui relance devant le joueur dans la situation « face à une relance »."""
    names = position_names(6)
    hero = names.index(position)
    if hero == 2:
        return None
    if hero in (0, 1):
        return "BTN"
    return names[ACTING_ORDER[ACTING_ORDER.index(hero) - 1]]


def preflop_observation(position: str, situation: str, hole: tuple[int, int], raise_bb: float, game: GameConfig) -> Observation | None:
    """Observation type pour la grille. ``None`` si la situation n'existe pas (UTG face à une relance)."""
    names = position_names(6)
    hero = names.index(position)
    bb, sb, stack0 = game.big_blind, game.small_blind, game.starting_stack
    bets = [sb, bb, 0, 0, 0, 0]
    history: list[ActionRecord] = [ActionRecord(0, 0, Kind.POST, sb, sb, False, 0), ActionRecord(0, 1, Kind.POST, bb, bb, False, 0)]
    folded = [False] * 6
    before = list(ACTING_ORDER[: ACTING_ORDER.index(hero)])
    raiser = None
    if situation == "vs_raise":
        name = raiser_for(position)
        if name is None:
            return None
        raiser = names.index(name)
    target = int(round(raise_bb * bb))
    for seat in before:
        pot = sum(bets)
        if situation == "vs_raise" and seat == raiser:
            history.append(ActionRecord(0, seat, Kind.RAISE, target - bets[seat], target, False, pot))
            bets[seat] = target
        elif situation == "unopened" and hero == 1 and seat == 0:
            history.append(ActionRecord(0, 0, Kind.CALL, bb - sb, bb, False, pot))
            bets[0] = bb
        else:
            history.append(ActionRecord(0, seat, Kind.FOLD, 0, bets[seat], False, pot))
            folded[seat] = True
    max_bet = max(bets)
    to_call = max_bet - bets[hero]
    stacks = tuple(stack0 - b for b in bets)
    increment = max(bb, max_bet - bb)
    max_to = stack0
    legal = Legal(can_fold=to_call > 0, call_amount=to_call, can_raise=True, min_raise_to=min(max_bet + increment, max_to), max_raise_to=max_to)
    return Observation(
        hand_id=0,
        seat=hero,
        position=position,
        positions=names,
        hole=hole,
        board=(),
        street=0,
        stacks=stacks,
        bets=tuple(bets),
        contributions=tuple(bets),
        starting_stacks=tuple([stack0] * 6),
        folded=tuple(folded),
        pot=sum(bets),
        legal=legal,
        history=tuple(history),
        big_blind=bb,
        small_blind=sb,
        decision_seed=derive_seed("fiche", position, situation, hole[0], hole[1]),
    )


def representative(cls: int) -> tuple[int, int]:
    """Combinaison représentative d'une classe (les règles préflop ne dépendent que de la classe)."""
    a, b = COMBOS[int(CLASS_COMBOS[cls][0])]
    return int(a), int(b)


def action_code(kind: str) -> int:
    return {"raise": RAISE, "call": CALL}.get(kind, FOLD)


def preflop_chart(strategy: Any, game: GameConfig, raise_bb: float) -> dict[tuple[str, str], np.ndarray | None]:
    """Action (0 couché, 1 paie/check, 2 relance) par classe, pour chaque position et situation."""
    chart: dict[tuple[str, str], np.ndarray | None] = {}
    for position in POSITIONS:
        for situation in SITUATIONS:
            if preflop_observation(position, situation, representative(0), raise_bb, game) is None:
                chart[(position, situation)] = None
                continue
            codes = np.zeros(N_CLASSES, dtype=np.int8)
            for cls in range(N_CLASSES):
                obs = preflop_observation(position, situation, representative(cls), raise_bb, game)
                assert obs is not None
                codes[cls] = action_code(strategy.decide(obs).kind)
            chart[(position, situation)] = codes
    return chart


def chart_shares(codes: np.ndarray) -> tuple[float, float]:
    """Part des combinaisons relancées et payées (en %, pondéré par le nombre de combos)."""
    from pokerlab.cards.ranges import CLASS_COMBO_COUNT

    weights = CLASS_COMBO_COUNT.astype(float)
    total = weights.sum()
    return 100 * float(weights[codes == RAISE].sum() / total), 100 * float(weights[codes == CALL].sum() / total)
