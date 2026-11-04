"""Joue une main complète avec PokerKit, à partir d'un paquet ordonné et de stratégies par siège.

Convention de distribution (indispensable au duplicate) : le siège PokerKit ``i``
(0 = SB, 1 = BB, ..., dernier = BTN) reçoit ``deck[2i], deck[2i+1]`` ; le tableau est
``deck[12:17]`` (6 joueurs). Les cartes sont donc liées aux positions, pas aux joueurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from pokerlab.cards.cards import to_pokerkit
from pokerlab.config import GameConfig
from pokerlab.engine.pk import create_state
from pokerlab.engine.types import Action, ActionRecord, Kind, Legal, Observation, position_names
from pokerlab.equity.allin import allin_ev_payoffs
from pokerlab.features.board import board_texture
from pokerlab.features.hand import read_hand
from pokerlab.rng import derive_seed

COUNTERS = {"hands": 0, "decisions": 0, "corrections": 0}


class StrategyLike(Protocol):
    name: str

    def decide(self, observation: Observation) -> Action: ...


@dataclass(slots=True)
class DecisionRecord:
    """Décision d'un siège suivi (pour l'export « une ligne par décision »)."""

    idx: int
    seat: int
    street: int
    position: str
    to_call: int
    pot: int
    stack: int
    n_opp: int
    facing: int
    kind: Kind
    amount: int
    to: int
    tag: str
    invested_before: int
    level: int
    draw: int
    board_label: str
    legal: Legal


@dataclass
class HandResult:
    """Résultat complet d'une main."""

    hand_id: int
    seed: int
    positions: tuple[str, ...]
    names: tuple[str, ...]
    holes: list[tuple[int, int]]
    board: list[int]
    history: list[ActionRecord]
    starting_stacks: list[int]
    payoffs: list[int]
    ev_payoffs: list[float]
    allin_street: int | None
    showdown: bool
    folded: list[bool]
    shown: list[bool]
    decisions: list[DecisionRecord] = field(default_factory=list)
    corrections: int = 0

    @property
    def allin_adjusted(self) -> bool:
        return self.allin_street is not None


def _legal(state) -> Legal:
    call = state.checking_or_calling_amount or 0
    can_raise = state.can_complete_bet_or_raise_to()
    lo = state.min_completion_betting_or_raising_to_amount if can_raise else 0
    hi = state.max_completion_betting_or_raising_to_amount if can_raise else 0
    return Legal(state.can_fold(), int(call), bool(can_raise), int(lo or 0), int(hi or 0))


def _facing(history: list[ActionRecord], street: int, to_call: int, stack: int) -> int:
    """0 rien à payer, 1 mise (préflop : blinds seules), 2 relance, 3 tapis, 4 sur-relance préflop (3-bet+)."""
    if to_call <= 0:
        return 0
    if to_call >= stack:
        return 3
    raises = sum(1 for a in history if a.street == street and a.kind in (Kind.BET, Kind.RAISE))
    if street == 0:
        return 1 if raises == 0 else (2 if raises == 1 else 4)
    return 2 if raises >= 2 else 1


def play_hand(
    players: Sequence[StrategyLike],
    cfg: GameConfig,
    deck: Sequence[int],
    stacks: Sequence[int],
    hand_id: int,
    seed: int,
    track: frozenset[int] | set[int] = frozenset(),
    forced: dict[int, Action] | None = None,
    fork_after: int | None = None,
    fork_salt: int = 0,
) -> HandResult:
    """Joue une main. ``forced`` impose une action à l'index de décision global donné ;
    après ``fork_after``, les graines de décision sont salées par ``fork_salt`` (rollouts)."""
    n = len(players)
    positions = position_names(n)
    state = create_state(cfg.small_blind, cfg.big_blind, list(stacks), cfg.memoize_hand_eval)
    start = list(stacks)
    holes = [(deck[2 * i], deck[2 * i + 1]) for i in range(n)]
    board_source = list(deck[2 * n:2 * n + 5])
    pk_holes = [to_pokerkit(h) for h in holes]
    for i in range(n):
        state.deal_hole(pk_holes[i])
    history: list[ActionRecord] = []
    for i in range(n):
        posted = start[i] - state.stacks[i]
        if posted:
            history.append(ActionRecord(0, i, Kind.POST, posted, posted, state.stacks[i] == 0, 0))
    board: list[int] = []
    folded = [False] * n
    seat_decisions = [0] * n
    decisions: list[DecisionRecord] = []
    ev_payoffs: list[float] | None = None
    allin_street: int | None = None
    corrections = 0
    global_idx = 0
    bb = cfg.big_blind
    while state.status:
        actor = state.actor_index
        if actor is not None:
            street = state.street_index or 0
            legal = _legal(state)
            cur_stacks = tuple(state.stacks)
            contributions = tuple(start[i] - cur_stacks[i] for i in range(n))
            pot = sum(contributions)
            if fork_after is not None and global_idx > fork_after:
                dseed = derive_seed(seed, actor, seat_decisions[actor], fork_salt)
            else:
                dseed = derive_seed(seed, actor, seat_decisions[actor])
            obs = Observation(
                hand_id=hand_id,
                seat=actor,
                position=positions[actor],
                positions=positions,
                hole=holes[actor],
                board=tuple(board),
                street=street,
                stacks=cur_stacks,
                bets=tuple(state.bets),
                contributions=contributions,
                starting_stacks=tuple(start),
                folded=tuple(folded),
                pot=pot,
                legal=legal,
                history=tuple(history),
                big_blind=bb,
                small_blind=cfg.small_blind,
                decision_seed=dseed,
                decision_index=global_idx,
            )
            if forced is not None and global_idx in forced:
                action = forced[global_idx]
            else:
                action = players[actor].decide(obs)
            before_bet = state.bets[actor]
            max_bet = max(state.bets)
            kind: Kind
            if action.kind == "fold" and legal.can_fold:
                state.fold()
                kind = Kind.FOLD
                folded[actor] = True
            elif action.kind == "raise" and legal.can_raise:
                target = min(max(int(action.amount), legal.min_raise_to), legal.max_raise_to)
                state.complete_bet_or_raise_to(target)
                kind = Kind.BET if max_bet == 0 else Kind.RAISE
            else:
                if action.kind not in ("call", "check"):
                    corrections += 1
                state.check_or_call()
                kind = Kind.CHECK if legal.call_amount == 0 else Kind.CALL
            if kind == Kind.CALL:
                after_bet = before_bet + legal.call_amount
            elif kind in (Kind.BET, Kind.RAISE):
                after_bet = target
            else:
                after_bet = before_bet
            added = after_bet - before_bet
            rec = ActionRecord(street, actor, kind, added, after_bet, added >= cur_stacks[actor] and added > 0, pot)
            if actor in track:
                if street > 0:
                    hr = read_hand(holes[actor], board)
                    level, draw = hr.level, (2 if hr.strong_draw else (1 if hr.any_draw else 0))
                    blabel = board_texture(board).label
                else:
                    level, draw, blabel = -1, 0, "préflop"
                decisions.append(
                    DecisionRecord(
                        idx=global_idx,
                        seat=actor,
                        street=street,
                        position=positions[actor],
                        to_call=legal.call_amount,
                        pot=pot,
                        stack=cur_stacks[actor],
                        n_opp=obs.n_active_opponents,
                        facing=_facing(history, street, legal.call_amount, cur_stacks[actor]),
                        kind=kind,
                        amount=added,
                        to=after_bet,
                        tag=action.tag,
                        invested_before=contributions[actor],
                        level=level,
                        draw=draw,
                        board_label=blabel,
                        legal=legal,
                    )
                )
            history.append(rec)
            seat_decisions[actor] += 1
            global_idx += 1
        elif state.can_burn_card("??"):
            if state.all_in_status and ev_payoffs is None and len(board) < 5:
                live = {i: holes[i] for i in range(n) if state.statuses[i]}
                if len(live) >= 2:
                    pots = [(p.amount, tuple(p.player_indices)) for p in state.pots]
                    contributions = [start[i] - state.stacks[i] for i in range(n)]
                    ev_payoffs = allin_ev_payoffs(
                        live,
                        board,
                        pots,
                        contributions,
                        cfg.allin_ev.exact_max_runouts,
                        cfg.allin_ev.mc_samples,
                        derive_seed(seed, "allin"),
                    )
                    allin_street = {0: 0, 3: 1, 4: 2}[len(board)]
            state.burn_card("??")
        elif state.board_dealing_count:
            count = state.board_dealing_count
            cards = board_source[len(board):len(board) + count]
            state.deal_board(to_pokerkit(cards))
            board.extend(cards)
        else:  # pragma: no cover - garde-fou
            raise RuntimeError(f"Main {hand_id} bloquée : état PokerKit inattendu")
    payoffs = list(state.payoffs)
    if sum(payoffs) != 0:  # pragma: no cover - garde-fou
        raise RuntimeError(f"Main {hand_id} : jetons non conservés {payoffs}")
    shown = [False] * n
    for op in state.operations:
        if type(op).__name__ == "HoleCardsShowingOrMucking" and op.hole_cards:
            shown[op.player_index] = True
    showdown = sum(1 for f in folded if not f) >= 2
    COUNTERS["hands"] += 1
    COUNTERS["decisions"] += global_idx
    COUNTERS["corrections"] += corrections
    names = tuple(getattr(p, "name", type(p).__name__) for p in players)
    return HandResult(
        hand_id=hand_id,
        seed=seed,
        positions=positions,
        names=names,
        holes=holes,
        board=board,
        history=history,
        starting_stacks=start,
        payoffs=payoffs,
        ev_payoffs=ev_payoffs if ev_payoffs is not None else [float(p) for p in payoffs],
        allin_street=allin_street,
        showdown=showdown,
        folded=folded,
        shown=shown,
        decisions=decisions,
        corrections=corrections,
    )
