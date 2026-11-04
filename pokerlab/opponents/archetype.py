"""Adversaires paramétrables (VPIP, PFR, agressivité, c-bet, fold to c-bet, bluff, tailles)."""

from __future__ import annotations

from typing import Any

from pokerlab.cards.ranges import hole_class
from pokerlab.engine.types import Action, Kind, Observation
from pokerlab.features.hand import level_of, read_hand
from pokerlab.features.situation import (
    AGGRESSIVE,
    facing_all_in,
    facing_fraction,
    limpers,
    no_bet_yet,
    preflop_aggressor,
    preflop_spot,
    raise_to_bb,
    street_actions,
)
from pokerlab.strategies.base import StrategyContext, check_or_fold, decision_rng, make_raise, pot_bet

PARAM_DEFAULTS: dict[str, Any] = {
    "vpip": 0.25,
    "pfr": 0.15,
    "three_bet": 0.05,
    "four_bet": 0.025,
    "limp": False,
    "cold_call_share": 0.75,
    "call_cap_bb": 4.0,
    "bb_defend_cap_bb": 3.5,
    "fold_to_3bet": 0.5,
    "preflop_trap": 0.0,
    "open_size_bb": 3.0,
    "aggression": 0.6,
    "cbet": 0.6,
    "fold_to_cbet": 0.4,
    "bluff": 0.08,
    "semi_bluff": 0.4,
    "thin_value": 0.25,
    "call_weak": 0.5,
    "float": 0.08,
    "raise_value": 0.35,
    "check_raise": 0.3,
    "bluff_raise": 0.02,
    "slowplay": 0.0,
    "bet_size_pot": 0.6,
    "size_jitter": 0.15,
    "stackoff_min_made": "overpair",
    "big_bet_fraction": 0.9,
    "raise_multiplier": 3.0,
    "trap_top_pct": 0.012,
    "call_3bet_max_stack": 0.35,
    "five_bet_factor": 1.2,
    "thin_size_factor": 0.7,
    "call_size_ref": 0.75,
    "allin_call_factor": 0.3,
    "min_bet_pot": 0.2,
}


class Archetype:
    """Joueur adverse modélisé par quelques fréquences (voir ``configs/opponents/archetypes.yaml``)."""

    def __init__(self, config: dict[str, Any], context: StrategyContext) -> None:
        self.config = config
        self.name: str = config["name"]
        self.label: str = config.get("label", self.name)
        p = dict(PARAM_DEFAULTS)
        p.update({k: v for k, v in config.items() if k in PARAM_DEFAULTS})
        self.p = p
        self.ctx = context
        self.stackoff = level_of(p["stackoff_min_made"])
        self.shove_fraction = context.game.shove_commit_fraction

    def _size(self, obs: Observation, rng, base: float) -> float:
        jitter = float(self.p["size_jitter"])
        return max(float(self.p["min_bet_pot"]), base * (1.0 + jitter * (2 * rng.random() - 1)))

    def _preflop(self, obs: Observation) -> Action:
        p = self.p
        rng = decision_rng(obs)
        pct = float(self.ctx.table.class_percentile[hole_class(obs.hole)])
        spot = preflop_spot(obs)
        if spot == "unopened":
            if pct <= p["pfr"]:
                size = p["open_size_bb"] + limpers(obs.history)
                return make_raise(obs, obs.chips(size), self.shove_fraction, "open")
            if pct <= p["vpip"] and p["limp"]:
                return Action.check_call("limp")
            return check_or_fold(obs)
        current = raise_to_bb(obs)
        if spot == "vs_open":
            trap = pct <= p["trap_top_pct"] and rng.random() < p["preflop_trap"]
            if pct <= p["three_bet"] and not trap:
                return make_raise(obs, p["raise_multiplier"] * max(obs.bets), self.shove_fraction, "3bet")
            if trap:
                return Action.check_call("trap")
            if pct <= p["vpip"] * p["cold_call_share"] and current <= p["call_cap_bb"]:
                return Action.check_call("call")
            if obs.position == "BB" and pct <= p["vpip"] and current <= p["bb_defend_cap_bb"]:
                return Action.check_call("call")
            return check_or_fold(obs)
        if spot == "vs_3bet":
            if pct <= p["four_bet"]:
                return make_raise(obs, obs.legal.max_raise_to, self.shove_fraction, "4bet")
            if pct <= p["pfr"] * (1.0 - p["fold_to_3bet"]) and obs.to_call <= p["call_3bet_max_stack"] * obs.stack:
                return Action.check_call("call")
            return check_or_fold(obs)
        if pct <= p["four_bet"] * p["five_bet_factor"]:
            return make_raise(obs, obs.legal.max_raise_to, self.shove_fraction, "5bet")
        return check_or_fold(obs)

    def _postflop(self, obs: Observation) -> Action:
        p = self.p
        rng = decision_rng(obs)
        hr = read_hand(obs.hole, obs.board)
        level = hr.level
        strong = level >= level_of("two_pair")
        good = level >= level_of("top_pair")
        pair = level >= level_of("weak_pair")
        aggressor = preflop_aggressor(obs.history)
        if obs.to_call == 0:
            if strong and rng.random() < p["slowplay"]:
                return Action.check_call("trap")
            if good and rng.random() < p["aggression"]:
                return make_raise(obs, pot_bet(obs, self._size(obs, rng, p["bet_size_pot"])), self.shove_fraction, "value")
            if aggressor == obs.seat and obs.street == 1 and no_bet_yet(obs) and rng.random() < p["cbet"]:
                return make_raise(obs, pot_bet(obs, self._size(obs, rng, p["bet_size_pot"])), self.shove_fraction, "cbet")
            if hr.strong_draw and obs.street < 3 and rng.random() < p["semi_bluff"]:
                return make_raise(obs, pot_bet(obs, self._size(obs, rng, p["bet_size_pot"])), self.shove_fraction, "semi-bluff")
            if pair and rng.random() < p["thin_value"]:
                return make_raise(obs, pot_bet(obs, self._size(obs, rng, p["bet_size_pot"] * p["thin_size_factor"])), self.shove_fraction, "thin")
            if rng.random() < p["bluff"]:
                return make_raise(obs, pot_bet(obs, self._size(obs, rng, p["bet_size_pot"])), self.shove_fraction, "bluff")
            return Action.check_call("check")
        acts = street_actions(obs.history, obs.street)
        trapped = any(a.seat == obs.seat and a.kind == Kind.CHECK for a in acts)
        frac = facing_fraction(obs)
        all_in = facing_all_in(obs)
        big = all_in or frac > p["big_bet_fraction"]
        if strong:
            chance = p["check_raise"] if trapped else p["raise_value"]
            if obs.legal.can_raise and rng.random() < chance:
                return make_raise(obs, p["raise_multiplier"] * max(obs.bets), self.shove_fraction, "value-raise")
            return Action.check_call("call-value")
        if good:
            if big:
                return Action.check_call("call-value") if level >= self.stackoff else Action.fold("fold")
            return Action.check_call("call-value")
        bets = [a for a in acts if a.kind in AGGRESSIVE]
        is_cbet = obs.street == 1 and len(bets) == 1 and bets[0].seat == aggressor
        if is_cbet and rng.random() < p["fold_to_cbet"]:
            return Action.fold("fold-cbet")
        ref = float(p["call_size_ref"])
        size_factor = 1.0 if frac <= ref else ref / frac
        if all_in:
            size_factor *= float(p["allin_call_factor"])
        if (pair or hr.any_draw) and rng.random() < p["call_weak"] * size_factor:
            if pair and level >= self.stackoff or not big:
                return Action.check_call("call-weak")
        if not big and rng.random() < p["float"] * size_factor:
            return Action.check_call("float")
        if not big and obs.legal.can_raise and rng.random() < p["bluff_raise"]:
            return make_raise(obs, p["raise_multiplier"] * max(obs.bets), self.shove_fraction, "bluff-raise")
        return Action.fold("fold")

    def decide(self, observation: Observation) -> Action:
        if observation.street == 0:
            return self._preflop(observation)
        return self._postflop(observation)
