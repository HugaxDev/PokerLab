"""Robot de référence : décisions à l'espérance (équité contre range estimée, cote du pot)."""

from __future__ import annotations

from typing import Any

from pokerlab.cards.ranges import class_weights, hole_class
from pokerlab.config import deep_merge
from pokerlab.engine.types import Action, Observation
from pokerlab.features.board import board_texture
from pokerlab.features.hand import read_hand
from pokerlab.features.situation import (
    in_position,
    limpers,
    no_bet_yet,
    pot_odds,
    preflop_aggressor,
    preflop_raises,
    preflop_spot,
)
from pokerlab.strategies.base import StrategyContext, check_or_fold, decision_rng, make_raise, pot_bet
from pokerlab.strategies.range_model import RangeModel

DEFAULTS: dict[str, Any] = {
    "preflop": {
        "open_pct": {"UTG": 0.15, "MP": 0.19, "CO": 0.27, "BTN": 0.45, "SB": 0.40, "BB": 0.25},
        "open_size_bb": 2.5,
        "per_limper_bb": 1.0,
        "iso_pct": 0.15,
        "three_bet_equity": 0.60,
        "three_bet_multiplier": 3.0,
        "four_bet_equity": 0.62,
        "realization_ip": 0.95,
        "realization_oop": 0.85,
        "multiway_discount": 0.85,
    },
    "postflop": {
        "sims": 300,
        "value_margin": 0.25,
        "raise_margin": 0.45,
        "bet_size_pot": 0.66,
        "raise_multiplier": 3.0,
        "realization": 0.9,
        "semi_bluff": {"enabled": True, "min_equity": 0.32, "max_opponents": 1, "size_pot": 0.5},
        "cbet": {"enabled": True, "max_opponents": 1, "size_pot": 0.5, "frequency": 0.6, "dry_only": False},
    },
}


class EquityBot:
    """Joue à l'EV : compare l'équité estimée à la cote du pot et à des seuils de value."""

    def __init__(self, config: dict[str, Any], context: StrategyContext) -> None:
        self.config = deep_merge(DEFAULTS, config)
        self.name: str = config["name"]
        self.label: str = config.get("label", self.name)
        self.ctx = context
        self.pre = self.config["preflop"]
        self.post = self.config["postflop"]
        self.shove_fraction = float(self.config.get("shove_commit_fraction", context.game.shove_commit_fraction))
        self.range_model = RangeModel(context.table, self.config.get("range_model"))

    # ------------------------------------------------------------------ préflop
    def _preflop(self, obs: Observation) -> Action:
        table = self.ctx.table
        cls = hole_class(obs.hole)
        pct = float(table.class_percentile[cls])
        spot = preflop_spot(obs)
        if spot == "unopened":
            n_limp = limpers(obs.history)
            threshold = self.pre["iso_pct"] if n_limp else self.pre["open_pct"].get(obs.position, 0.2)
            if pct <= threshold:
                size = self.pre["open_size_bb"] + self.pre["per_limper_bb"] * n_limp
                return make_raise(obs, obs.chips(size), self.shove_fraction, "open")
            return check_or_fold(obs, "check" if obs.to_call == 0 else "fold")
        raiser = preflop_raises(obs.history)[-1].seat
        _, w = self.range_model.villain_range(obs, raiser)
        eq = table.equity_vs_class_weights(cls, class_weights(w))
        extra = max(0, obs.n_active_opponents - 1)
        eq *= float(self.pre["multiway_discount"]) ** extra
        all_in_call = obs.to_call >= obs.stack
        real = 1.0 if all_in_call else float(self.pre["realization_ip" if in_position(obs) else "realization_oop"])
        if spot == "vs_open" and eq >= self.pre["three_bet_equity"] and obs.legal.can_raise:
            return make_raise(obs, self.pre["three_bet_multiplier"] * max(obs.bets), self.shove_fraction, "3bet")
        if spot != "vs_open" and eq >= self.pre["four_bet_equity"] and obs.legal.can_raise:
            return make_raise(obs, obs.legal.max_raise_to, self.shove_fraction, "4bet")
        if eq * real >= pot_odds(obs):
            return Action.check_call("call")
        return check_or_fold(obs, "fold")

    # ----------------------------------------------------------------- postflop
    def _equity(self, obs: Observation) -> float:
        sigs, ranges = [], []
        for v in obs.active_opponents:
            sig, w = self.range_model.villain_range(obs, v)
            sigs.append(sig)
            ranges.append(w)
        return self.ctx.estimate_equity(obs.hole, obs.board, ranges, (self.name, tuple(sigs)), int(self.post["sims"]))

    def _postflop(self, obs: Observation) -> Action:
        eq = self._equity(obs)
        n = max(1, obs.n_active_opponents)
        fair = 1.0 / (n + 1)
        value_thr = fair + (1 - fair) * float(self.post["value_margin"])
        raise_thr = fair + (1 - fair) * float(self.post["raise_margin"])
        if obs.to_call == 0:
            if eq >= value_thr:
                return make_raise(obs, pot_bet(obs, float(self.post["bet_size_pot"])), self.shove_fraction, "value")
            sb = self.post["semi_bluff"]
            if (
                sb["enabled"]
                and obs.street < 3
                and n <= int(sb["max_opponents"])
                and eq >= float(sb["min_equity"])
                and read_hand(obs.hole, obs.board).strong_draw
            ):
                return make_raise(obs, pot_bet(obs, float(sb["size_pot"])), self.shove_fraction, "semi-bluff")
            cb = self.post["cbet"]
            if (
                cb["enabled"]
                and obs.street == 1
                and preflop_aggressor(obs.history) == obs.seat
                and n <= int(cb["max_opponents"])
                and no_bet_yet(obs)
                and (not cb["dry_only"] or board_texture(obs.board).dry)
                and decision_rng(obs, 0xCB).random() < float(cb["frequency"])
            ):
                return make_raise(obs, pot_bet(obs, float(cb["size_pot"])), self.shove_fraction, "cbet")
            return Action.check_call("check")
        if eq >= raise_thr and obs.legal.can_raise:
            return make_raise(obs, float(self.post["raise_multiplier"]) * max(obs.bets), self.shove_fraction, "value-raise")
        all_in_call = obs.to_call >= obs.stack or obs.street == 3
        real = 1.0 if all_in_call else float(self.post["realization"])
        if eq * real >= pot_odds(obs):
            return Action.check_call("call-equity")
        return Action.fold("fold")

    def decide(self, observation: Observation) -> Action:
        if observation.street == 0:
            return self._preflop(observation)
        return self._postflop(observation)
