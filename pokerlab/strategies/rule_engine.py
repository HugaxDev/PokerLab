"""Moteur de stratégie à règles, entièrement piloté par une config YAML.

Préflop : 4 spots (``unopened``, ``vs_open``, ``vs_3bet``, ``vs_4bet``). Chaque spot a une
range de relance (taille en bb, multiplicateur ou tapis) et des paliers de paie
(range + relance maximale en bb), avec surcharges possibles par position.
Postflop : règles ordonnées (value bet, semi-bluff, c-bet, bluff ; face à une mise :
relance, garde-fou, paie, paie de tirage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pokerlab.cards.cards import RANK_INDEX
from pokerlab.cards.ranges import hole_class, range_classes
from pokerlab.config import deep_merge
from pokerlab.engine.types import Action, Kind, Observation
from pokerlab.features.board import board_texture
from pokerlab.features.hand import HandRead, draw_equity, level_of, read_hand
from pokerlab.features.situation import (
    AGGRESSIVE,
    bets_on_street,
    facing_all_in,
    facing_fraction,
    limpers,
    no_bet_yet,
    pot_odds,
    preflop_aggressor,
    preflop_spot,
    raise_to_bb,
)
from pokerlab.strategies.base import StrategyContext, check_or_fold, decision_rng, make_raise, pot_bet
from pokerlab.strategies.range_model import RangeModel

SPOTS = ("unopened", "vs_open", "vs_3bet", "vs_4bet")

POSTFLOP_DEFAULTS: dict[str, Any] = {
    "value_bet": {"enabled": True, "tiers": [{"min_made": "top_pair", "size_pot": 0.667}], "streets": [1, 2, 3]},
    "draw_bet": {"enabled": False, "draws": "strong", "size_pot": 0.33, "streets": [1, 2]},
    "cbet": {"enabled": False, "size_pot": 0.33, "max_opponents": 1, "streets": [1], "frequency": 1.0},
    "bluff": {
        "enabled": False,
        "size_pot": 0.5,
        "frequency": 1.0,
        "streets": [1],
        "require_preflop_raiser": True,
        "max_opponents": 1,
        "require_dry": True,
        "dry_board": {"connected_max_span": 4, "paired_wet_max_gap": 2},
    },
    "raise_vs_bet": {"enabled": True, "min_made": "two_pair", "multiplier": 3.0},
    "call_vs_bet": {"min_made": "top_pair"},
    "call_draws": {"enabled": False, "draws": "strong", "use_pot_odds": True, "implied_odds_factor": 0.0},
    "guardrail": {
        "enabled": False,
        "trigger_fraction": 0.5,
        "trigger_all_in": True,
        "trigger_scope": "bet_or_raise",
        "min_made": "two_pair",
        "allow_overpair_from": None,
        "seuil": None,          # 0.33 / 0.5 / 0.75 / 1.0 (fraction du pot) ou « tapis » ; remplace trigger_fraction
        "force_min": None,      # top_paire_bon_kicker / overpair / overpair_qq / deux_paires / brelan ; remplace min_made
        "bon_kicker": "J",      # top paire « bon kicker » : kicker valet ou mieux
        "equity_check": True,
        "equity_sims": 300,
    },
    "never_bet_without_pair": False,
}


# Garde-fou : les deux paramètres réglables (sweep, configs) et l'échelle de force pour payer.
GUARD_SEUILS = (0.33, 0.5, 0.75, 1.0, "tapis")
GUARD_FORCES = ("top_paire_bon_kicker", "overpair", "overpair_qq", "deux_paires", "brelan")
_FORCE_RANK = {name: i + 1 for i, name in enumerate(GUARD_FORCES)}


def guard_strength(hole: tuple[int, ...] | list[int], hr: HandRead, good_kicker: int) -> int:
    """Rang sur l'échelle du garde-fou : 0 sous top paire bon kicker, 1 top paire bon kicker, 2 overpair,
    3 overpair QQ+, 4 deux paires, 5 brelan ou mieux (quinte, couleur, full...)."""
    lv = hr.level
    if lv >= level_of("trips"):
        return 5
    if lv == level_of("two_pair"):
        return 4
    if lv == level_of("overpair"):
        return 3 if (hole[0] >> 2) >= RANK_INDEX["Q"] else 2
    if lv == level_of("top_pair"):
        return 1 if hr.kicker >= good_kicker else 0
    return 0


def guard_force_rank(name: str) -> int:
    key = str(name).lower()
    if key not in _FORCE_RANK:
        raise ValueError(f"force_min inconnue : {name!r} (attendu : {', '.join(GUARD_FORCES)})")
    return _FORCE_RANK[key]


class RangeSpec:
    """Range par position : chaîne unique ou dict ``{default: ..., BTN: ...}``."""

    def __init__(self, spec: str | dict[str, str] | None) -> None:
        if isinstance(spec, dict):
            self.default = range_classes(spec.get("default", "") or "")
            self.by_pos = {k: range_classes(v or "") for k, v in spec.items() if k != "default"}
        else:
            self.default = range_classes(spec or "")
            self.by_pos = {}

    def classes(self, position: str) -> frozenset[int]:
        return self.by_pos.get(position, self.default)


@dataclass
class PreflopNode:
    """Règles d'un spot préflop."""

    enabled: bool
    raise_range: RangeSpec
    raise_size_bb: float | None
    raise_multiplier: float | None
    raise_allin: bool
    per_limper_bb: float
    call: list[tuple[RangeSpec, float]]
    by_position: dict[str, "PreflopNode"] = field(default_factory=dict)

    @staticmethod
    def build(data: dict[str, Any], with_overrides: bool = True) -> "PreflopNode":
        node = PreflopNode(
            enabled=bool(data.get("enabled", True)),
            raise_range=RangeSpec(data.get("raise_range")),
            raise_size_bb=data.get("raise_size_bb"),
            raise_multiplier=data.get("raise_multiplier"),
            raise_allin=bool(data.get("raise_allin", False)),
            per_limper_bb=float(data.get("per_limper_bb", 1.0)),
            call=[(RangeSpec(t.get("range")), float(t.get("max_bb", 0.0))) for t in data.get("call", []) or []],
        )
        if with_overrides:
            base = {k: v for k, v in data.items() if k != "by_position"}
            for pos, override in (data.get("by_position") or {}).items():
                node.by_position[pos] = PreflopNode.build(deep_merge(base, override), with_overrides=False)
        return node

    def at(self, position: str) -> "PreflopNode":
        return self.by_position.get(position, self)


class RuleStrategy:
    """Stratégie à règles paramétrée par sa config YAML (voir ``configs/strategies``)."""

    def __init__(self, config: dict[str, Any], context: StrategyContext) -> None:
        self.config = config
        self.name: str = config["name"]
        self.label: str = config.get("label", self.name)
        self.ctx = context
        self.shove_fraction = float(config.get("shove_commit_fraction", context.game.shove_commit_fraction))
        pre = config.get("preflop", {})
        self.pre = {spot: PreflopNode.build(pre.get(spot, {})) for spot in SPOTS}
        post = deep_merge(POSTFLOP_DEFAULTS, config.get("postflop", {}))
        self.post = post
        self.value_tiers = sorted(
            ((level_of(t["min_made"]), float(t["size_pot"])) for t in post["value_bet"].get("tiers", [])),
            reverse=True,
        )
        self.raise_min = level_of(post["raise_vs_bet"]["min_made"])
        self.call_min = level_of(post["call_vs_bet"]["min_made"])
        g = post["guardrail"]
        self.guard_min = level_of(g["min_made"])
        self.guard_force = guard_force_rank(g["force_min"]) if g.get("force_min") else None
        self.guard_kicker = RANK_INDEX[str(g.get("bon_kicker", "J")).upper()[0]]
        seuil = g.get("seuil")
        if seuil is None:
            self.guard_fraction = float(g["trigger_fraction"])
            self.guard_all_in = bool(g["trigger_all_in"])
        elif str(seuil).lower() == "tapis":
            self.guard_fraction = float("inf")
            self.guard_all_in = True
        else:
            self.guard_fraction = float(seuil)
            self.guard_all_in = True
        self.range_model = RangeModel(context.table, config.get("range_model"))

    # ------------------------------------------------------------------ préflop
    def _preflop(self, obs: Observation) -> Action:
        spot = preflop_spot(obs)
        node = self.pre[spot].at(obs.position)
        if not node.enabled:
            return check_or_fold(obs, "check" if obs.to_call == 0 else "fold")
        cls = hole_class(obs.hole)
        pos = obs.position
        if cls in node.raise_range.classes(pos) and obs.legal.can_raise:
            tag = {"unopened": "open", "vs_open": "3bet", "vs_3bet": "4bet", "vs_4bet": "5bet"}[spot]
            if node.raise_allin:
                target = obs.legal.max_raise_to
            elif node.raise_size_bb is not None:
                target = obs.chips(float(node.raise_size_bb) + node.per_limper_bb * limpers(obs.history))
            else:
                target = float(node.raise_multiplier or 3.0) * max(obs.bets)
            return make_raise(obs, target, self.shove_fraction, tag)
        if cls in node.raise_range.classes(pos):
            return Action.check_call("call")
        if obs.to_call > 0:
            current = raise_to_bb(obs)
            for spec, max_bb in node.call:
                if cls in spec.classes(pos) and current <= max_bb + 1e-9:
                    return Action.check_call("call" if spot != "unopened" else "limp")
        return check_or_fold(obs, "check" if obs.to_call == 0 else "fold")

    # ----------------------------------------------------------------- postflop
    def _hero_bet_postflop_before(self, obs: Observation) -> bool:
        return any(a.seat == obs.seat and a.street >= 1 and a.kind in AGGRESSIVE for a in obs.history)

    def _bet(self, obs: Observation, fraction: float, tag: str) -> Action:
        return make_raise(obs, pot_bet(obs, fraction), self.shove_fraction, tag)

    def _no_bet(self, obs: Observation, hr: HandRead) -> Action:
        p = self.post
        street = obs.street
        if p["value_bet"]["enabled"] and street in p["value_bet"]["streets"]:
            for min_level, size in self.value_tiers:
                if hr.level >= min_level:
                    return self._bet(obs, size, "value")
        db = p["draw_bet"]
        if db["enabled"] and street in db["streets"] and self._draw_ok(hr, db["draws"]):
            return self._bet(obs, float(db["size_pot"]), "semi-bluff")
        aggressor = preflop_aggressor(obs.history)
        cb = p["cbet"]
        if (
            cb["enabled"]
            and street in cb["streets"]
            and aggressor == obs.seat
            and obs.n_active_opponents <= int(cb["max_opponents"])
            and no_bet_yet(obs)
            and not self._hero_bet_postflop_before(obs)
            and decision_rng(obs, 0xCB).random() < float(cb["frequency"])
        ):
            return self._bet(obs, float(cb["size_pot"]), "cbet")
        bl = p["bluff"]
        if bl["enabled"] and street in bl["streets"] and not self._hero_bet_postflop_before(obs):
            ok = obs.n_active_opponents <= int(bl["max_opponents"])
            if bl["require_preflop_raiser"]:
                ok = ok and aggressor == obs.seat
            if ok and bl["require_dry"]:
                dry_cfg = bl.get("dry_board", {})
                ok = board_texture(
                    obs.board,
                    int(dry_cfg.get("connected_max_span", 4)),
                    int(dry_cfg.get("paired_wet_max_gap", 2)),
                ).dry
            if ok and decision_rng(obs, 0xB1).random() < float(bl["frequency"]):
                return self._bet(obs, float(bl["size_pot"]), "bluff")
        return Action.check_call("check")

    @staticmethod
    def _draw_ok(hr: HandRead, kind: str) -> bool:
        return hr.strong_draw if kind == "strong" else hr.any_draw

    def _equity(self, obs: Observation, sims: int) -> float:
        sigs, ranges = [], []
        for v in obs.active_opponents:
            sig, w = self.range_model.villain_range(obs, v)
            sigs.append(sig)
            ranges.append(w)
        return self.ctx.estimate_equity(obs.hole, obs.board, ranges, (self.name, tuple(sigs)), sims)

    def _guardrail_triggered(self, obs: Observation) -> bool:
        """Garde-fou actif : tapis adverse, ou mise/relance au-delà de ``trigger_fraction`` du pot."""
        g = self.post["guardrail"]
        if not g["enabled"] or obs.to_call <= 0:
            return False
        if self.guard_all_in and facing_all_in(obs):
            return True
        if facing_fraction(obs) <= self.guard_fraction:
            return False
        if g.get("trigger_scope", "bet_or_raise") == "raise_only":
            return bets_on_street(obs) >= 2
        return True

    def _strong_overpair(self, obs: Observation, hr: HandRead) -> bool:
        """Overpair servie au moins égale à ``allow_overpair_from`` (ex. Q -> QQ+)."""
        rank = self.post["guardrail"].get("allow_overpair_from")
        if not rank or hr.level != level_of("overpair"):
            return False
        return obs.hole[0] >> 2 == obs.hole[1] >> 2 and (obs.hole[0] >> 2) >= RANK_INDEX[str(rank).upper()[0]]

    def _guard_pass(self, obs: Observation, hr: HandRead) -> bool:
        """Main assez forte pour payer malgré le garde-fou."""
        if self.guard_force is not None:
            return guard_strength(obs.hole, hr, self.guard_kicker) >= self.guard_force
        return hr.level >= self.guard_min or self._strong_overpair(obs, hr)

    def _facing(self, obs: Observation, hr: HandRead, allow_raise: bool = True) -> Action:
        p = self.post
        g = p["guardrail"]
        rv = p["raise_vs_bet"]
        big = self._guardrail_triggered(obs)
        if allow_raise and rv["enabled"] and hr.level >= self.raise_min and obs.legal.can_raise:
            return make_raise(obs, float(rv["multiplier"]) * max(obs.bets), self.shove_fraction, "value-raise")
        if big:
            if self._guard_pass(obs, hr):
                return Action.check_call("call-value")
            if g["equity_check"]:
                eq = self._equity(obs, int(g["equity_sims"]))
                if eq >= pot_odds(obs):
                    return Action.check_call("call-equity")
            return Action.fold("fold")
        if hr.level >= self.call_min:
            return Action.check_call("call-value")
        cd = p["call_draws"]
        if cd["enabled"] and obs.street < 3 and self._draw_ok(hr, cd["draws"]):
            if not cd["use_pot_odds"]:
                return Action.check_call("call-draw")
            all_in_call = obs.to_call >= obs.stack
            eq = draw_equity(hr.outs, len(obs.board), two_cards=all_in_call)
            implied = 0.0 if all_in_call else float(cd["implied_odds_factor"]) * min(obs.stack - obs.to_call, obs.pot)
            if eq >= obs.to_call / (obs.pot + obs.to_call + implied):
                return Action.check_call("call-draw")
        return Action.fold("fold")

    def _postflop(self, obs: Observation) -> Action:
        hr = read_hand(obs.hole, obs.board)
        action = self._no_bet(obs, hr) if obs.to_call == 0 else self._facing(obs, hr)
        if self.post["never_bet_without_pair"] and action.kind == "raise" and hr.level < level_of("weak_pair"):
            return Action.check_call("check") if obs.to_call == 0 else self._facing(obs, hr, allow_raise=False)
        return action

    def decide(self, observation: Observation) -> Action:
        if observation.street == 0:
            return self._preflop(observation)
        return self._postflop(observation)


__all__ = ["RuleStrategy", "PreflopNode", "RangeSpec", "POSTFLOP_DEFAULTS", "Kind"]
