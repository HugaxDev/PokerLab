"""Chaque stratégie respecte ses règles : situations construites + propriétés sur mains jouées."""

from __future__ import annotations

import pytest
from conftest import BLINDS, make_obs, rec

from pokerlab.cards.cards import parse_cards
from pokerlab.engine.hand_runner import play_hand
from pokerlab.engine.types import Action, Kind, Observation
from pokerlab.experiments.common import seating
from pokerlab.features.board import board_texture
from pokerlab.features.hand import LEVEL, read_hand
from pokerlab.features.situation import preflop_aggressor
from pokerlab.opponents.compositions import Composition
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy

H = lambda s: tuple(parse_cards(s))  # noqa: E731

# Préflop : UTG, MP, CO se couchent, BTN (5) relance à 3 bb, SB se couche, BB (1) paie.
HU_PRE = BLINDS + (rec(0, 2, Kind.FOLD), rec(0, 3, Kind.FOLD), rec(0, 4, Kind.FOLD), rec(0, 5, Kind.RAISE, 300, 300),
                   rec(0, 0, Kind.FOLD), rec(0, 1, Kind.CALL, 200, 300))
HU_FOLDED = (True, False, True, True, True, False)
MW_PRE = BLINDS + (rec(0, 2, Kind.FOLD), rec(0, 3, Kind.FOLD), rec(0, 4, Kind.FOLD), rec(0, 5, Kind.RAISE, 300, 300),
                   rec(0, 0, Kind.CALL, 250, 300), rec(0, 1, Kind.CALL, 200, 300))
MW_FOLDED = (False, False, True, True, True, False)
STACKS = (9700, 9700, 10000, 10000, 10000, 9700)


def flop_obs(hole: str, board: str, *, multiway: bool = False, facing: int = 0, seat: int = 5) -> Observation:
    history = MW_PRE if multiway else HU_PRE
    folded = MW_FOLDED if multiway else HU_FOLDED
    pot = 900 if multiway else 650
    bets = [0] * 6
    if facing:
        history = history + (rec(1, 1, Kind.BET, facing, facing),)
        bets[1] = facing
        pot += facing
    stacks = list(STACKS)
    if facing:
        stacks[1] -= facing
    return make_obs(H(hole), tuple(parse_cards(board)), seat=seat, history=history, to_call=facing, pot=pot,
                    bets=tuple(bets), folded=folded, stacks=tuple(stacks))


def pre_obs(hole: str, seat: int, history=BLINDS, to_call: int = 0, bets=(50, 100, 0, 0, 0, 0)) -> Observation:
    pot = sum(a.amount for a in history)
    stacks = tuple(10000 - b for b in bets)
    return make_obs(H(hole), (), seat=seat, history=history, to_call=to_call, pot=pot, bets=bets, stacks=stacks)


# ---------------------------------------------------------------------------- serré
def test_serre_preflop(lab) -> None:
    s = build_strategy("serre", lab)
    assert s.decide(pre_obs("7s6s", 2, to_call=100)).kind == "fold"
    a = s.decide(pre_obs("AsJs", 2, to_call=100))
    assert a.kind == "raise" and a.amount == 300
    assert s.decide(pre_obs("AsJd", 2, to_call=100)).kind == "fold"  # AJo hors range
    opened = BLINDS + (rec(0, 2, Kind.RAISE, 300, 300),)
    bets = (50, 100, 300, 0, 0, 0)
    assert s.decide(pre_obs("JsJd", 5, opened, 300, bets)).kind == "call"
    three = s.decide(pre_obs("QsQd", 5, opened, 300, bets))
    assert three.kind == "raise" and three.amount == 900
    big = BLINDS + (rec(0, 2, Kind.RAISE, 500, 500),)
    assert s.decide(pre_obs("JsJd", 5, big, 500, (50, 100, 500, 0, 0, 0))).kind == "fold"


def test_serre_postflop(lab) -> None:
    s = build_strategy("serre", lab)
    bet = s.decide(flop_obs("AsKd", "Kc7h2s"))
    assert bet.kind == "raise" and bet.amount == pytest.approx(round(0.667 * 650), abs=1)
    assert s.decide(flop_obs("9s9d", "Kc7h2s", facing=300)).kind == "fold"
    assert s.decide(flop_obs("AsKd", "Kc7h2s", facing=300)).kind == "call"   # mise <= 50 % du pot : paie avec top paire
    assert s.decide(flop_obs("AhTh", "Kh7h2s", facing=300)).kind == "fold"  # pas de paie de tirage
    assert s.decide(flop_obs("AsQd", "Kc7h2s")).kind == "call"  # check


SERRE_FAMILY = ("serre", "large", "zero_bluff", "position", "bluff")


@pytest.mark.parametrize("name", SERRE_FAMILY)
def test_guardrail_situations(lab, name: str) -> None:
    """Garde-fou uniforme : face à un tapis ou une mise/relance > 50 % du pot, deux paires+ ou overpair QQ+."""
    s = build_strategy(name, lab)
    assert s.decide(flop_obs("AsKd", "Kc7h2s", facing=300)).kind == "call"    # 46 % du pot : top paire suffit
    assert s.decide(flop_obs("AsKd", "Kc7h2s", facing=400)).kind == "fold"    # top paire, mise de 62 % du pot
    assert s.decide(flop_obs("AsKd", "Kc7h2s", facing=9700)).kind == "fold"   # top paire face à un tapis
    assert s.decide(flop_obs("QsQd", "Jc7h2s", facing=9700)).kind == "call"   # overpair QQ
    assert s.decide(flop_obs("AsAd", "Kc7h2s", facing=400)).kind == "call"    # overpair AA
    assert s.decide(flop_obs("JsJd", "Tc7h2s", facing=9700)).kind == "fold"   # overpair JJ : insuffisant
    assert s.decide(flop_obs("QsQd", "Kc7h2s", facing=9700)).kind == "fold"   # QQ sous un roi : pas une overpair
    assert s.decide(flop_obs("Ks7d", "Kc7h2s", facing=9700)).kind in ("call", "raise")  # deux paires


def test_guardrail_scope_unchanged(lab) -> None:
    """joueur garde son garde-fou (avec exception d'équité) ; equity_bot et random_legal n'en ont pas."""
    from pokerlab.config import load_strategy_config

    ref = load_strategy_config("serre")["postflop"]["guardrail"]
    for name in SERRE_FAMILY:
        assert load_strategy_config(name)["postflop"]["guardrail"] == ref, name
    assert ref["trigger_scope"] == "bet_or_raise"
    joueur = load_strategy_config("joueur")["postflop"]["guardrail"]
    assert joueur["equity_check"] is True and "allow_overpair_from" not in joueur
    for name in ("equity_bot", "random_legal"):
        assert "guardrail" not in (load_strategy_config(name).get("postflop") or {})


@pytest.mark.parametrize("name", ("serre", "position"))
def test_guardrail_can_be_ablated(lab, name: str) -> None:
    """L'ablation « garde-fou » (postflop.guardrail.enabled: false) rend l'ancien comportement."""
    from pokerlab.experiments.variants import variant
    from pokerlab.strategies.registry import resolve_spec

    ablated = variant(resolve_spec(name), {"postflop.guardrail.enabled": False}, "sans garde-fou")
    s = build_strategy(ablated, lab)
    assert s.decide(flop_obs("AsKd", "Kc7h2s", facing=9700)).kind == "call"   # top paire paie le tapis sans garde-fou
    assert build_strategy(name, lab).decide(flop_obs("AsKd", "Kc7h2s", facing=9700)).kind == "fold"


@pytest.mark.parametrize("name", SERRE_FAMILY)
def test_guardrail_in_play(lab, name: str) -> None:
    """Propriété : sur des mains jouées, aucune stratégie de la famille serré ne continue face à un tapis
    ou une grosse mise sans deux paires ou une overpair QQ+."""
    from pokerlab.features.situation import facing_all_in, facing_fraction

    strat = build_strategy(name, lab)
    seen: list[tuple[Observation, Action]] = []

    class Spy:
        def __init__(self) -> None:
            self.name = strat.name

        def decide(self, obs: Observation) -> Action:
            act = strat.decide(obs)
            seen.append((obs, act))
            return act

    comp = Composition.from_dict({"name": "t", "mode": "sample", "weights": {"maniac": 2, "LAG": 1, "calling_station": 1}})
    spy = Spy()
    for d in range(3):
        opponents = [build_strategy(x, lab) for x in comp.opponents(lab.seed, d, 5)]
        seed = derive_seed(777, d)
        for r in range(6):
            play_hand(seating(spy, opponents, r, 6), lab.game, shuffled_deck(seed), [10000] * 6, d * 6 + r, seed)
    for obs, act in seen:
        if obs.street == 0 or obs.to_call == 0 or act.kind == "fold":
            continue
        if facing_all_in(obs) or facing_fraction(obs) > 0.5:
            hr = read_hand(obs.hole, obs.board)
            pocket = obs.hole[0] >> 2 == obs.hole[1] >> 2
            assert hr.level >= LEVEL["two_pair"] or (hr.level == LEVEL["overpair"] and pocket and obs.hole[0] >> 2 >= 10), obs


# -------------------------------------------------------------------------- position
def test_position_rules(lab) -> None:
    s = build_strategy("position", lab)
    assert s.decide(pre_obs("5s4s", 5, to_call=100)).kind == "raise"  # BTN : range large
    assert s.decide(pre_obs("5s4s", 2, to_call=100)).kind == "fold"   # UTG : range serrée
    limped = BLINDS + (rec(0, 2, Kind.CALL, 100, 100), rec(0, 3, Kind.FOLD), rec(0, 4, Kind.FOLD), rec(0, 5, Kind.FOLD), rec(0, 0, Kind.FOLD))
    assert s.decide(pre_obs("7s2d", 1, limped, 0, (0, 100, 100, 0, 0, 0))).kind == "call"  # check gratuit en BB
    opened = BLINDS + (rec(0, 5, Kind.RAISE, 300, 300), rec(0, 0, Kind.FOLD))
    assert s.decide(pre_obs("5s4s", 1, opened, 200, (0, 100, 0, 0, 0, 300))).kind == "fold"
    assert s.decide(pre_obs("AsJd", 1, opened, 200, (0, 100, 0, 0, 0, 300))).kind == "call"


# ----------------------------------------------------------------------------- bluff
def test_bluff_conditions(lab) -> None:
    s = build_strategy("bluff", lab)
    a = s.decide(flop_obs("6s5d", "Kd9c2h"))
    assert a.kind == "raise" and a.tag == "bluff" and a.amount == pytest.approx(325, abs=1)
    assert s.decide(flop_obs("6s5d", "Kd9c2h", multiway=True)).kind == "call"   # multiway : pas de bluff
    assert s.decide(flop_obs("6s5d", "Kd9d2h")).kind == "call"                  # deux couleurs : pas sec
    assert s.decide(flop_obs("AsQd", "9d8c7h")).kind == "call"                  # connecté
    not_pfr = flop_obs("6s5d", "Kd9c2h", seat=1)
    assert s.decide(not_pfr).kind == "call"                                     # pas relanceur préflop


def test_bluff_never_multiway_in_play(lab) -> None:
    """Propriété : toute action étiquetée bluff respecte les 4 conditions."""
    bluff = build_strategy("bluff", lab)
    seen: list[tuple[Observation, Action]] = []

    class Spy:
        name = bluff.name

        def decide(self, obs: Observation) -> Action:
            act = bluff.decide(obs)
            seen.append((obs, act))
            return act

    comp = Composition.from_dict({"name": "t", "mode": "sample", "weights": {"calling_station": 2, "nit": 1, "TAG": 1, "LAG": 1}})
    spy = Spy()
    for d in range(6):
        opponents = [build_strategy(x, lab) for x in comp.opponents(lab.seed, d, 5)]
        seed = derive_seed(123, d)
        for r in range(6):
            play_hand(seating(spy, opponents, r, 6), lab.game, shuffled_deck(seed), [10000] * 6, d * 6 + r, seed)
    bluffs = [(o, a) for o, a in seen if a.tag == "bluff"]
    for obs, _ in bluffs:
        assert obs.n_active_opponents == 1 and obs.street == 1
        assert preflop_aggressor(obs.history) == obs.seat
        assert board_texture(obs.board).dry
    per_hand: dict[int, int] = {}
    for obs, _ in bluffs:
        per_hand[obs.hand_id] = per_hand.get(obs.hand_id, 0) + 1
    assert all(v == 1 for v in per_hand.values())


# ------------------------------------------------------------------------ zéro bluff
def test_zero_bluff_never_bets_without_pair(lab) -> None:
    zb = build_strategy("zero_bluff", lab)
    assert zb.decide(flop_obs("AhTh", "Kh7h2s")).kind == "call"            # tirage seul : check
    assert zb.decide(flop_obs("AhTh", "Kh7h2s", facing=200)).kind == "call" # paie le tirage (bonne cote)
    assert zb.decide(flop_obs("AhTh", "Kh7h2s", facing=2000)).kind == "fold"  # cote insuffisante
    weak = zb.decide(flop_obs("5s5d", "Kc7h2s"))
    assert weak.kind == "raise"                                             # mise avec toute paire
    seen: list[tuple[Observation, Action]] = []

    class Spy:
        name = zb.name

        def decide(self, obs: Observation) -> Action:
            act = zb.decide(obs)
            seen.append((obs, act))
            return act

    comp = Composition.from_dict({"name": "t", "mode": "sample", "weights": {"maniac": 1, "calling_station": 1, "LAG": 1}})
    spy = Spy()
    for d in range(8):
        opponents = [build_strategy(x, lab) for x in comp.opponents(lab.seed, d, 5)]
        seed = derive_seed(321, d)
        for r in range(6):
            play_hand(seating(spy, opponents, r, 6), lab.game, shuffled_deck(seed), [10000] * 6, d * 6 + r, seed)
    post_raises = [(o, a) for o, a in seen if o.street > 0 and a.kind == "raise"]
    assert post_raises, "aucune mise postflop observée"
    for obs, _ in post_raises:
        assert read_hand(obs.hole, obs.board).level >= LEVEL["weak_pair"]


# ----------------------------------------------------------------------------- large
def test_large_calls_draws_on_odds(lab) -> None:
    lg = build_strategy("large", lab)
    assert lg.decide(flop_obs("AhTh", "Kh7h2s", facing=200)).kind == "call"
    assert lg.decide(flop_obs("AhTh", "Kh7h2s", facing=2000)).kind == "fold"
    assert lg.decide(flop_obs("As4d", "Kc7h2s", facing=200)).kind == "fold"


# ---------------------------------------------------------------------------- joueur
def test_joueur_rules(lab) -> None:
    j = build_strategy("joueur", lab)
    cb = j.decide(flop_obs("6s5d", "Kd9c2h"))
    assert cb.kind == "raise" and cb.tag == "cbet" and cb.amount == pytest.approx(round(0.33 * 650), abs=1)
    small = j.decide(flop_obs("5s5d", "Kc7h2s", facing=200))
    assert small.kind == "call"                                     # petite mise : paie avec une paire
    shove = flop_obs("9s9d", "AcKh2s", facing=9700)
    assert j.decide(shove).kind == "fold"                           # garde-fou : tapis, paire faible, mauvaise équité
    assert j.decide(flop_obs("Ks2d", "Kc7h2s", facing=9700)).kind in ("call", "raise")  # deux paires : suit
    opened = BLINDS + (rec(0, 2, Kind.RAISE, 500, 500),)
    assert j.decide(pre_obs("KsTd", 5, opened, 500, (50, 100, 500, 0, 0, 0))).kind == "call"   # <= 5 bb avec deux cartes >= 10
    assert j.decide(pre_obs("Ks4d", 5, opened, 500, (50, 100, 500, 0, 0, 0))).kind == "fold"   # roi quelconque : <= 4 bb seulement


def test_random_legal_and_equity_bot_return_actions(lab) -> None:
    for name in ("random_legal", "equity_bot"):
        s = build_strategy(name, lab)
        a = s.decide(flop_obs("AsKd", "Kc7h2s", facing=400))
        assert a.kind in ("fold", "call", "raise")
