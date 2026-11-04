"""Garde-fou : échelle de force, seuils, et équivalence exacte avec l'ancien codage des configs."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from pokerlab.cards.cards import parse_cards
from pokerlab.config import LabConfig
from pokerlab.experiments.common import DuplicateTask, TaskResult, run_duplicate_task
from pokerlab.features.hand import read_hand
from pokerlab.strategies.registry import resolve_spec
from pokerlab.strategies.rule_engine import GUARD_FORCES, guard_force_rank, guard_strength

J = 9  # rang du valet (2 = 0)


@pytest.mark.parametrize("hole, board, rank", [
    ("AhJd", "As7c2d", 1),   # top paire, kicker valet : bon kicker
    ("AhTd", "As7c2d", 0),   # top paire, kicker 10 : pas assez
    ("TT", "9s7c2d", 2),     # overpair TT
    ("QQ", "9s7c2d", 3),     # overpair QQ+
    ("Ah7d", "As7c2d", 4),   # deux paires
    ("7h7d", "As7c2d", 5),   # brelan
    ("6h5d", "4s3c2d", 5),   # quinte : au-dessus du brelan
    ("Kh9d", "As7c2d", 0),   # rien
])
def test_guard_strength_ladder(hole: str, board: str, rank: int) -> None:
    if len(hole) == 2:
        hole = f"{hole[0]}h{hole[1]}d"
    h = parse_cards(hole)
    b = parse_cards(board)
    assert guard_strength(h, read_hand(h, b), J) == rank


def test_force_names() -> None:
    assert [guard_force_rank(f) for f in GUARD_FORCES] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        guard_force_rank("paire")


def _legacy(spec: dict, force: str) -> dict:
    """Même stratégie écrite avec l'ancien codage (trigger_fraction, min_made, allow_overpair_from)."""
    old = copy.deepcopy(spec)
    g = old["postflop"]["guardrail"]
    g.pop("seuil")
    g.pop("force_min")
    g["trigger_fraction"] = 0.5
    g["trigger_all_in"] = True
    g["min_made"] = "two_pair"
    if force == "overpair_qq":
        g["allow_overpair_from"] = "Q"
    old["name"] = spec["name"]
    return old


def _hands(lab: LabConfig, spec: dict, decks: int) -> pd.DataFrame:
    comp = {"name": "garde", "mode": "sample", "weights": {"maniac": 1}}   # beaucoup de grosses mises : le garde-fou se déclenche souvent
    out = run_duplicate_task(DuplicateTask(lab.raw, spec, comp, 0, decks, {"strategy": spec["name"]}, record_decisions=False))
    merged = TaskResult()
    merged.merge(out)
    return merged.hands.frame()[["deck", "rot", "result", "result_ev"]].reset_index(drop=True)


@pytest.mark.parametrize("name, force", [("serre", "overpair_qq"), ("joueur", "deux_paires")])
def test_new_keys_equal_legacy_encoding(name: str, force: str) -> None:
    """seuil 0.5 + force_min = ancien trigger_fraction 0.5 + min_made (+ overpair QQ+) : mêmes mains, au jeton près."""
    from pokerlab.strategies.rule_engine import RuleStrategy

    lab = LabConfig.load()
    spec = resolve_spec(name)
    assert spec["postflop"]["guardrail"]["force_min"] == force
    calls = []
    original = RuleStrategy._guard_pass

    def counted(self, obs, hr):
        calls.append(1)
        return original(self, obs, hr)

    RuleStrategy._guard_pass = counted
    try:
        new = _hands(lab, spec, 20)
    finally:
        RuleStrategy._guard_pass = original
    assert len(calls) >= 2  # le garde-fou s'est bien déclenché sur ces donnes
    pd.testing.assert_frame_equal(new, _hands(lab, _legacy(spec, force), 20))
