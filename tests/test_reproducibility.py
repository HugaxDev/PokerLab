"""Reproductibilité : même graine, même résultat, avec 1 worker ou N workers. Duplicate cohérent."""

from __future__ import annotations

import pandas as pd

from pokerlab.config import LabConfig
from pokerlab.experiments.common import DuplicateTask, TaskResult, run_duplicate_task, run_tasks, split_range
from pokerlab.strategies.registry import resolve_spec

COMP = {"name": "test_mix", "mode": "sample", "weights": {"calling_station": 1, "maniac": 1, "nit_trappeur": 1, "TAG": 1, "equity_bot": 1}}


def _run(lab: LabConfig, hero: str, decks: int, workers: int, comp=COMP) -> pd.DataFrame:
    tasks = [DuplicateTask(lab.raw, resolve_spec(hero), comp, a, b, {"strategy": hero}) for a, b in split_range(0, decks, max(1, workers))]
    outs = run_tasks(run_duplicate_task, tasks, workers, lab.mp_start_method)
    merged = TaskResult()
    for o in outs:
        merged.merge(o)
    return merged.hands.frame()


def test_same_seed_same_result(lab) -> None:
    a = _run(lab, "joueur", 3, 1)
    b = _run(lab, "joueur", 3, 1)
    pd.testing.assert_frame_equal(a, b)


def test_workers_do_not_change_results(lab) -> None:
    a = _run(lab, "equity_bot", 4, 1)
    b = _run(lab, "equity_bot", 4, 3)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_cards(lab) -> None:
    other = LabConfig.load(overrides={"seed": lab.seed + 1, "workers": 1})
    a = _run(lab, "serre", 1, 1)
    b = _run(other, "serre", 1, 1)
    assert not (a[["c1", "c2"]].to_numpy() == b[["c1", "c2"]].to_numpy()).all()


def test_duplicate_sum_is_zero_when_everyone_plays_the_same(lab) -> None:
    """Même stratégie à tous les sièges : rejouer le deck sur chaque siège rejoue la même partie,
    donc la somme des résultats du héros sur les 6 rotations vaut exactement 0 (réel et EV)."""
    for strat in ("joueur", "calling_station"):
        comp = {"name": "miroir", "mode": "fixed", "seats": [strat] * 5}
        df = _run(lab, strat, 3, 1, comp)
        sums = df.groupby("deck")[["result", "result_ev"]].sum()
        assert (sums["result"].abs() < 1e-9).all(), sums
        assert (sums["result_ev"].abs() < 1e-6).all(), sums
        assert (df.groupby("deck")["rot"].nunique() == 6).all()


def test_duplicate_deals_same_cards_per_seat(lab) -> None:
    df = _run(lab, "serre", 2, 1)
    other = _run(lab, "large", 2, 1)
    key = ["deck", "rot", "c1", "c2", "b0"]
    pd.testing.assert_frame_equal(df[key[:4]].reset_index(drop=True), other[key[:4]].reset_index(drop=True))
