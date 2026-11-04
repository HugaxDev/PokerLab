"""Statistiques, Holm, journal, historiques, et contrôle : random_legal perd contre equity_bot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pokerlab.analysis.stats import holm, max_drawdown, paired_test, summarize
from pokerlab.config import PROJECT_ROOT, load_yaml
from pokerlab.engine.hand_runner import play_hand
from pokerlab.engine.history import format_hand
from pokerlab.experiments.common import DuplicateTask, run_duplicate_task
from pokerlab.journal.importer import journal_stats, load_journal, percentile_of
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy, resolve_spec


def test_holm() -> None:
    adj = holm([0.01, 0.04, 0.03, 0.2])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.2])


def test_paired_and_summary() -> None:
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.05, 1.0, 4000)
    mean, half, p = paired_test(diffs)
    assert mean == pytest.approx(diffs.mean() * 100)
    assert p < 0.05 and half > 0
    _, _, p_boot = paired_test(diffs, method="bootstrap", resamples=500, block=20)
    assert p_boot < 0.1
    s = summarize(diffs, diffs, 1)
    assert s.low < s.bb100 < s.high and s.hands_needed > 0
    dd, longest = max_drawdown(np.array([1.0, 3.0, 1.0, 2.0, 4.0, 0.0]))
    assert dd == 4.0 and longest == 2


def test_journal_example() -> None:
    phases = {str(k): v for k, v in load_yaml(PROJECT_ROOT / "journal" / "phases.yaml")["phases"].items()}
    j = load_journal(PROJECT_ROOT / "journal" / "exemple_journal.csv", phases)
    df = j.hands
    assert len(df) == 10
    first = df[df["phase"] == "1"]
    assert first["result_bb"].round(2).tolist() == [7.0, -1.0, -10.0, -3.0, -0.5]
    stats = journal_stats(df)
    assert set(stats["phase"]) == {"1", "2"}
    assert int(stats.loc[stats["phase"] == "1", "mains hors règle"].iloc[0]) == 1
    row2 = df[(df["phase"] == "1") & (df["main"] == 2)].iloc[0]
    assert not row2["vpip"]  # « couché sur relance » n'est pas une relance
    assert percentile_of(0.0, np.array([-1.0, 1.0])) == 50.0


def test_history_text(lab) -> None:
    players = [build_strategy(n, lab) for n in ("maniac", "calling_station", "serre", "joueur", "TAG", "nit")]
    seed = derive_seed(77, 1)
    hr = play_hand(players, lab.game, shuffled_deck(seed), [10000] * 6, 1, seed)
    text = format_hand(hr, lab.game, "test")
    assert "*** HOLE CARDS ***" in text and "*** SUMMARY ***" in text and "posts big blind €0.02" in text


def test_random_legal_loses_against_equity_bot(lab) -> None:
    """Contrôle du moteur et des stats : le robot aléatoire doit perdre nettement."""
    comp = {"name": "refs", "mode": "fixed", "seats": ["equity_bot"] * 5}
    out = run_duplicate_task(DuplicateTask(lab.raw, resolve_spec("random_legal"), comp, 0, 20, {}, record_decisions=False))
    df = out.hands.frame()
    bb100 = df.groupby("deck")["result_ev"].mean().mean() * 100
    assert len(df) == 120
    assert bb100 < -50, bb100
