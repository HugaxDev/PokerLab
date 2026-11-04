"""Fiche de stratégie : grilles, tailles, règles, blocs campagne (données synthétiques), journal hors fiche.

Aucun de ces tests ne joue de main : la fiche interroge seulement les décisions de la stratégie.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pokerlab.cards.ranges import CLASS_INDEX_BY_NAME
from pokerlab.config import PROJECT_ROOT, list_strategies, load_strategy_config, load_yaml
from pokerlab.engine.types import Action, Kind
from pokerlab.fiche import CALL, FOLD, RAISE
from pokerlab.fiche.build import build_fiche, chart_for
from pokerlab.fiche.content import postflop_rules
from pokerlab.journal.importer import flag_off_sheet, load_journal

C = CLASS_INDEX_BY_NAME


def test_serre_chart(lab) -> None:
    chart = chart_for("serre", lab, 3.0)
    utg = chart[("UTG", "unopened")]
    assert utg[C["AA"]] == RAISE and utg[C["AJs"]] == RAISE and utg[C["KQs"]] == RAISE
    assert utg[C["AJo"]] == FOLD and utg[C["72o"]] == FOLD and utg[C["88"]] == FOLD
    btn = chart[("BTN", "vs_raise")]
    assert btn[C["QQ"]] == RAISE and btn[C["AKo"]] == RAISE
    assert btn[C["JJ"]] == CALL and btn[C["AQo"]] == CALL and btn[C["AJs"]] == FOLD
    assert chart[("UTG", "vs_raise")] is None
    assert chart[("BB", "unopened")][C["72o"]] == CALL  # check gratuit après complément de la SB


def test_position_chart(lab) -> None:
    chart = chart_for("position", lab, 3.0)
    assert chart[("BTN", "unopened")][C["54s"]] == RAISE
    assert chart[("UTG", "unopened")][C["54s"]] == FOLD
    assert chart[("BB", "vs_raise")][C["54s"]] == FOLD


def test_postflop_rules_are_short() -> None:
    for name in list_strategies():
        rules = postflop_rules(load_strategy_config(name))
        assert 1 <= len(rules) <= 10, name
    serre = " ".join(postflop_rules(load_strategy_config("serre")))
    assert "QQ+" in serre and "deux paires" in serre


def test_fiche_without_campaign(lab, tmp_path: Path) -> None:
    res = build_fiche("serre", lab, None, tmp_path / "f.html")
    html = res["path"].read_text(encoding="utf-8")
    assert html.count("Pas encore de chiffres") == 2
    assert "python -m pokerlab compare" in html and "python -m pokerlab audit --set strategy=serre" in html
    assert "0,06" in html and "0,15" in html  # ouverture 3 bb en 0,01/0,02 et 0,02/0,05
    assert html.count('class="grid"') == 11  # 6 positions x 2 situations, sauf UTG face à une relance
    assert "bb/100 all-in adjusted" not in html
    assert not any(res["filled"].values())


def _fake_campaign(root: Path) -> None:
    rng = np.random.default_rng(0)
    comp = root / "compare"
    comp.mkdir(parents=True)
    n = 600
    pd.DataFrame({
        "strategy": ["serre"] * n,
        "composition": ["micro_play_money"] * n,
        "deck": np.repeat(np.arange(n // 6), 6),
        "rot": np.tile(np.arange(6), n // 6),
        "result": rng.normal(0.05, 8, n),
        "result_ev": rng.normal(0.05, 5, n),
    }).to_csv(comp / "mains.csv", index=False)
    (comp / "resume.json").write_text(json.dumps({"experience": "compare", "config": {"strategies": ["serre"]}}), encoding="utf-8")
    aud = root / "audit"
    aud.mkdir()
    rows = []
    for i in range(40):
        rows.append({"strategy": "serre", "street": 2, "position": "CO", "hand_class": C["AKo"], "level": 3, "draw": 0,
                     "board_label": "connecté", "facing": 2, "kind": int(Kind.CALL), "invested_bb": 5.0, "result_ev": -10.0,
                     "loss_cv": 2.0 + 0.1 * (i % 3), "best_action": "se couche"})
    for i in range(10):
        rows.append({"strategy": "serre", "street": 0, "position": "SB", "hand_class": C["99"], "level": -1, "draw": 0,
                     "board_label": "préflop", "facing": 1, "kind": int(Kind.FOLD), "invested_bb": 0.5, "result_ev": -0.5,
                     "loss_cv": 0.4, "best_action": "relance 3x"})
    pd.DataFrame(rows).to_csv(aud / "decisions.csv", index=False)
    (aud / "resume.json").write_text(json.dumps({"experience": "audit", "config": {"strategy": "serre"}}), encoding="utf-8")


def test_fiche_with_campaign(lab, tmp_path: Path) -> None:
    _fake_campaign(tmp_path / "camp")
    res = build_fiche("serre", lab, tmp_path / "camp", tmp_path / "f.html")
    html = res["path"].read_text(encoding="utf-8")
    assert res["filled"] == {"interdits": True, "resultat": True}
    assert "Ne paie pas avec top paire face à une relance au turn : couche-toi." in html
    assert "Ne te couche pas avec paire moyenne (66-99) sans relance devant avant le flop : relance 3x." in html
    assert "50 mains" in html and "200 mains" in html and "1000 mains" in html
    assert "Pas encore de chiffres" not in html


def test_journal_flags_off_sheet(lab) -> None:
    phases = {str(k): v for k, v in load_yaml(PROJECT_ROOT / "journal" / "phases.yaml")["phases"].items()}
    df = load_journal(PROJECT_ROOT / "journal" / "exemple_journal.csv", phases).hands
    flagged = flag_off_sheet(df, {s: chart_for(s, lab, 3.0) for s in ("serre", "joueur")})
    jt = flagged[(flagged["phase"] == "1") & (flagged["main"] == 4)].iloc[0]
    assert jt["classe"] == "JTs" and jt["position"] == "UTG" and bool(jt["hors_fiche"])
    ak = flagged[(flagged["phase"] == "1") & (flagged["main"] == 1)].iloc[0]
    assert not bool(ak["hors_fiche"])
    fold_vs_raise = flagged[(flagged["phase"] == "1") & (flagged["main"] == 2)].iloc[0]
    assert fold_vs_raise["situation"] == "vs_raise" and fold_vs_raise["joue"] == "couché"


def test_audit_rollout_uses_hero_information_only(lab) -> None:
    """Rollouts d'audit : historique imposé, cartes du héros et tableau vu gardés, cartes adverses
    retirées dans leur range estimée (pas les vraies), paquet sans doublon. Joue 2 mains."""
    from pokerlab.engine.hand_runner import play_hand
    from pokerlab.experiments.audit import _Capture, forced_prefix, posterior_deck
    from pokerlab.experiments.common import seating
    from pokerlab.rng import derive_seed, shuffled_deck
    from pokerlab.strategies.range_model import RangeModel
    from pokerlab.strategies.registry import build_strategy, get_context

    hero = build_strategy("joueur", lab)
    opponents = [build_strategy(x, lab) for x in ("calling_station", "maniac", "calling_station", "LAG", "maniac")]
    seed = derive_seed(4242, 0)
    deck = shuffled_deck(seed)
    rot = 5
    cap = _Capture(hero, 3)
    original = play_hand(seating(cap, opponents, rot, 6), lab.game, deck, [10000] * 6, 0, seed)
    obs = cap.obs
    assert obs is not None and obs.seat == rot
    rdeck = posterior_deck(deck, obs, 6, RangeModel(get_context(lab).table, {"looseness": 1.3}), 7)
    assert sorted(rdeck) == list(range(52))
    assert (rdeck[2 * rot], rdeck[2 * rot + 1]) == tuple(obs.hole)
    dealt = {0: 0, 1: 3, 2: 4, 3: 5}[obs.street]
    assert rdeck[12:12 + dealt] == list(obs.board)
    prefix = forced_prefix(original.history, 3)
    replay = play_hand(seating(hero, opponents, rot, 6), lab.game, rdeck, [10000] * 6, 0, seed,
                       forced={**prefix, 3: Action.fold() if obs.legal.can_fold else Action.check_call()}, fork_after=3, fork_salt=1)
    voluntary = [a for a in original.history if a.kind != Kind.POST][:3]
    assert [(a.seat, a.kind, a.to) for a in replay.history if a.kind != Kind.POST][:3] == [(a.seat, a.kind, a.to) for a in voluntary]
