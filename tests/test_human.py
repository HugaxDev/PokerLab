"""Adversaires « humains » : tirage, stratégie mixte, tilt, adaptation, état par session et déterminisme."""

from __future__ import annotations

import random
import statistics

import pandas as pd

from pokerlab.config import load_table
from pokerlab.experiments.common import (
    DuplicateTask,
    TaskResult,
    deck_seed,
    lineup_unit,
    play_hand,
    record_seat,
    replay_duplicate_hand,
    run_duplicate_task,
    seating,
)
from pokerlab.opponents.compositions import Composition
from pokerlab.opponents.human import HumanState, draw_params, human_config, is_stateful
from pokerlab.rng import shuffled_deck
from pokerlab.strategies.registry import build_player, build_strategy, resolve_spec

HUMAN = {"name": "humains_test", "mode": "sample", "session_decks": 3,
         "weights": {"humain_debutant": 1, "humain_agressif": 1, "humain_regulier": 1}}


def _frame(out: TaskResult) -> pd.DataFrame:
    merged = TaskResult()
    merged.merge(out)
    return merged.hands.frame()[["deck", "rot", "result", "result_ev", "lineup"]].sort_values(["deck", "rot"]).reset_index(drop=True)


def test_options_are_off_by_default_and_on_for_humains_simules() -> None:
    assert not is_stateful(resolve_spec("maniac"))
    comp = load_table("humains_simules")
    assert comp["session_decks"] > 1
    for name in comp["weights"]:
        h = human_config(resolve_spec(name))
        assert all(h[k]["enabled"] for k in ("tirage", "mixte", "tilt", "adaptation")), name


def test_draw_params_spread_around_archetype() -> None:
    base = dict(resolve_spec("humain_regulier"))
    from pokerlab.opponents.archetype import PARAM_DEFAULTS

    full = {**PARAM_DEFAULTS, **{k: v for k, v in base.items() if k in PARAM_DEFAULTS}}
    draws = [draw_params(full, 0.25, s) for s in range(400)]
    vpips = [d["vpip"] for d in draws]
    assert abs(statistics.mean(vpips) - full["vpip"]) < 0.02          # centré sur la valeur de l'archétype
    assert statistics.pstdev(vpips) > 0.03                              # mais chaque joueur est différent
    assert all(d["pfr"] <= d["vpip"] and 0 <= d["fold_to_cbet"] <= 0.98 for d in draws)
    assert draw_params(full, 0.25, 7) == draw_params(full, 0.25, 7)


def test_mixed_threshold_is_a_frequency() -> None:
    st = HumanState(human_config({"humain": {"mixte": {"enabled": True, "douceur": 0.04}}}))
    rng = random.Random(1)
    at = sum(st.mixed_below(0.25, 0.25, rng) for _ in range(4000)) / 4000
    near = sum(st.mixed_below(0.29, 0.25, rng) for _ in range(4000)) / 4000
    assert 0.45 < at < 0.55 and 0.15 < near < 0.35
    assert st.mixed_below(0.01, 0.25, rng) and not st.mixed_below(0.9, 0.25, rng)


def test_tilt_after_big_loss() -> None:
    cfg = human_config({"humain": {"tilt": {"enabled": True, "perte_bb": 30, "mains": 3, "elargissement": 1.5, "agressivite": 2.0}}})
    st = HumanState(cfg)
    p = {"vpip": 0.2, "pfr": 0.1, "three_bet": 0.04, "aggression": 0.4, "cbet": 0.5, "bluff": 0.1, "semi_bluff": 0.3,
         "bluff_raise": 0.02, "thin_value": 0.2, "fold_to_cbet": 0.5}
    st.end_hand(-10.0, None)
    assert st.apply_tilt(p) == p
    st.end_hand(-45.0, None)
    q = st.apply_tilt(p)
    assert q["vpip"] == 0.2 * 1.5 and q["aggression"] == 0.8 and q["fold_to_cbet"] == 0.25
    for _ in range(3):
        st.end_hand(1.0, None)
    assert st.apply_tilt(p) == p          # fin du tilt après 3 mains


def test_adaptation_to_hero() -> None:
    cfg = human_config({"humain": {"adaptation": {"enabled": True, "fenetre": 20, "min_mains": 10, "force": 1.0}}})
    loose = HumanState(cfg)
    for k in range(20):   # héros qui relance une main sur deux et montre des bluffs
        loose.end_hand(0.0, (1, int(k % 2 == 0), 1, int(k % 2 == 0)))
    pre, post = loose.adaptation_factors()
    assert pre > 1.2 and post > 1.2        # défend plus face à lui
    tight = HumanState(cfg)
    for k in range(20):   # héros serré qui ne montre que de la valeur
        tight.end_hand(0.0, (int(k % 10 == 0), int(k % 10 == 0), int(k % 5 == 0), 0))
    pre, post = tight.adaptation_factors()
    assert pre < 0.9 and post < 1.0        # se couche plus face à lui
    assert HumanState(cfg).adaptation_factors() is None   # pas assez observé


def test_stateless_compositions_unchanged(lab) -> None:
    """Compositions sans option humaine : mêmes mains que l'ancienne boucle (adversaires construits à chaque deck)."""
    comp_d = load_table("micro_play_money")
    hero = resolve_spec("joueur")
    new = _frame(run_duplicate_task(DuplicateTask(lab.raw, hero, comp_d, 0, 2, {"strategy": "joueur"}, record_decisions=False)))
    comp = Composition.from_dict(comp_d)
    hero_s = build_strategy(hero, lab)
    n = lab.game.players
    old = TaskResult()
    for d in range(2):
        names = comp.opponents(lab.seed, lineup_unit(d, "deck"), n - 1)
        opps = [build_strategy(resolve_spec(x), lab) for x in names]
        seed = deck_seed(lab, d)
        for r in range(n):
            hr = play_hand(seating(hero_s, opps, r, n), lab.game, shuffled_deck(seed), [lab.game.starting_stack] * n, d * n + r, seed, track={r})
            record_seat(old, hr, r, lab, {"strategy": "joueur", "deck": d, "rot": r, "hand_id": d * n + r, "lineup": "|".join(names)}, False)
    pd.testing.assert_frame_equal(new, _frame(old))


def test_human_sessions_do_not_depend_on_task_split(lab) -> None:
    """Même résultat en une tâche ou découpé au milieu d'une session (l'état est rejoué)."""
    hero = resolve_spec("joueur")
    one = _frame(run_duplicate_task(DuplicateTask(lab.raw, hero, HUMAN, 0, 4, {"strategy": "joueur"}, record_decisions=False)))
    parts = TaskResult()
    for a, b in ((0, 2), (2, 4)):
        parts.merge(run_duplicate_task(DuplicateTask(lab.raw, hero, HUMAN, a, b, {"strategy": "joueur"}, record_decisions=False)))
    pd.testing.assert_frame_equal(one, _frame(parts))
    assert one.loc[one["deck"] < 3, "lineup"].nunique() == 1          # mêmes adversaires pendant la session de 3 decks
    hr, _ = replay_duplicate_hand(lab, hero, HUMAN, 2, 4)             # rejeu d'une main en cours de session (historiques, audit)
    row = one[(one["deck"] == 2) & (one["rot"] == 4)].iloc[0]
    assert hr.payoffs[4] / lab.game.chips_per_bb == row["result"]


def test_each_seat_has_its_own_human(lab) -> None:
    a = build_player(resolve_spec("humain_regulier"), lab)
    b = build_player(resolve_spec("humain_regulier"), lab)
    assert a is not b
    a.begin_session(1)
    b.begin_session(2)
    assert a.base_p["vpip"] != b.base_p["vpip"]
    assert build_player(resolve_spec("maniac"), lab) is build_player(resolve_spec("maniac"), lab)
