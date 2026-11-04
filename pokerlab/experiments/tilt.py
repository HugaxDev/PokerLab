"""Mesure du tilt : part des sièges adverses en tilt au début de chaque main (mesurée en jouant, pas déduite des réglages).

``python -m pokerlab mesure-tilt --composition humains_simules --strategy joueur --decks 3000``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pokerlab.config import LabConfig, load_table
from pokerlab.experiments.common import SessionTable, TaskResult, lab_from_raw, run_tasks, split_range
from pokerlab.opponents.compositions import Composition
from pokerlab.strategies.registry import build_strategy, resolve_spec


@dataclass
class TiltTask:
    lab_raw: dict[str, Any]
    hero: dict[str, Any]
    composition: dict[str, Any]
    deck_start: int
    deck_end: int
    space: str = "mesure_tilt"


def run_tilt_task(task: TiltTask) -> TaskResult:
    """Compte, par archétype, les sièges en tilt au début de chaque main (et les sièges observés)."""
    lab = lab_from_raw(task.lab_raw)
    n = lab.game.players
    comp = Composition.from_dict(task.composition)
    hero = build_strategy(task.hero, lab)
    table = SessionTable(lab, comp, hero, task.space, lambda d: tuple(range(n)))
    out = TaskResult()
    counts: dict[str, list[int]] = {}
    for d in range(task.deck_start, task.deck_end):
        for r in range(n):
            names, opps = table.at(d, r)
            for name, opp in zip(names, opps):
                c = counts.setdefault(name, [0, 0])
                c[1] += 1
                h = getattr(opp, "human", None)
                c[0] += int(h is not None and h.tilted)
            hr = table.play(d, r)
            table.finish(hr, d, r)
            out.hands_played += 1
    out.flags_by_name = {k: list(v) for k, v in counts.items()}   # [sièges en tilt, sièges observés]
    return out


def measure_tilt(lab: LabConfig, composition: str | dict[str, Any], hero: str = "joueur", decks: int = 3000,
                 workers: int = 1, space: str = "mesure_tilt") -> dict[str, Any]:
    """Taux de tilt global et par archétype ; IC 95 % par blocs de sessions (les épisodes de tilt sont corrélés)."""
    cdict = load_table(composition) if isinstance(composition, str) else composition
    size = Composition.from_dict(cdict).session_decks
    decks = max(size, decks - decks % size)
    bounds = [(a - a % size, b - b % size) for a, b in split_range(0, decks, max(20, workers * 4))]
    bounds = [(a, b) for a, b in bounds if b > a]
    tasks = [TiltTask(lab.raw, resolve_spec(hero), cdict, a, b, space) for a, b in bounds]
    outs = run_tasks(run_tilt_task, tasks, workers, lab.mp_start_method)
    total: dict[str, list[int]] = {}
    per_task = []
    for o in outs:
        t = [0, 0]
        for k, (tilted, seen) in o.flags_by_name.items():
            acc = total.setdefault(k, [0, 0])
            acc[0] += tilted
            acc[1] += seen
            t[0] += tilted
            t[1] += seen
        per_task.append(t[0] / t[1] if t[1] else np.nan)
    tilted = sum(v[0] for v in total.values())
    seen = sum(v[1] for v in total.values())
    rate = tilted / seen if seen else float("nan")
    arr = np.array([x for x in per_task if np.isfinite(x)])
    half = 1.96 * arr.std(ddof=1) / np.sqrt(arr.size) if arr.size > 1 else float("nan")
    return {
        "composition": cdict.get("name"),
        "heros": hero,
        "mains": int(sum(o.hands_played for o in outs)),
        "taux": rate,
        "ic": float(half),
        "par_archetype": {k: v[0] / v[1] for k, v in sorted(total.items()) if v[1]},
    }
