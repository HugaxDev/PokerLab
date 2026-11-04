"""Campagnes duplicate par blocs, avec arrêt automatique sur la précision du bb/100.

Une cellule = (stratégie ou variante, composition). Toutes les cellules jouent les mêmes
decks (mêmes graines) : les comparaisons sont appariées deck par deck.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from pokerlab.analysis.stats import z_value
from pokerlab.config import LabConfig
from pokerlab.experiments.common import DuplicateTask, TaskResult, run_duplicate_task, run_tasks, split_range


@dataclass
class Cell:
    """Une stratégie (config complète) jouée contre une composition."""

    key: str
    hero: dict[str, Any]
    composition: dict[str, Any]
    keys: dict[str, Any]
    deck_space: str = "deck"
    deck_offset: int = 0
    record_decisions: bool = True
    decision_decks: int | None = None


@dataclass
class Autostop:
    """Arrêt quand la demi-largeur de l'IC du bb/100 passe sous le seuil (ou au maximum de mains)."""

    enabled: bool = True
    block_decks: int = 500
    ci_halfwidth_bb100: float = 3.0
    max_hands: int = 200_000
    min_decks: int = 100
    metric: str = "result_ev"
    decks: int = 200
    confidence: float = 0.95
    max_minutes: float | None = None
    stop_file: str | None = None

    @staticmethod
    def from_dict(data: dict[str, Any] | None, decks: int | None = None) -> "Autostop":
        data = dict(data or {})
        metric = data.pop("metric", "ev")
        if "max_hands_per_strategy" in data:
            data["max_hands"] = data.pop("max_hands_per_strategy")
        out = Autostop(**data)
        out.metric = "result_ev" if metric in ("ev", "result_ev") else "result"
        if decks is not None:
            out.decks = int(decks)
        return out


@dataclass
class CellProgress:
    decks: int = 0
    deck_means: list[float] = field(default_factory=list)
    stopped: str = ""

    def halfwidth(self, confidence: float) -> float:
        if len(self.deck_means) < 2:
            return math.inf
        arr = np.asarray(self.deck_means)
        return z_value(confidence) * float(arr.std(ddof=1)) / math.sqrt(arr.size) * 100

    def bb100(self) -> float:
        return float(np.mean(self.deck_means)) * 100 if self.deck_means else float("nan")


def _deck_means(out: TaskResult, metric: str) -> list[float]:
    decks = out.hands.column("deck")
    if decks.size == 0:
        return []
    df = pd.DataFrame({"deck": decks, "v": out.hands.column(metric)})
    return df.groupby("deck", sort=True)["v"].mean().tolist()


def run_cells(
    cells: list[Cell],
    lab: LabConfig,
    autostop: Autostop,
    workers: int,
    record_decisions: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, TaskResult], dict[str, CellProgress]]:
    """Joue toutes les cellules par blocs de decks jusqu'à la précision visée."""
    n = lab.game.players
    results = {c.key: TaskResult() for c in cells}
    progress = {c.key: CellProgress() for c in cells}
    active = list(cells)
    max_decks = max(1, autostop.max_hands // n)
    round_no = 0
    started = time.perf_counter()
    total_played = 0
    stop_path = Path(autostop.stop_file) if autostop.stop_file else None
    if stop_path is not None and stop_path.exists():
        stop_path.unlink()
    while active:
        round_no += 1
        t0 = time.perf_counter()
        tasks: list[DuplicateTask] = []
        owners: list[str] = []
        for c in active:
            prog = progress[c.key]
            size = autostop.block_decks if autostop.enabled else autostop.decks
            size = max(0, min(size, max_decks - prog.decks))
            if size == 0:
                continue
            start = c.deck_offset + prog.decks
            for a, b in split_range(start, start + size, max(1, workers * 2)):
                limit = None if c.decision_decks is None else c.deck_offset + c.decision_decks
                tasks.append(DuplicateTask(lab.raw, c.hero, c.composition, a, b, c.keys, record_decisions=record_decisions and c.record_decisions,
                                           deck_space=c.deck_space, decision_deck_limit=limit))
                owners.append(c.key)
            prog.decks += size
        outs = run_tasks(run_duplicate_task, tasks, workers, lab.mp_start_method)
        for owner, out in zip(owners, outs):
            results[owner].merge(out)
            progress[owner].deck_means.extend(_deck_means(out, autostop.metric))
        played = sum(o.hands_played for o in outs)
        total_played += played
        elapsed = time.perf_counter() - started
        log(f"  bloc {round_no} : {played} mains en {time.perf_counter() - t0:.1f} s — total {total_played} mains, "
            f"{elapsed / 60:.1f} min, {played / max(time.perf_counter() - t0, 1e-9):.0f} mains/s, {len(active)} cellule(s) actives")
        next_active = []
        for c in active:
            prog = progress[c.key]
            hw = prog.halfwidth(autostop.confidence)
            if not autostop.enabled:
                prog.stopped = f"nombre de decks fixé ({prog.decks})"
                continue
            if prog.decks >= autostop.min_decks and 0 < hw <= autostop.ci_halfwidth_bb100:
                prog.stopped = f"précision atteinte (± {hw:.2f} bb/100)"
            elif prog.decks >= max_decks:
                prog.stopped = f"maximum de mains atteint (± {hw:.2f} bb/100)"
            else:
                next_active.append(c)
            log(f"    {c.key:40s} {prog.decks * n:>8d} mains  {prog.bb100():+8.1f} ± {hw:.1f} bb/100 {prog.stopped}")
        active = next_active
        reason = ""
        if active and autostop.max_minutes is not None and time.perf_counter() - started > autostop.max_minutes * 60:
            reason = f"durée maximale atteinte ({autostop.max_minutes:g} min) : résultats partiels"
        elif active and stop_path is not None and stop_path.exists():
            reason = "arrêt demandé (fichier STOP) : résultats partiels"
        if reason:
            for c in active:
                prog = progress[c.key]
                prog.stopped = f"{reason} (± {prog.halfwidth(autostop.confidence):.2f} bb/100)"
            log(f"  ARRÊT : {reason} ; {len(active)} cellule(s) interrompue(s)")
            active = []
    return results, progress


def collect(results: dict[str, TaskResult]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[int]], int, int]:
    """Fusionne les cellules : mains, décisions, compteurs par joueur, mains jouées, corrections."""
    merged = TaskResult()
    for key in results:
        merged.merge(results[key])
    hands = merged.hands.frame()
    decisions = merged.decisions.frame()
    return hands, decisions, merged.flags_by_name, merged.hands_played, merged.corrections
