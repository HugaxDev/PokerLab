"""Briques communes des expériences : tâches parallèles déterministes et collecte des résultats.

Chaque main dépend seulement de (graine globale, index de deck, rotation, joueurs) : le
découpage en tâches et le nombre de workers ne changent donc pas les résultats ; la fusion
se fait dans l'ordre des tâches.
"""

from __future__ import annotations

import atexit
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from pokerlab.cards.ranges import hole_class
from pokerlab.config import EquityConfig, GameConfig, LabConfig
from pokerlab.engine.flags import FLAG_NAMES, N_FLAGS, line_of, seat_flags, street_end
from pokerlab.engine.hand_runner import COUNTERS, HandResult, play_hand
from pokerlab.engine.types import position_names
from pokerlab.features.hand import level_of
from pokerlab.opponents.compositions import Composition
from pokerlab.rng import derive_seed, shuffled_deck
from pokerlab.strategies.registry import build_strategy, resolve_spec

_LAB_CACHE: dict[str, LabConfig] = {}


def lab_from_raw(raw: dict[str, Any]) -> LabConfig:
    """Reconstruit (avec cache) la config globale dans un worker."""
    key = json.dumps(raw, sort_keys=True, default=str)
    lab = _LAB_CACHE.get(key)
    if lab is None:
        lab = LabConfig(
            game=GameConfig.from_dict(raw.get("game", {})),
            equity=EquityConfig.from_dict(raw.get("equity", {})),
            seed=int(raw.get("seed", 1)),
            workers=int(raw.get("workers", 0)),
            mp_start_method=str(raw.get("mp_start_method", "forkserver")),
            raw=raw,
        )
        _LAB_CACHE[key] = lab
    return lab


def lineup_unit(deck: int, space: str) -> int:
    """Unité de tirage des adversaires (différente entre graines d'entraînement et de validation)."""
    return deck if space == "deck" else derive_seed(space, deck)


def deck_seed(lab: LabConfig, deck: int, space: str = "deck") -> int:
    """Graine du deck ``deck`` : identique pour toutes les stratégies (nombres aléatoires communs)."""
    return derive_seed(lab.seed, space, deck)


WIDE_FLOATS = frozenset({"result", "result_ev", "net_after", "loss_cv", "loss_naive", "ev_taken", "ev_best"})


def _compact_array(name: str, values: list) -> np.ndarray:
    """Liste Python -> tableau numpy au type le plus petit (mémoire des grosses campagnes)."""
    arr = np.asarray(values)
    if arr.dtype.kind in "iu" and arr.size:
        lo, hi = int(arr.min()), int(arr.max())
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            info = np.iinfo(dtype)
            if info.min <= lo and hi <= info.max:
                return arr.astype(dtype)
    if arr.dtype.kind == "f" and name not in WIDE_FLOATS:
        return arr.astype(np.float32)
    if arr.dtype.kind in "U":
        return arr.astype(object)
    return arr


@dataclass
class RowBuffer:
    """Accumulateur de lignes : listes pendant la tâche, puis blocs numpy compacts (``compact``)."""

    cols: dict[str, list] = field(default_factory=dict)
    chunks: list[dict[str, np.ndarray]] = field(default_factory=list)

    def add(self, **values: Any) -> None:
        for k, v in values.items():
            self.cols.setdefault(k, []).append(v)

    def compact(self) -> None:
        if self.cols:
            self.chunks.append({k: _compact_array(k, v) for k, v in self.cols.items()})
            self.cols = {}

    def extend(self, other: "RowBuffer") -> None:
        other.compact()
        self.compact()
        self.chunks.extend(other.chunks)

    def column(self, name: str) -> np.ndarray:
        self.compact()
        parts = [c[name] for c in self.chunks if name in c]
        return np.concatenate(parts) if parts else np.array([])

    def frame(self) -> pd.DataFrame:
        self.compact()
        if not self.chunks:
            return pd.DataFrame()
        names = list(self.chunks[0])
        data = {}
        for k in names:
            parts = [c[k] for c in self.chunks]
            col = np.concatenate(parts) if parts[0].dtype != object else np.concatenate([p.astype(object) for p in parts])
            if col.dtype == object:
                data[k] = pd.Categorical(col)
            else:
                data[k] = col
        df = pd.DataFrame(data)
        for k in names:
            if isinstance(df[k].dtype, pd.CategoricalDtype):
                df[k] = df[k].astype(str) if df[k].cat.categories.size > 5000 else df[k]
        return df

    def __len__(self) -> int:
        self.compact()
        return sum(len(next(iter(c.values()))) for c in self.chunks if c)


@dataclass
class TaskResult:
    """Résultat d'une tâche (bloc de decks ou de sessions)."""

    hands: RowBuffer = field(default_factory=RowBuffer)
    decisions: RowBuffer = field(default_factory=RowBuffer)
    flags_by_name: dict[str, list[int]] = field(default_factory=dict)
    hands_played: int = 0
    corrections: int = 0

    def compact(self) -> "TaskResult":
        self.hands.compact()
        self.decisions.compact()
        return self

    def merge(self, other: "TaskResult") -> None:
        self.hands.extend(other.hands)
        self.decisions.extend(other.decisions)
        for k, v in other.flags_by_name.items():
            acc = self.flags_by_name.setdefault(k, [0] * N_FLAGS)
            for i, x in enumerate(v):
                acc[i] += x
        self.hands_played += other.hands_played
        self.corrections += other.corrections


def add_flags(result: TaskResult, name: str, flags: Sequence[int]) -> None:
    acc = result.flags_by_name.setdefault(name, [0] * N_FLAGS)
    for i, x in enumerate(flags):
        acc[i] += x


def record_seat(
    result: TaskResult,
    hr: HandResult,
    seat: int,
    lab: LabConfig,
    keys: dict[str, Any],
    record_decisions: bool,
) -> None:
    """Ajoute la ligne « main » (et les décisions) d'un siège suivi."""
    bb = lab.game.chips_per_bb
    value_min = level_of(lab.raw.get("analysis", {}).get("value_min_level", "middle_pair"))
    flags = seat_flags(hr, seat)
    board = list(hr.board) + [-1] * (5 - len(hr.board))
    c1, c2 = hr.holes[seat]
    row = dict(keys)
    row.update(
        seat=seat,
        position=hr.positions[seat],
        hand_class=hole_class((c1, c2)),
        c1=c1,
        c2=c2,
        b0=board[0],
        b1=board[1],
        b2=board[2],
        b3=board[3],
        b4=board[4],
        result=hr.payoffs[seat] / bb,
        result_ev=hr.ev_payoffs[seat] / bb,
        allin_street=-1 if hr.allin_street is None else hr.allin_street,
        showdown=bool(hr.showdown and not hr.folded[seat]),
        street_end=street_end(hr, seat),
        line=line_of(hr, seat, value_min),
        hand_seed=hr.seed,
    )
    for name, value in zip(FLAG_NAMES, flags):
        row["f_" + name] = value
    result.hands.add(**row)
    add_flags(result, hr.names[seat], flags)
    if record_decisions:
        for d in hr.decisions:
            if d.seat != seat:
                continue
            drow = dict(keys)
            drow.update(
                seat=seat,
                idx=d.idx,
                street=d.street,
                position=d.position,
                hand_class=row["hand_class"],
                level=d.level,
                draw=d.draw,
                board_label=d.board_label,
                facing=d.facing,
                kind=int(d.kind),
                tag=d.tag,
                amount_bb=d.amount / bb,
                to_bb=d.to / bb,
                to_call_bb=d.to_call / bb,
                pot_bb=d.pot / bb,
                stack_bb=d.stack / bb,
                n_opp=d.n_opp,
                invested_bb=d.invested_before / bb,
                can_fold=d.legal.can_fold,
                can_raise=d.legal.can_raise,
                min_to_bb=d.legal.min_raise_to / bb,
                max_to_bb=d.legal.max_raise_to / bb,
                result=row["result"],
                result_ev=row["result_ev"],
            )
            result.decisions.add(**drow)


def seating(hero: Any, opponents: Sequence[Any], rotation: int, n: int) -> list[Any]:
    """Place le héros au siège PokerKit ``rotation`` et les adversaires dans l'ordre à sa gauche."""
    players: list[Any] = [None] * n
    players[rotation] = hero
    for k, opp in enumerate(opponents):
        players[(rotation + 1 + k) % n] = opp
    return players


@dataclass
class DuplicateTask:
    """Bloc de decks joués en duplicate pour un héros contre une composition."""

    lab_raw: dict[str, Any]
    hero: dict[str, Any]
    composition: dict[str, Any]
    deck_start: int
    deck_end: int
    keys: dict[str, Any]
    rotations: tuple[int, ...] | None = None
    record_decisions: bool = True
    deck_space: str = "deck"
    duplicate: bool = True
    decision_deck_limit: int | None = None


def run_duplicate_task(task: DuplicateTask) -> TaskResult:
    """Joue chaque deck du bloc avec le héros sur chaque siège (mêmes cartes par siège)."""
    lab = lab_from_raw(task.lab_raw)
    n = lab.game.players
    comp = Composition.from_dict(task.composition)
    hero = build_strategy(task.hero, lab)
    out = TaskResult()
    start_count = COUNTERS["hands"]
    start_corr = COUNTERS["corrections"]
    stacks = [lab.game.starting_stack] * n
    for d in range(task.deck_start, task.deck_end):
        names = comp.opponents(lab.seed, lineup_unit(d, task.deck_space), n - 1)
        opponents = [build_strategy(resolve_spec(x), lab) for x in names]
        seed = deck_seed(lab, d, task.deck_space)
        deck = shuffled_deck(seed)
        if task.rotations is not None:
            rotations: Iterable[int] = task.rotations
        elif task.duplicate:
            rotations = range(n)
        else:
            rotations = [int(seed % n)]
        for r in rotations:
            players = seating(hero, opponents, r, n)
            hand_id = d * n + r
            hr = play_hand(players, lab.game, deck, stacks, hand_id, seed, track={r})
            keys = dict(task.keys)
            keys.update(deck=d, rot=r, hand_id=hand_id, lineup="|".join(names))
            record = task.record_decisions and (task.decision_deck_limit is None or d < task.decision_deck_limit)
            record_seat(out, hr, r, lab, keys, record)
            for i in range(n):
                if i != r:
                    add_flags(out, hr.names[i], seat_flags(hr, i))
    out.hands_played = COUNTERS["hands"] - start_count
    out.corrections = COUNTERS["corrections"] - start_corr
    return out.compact()


def replay_duplicate_hand(
    lab: LabConfig, hero: dict[str, Any], composition: dict[str, Any], deck: int, rotation: int, deck_space: str = "deck"
) -> tuple[HandResult, list[str]]:
    """Rejoue à l'identique une main d'une campagne duplicate (pour les historiques)."""
    n = lab.game.players
    comp = Composition.from_dict(composition)
    names = comp.opponents(lab.seed, lineup_unit(deck, deck_space), n - 1)
    opponents = [build_strategy(resolve_spec(x), lab) for x in names]
    hero_s = build_strategy(hero, lab)
    seed = deck_seed(lab, deck, deck_space)
    players = seating(hero_s, opponents, rotation, n)
    hr = play_hand(players, lab.game, shuffled_deck(seed), [lab.game.starting_stack] * n, deck * n + rotation, seed, track={rotation})
    labels = seating(hero_s.name, names, rotation, n)
    return hr, labels


def worker_count(lab: LabConfig, override: int | None = None) -> int:
    value = override if override is not None else lab.workers
    return max(1, value if value > 0 else (os.cpu_count() or 1))


_POOLS: dict[tuple[int, str], ProcessPoolExecutor] = {}


def _pool(workers: int, start_method: str) -> ProcessPoolExecutor:
    """Pool de processus réutilisé pendant toute la vie du programme (évite de relancer les workers)."""
    key = (workers, start_method)
    pool = _POOLS.get(key)
    if pool is None:
        try:
            ctx = get_context(start_method)
        except ValueError:
            ctx = get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        _POOLS[key] = pool
    return pool


def shutdown_pools() -> None:
    for pool in _POOLS.values():
        pool.shutdown(wait=True, cancel_futures=True)
    _POOLS.clear()


atexit.register(shutdown_pools)


def run_tasks(fn: Callable[[Any], TaskResult], tasks: Sequence[Any], workers: int, start_method: str = "forkserver") -> list[TaskResult]:
    """Exécute les tâches (en parallèle si workers > 1) ; résultats dans l'ordre des tâches."""
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    return list(_pool(workers, start_method).map(fn, tasks))


def split_range(start: int, end: int, parts: int) -> list[tuple[int, int]]:
    """Découpe [start, end) en blocs contigus (au plus ``parts``)."""
    total = end - start
    if total <= 0:
        return []
    parts = max(1, min(parts, total))
    size = -(-total // parts)
    return [(s, min(s + size, end)) for s in range(start, end, size)]


def deck_means(hands: pd.DataFrame, column: str = "result_ev", by: Sequence[str] = ()) -> pd.DataFrame:
    """Moyenne par deck (unité appariée du duplicate)."""
    keys = list(by) + ["deck"]
    return hands.groupby(keys, sort=True)[column].mean().reset_index()


def positions_for(lab: LabConfig) -> tuple[str, ...]:
    return position_names(lab.game.players)


def results_array(hands: pd.DataFrame, column: str) -> np.ndarray:
    return hands[column].to_numpy(dtype=float)
