"""Tables d'équité préflop précalculées (169 classes) avec cache disque."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pokerlab.cards.ranges import (
    CLASS_COMBO_COUNT,
    CLASS_COMBOS,
    COMBO_CARDS,
    COMBO_CLASS,
    N_CLASSES,
    N_COMBOS,
)
from pokerlab.config import EquityConfig
from pokerlab.equity.evaluator import CARD_BITS, evaluate_packed
from pokerlab.equity.montecarlo import sample_board

_FLAT = np.concatenate(CLASS_COMBOS).astype(np.int64)
_OFFSET = np.concatenate([[0], np.cumsum(CLASS_COMBO_COUNT)[:-1]]).astype(np.int64)
_COUNT = CLASS_COMBO_COUNT.astype(np.int64)


def _sample_class_combos(rng: np.random.Generator, classes: np.ndarray) -> np.ndarray:
    u = rng.random(classes.size)
    return _FLAT[_OFFSET[classes] + np.minimum((u * _COUNT[classes]).astype(np.int64), _COUNT[classes] - 1)]


def _matchup_shares(rng: np.random.Generator, ci: np.ndarray, cj: np.ndarray | None) -> np.ndarray:
    """Part de pot de la classe ``ci`` contre ``cj`` (ou une main aléatoire si None)."""
    n = ci.size
    a = _sample_class_combos(rng, ci)
    b = _sample_class_combos(rng, cj) if cj is not None else rng.integers(0, N_COMBOS, n)
    for _ in range(60):
        ca, cb = COMBO_CARDS[a], COMBO_CARDS[b]
        bad = (ca[:, 0] == cb[:, 0]) | (ca[:, 0] == cb[:, 1]) | (ca[:, 1] == cb[:, 0]) | (ca[:, 1] == cb[:, 1])
        if not bad.any():
            break
        rows = np.flatnonzero(bad)
        b[rows] = _sample_class_combos(rng, cj[rows]) if cj is not None else rng.integers(0, N_COMBOS, rows.size)
    ca, cb = COMBO_CARDS[a].astype(np.int64), COMBO_CARDS[b].astype(np.int64)
    blocked = np.zeros((n, 52), dtype=bool)
    np.put_along_axis(blocked, np.concatenate([ca, cb], axis=1), True, axis=1)
    board = sample_board(rng, blocked, 5)
    bb = CARD_BITS[board].sum(axis=1)
    sa = evaluate_packed(bb + CARD_BITS[ca[:, 0]] + CARD_BITS[ca[:, 1]])
    sb = evaluate_packed(bb + CARD_BITS[cb[:, 0]] + CARD_BITS[cb[:, 1]])
    return np.where(sa > sb, 1.0, np.where(sa == sb, 0.5, 0.0))


def compute_vs_random(sims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ci = np.repeat(np.arange(N_CLASSES), sims)
    shares = _matchup_shares(rng, ci, None)
    return shares.reshape(N_CLASSES, sims).mean(axis=1)


def compute_matrix(sims: int, seed: int, chunk_pairs: int = 400) -> np.ndarray:
    """Matrice 169x169 : équité heads-up de la classe ligne contre la classe colonne."""
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(N_CLASSES) for j in range(i + 1, N_CLASSES)]
    matrix = np.full((N_CLASSES, N_CLASSES), 0.5)
    for start in range(0, len(pairs), chunk_pairs):
        block = np.array(pairs[start:start + chunk_pairs], dtype=np.int64)
        ci = np.repeat(block[:, 0], sims)
        cj = np.repeat(block[:, 1], sims)
        eq = _matchup_shares(rng, ci, cj).reshape(len(block), sims).mean(axis=1)
        matrix[block[:, 0], block[:, 1]] = eq
        matrix[block[:, 1], block[:, 0]] = 1.0 - eq
    return matrix


@dataclass
class PreflopTable:
    """Équités préflop : classe contre main aléatoire, et classe contre classe."""

    vs_random: np.ndarray
    matrix: np.ndarray | None
    sims: int

    def __post_init__(self) -> None:
        strength = self.vs_random[COMBO_CLASS]
        order = np.argsort(-strength, kind="stable")
        pct = np.empty(N_COMBOS)
        pct[order] = (np.arange(N_COMBOS) + 1) / N_COMBOS
        # Toutes les combos d'une même classe partagent le percentile le plus faible de la classe.
        class_pct = np.zeros(N_CLASSES)
        np.maximum.at(class_pct, COMBO_CLASS, pct)
        self.class_percentile = class_pct
        self.combo_percentile = class_pct[COMBO_CLASS]

    def top_fraction(self, fraction: float) -> np.ndarray:
        """Poids (1326,) des meilleures mains représentant ``fraction`` des combos."""
        return (self.combo_percentile <= max(fraction, 1e-9) + 1e-12).astype(np.float64)

    def band(self, low: float, high: float) -> np.ndarray:
        """Mains entre les percentiles ``low`` (exclu) et ``high`` (inclus)."""
        p = self.combo_percentile
        return ((p > low + 1e-12) & (p <= high + 1e-12)).astype(np.float64)

    def equity_vs_class_weights(self, hero_class: int, class_w: np.ndarray) -> float:
        if self.matrix is None:
            raise RuntimeError("Matrice préflop absente : lancer `python -m pokerlab precompute`.")
        total = class_w.sum()
        if total <= 0:
            return float(self.vs_random[hero_class])
        return float(self.matrix[hero_class] @ class_w / total)


_TABLE: PreflopTable | None = None
_TABLE_KEY: tuple | None = None


def build_table(cfg: EquityConfig, with_matrix: bool = True, verbose: bool = False) -> PreflopTable:
    t0 = time.perf_counter()
    vs_random = compute_vs_random(max(cfg.preflop_sims_per_matchup * 5, 1000), cfg.seed)
    matrix = compute_matrix(cfg.preflop_sims_per_matchup, cfg.seed + 1) if with_matrix else None
    if verbose:
        print(f"Table préflop calculée en {time.perf_counter() - t0:.1f} s")
    return PreflopTable(vs_random=vs_random, matrix=matrix, sims=cfg.preflop_sims_per_matchup)


def save_table(table: PreflopTable, path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vs_random=table.vs_random, matrix=table.matrix, sims=table.sims, seed=seed)


def load_table_file(path: Path) -> tuple[PreflopTable, int] | None:
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return PreflopTable(vs_random=data["vs_random"], matrix=data["matrix"], sims=int(data["sims"])), int(data["seed"])


def get_preflop_table(cfg: EquityConfig, verbose: bool = False) -> PreflopTable:
    """Table préflop du processus : chargée depuis le disque, sinon calculée puis sauvée."""
    global _TABLE, _TABLE_KEY
    key = (str(cfg.table_path()), cfg.preflop_sims_per_matchup, cfg.seed)
    if _TABLE is not None and _TABLE_KEY == key:
        return _TABLE
    loaded = load_table_file(cfg.table_path())
    if loaded is not None and loaded[0].sims >= cfg.preflop_sims_per_matchup and loaded[1] == cfg.seed:
        table = loaded[0]
    else:
        table = build_table(cfg, verbose=verbose)
        save_table(table, cfg.table_path(), cfg.seed)
    _TABLE, _TABLE_KEY = table, key
    return table


def set_preflop_table(table: PreflopTable) -> None:
    """Force la table du processus (tests)."""
    global _TABLE, _TABLE_KEY
    _TABLE, _TABLE_KEY = table, None


def current_table() -> PreflopTable:
    if _TABLE is None:
        raise RuntimeError("Table préflop non initialisée.")
    return _TABLE
