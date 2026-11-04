"""Statistiques : bb/100, intervalles de confiance, tests appariés, correction de Holm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats as sps


def z_value(confidence: float) -> float:
    return float(sps.norm.ppf(0.5 + confidence / 2))


@dataclass(frozen=True)
class WinrateSummary:
    """Résumé d'un taux de gain. ``unit`` = nombre de mains par unité appariée (6 en duplicate)."""

    hands: int
    units: int
    bb100: float
    ci_half: float
    sd_hand_bb100: float
    sd_effective_bb100: float
    hands_needed: float

    @property
    def low(self) -> float:
        return self.bb100 - self.ci_half

    @property
    def high(self) -> float:
        return self.bb100 + self.ci_half


def summarize(unit_means: np.ndarray, hand_values: np.ndarray, hands_per_unit: float, confidence: float = 0.95) -> WinrateSummary:
    """Taux de gain à partir des moyennes par unité (deck) et des résultats par main.

    - ``sd_hand_bb100`` : écart type brut sur 100 mains (σ main x 10) ;
    - ``sd_effective_bb100`` : écart type équivalent après réduction de variance ;
    - ``hands_needed`` : mains pour que l'IC exclue 0 au taux observé.
    """
    unit_means = np.asarray(unit_means, dtype=float)
    hand_values = np.asarray(hand_values, dtype=float)
    k = unit_means.size
    z = z_value(confidence)
    mean = float(unit_means.mean()) if k else float("nan")
    sd_unit = float(unit_means.std(ddof=1)) if k > 1 else float("nan")
    ci = z * sd_unit / math.sqrt(k) * 100 if k > 1 else float("inf")
    sd_hand = float(hand_values.std(ddof=1)) if hand_values.size > 1 else float("nan")
    sd_eff = sd_unit * math.sqrt(hands_per_unit) * 10 if k > 1 else float("nan")
    if k > 1 and abs(mean) > 1e-12 and not math.isnan(sd_unit):
        needed = (z * sd_unit / abs(mean)) ** 2 * hands_per_unit
    else:
        needed = float("inf")
    return WinrateSummary(int(hand_values.size), k, mean * 100, ci, sd_hand * 10, sd_eff, needed)


def block_bootstrap_mean(values: np.ndarray, block: int, resamples: int, seed: int) -> np.ndarray:
    """Moyennes bootstrap par blocs mobiles (préserve l'autocorrélation)."""
    values = np.asarray(values, dtype=float)
    n = values.size
    block = max(1, min(block, n))
    n_blocks = -(-n // block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(resamples, -1)[:, :n]
    return values[idx].mean(axis=1)


@dataclass(frozen=True)
class PairedTest:
    """Comparaison appariée A - B (en bb/100)."""

    a: str
    b: str
    diff_bb100: float
    ci_half: float
    p_value: float
    p_holm: float
    n_units: int

    @property
    def significant(self) -> bool:
        return self.p_holm < 0.05

    def verdict(self) -> str:
        winner, loser = (self.a, self.b) if self.diff_bb100 >= 0 else (self.b, self.a)
        state = "significatif" if self.significant else "pas encore significatif"
        return f"{winner} bat {loser} : {state} ({abs(self.diff_bb100):.1f} ± {self.ci_half:.1f} bb/100, p Holm = {self.p_holm:.3g})"


def paired_test(
    diffs: np.ndarray,
    method: str = "t",
    confidence: float = 0.95,
    block: int = 50,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(moyenne bb/100, demi-IC bb/100, p-valeur bilatérale) des différences appariées par deck."""
    diffs = np.asarray(diffs, dtype=float)
    n = diffs.size
    if n < 2:
        return float(diffs.mean() * 100) if n else 0.0, float("inf"), 1.0
    mean = float(diffs.mean())
    if method == "bootstrap":
        boot = block_bootstrap_mean(diffs, block, resamples, seed)
        lo, hi = np.quantile(boot, [(1 - confidence) / 2, 0.5 + confidence / 2])
        centered = boot - mean
        p = float((np.abs(centered) >= abs(mean)).mean())
        return mean * 100, float(hi - lo) / 2 * 100, max(p, 1.0 / resamples)
    sd = float(diffs.std(ddof=1))
    if sd == 0:
        return mean * 100, 0.0, 1.0 if mean == 0 else 0.0
    t = mean / (sd / math.sqrt(n))
    p = float(2 * sps.t.sf(abs(t), df=n - 1))
    half = float(sps.t.ppf(0.5 + confidence / 2, df=n - 1)) * sd / math.sqrt(n)
    return mean * 100, half * 100, p


def holm(pvalues: Sequence[float]) -> list[float]:
    """Correction de Holm-Bonferroni (p-valeurs ajustées, monotones)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[i])
        running = max(running, value)
        adjusted[i] = running
    return adjusted


def max_drawdown(cumulative: np.ndarray) -> tuple[float, int]:
    """(drawdown maximal, plus longue durée sous le dernier sommet en mains)."""
    if cumulative.size == 0:
        return 0.0, 0
    peak = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    dd = peak - cumulative
    longest = 0
    current = 0
    for below in dd > 1e-9:
        current = current + 1 if below else 0
        longest = max(longest, current)
    return float(dd.max()), int(longest)


def longest_losing_streak(values: np.ndarray) -> int:
    """Plus longue suite de résultats négatifs consécutifs."""
    best = cur = 0
    for v in values:
        cur = cur + 1 if v < 0 else 0
        best = max(best, cur)
    return best
