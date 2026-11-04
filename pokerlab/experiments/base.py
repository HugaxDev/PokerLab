"""Contexte d'expérience : config, dossier de sortie, workers, export final."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from pokerlab.analysis.stats import PairedTest, holm, paired_test
from pokerlab.config import PROJECT_ROOT, LabConfig, load_table
from pokerlab.engine.hand_runner import COUNTERS
from pokerlab.experiments.common import worker_count
from pokerlab.report.export import readable_decisions, readable_hands, write_histories, write_json, write_tables
from pokerlab.report.html import Report


@dataclass
class ExperimentContext:
    """Ce dont une expérience a besoin pour tourner et écrire ses sorties."""

    name: str
    lab: LabConfig
    cfg: dict[str, Any]
    out_dir: Path
    workers: int
    log: Callable[[str], None] = print
    started: float = field(default_factory=time.perf_counter)
    hands_played: int = 0

    @staticmethod
    def create(name: str, lab: LabConfig, cfg: dict[str, Any], out: str | Path | None = None, workers: int | None = None,
               log: Callable[[str], None] = print) -> "ExperimentContext":
        out_dir = Path(out) if out else PROJECT_ROOT / "out" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        return ExperimentContext(name, lab, cfg, out_dir, worker_count(lab, workers), log)

    @property
    def confidence(self) -> float:
        return float(self.lab.raw.get("analysis", {}).get("confidence", 0.95))

    @property
    def analysis(self) -> dict[str, Any]:
        return self.lab.raw.get("analysis", {})

    def elapsed(self) -> float:
        return time.perf_counter() - self.started


def composition_dict(name: str) -> dict[str, Any]:
    return load_table(name)


def pairwise_tests(
    deck_means: dict[str, pd.Series],
    pairs: Sequence[tuple[str, str]] | None = None,
    method: str = "t",
    confidence: float = 0.95,
    block: int = 50,
    resamples: int = 2000,
    seed: int = 0,
    labels: dict[str, str] | None = None,
) -> list[PairedTest]:
    """Tests appariés (différences par deck) entre stratégies, sans correction (Holm à part)."""
    labels = labels or {}
    names = list(deck_means)
    pairs = list(pairs) if pairs is not None else list(combinations(names, 2))
    raw = []
    for a, b in pairs:
        joined = pd.concat([deck_means[a].rename("a"), deck_means[b].rename("b")], axis=1, join="inner")
        diff = (joined["a"] - joined["b"]).to_numpy(dtype=float)
        mean, half, p = paired_test(diff, method, confidence, block, resamples, seed)
        raw.append((a, b, mean, half, p, diff.size))
    return [PairedTest(labels.get(a, a), labels.get(b, b), m, h, p, 1.0, n) for a, b, m, h, p, n in raw]


def apply_holm(tests: list[PairedTest]) -> list[PairedTest]:
    adjusted = holm([t.p_value for t in tests])
    return [PairedTest(t.a, t.b, t.diff_bb100, t.ci_half, t.p_value, p, t.n_units) for t, p in zip(tests, adjusted)]


def save_outputs(
    ctx: ExperimentContext,
    report: Report,
    hands: pd.DataFrame | None = None,
    decisions: pd.DataFrame | None = None,
    histories: Sequence[str] = (),
    summary: dict[str, Any] | None = None,
    prefix: str = "",
) -> dict[str, Path]:
    """Écrit rapport HTML, exports CSV/Parquet, historiques et résumé JSON."""
    paths: dict[str, Path] = {}
    if hands is not None and not hands.empty:
        write_tables(ctx.out_dir, prefix + "mains", readable_hands(hands))
        paths["mains"] = ctx.out_dir / f"{prefix}mains.csv"
    if decisions is not None and not decisions.empty:
        write_tables(ctx.out_dir, prefix + "decisions", readable_decisions(decisions))
        paths["decisions"] = ctx.out_dir / f"{prefix}decisions.csv"
    if histories:
        paths["historiques"] = write_histories(ctx.out_dir / f"{prefix}historiques.txt", histories)
    data = dict(summary or {})
    data.setdefault("experience", ctx.name)
    data.setdefault("config", ctx.cfg)
    data.setdefault("mains_jouees", ctx.hands_played)
    data.setdefault("duree_s", round(ctx.elapsed(), 1))
    paths["resume"] = write_json(ctx.out_dir / f"{prefix}resume.json", data)
    paths["rapport"] = report.save(ctx.out_dir / f"{prefix}rapport.html")
    return paths


def hands_counter() -> int:
    return COUNTERS["hands"]


def finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]
