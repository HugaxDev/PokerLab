"""Commande ``bench`` : mesure du débit réel (mains/s), mono-processus et multi-processus."""

from __future__ import annotations

import os
import platform
import time
from typing import Any

from pokerlab.experiments.base import ExperimentContext, composition_dict
from pokerlab.experiments.common import DuplicateTask, run_duplicate_task, run_tasks, split_range
from pokerlab.report.export import write_json
from pokerlab.strategies.registry import get_context, resolve_spec


def _tasks(ctx: ExperimentContext, specs: list[str], comp: dict, hands: int, parts: int) -> list[DuplicateTask]:
    n = ctx.lab.game.players
    decks = max(1, hands // (n * len(specs)))
    tasks = []
    for s in specs:
        for a, b in split_range(0, decks, parts):
            tasks.append(DuplicateTask(ctx.lab.raw, resolve_spec(s), comp, a, b, {"strategy": s}, record_decisions=False))
    return tasks


def _measure(ctx: ExperimentContext, tasks: list[DuplicateTask], workers: int) -> tuple[int, float]:
    t0 = time.perf_counter()
    outs = run_tasks(run_duplicate_task, tasks, workers, ctx.lab.mp_start_method)
    return sum(o.hands_played for o in outs), time.perf_counter() - t0


def run(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    get_context(ctx.lab)
    comp = composition_dict(cfg.get("composition", "micro_play_money"))
    specs = list(cfg.get("strategies", ["serre"]))
    hands = int(cfg.get("hands", 600))
    single = _tasks(ctx, specs, comp, hands, 1)
    h1, t1 = _measure(ctx, single, 1)
    warm = _tasks(ctx, specs[:1], comp, ctx.lab.game.players * ctx.workers, ctx.workers)
    hw, _ = _measure(ctx, warm, ctx.workers)  # démarrage des workers, exclu de la mesure
    multi = _tasks(ctx, specs, comp, hands, ctx.workers)
    h2, t2 = _measure(ctx, multi, ctx.workers)
    eq = _tasks(ctx, ["equity_bot"], comp, max(60, hands // 10), 1)
    h3, t3 = _measure(ctx, eq, 1)
    ctx.hands_played = h1 + hw + h2 + h3
    result = {
        "machine": platform.platform(),
        "processeur": platform.processor() or platform.machine(),
        "coeurs": os.cpu_count(),
        "workers": ctx.workers,
        "strategies": specs,
        "mono_processus": {"mains": h1, "secondes": round(t1, 2), "mains_par_s": round(h1 / t1, 1)},
        "multi_processus": {"mains": h2, "secondes": round(t2, 2), "mains_par_s": round(h2 / t2, 1),
                            "note": "workers déjà démarrés (échauffement exclu)"},
        "equity_bot_mono": {"mains": h3, "secondes": round(t3, 2), "mains_par_s": round(h3 / t3, 1)},
    }
    write_json(ctx.out_dir / "bench.json", result)
    ctx.log(f"mono-processus : {result['mono_processus']['mains_par_s']} mains/s ; "
            f"{ctx.workers} workers : {result['multi_processus']['mains_par_s']} mains/s ; "
            f"equity_bot (mono) : {result['equity_bot_mono']['mains_par_s']} mains/s")
    return result
