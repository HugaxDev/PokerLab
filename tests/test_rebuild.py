"""Rapports régénérés sans re-simuler (compare, sweep) : mêmes chiffres que la campagne."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pokerlab.config import PROJECT_ROOT
from pokerlab.experiments.base import ExperimentContext


def test_sweep_rebuild_keeps_grid(lab, tmp_path: Path) -> None:
    from pokerlab.experiments.variants import rebuild_sweep_report, run_sweep

    cfg = {"strategy": "joueur", "composition": "micro_play_money", "decks": 1,
           "params": [{"path": "garde_fou_seuil", "label": "seuil", "values": [0.5, "tapis"]},
                      {"path": "garde_fou_force", "label": "force_min", "values": ["deux_paires"]}]}
    ctx = ExperimentContext.create("sweep", lab, cfg, tmp_path, 1, log=lambda m: None)
    run_sweep(ctx)
    before = json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))
    ctx2 = ExperimentContext.create("sweep", lab, {}, tmp_path, 1, log=lambda m: None)
    rebuild_sweep_report(ctx2)
    after = json.loads((tmp_path / "resume.json").read_text(encoding="utf-8"))
    assert after["grille"] == before["grille"] and after["mains_jouees"] == before["mains_jouees"]
    html = (tmp_path / "rapport.html").read_text(encoding="utf-8")
    assert '"type":"category"' in html          # « tapis » reste une catégorie, pas un nombre manquant


def test_compare_rebuild_from_saved_data(lab, tmp_path: Path) -> None:
    from pokerlab.experiments.compare import rebuild_report

    src = PROJECT_ROOT / "out" / "smoke" / "compare"
    if not (src / "mains.parquet").exists():
        pytest.skip("out/smoke absent : lance python -m pokerlab smoke")
    shutil.copytree(src, tmp_path / "compare")
    ctx = ExperimentContext.create("compare", lab, {}, tmp_path / "compare", 1, log=lambda m: None)
    res = rebuild_report(ctx)
    assert res["report"].verdict[0].startswith(("Gagnante", "PAS DE GAGNANTE"))
    assert (tmp_path / "compare" / "rapport.html").stat().st_size > 100_000
