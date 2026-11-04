"""Stockage des mains réelles importées (Parquet) et import depuis des fichiers PokerStars."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pokerlab.config import PROJECT_ROOT, LabConfig
from pokerlab.journal.population import merge_opponents
from pokerlab.journal.pokerstars import assign_sessions, import_texts, read_files
from pokerlab.strategies.registry import build_strategy, get_context, resolve_spec


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_opponents(folder: str | Path) -> pd.DataFrame:
    """Compteurs par adversaire et par main (vide si rien n'a été importé)."""
    pq = resolve(folder) / "adversaires.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.DataFrame()


def load_hands(folder: str | Path) -> pd.DataFrame:
    """Mains importées d'un dossier (vide si rien n'a été importé)."""
    pq = resolve(folder) / "mains.parquet"
    return pd.read_parquet(pq) if pq.exists() else pd.DataFrame()


def import_paths(paths: Iterable[str | Path], lab: LabConfig, strategy: str, dest: str | Path, hero: str | None = None,
                 gap_minutes: float = 30.0, source: str = "reel") -> dict[str, Any]:
    """Importe des historiques PokerStars dans ``dest`` (fusion, sans doublon par numéro de main)."""
    get_context(lab)
    strat = build_strategy(resolve_spec(strategy), lab)
    texts = read_files(paths)
    opp_rows: list[dict[str, Any]] = []
    rows, skipped = import_texts(texts, strat, lab.game.chips_per_bb, hero, lab.raw.get("analysis", {}).get("value_min_level", "middle_pair"),
                                 opponents=opp_rows)
    dest = resolve(dest)
    dest.mkdir(parents=True, exist_ok=True)
    opp_path = dest / "adversaires.parquet"
    opponents = merge_opponents(pd.read_parquet(opp_path) if opp_path.exists() else pd.DataFrame(), opp_rows)
    if not opponents.empty:
        opponents.to_parquet(opp_path, index=False)
    old = load_hands(dest)
    new = pd.DataFrame(rows)
    if not new.empty:
        new["strategie"] = strategy
        new["source"] = source
    merged = pd.concat([old, new], ignore_index=True) if not old.empty else new
    added = len(new)
    if not merged.empty:
        merged = merged.drop_duplicates("hand_id", keep="last")
        records = merged.to_dict("records")
        assign_sessions(records, gap_minutes)
        merged = pd.DataFrame(records)
        merged.to_parquet(dest / "mains.parquet", index=False)
    log = dest / "imports.json"
    history = json.loads(log.read_text(encoding="utf-8")) if log.exists() else []
    history.append({"date": datetime.now().isoformat(timespec="seconds"), "fichiers": [str(p) for p in paths],
                    "mains_lues": added, "mains_ignorees": skipped, "total": int(len(merged)), "strategie": strategy})
    log.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"lues": added, "ignorees": skipped, "total": int(len(merged)), "dossier": str(dest),
            "adversaires": int(opponents["joueur"].nunique()) if not opponents.empty else 0}


def import_drop_folder(folder: str | Path, lab: LabConfig, strategy: str, dest: str | Path, hero: str | None = None,
                       gap_minutes: float = 30.0) -> dict[str, Any] | None:
    """Importe les nouveaux .txt déposés dans ``folder`` puis les range dans ``folder/importes``."""
    folder = resolve(folder)
    files = sorted(folder.glob("*.txt")) if folder.exists() else []
    if not files:
        return None
    res = import_paths(files, lab, strategy, dest, hero, gap_minutes)
    done = folder / "importes"
    done.mkdir(exist_ok=True)
    for f in files:
        shutil.move(str(f), str(done / f.name))
    return res
