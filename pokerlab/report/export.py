"""Exports bruts : une ligne par main, une ligne par décision (CSV + Parquet), historiques texte."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from pokerlab.cards.cards import CARD_STR
from pokerlab.cards.ranges import CLASS_NAMES
from pokerlab.engine.flags import LINE_LABELS, STREET_END_LABELS
from pokerlab.engine.types import KIND_LABELS, STREET_NAMES, Kind


def readable_hands(hands: pd.DataFrame) -> pd.DataFrame:
    """Ajoute cartes, tableau et libellés lisibles."""
    if hands.empty:
        return hands
    out = hands.copy()
    out["cartes"] = [CARD_STR[a] + CARD_STR[b] for a, b in zip(out["c1"], out["c2"])]
    out["classe"] = [CLASS_NAMES[int(c)] for c in out["hand_class"]]
    boards = []
    for row in out[["b0", "b1", "b2", "b3", "b4"]].itertuples(index=False):
        boards.append("".join(CARD_STR[int(c)] for c in row if c >= 0))
    out["tableau"] = boards
    out["fin_de_main"] = [STREET_END_LABELS[int(s)] for s in out["street_end"]]
    out["ligne_jouee"] = [LINE_LABELS[int(s)] for s in out["line"]]
    return out


def readable_decisions(dec: pd.DataFrame) -> pd.DataFrame:
    if dec.empty:
        return dec
    out = dec.copy()
    out["street_nom"] = [STREET_NAMES[int(s)] for s in out["street"]]
    out["action"] = [KIND_LABELS[Kind(int(k))] for k in out["kind"]]
    out["classe"] = [CLASS_NAMES[int(c)] for c in out["hand_class"]]
    return out


def write_tables(out_dir: Path, name: str, df: pd.DataFrame, parquet: bool = True) -> list[Path]:
    """Écrit ``name.csv`` et ``name.parquet``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=False)
    paths.append(csv_path)
    if parquet:
        pq = out_dir / f"{name}.parquet"
        try:
            df.to_parquet(pq, index=False)
            paths.append(pq)
        except (ImportError, ValueError) as exc:  # pragma: no cover - pyarrow absent
            print(f"Parquet non écrit ({exc}) ; CSV disponible.")
    return paths


def write_histories(path: Path, texts: Iterable[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(texts), encoding="utf-8")
    return path


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path
