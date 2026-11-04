"""Stats de joueur mesurées (pas déclarées) à partir des compteurs par main."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from pokerlab.engine.flags import FLAG_NAMES

STAT_LABELS = {
    "VPIP": "VPIP %",
    "PFR": "PFR %",
    "3bet": "3-bet %",
    "AF": "AF",
    "WTSD": "WTSD %",
    "W$SD": "W$SD %",
    "cbet": "C-bet %",
    "fold_cbet": "Fold to c-bet %",
}


def _ratio(a: float, b: float, pct: bool = True) -> float:
    if b <= 0:
        return float("nan")
    return 100.0 * a / b if pct else a / b


def stats_from_flags(flags: Sequence[int] | Mapping[str, float]) -> dict[str, float]:
    """VPIP, PFR, 3-bet, AF, WTSD, W$SD, c-bet, fold to c-bet."""
    if isinstance(flags, Mapping):
        f = {k: float(flags.get(k, 0)) for k in FLAG_NAMES}
    else:
        f = {k: float(v) for k, v in zip(FLAG_NAMES, flags)}
    return {
        "hands": f["hands"],
        "VPIP": _ratio(f["vpip"], f["hands"]),
        "PFR": _ratio(f["pfr"], f["hands"]),
        "3bet": _ratio(f["tb"], f["tb_opp"]),
        "AF": _ratio(f["post_aggr"], f["post_calls"], pct=False),
        "WTSD": _ratio(f["wtsd"], f["saw_flop"]),
        "W$SD": _ratio(f["wsd"], f["wtsd"]),
        "cbet": _ratio(f["cbet"], f["cbet_opp"]),
        "fold_cbet": _ratio(f["fcb"], f["fcb_opp"]),
        "unopened_entry": _ratio(f["unop_vpip"], f["unop_opp"]),
        "unopened_raise": _ratio(f["unop_pfr"], f["unop_opp"]),
    }


def stats_from_hands(hands: pd.DataFrame) -> dict[str, float]:
    sums = {k: float(hands["f_" + k].sum()) for k in FLAG_NAMES}
    return stats_from_flags(sums)


def stats_table(flags_by_name: Mapping[str, Sequence[int]]) -> pd.DataFrame:
    rows = []
    for name, flags in flags_by_name.items():
        row = {"joueur": name}
        row.update(stats_from_flags(flags))
        rows.append(row)
    return pd.DataFrame(rows)
