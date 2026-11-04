"""Indicateurs mesurés par siège et par main (VPIP, PFR, 3-bet, c-bet...) et ligne jouée."""

from __future__ import annotations

from pokerlab.engine.hand_runner import HandResult
from pokerlab.engine.types import Kind

FLAG_NAMES = (
    "hands",
    "vpip",
    "pfr",
    "tb_opp",
    "tb",
    "cbet_opp",
    "cbet",
    "fcb_opp",
    "fcb",
    "saw_flop",
    "wtsd",
    "wsd",
    "post_aggr",
    "post_calls",
    "unop_opp",
    "unop_vpip",
    "unop_pfr",
)
N_FLAGS = len(FLAG_NAMES)
AGG = (Kind.BET, Kind.RAISE)
STREET_END_LABELS = ("préflop", "flop", "turn", "river", "showdown")
LINE_LABELS = ("préflop", "check/abandon", "paie", "value bet", "semi-bluff", "bluff")


def seat_flags(hr: HandResult, seat: int) -> list[int]:
    """Compteurs d'une main pour un siège (dans l'ordre de ``FLAG_NAMES``)."""
    f = [0] * N_FLAGS
    f[0] = 1
    raises = 0
    aggressor = None
    fold_street = None
    for a in hr.history:
        if a.kind == Kind.POST:
            continue
        if a.seat == seat and a.kind == Kind.FOLD and fold_street is None:
            fold_street = a.street
        if a.street == 0:
            if a.seat == seat:
                if raises == 0 and not f[14]:
                    f[14] = 1
                    f[15] = int(a.kind in (Kind.CALL, Kind.BET, Kind.RAISE))
                    f[16] = int(a.kind in AGG)
                if a.kind in (Kind.CALL, Kind.BET, Kind.RAISE):
                    f[1] = 1
                if a.kind in AGG:
                    f[2] = 1
                if raises == 1 and aggressor != seat and not f[3]:
                    f[3] = 1
                    if a.kind in AGG:
                        f[4] = 1
            if a.kind in AGG:
                raises += 1
                aggressor = a.seat
        else:
            if a.seat == seat:
                if a.kind in AGG:
                    f[12] += 1
                elif a.kind == Kind.CALL:
                    f[13] += 1
    saw_flop = len(hr.board) >= 3 and (fold_street is None or fold_street >= 1)
    f[9] = int(saw_flop)
    if saw_flop and aggressor is not None:
        flop = [a for a in hr.history if a.street == 1]
        if seat == aggressor:
            for a in flop:
                if a.kind in AGG and a.seat != seat:
                    break
                if a.seat == seat:
                    f[5] = 1
                    f[6] = int(a.kind == Kind.BET)
                    break
        else:
            cbet_seen = False
            for a in flop:
                if not cbet_seen:
                    if a.kind in AGG:
                        if a.seat != aggressor:
                            break
                        cbet_seen = True
                    continue
                if a.kind == Kind.RAISE and a.seat != seat:
                    break
                if a.seat == seat:
                    f[7] = 1
                    f[8] = int(a.kind == Kind.FOLD)
                    break
    if hr.showdown and not hr.folded[seat]:
        f[10] = 1
        f[11] = int(hr.payoffs[seat] > 0)
    return f


def street_end(hr: HandResult, seat: int) -> int:
    """Street où le siège a quitté la main (4 = showdown)."""
    for a in hr.history:
        if a.seat == seat and a.kind == Kind.FOLD:
            return a.street
    if hr.showdown:
        return 4
    return max((a.street for a in hr.history), default=0)


def line_of(hr: HandResult, seat: int, value_min_level: int) -> int:
    """Ligne jouée (index dans ``LINE_LABELS``) : dernière action volontaire postflop."""
    post = [d for d in hr.decisions if d.seat == seat and d.street >= 1]
    if not post:
        return 0
    voluntary = [d for d in post if d.kind in (Kind.CALL, Kind.BET, Kind.RAISE)]
    if not voluntary:
        return 1
    last = voluntary[-1]
    if last.kind == Kind.CALL:
        return 2
    if last.level >= value_min_level:
        return 3
    return 4 if last.draw > 0 else 5
