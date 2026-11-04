"""Import d'historiques de mains PokerStars (format texte anglais, Hold'em No Limit).

Chaque main est relue action par action ; à chaque décision du héros on reconstruit
l'``Observation`` du moteur et on demande à la stratégie suivie ce qu'elle aurait joué :
c'est ce qui permet de mesurer le respect de la fiche, au préflop comme après le flop.
Les montants sont convertis en jetons moteur (1 bb = ``chips_per_bb``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from pokerlab.cards.cards import CARD_STR, parse_card
from pokerlab.cards.ranges import hole_class
from pokerlab.engine.flags import FLAG_NAMES, line_of, seat_flags, street_end
from pokerlab.engine.hand_runner import DecisionRecord, HandResult, _facing
from pokerlab.features.board import board_texture
from pokerlab.engine.types import ActionRecord, Kind, Legal, Observation, position_names
from pokerlab.features.hand import level_of, read_hand
from pokerlab.rng import derive_seed

HEADER = re.compile(
    r"^(?:PokerStars|PokerLab)(?: Zoom| Home Game)? Hand #(\w+):\s+Hold'em No Limit \(([^)]*)\)(?:\s+-\s+(\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2}))?"
)
TABLE = re.compile(r"^Table '([^']*)' (\d+)-max (?:\(.*\) )?Seat #(\d+) is the button")
SEAT = re.compile(r"^Seat (\d+): (.+?) \(([^ ]+) in chips")
POST = re.compile(r"^(.+?): posts (small blind|big blind|small & big blinds|the ante) ([^ ]+)")
DEALT = re.compile(r"^Dealt to (.+?) \[(\S\S) (\S\S)\]")
STREET = re.compile(r"^\*\*\* (FLOP|TURN|RIVER) \*\*\* (?:\[([^\]]*)\])?(?: \[([^\]]*)\])?")
ACTION = re.compile(r"^(.+?): (folds|checks|calls ([^ ]+)|bets ([^ ]+)|raises ([^ ]+) to ([^ ]+))( and is all-in)?\s*$")
UNCALLED = re.compile(r"^Uncalled bet \(([^)]+)\) returned to (.+)$")
COLLECTED = re.compile(r"^(.+?) collected ([^ ]+) from (?:side |main )?pot")
SHOWS = re.compile(r"^(.+?): shows \[(\S\S) (\S\S)\]")
STREETS = {"FLOP": 1, "TURN": 2, "RIVER": 3}
KIND_OF = {"folds": Kind.FOLD, "checks": Kind.CHECK, "calls": Kind.CALL, "bets": Kind.BET, "raises": Kind.RAISE}


def money(text: str) -> float:
    """'$0.06', '€1,250.50', '600' -> nombre."""
    return float(re.sub(r"[^\d.]", "", text.replace(",", "")) or 0)


@dataclass
class ParsedHand:
    """Main relue depuis le texte."""

    hand_id: str
    date: datetime | None
    table: str
    small_blind: float
    big_blind: float
    currency: str
    button: int
    seats: dict[int, tuple[str, float]]
    hero: str | None
    hero_cards: tuple[int, int] | None
    lines: list[tuple[str, Any]] = field(default_factory=list)
    raw: str = ""


def split_hands(text: str) -> list[str]:
    """Découpe un fichier en mains (lignes commençant par # ignorées)."""
    lines = [ln.rstrip("\r") for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    blocks: list[list[str]] = []
    for ln in lines:
        if HEADER.match(ln.strip().lstrip("﻿")):
            blocks.append([ln.strip().lstrip("﻿")])
        elif blocks:
            blocks[-1].append(ln)
    return ["\n".join(b).strip() for b in blocks if b]


def parse_hand(text: str, hero: str | None = None) -> ParsedHand | None:
    """Relit une main. ``None`` si le format n'est pas reconnu (variante, omaha...)."""
    lines = text.splitlines()
    m = HEADER.match(lines[0].strip())
    if not m:
        return None
    stakes = m.group(2)
    nums = [money(x) for x in re.findall(r"[^/ ]+", stakes.split(" ")[0])]
    if len(nums) < 2:
        return None
    currency = "€" if "€" in stakes else ("$" if "$" in stakes else ("£" if "£" in stakes else ""))
    date = datetime.strptime(m.group(3), "%Y/%m/%d %H:%M:%S") if m.group(3) else None
    hand = ParsedHand(m.group(1), date, "", nums[0], nums[1], currency, 0, {}, None, None, raw=text)
    dealt: list[tuple[str, tuple[int, int]]] = []
    for ln in lines[1:]:
        ln = ln.strip()
        if (t := TABLE.match(ln)):
            hand.table, hand.button = t.group(1), int(t.group(3))
        elif (t := SEAT.match(ln)) and "*** SUMMARY ***" not in [x[0] for x in hand.lines]:
            if not hand.lines:
                hand.seats[int(t.group(1))] = (t.group(2), money(t.group(3)))
        elif (t := POST.match(ln)):
            hand.lines.append(("post", (t.group(1), t.group(2), money(t.group(3)))))
        elif (t := DEALT.match(ln)):
            dealt.append((t.group(1), (parse_card(t.group(2)), parse_card(t.group(3)))))
        elif (t := STREET.match(ln)):
            cards = (t.group(3) or t.group(2) or "").split()
            hand.lines.append(("street", (STREETS[t.group(1)], [parse_card(c) for c in cards])))
        elif (t := UNCALLED.match(ln)):
            hand.lines.append(("uncalled", (t.group(2).strip(), money(t.group(1)))))
        elif (t := COLLECTED.match(ln)):
            hand.lines.append(("collected", (t.group(1).strip(), money(t.group(2)))))
        elif (t := SHOWS.match(ln)):
            hand.lines.append(("shows", (t.group(1), (parse_card(t.group(2)), parse_card(t.group(3))))))
        elif ln.startswith("*** SHOW DOWN ***"):
            hand.lines.append(("showdown", None))
        elif ln.startswith("*** SUMMARY ***"):
            hand.lines.append(("summary", None))
            break
        elif (t := ACTION.match(ln)):
            verb = t.group(2).split(" ")[0]
            amount = to = 0.0
            if verb == "calls":
                amount = money(t.group(3))
            elif verb == "bets":
                amount = to = money(t.group(4))
            elif verb == "raises":
                amount, to = money(t.group(5)), money(t.group(6))
            hand.lines.append(("action", (t.group(1), KIND_OF[verb], amount, to, bool(t.group(7)))))
    if hero is not None:
        hand.hero = hero
        hand.hero_cards = next((c for n, c in dealt if n == hero), None)
    elif dealt:
        hand.hero, hand.hero_cards = dealt[0]
    return hand


@dataclass
class Replayed:
    """Main rejouée : résultat moteur, décisions du héros et données de replay."""

    result: HandResult
    hero_index: int
    decisions: list[dict[str, Any]]
    replay: dict[str, Any]


def replay_hand(hand: ParsedHand, strategy: Any | None, chips_per_bb: int = 100) -> Replayed | None:
    """Rejoue la main dans les conventions du moteur et compare chaque décision du héros à la stratégie."""
    if hand.hero is None or hand.hero_cards is None or not hand.seats:
        return None
    seats = sorted(hand.seats)
    n = len(seats)
    if n < 2:
        return None
    if n == 2 and hand.button in seats:
        order = [hand.button, next(s for s in seats if s != hand.button)]  # tête-à-tête : le bouton est petite blind
    else:
        order = [s for s in seats if s > hand.button] + [s for s in seats if s <= hand.button]
    names = [hand.seats[s][0] for s in order]
    index = {name: i for i, name in enumerate(names)}
    if hand.hero not in index:
        return None
    hero = index[hand.hero]
    positions = position_names(n)
    scale = chips_per_bb / hand.big_blind

    def chips(x: float) -> int:
        return int(round(x * scale))

    start = [chips(hand.seats[s][1]) for s in order]
    stacks = list(start)
    bets = [0] * n
    folded = [False] * n
    board: list[int] = []
    history: list[ActionRecord] = []
    street = 0
    last_inc = chips_per_bb
    collected = [0] * n
    shown = [False] * n
    holes: list[tuple[int, int]] = [(-1, -1)] * n
    holes[hero] = hand.hero_cards
    showdown_seen = False
    decisions: list[dict[str, Any]] = []
    records: list[DecisionRecord] = []
    global_idx = 0
    sb_chips, bb_chips = chips(hand.small_blind), chips_per_bb
    steps: list[dict[str, Any]] = []
    situation: str | None = None

    def pot() -> int:
        return sum(start[i] - stacks[i] for i in range(n))

    for kind, data in hand.lines:
        if kind == "post":
            name, what, amount = data
            if name not in index:
                continue
            i = index[name]
            add = chips(amount)
            stacks[i] -= add
            bets[i] += add
            history.append(ActionRecord(0, i, Kind.POST, add, bets[i], stacks[i] == 0, pot() - add))
            steps.append({"street": 0, "seat": i, "text": f"poste {amount:g}", "pot": pot() / chips_per_bb, "board": []})
        elif kind == "street":
            street, cards = data
            board.extend(cards)
            bets = [0] * n
            last_inc = chips_per_bb
            steps.append({"street": street, "seat": None, "text": "cartes : " + " ".join(CARD_STR[c] for c in cards),
                          "pot": pot() / chips_per_bb, "board": list(board)})
        elif kind == "action":
            name, k, amount, to, all_in = data
            if name not in index:
                continue
            i = index[name]
            max_bet = max(bets)
            to_call = min(max_bet - bets[i], stacks[i])
            rec_action = None
            if i == hero:
                can_raise = stacks[i] > to_call
                legal = Legal(can_fold=to_call > 0, call_amount=to_call, can_raise=can_raise,
                              min_raise_to=min(max_bet + last_inc, stacks[i] + bets[i]) if can_raise else 0,
                              max_raise_to=stacks[i] + bets[i] if can_raise else 0)
                obs = Observation(
                    hand_id=0, seat=i, position=positions[i], positions=positions, hole=tuple(hand.hero_cards), board=tuple(board),
                    street=street, stacks=tuple(stacks), bets=tuple(bets), contributions=tuple(start[j] - stacks[j] for j in range(n)),
                    starting_stacks=tuple(start), folded=tuple(folded), pot=pot(), legal=legal, history=tuple(history),
                    big_blind=bb_chips, small_blind=sb_chips, decision_seed=derive_seed("reel", hand.hand_id, global_idx),
                    decision_index=global_idx,
                )
                if strategy is not None:
                    act = strategy.decide(obs)
                    rec_action = {"kind": act.kind, "amount_bb": act.amount / chips_per_bb if act.kind == "raise" else 0.0}
                played = {Kind.FOLD: "fold", Kind.CHECK: "call", Kind.CALL: "call"}.get(k, "raise")
                hr_read = read_hand(hand.hero_cards, board) if street > 0 else None
                level = hr_read.level if hr_read else -1
                draw = (2 if hr_read.strong_draw else (1 if hr_read.any_draw else 0)) if hr_read else 0
                decisions.append({
                    "idx": global_idx, "street": street, "position": positions[i], "to_call_bb": to_call / chips_per_bb,
                    "pot_bb": pot() / chips_per_bb, "kind": int(k), "level": level, "draw": draw,
                    "invested_bb": (start[i] - stacks[i]) / chips_per_bb, "hand_class": hole_class(hand.hero_cards),
                    "board_label": board_texture(board).label if street > 0 else "préflop",
                    "played": played, "recommended": rec_action["kind"] if rec_action else None,
                    "match": (rec_action is None) or rec_action["kind"] == played,
                    "recommended_amount_bb": rec_action["amount_bb"] if rec_action else 0.0,
                })
                facing = _facing(history, street, to_call, stacks[i])
                if street == 0 and situation is None:
                    situation = "vs_raise" if any(a.street == 0 and a.kind == Kind.RAISE for a in history) else "unopened"
                decisions[-1]["facing"] = facing
                records.append(DecisionRecord(global_idx, i, street, positions[i], to_call, pot(), stacks[i], sum(1 for j in range(n) if not folded[j] and j != i),
                                              facing, k, 0, 0, "", start[i] - stacks[i], level, draw,
                                              board_texture(board).label if street > 0 else "préflop", legal))
            if k == Kind.FOLD:
                folded[i] = True
                added, new_to = 0, bets[i]
            elif k == Kind.CHECK:
                added, new_to = 0, bets[i]
            elif k == Kind.CALL:
                added = min(chips(amount), stacks[i])
                new_to = bets[i] + added
            else:
                new_to = chips(to)
                added = new_to - bets[i]
                last_inc = max(last_inc, new_to - max_bet)
            stacks[i] -= added
            bets[i] = new_to
            history.append(ActionRecord(street, i, k, added, new_to, all_in or stacks[i] == 0, pot() - added))
            text = {Kind.FOLD: "se couche", Kind.CHECK: "check", Kind.CALL: f"paie {added / chips_per_bb:.1f} bb",
                    Kind.BET: f"mise {new_to / chips_per_bb:.1f} bb", Kind.RAISE: f"relance à {new_to / chips_per_bb:.1f} bb"}[k]
            step = {"street": street, "seat": i, "text": text + (" (tapis)" if all_in else ""), "pot": pot() / chips_per_bb, "board": list(board)}
            if i == hero and rec_action is not None:
                rk = rec_action["kind"]
                step["fiche"] = {"fold": "se coucher", "call": "payer / checker", "raise": f"relancer ({rec_action['amount_bb']:.1f} bb)"}[rk]
                step["respect"] = decisions[-1]["match"]
            steps.append(step)
            global_idx += 1
        elif kind == "uncalled":
            name, amount = data
            if name in index:
                stacks[index[name]] += chips(amount)
        elif kind == "collected":
            name, amount = data
            if name in index:
                collected[index[name]] += chips(amount)
        elif kind == "shows":
            name, cards = data
            if name in index:
                shown[index[name]] = True
                holes[index[name]] = cards
        elif kind == "showdown":
            showdown_seen = True
    final = [stacks[i] + collected[i] for i in range(n)]
    payoffs = [final[i] - start[i] for i in range(n)]
    showdown = showdown_seen and sum(1 for f in folded if not f) >= 2
    result = HandResult(
        hand_id=0, seed=0, positions=positions, names=tuple(names), holes=holes, board=board, history=history,
        starting_stacks=start, payoffs=payoffs, ev_payoffs=[float(p) for p in payoffs], allin_street=None,
        showdown=showdown, folded=folded, shown=shown, decisions=records,
    )
    replay = {
        "id": hand.hand_id, "table": hand.table, "date": hand.date.isoformat(sep=" ") if hand.date else "",
        "players": [{"name": names[i], "position": positions[i], "stack_bb": start[i] / chips_per_bb,
                     "cards": [CARD_STR[c] for c in holes[i]] if holes[i][0] >= 0 else [], "hero": i == hero} for i in range(n)],
        "steps": steps, "board": [CARD_STR[c] for c in board],
        "result_bb": payoffs[hero] / chips_per_bb,
    }
    replay["situation"] = situation or ""
    return Replayed(result, hero, decisions, replay)


def hand_row(hand: ParsedHand, rep: Replayed, chips_per_bb: int, value_min_level: int) -> dict[str, Any]:
    """Une ligne « main » au format des campagnes (colonnes compatibles avec l'analyse)."""
    hr, i = rep.result, rep.hero_index
    flags = seat_flags(hr, i)
    board = list(hr.board) + [-1] * (5 - len(hr.board))
    c1, c2 = hand.hero_cards  # type: ignore[misc]
    n_dec = len(rep.decisions)
    n_off = sum(1 for d in rep.decisions if not d["match"])
    pre = [d for d in rep.decisions if d["street"] == 0]
    row = {
        "hand_id": hand.hand_id, "date": hand.date, "table": hand.table, "position": hr.positions[i],
        "hand_class": hole_class((c1, c2)), "c1": c1, "c2": c2,
        "b0": board[0], "b1": board[1], "b2": board[2], "b3": board[3], "b4": board[4],
        "result": hr.payoffs[i] / chips_per_bb, "result_ev": hr.payoffs[i] / chips_per_bb,
        "result_cur": hr.payoffs[i] / chips_per_bb * hand.big_blind, "currency": hand.currency, "big_blind": hand.big_blind,
        "showdown": bool(hr.showdown and not hr.folded[i]), "street_end": street_end(hr, i), "line": line_of(hr, i, value_min_level),
        "decisions": n_dec, "decisions_hors_fiche": n_off, "hors_fiche": n_off > 0,
        "situation": rep.replay.get("situation", ""),
        "hors_fiche_preflop": any(not d["match"] for d in pre),
        "replay": json.dumps(rep.replay, ensure_ascii=False),
        "decisions_json": json.dumps(rep.decisions, ensure_ascii=False),
        "raw": hand.raw,
    }
    for name, value in zip(FLAG_NAMES, flags):
        row["f_" + name] = value
    return row


def import_texts(texts: Iterable[str], strategy: Any | None, chips_per_bb: int = 100, hero: str | None = None,
                 value_min_level: str = "middle_pair") -> tuple[list[dict[str, Any]], int]:
    """Relit des fichiers d'historiques ; renvoie (lignes, mains ignorées)."""
    rows, skipped = [], 0
    vmin = level_of(value_min_level)
    for text in texts:
        for block in split_hands(text):
            hand = parse_hand(block, hero)
            rep = replay_hand(hand, strategy, chips_per_bb) if hand else None
            if hand is None or rep is None:
                skipped += 1
                continue
            rows.append(hand_row(hand, rep, chips_per_bb, vmin))
    return rows, skipped


def assign_sessions(rows: list[dict[str, Any]], gap_minutes: float) -> None:
    """Numérote les sessions : nouvelle session après une pause de plus de ``gap_minutes``."""
    rows.sort(key=lambda r: (r["date"] or datetime.min, r["hand_id"]))
    session, last = 0, None
    for r in rows:
        d = r["date"]
        if last is not None and d is not None and (d - last).total_seconds() > gap_minutes * 60:
            session += 1
        r["session"] = session
        last = d if d is not None else last


def read_files(paths: Iterable[str | Path]) -> list[str]:
    texts = []
    for p in paths:
        p = Path(p)
        files = sorted(p.glob("**/*.txt")) if p.is_dir() else [p]
        for f in files:
            texts.append(f.read_text(encoding="utf-8-sig", errors="replace"))
    return texts
