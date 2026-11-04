"""Lecture du tableau (sec / mouillé) et de la main (niveaux, tirages)."""

from __future__ import annotations

import pytest

from pokerlab.cards.cards import parse_cards
from pokerlab.features.board import board_texture
from pokerlab.features.hand import LEVEL, read_hand


@pytest.mark.parametrize(
    "board, dry",
    [
        ("Kd7c2s", True),
        ("Kd7d2s", False),   # deux cartes de même couleur
        ("9c8d6s", False),   # trois rangs dans un écart <= 4
        ("Ac3d5s", False),   # roue (As bas)
        ("KdKc4s", True),    # paire sèche
        ("8d8c9s", False),   # paire mouillée
        ("Qs8d3c", True),
    ],
)
def test_dry(board: str, dry: bool) -> None:
    assert board_texture(parse_cards(board)).dry is dry


@pytest.mark.parametrize(
    "hole, board, level",
    [
        ("AsKd", "Kc7h2s", "top_pair"),
        ("QsQd", "Jc7h2s", "overpair"),
        ("9s9d", "Kc7h2s", "middle_pair"),
        ("5s5d", "Kc7h2s", "weak_pair"),
        ("Ks7d", "Kc7h2s", "two_pair"),
        ("7s7d", "Kc7h2s", "trips"),
        ("AsQd", "Kc7h2s", "nothing"),
        ("Ah2h", "KhQh3h", "flush"),
        ("9s8d", "7c6h5s", "straight"),
        ("AsQd", "KcKh2s", "nothing"),      # paire du tableau seulement
        ("7s2d", "7c7h2s", "full_house"),
    ],
)
def test_levels(hole: str, board: str, level: str) -> None:
    assert read_hand(parse_cards(hole), parse_cards(board)).level == LEVEL[level]


def test_draws() -> None:
    fd = read_hand(parse_cards("AhTh"), parse_cards("Kh7h2s"))
    assert fd.flush_draw and fd.strong_draw and fd.level == LEVEL["nothing"]
    oesd = read_hand(parse_cards("9s8d"), parse_cards("7c6h2s"))
    assert oesd.straight_draw == 2 and oesd.outs == 8
    gut = read_hand(parse_cards("9s8d"), parse_cards("Jc7h2s"))
    assert gut.straight_draw == 1 and not gut.strong_draw
    river = read_hand(parse_cards("AhTh"), parse_cards("Kh7h2s3c4d"))
    assert not river.any_draw
