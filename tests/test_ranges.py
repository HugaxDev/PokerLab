"""Parseur de ranges : cas classiques, cas limites, nombre exact de combinaisons."""

from __future__ import annotations

import pytest

from pokerlab.cards.ranges import CLASS_COMBO_COUNT, CLASS_NAMES, COMBOS, N_COMBOS, combo_count, parse_range, range_classes


def test_totals() -> None:
    assert N_COMBOS == len(COMBOS) == 1326
    assert combo_count("any") == 1326
    assert int(CLASS_COMBO_COUNT.sum()) == 1326
    for idx, name in enumerate(CLASS_NAMES):
        expected = 6 if len(name) == 2 else (4 if name.endswith("s") else 12)
        assert CLASS_COMBO_COUNT[idx] == expected, name


@pytest.mark.parametrize(
    "text, combos",
    [
        ("AA", 6),
        ("AKs", 4),
        ("AKo", 12),
        ("AK", 16),
        ("22+", 78),
        ("99+", 36),
        ("A2s+", 48),
        ("KTo+", 36),
        ("T9s-54s", 24),
        ("A2s-A5s", 16),
        ("22-55", 24),
        ("AsKs", 1),
        ("AKs, AKo", 16),
        ("99+, AK, AQ, AJs, KQs", 36 + 16 + 16 + 4 + 4),
        ("AA, AA, KK", 12),
        ("random", 1326),
    ],
)
def test_counts(text: str, combos: int) -> None:
    assert combo_count(text) == combos


def test_classes_and_order_insensitive() -> None:
    assert range_classes("KTo+") == range_classes("KQo, KJo, KTo")
    assert range_classes("T9s-54s") == range_classes("54s-T9s")
    assert {CLASS_NAMES[c] for c in range_classes("A2s+")} == {f"A{r}s" for r in "23456789TJQK"}
    assert {CLASS_NAMES[c] for c in range_classes("22+")} == {r + r for r in "23456789TJQKA"}


def test_weights() -> None:
    w = parse_range("AKs:0.5, QQ")
    assert w.sum() == pytest.approx(4 * 0.5 + 6)


@pytest.mark.parametrize("bad", ["AAs", "XY", "A", "AK-Q2", "AKs-QJo"])
def test_invalid(bad: str) -> None:
    parse_range.cache_clear()
    with pytest.raises(ValueError):
        parse_range(bad)
