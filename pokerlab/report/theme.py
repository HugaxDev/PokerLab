"""Palette et encre du rapport (palette catégorielle validée, clair et sombre)."""

from __future__ import annotations

CATEGORICAL_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
CATEGORICAL_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767")

SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MID = "#f0efec"
NEG = "#e34948"
POS = "#2a78d6"
POS_DEEP = "#184f95"
NEG_DEEP = "#a8302f"
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

# Correspondance clair -> sombre appliquée aux figures en mode sombre (remplacement en une passe).
DARK_MAP = {
    **dict(zip(CATEGORICAL_LIGHT, CATEGORICAL_DARK)),
    SURFACE: "#1a1a19",
    PAGE: "#0d0d0d",
    INK: "#f0efec",
    INK_2: "#c3c2b7",
    GRID: "#2c2c2a",
    AXIS: "#383835",
    MID: "#383835",
    POS_DEEP: "#86b6ef",
    NEG_DEEP: "#f09a99",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def strategy_colors(names: list[str], declared: dict[str, str | None]) -> dict[str, str]:
    """Couleur fixe par stratégie : celle de sa config, sinon le prochain slot libre (jamais cyclé)."""
    out: dict[str, str] = {}
    used = {c for c in declared.values() if c}
    free = [c for c in CATEGORICAL_LIGHT if c not in used]
    for name in names:
        color = declared.get(name)
        if not color:
            color = free.pop(0) if free else MUTED
        out[name] = color
    return out
