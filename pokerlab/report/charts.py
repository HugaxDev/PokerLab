"""Graphiques plotly (dicts JSON) : couleurs fixes par stratégie, survol détaillé."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from pokerlab.cards.ranges import CLASS_NAMES, GRID_LABELS
from pokerlab.report import theme
from pokerlab.report.html import base_layout


def _clean(values: Sequence[float]) -> list[float | None]:
    return [None if (v is None or (isinstance(v, float) and (v != v or abs(v) == float("inf")))) else float(v) for v in values]


def winrate_bars(names: Sequence[str], ev: Sequence[float], ci: Sequence[float], real: Sequence[float], colors: Mapping[str, str], labels: Mapping[str, str]) -> dict:
    """bb/100 all-in adjusted (avec IC 95 %) et réel, par stratégie."""
    y = [labels.get(n, n) for n in names]
    data = [
        {
            "type": "bar",
            "orientation": "h",
            "name": "all-in adjusted (IC 95 %)",
            "y": y,
            "x": _clean(ev),
            "marker": {"color": [colors[n] for n in names], "line": {"color": theme.SURFACE, "width": 2}},
            "error_x": {"type": "data", "array": _clean(ci), "color": theme.INK_2, "thickness": 1.5, "width": 4},
            "hovertemplate": "%{y}<br>EV : %{x:.1f} bb/100 ± %{error_x.array:.1f}<extra></extra>",
        },
        {
            "type": "bar",
            "orientation": "h",
            "name": "réel",
            "y": y,
            "x": _clean(real),
            "marker": {"color": [colors[n] for n in names], "opacity": 0.45, "line": {"color": theme.SURFACE, "width": 2}},
            "hovertemplate": "%{y}<br>réel : %{x:.1f} bb/100<extra></extra>",
        },
    ]
    layout = base_layout("bb/100 par stratégie", height=max(320, 46 * len(names) + 120), barmode="group", showlegend=False,
                         xaxis={"title": {"text": "bb/100"}}, yaxis={"autorange": "reversed"})
    return {"data": data, "layout": layout}


def diverging_scale() -> list[list[Any]]:
    return [[0.0, theme.NEG_DEEP], [0.25, theme.NEG], [0.5, theme.MID], [0.75, theme.POS], [1.0, theme.POS_DEEP]]


def matrix_heatmap(z: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str, unit: str = "bb/100", text: np.ndarray | None = None) -> dict:
    """Heatmap divergente centrée sur 0 (rouge = perd, bleu = gagne)."""
    finite = np.abs(z[np.isfinite(z)]) if np.isfinite(z).any() else np.array([1.0])
    lim = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    txt = text if text is not None else np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v:.1f}")(z)
    data = [{
        "type": "heatmap",
        "z": [[None if not np.isfinite(v) else float(v) for v in row] for row in z],
        "x": list(cols),
        "y": list(rows),
        "zmin": -lim,
        "zmax": lim,
        "colorscale": diverging_scale(),
        "text": txt.tolist(),
        "texttemplate": "%{text}",
        "textfont": {"size": 11},
        "xgap": 2,
        "ygap": 2,
        "colorbar": {"title": {"text": unit}, "thickness": 10},
        "hovertemplate": "%{y} · %{x}<br>%{z:.2f} " + unit + "<extra></extra>",
    }]
    # axes catégoriels : des libellés comme 0.33 / 0.5 / tapis ne doivent pas être lus comme des nombres
    layout = base_layout(title, height=max(300, 40 * len(rows) + 140), yaxis={"autorange": "reversed", "gridcolor": theme.SURFACE, "type": "category"},
                         xaxis={"gridcolor": theme.SURFACE, "type": "category"})
    return {"data": data, "layout": layout}


def cumulative_lines(curves: Mapping[str, dict[str, Any]], colors: Mapping[str, str], labels: Mapping[str, str], title: str) -> dict:
    """Gains cumulés ; survol : main, position, cartes, action, montant, cumul."""
    data = []
    for name, c in curves.items():
        data.append({
            "type": "scattergl" if len(c["x"]) > 2000 else "scatter",
            "mode": "lines",
            "name": labels.get(name, name) + c.get("suffix", ""),
            "x": list(c["x"]),
            "y": _clean(c["y"]),
            "line": {"color": colors.get(name, theme.MUTED), "width": 2, "dash": c.get("dash", "solid")},
            "customdata": c.get("custom"),
            "hovertemplate": (
                "main %{customdata[0]} · %{customdata[1]} · %{customdata[2]}<br>"
                "action : %{customdata[3]}<br>montant : %{customdata[4]} bb<br>cumul : %{y:.1f} bb<extra>%{fullData.name}</extra>"
            ) if c.get("custom") is not None else "%{y:.1f} bb<extra>%{fullData.name}</extra>",
        })
    layout = base_layout(title, height=420, xaxis={"title": {"text": "mains"}}, yaxis={"title": {"text": "bb cumulées"}}, hovermode="x unified" if len(curves) <= 3 else "closest")
    return {"data": data, "layout": layout}


def showdown_lines(x: Sequence[int], total: Sequence[float], sd: Sequence[float], nsd: Sequence[float], title: str) -> dict:
    """Lignes classiques : total, avec showdown (bleu), sans showdown (rouge)."""
    data = [
        {"type": "scatter", "mode": "lines", "name": "total", "x": list(x), "y": _clean(total), "line": {"color": theme.INK_2, "width": 2}},
        {"type": "scatter", "mode": "lines", "name": "avec showdown (bleu)", "x": list(x), "y": _clean(sd), "line": {"color": theme.POS, "width": 2}},
        {"type": "scatter", "mode": "lines", "name": "sans showdown (rouge)", "x": list(x), "y": _clean(nsd), "line": {"color": theme.NEG, "width": 2}},
    ]
    layout = base_layout(title, height=340, hovermode="x unified", xaxis={"title": {"text": "mains"}}, yaxis={"title": {"text": "bb cumulées"}})
    return {"data": data, "layout": layout}


def grouped_bars(categories: Sequence[str], series: Mapping[str, Sequence[float]], colors: Mapping[str, str], labels: Mapping[str, str], title: str, unit: str) -> dict:
    data = []
    for name, values in series.items():
        data.append({
            "type": "bar",
            "name": labels.get(name, name),
            "x": list(categories),
            "y": _clean(values),
            "marker": {"color": colors.get(name, theme.MUTED), "line": {"color": theme.SURFACE, "width": 2}},
            "hovertemplate": "%{x}<br>%{y:.2f} " + unit + "<extra>%{fullData.name}</extra>",
        })
    layout = base_layout(title, height=380, barmode="group", yaxis={"title": {"text": unit}})
    return {"data": data, "layout": layout}


def class_heatmaps(grids: Mapping[str, tuple[np.ndarray, np.ndarray]], labels: Mapping[str, str], title: str) -> dict:
    """Grille 13x13 (bb/main) ; un menu remplace les valeurs pour choisir la stratégie."""
    names = list(grids)
    all_vals = np.concatenate([g[0][np.isfinite(g[0])] for g in grids.values()]) if grids else np.array([1.0])
    lim = float(np.quantile(np.abs(all_vals), 0.95)) if all_vals.size else 1.0
    lim = lim if lim > 0 else 1.0
    zs, customs = [], []
    for name in names:
        means, counts = grids[name]
        z = means.reshape(13, 13)
        n = counts.reshape(13, 13)
        zs.append([[None if not np.isfinite(v) else float(v) for v in row] for row in z])
        customs.append([[[CLASS_NAMES[r * 13 + c], int(n[r, c])] for c in range(13)] for r in range(13)])
    data = [{
        "type": "heatmap",
        "z": zs[0] if zs else [],
        "x": list(range(13)),
        "y": list(range(13)),
        "zmin": -lim,
        "zmax": lim,
        "colorscale": diverging_scale(),
        "customdata": customs[0] if customs else [],
        "xgap": 2,
        "ygap": 2,
        "colorbar": {"title": {"text": "bb/main"}, "thickness": 10},
        "hovertemplate": "%{customdata[0]}<br>%{z:.2f} bb/main<br>%{customdata[1]} mains<extra></extra>",
    }]
    buttons = [{"label": labels.get(n, n), "method": "restyle", "args": [{"z": [zs[k]], "customdata": [customs[k]]}]} for k, n in enumerate(names)]
    ticks = {"tickmode": "array", "tickvals": list(range(13)), "ticktext": GRID_LABELS, "gridcolor": theme.SURFACE, "zeroline": False}
    layout = base_layout(title, height=520, yaxis={**ticks, "autorange": "reversed", "scaleanchor": "x"},
                         xaxis={**ticks, "side": "top"},
                         annotations=[{"x": c, "y": r, "text": CLASS_NAMES[r * 13 + c], "showarrow": False,
                                       "font": {"size": 8, "color": theme.INK_2}} for r in range(13) for c in range(13)],
                         updatemenus=[{"buttons": buttons, "x": 1.0, "y": 1.2, "xanchor": "right", "yanchor": "bottom", "bgcolor": theme.SURFACE,
                                       "font": {"color": theme.INK}, "bordercolor": theme.AXIS}] if len(names) > 1 else [])
    layout["margin"]["t"] = 96
    return {"data": data, "layout": layout}


def boxes(series: Mapping[str, Sequence[float]], colors: Mapping[str, str], labels: Mapping[str, str], title: str, unit: str) -> dict:
    data = []
    for name, values in series.items():
        data.append({
            "type": "box",
            "name": labels.get(name, name),
            "y": _clean(list(values)),
            "boxmean": True,
            "marker": {"color": colors.get(name, theme.MUTED), "size": 4},
            "line": {"color": colors.get(name, theme.MUTED), "width": 1.5},
            "boxpoints": False,
        })
    layout = base_layout(title, height=420, showlegend=False, yaxis={"title": {"text": unit}})
    return {"data": data, "layout": layout}


def histogram(values: Sequence[float], color: str, title: str, unit: str, marker_x: float | None = None) -> dict:
    data = [{
        "type": "histogram",
        "x": _clean(list(values)),
        "marker": {"color": color, "line": {"color": theme.SURFACE, "width": 2}},
        "nbinsx": 60,
        "hovertemplate": "%{x}<br>%{y} sessions<extra></extra>",
        "name": "sessions simulées",
    }]
    shapes = []
    annotations = []
    if marker_x is not None:
        shapes.append({"type": "line", "x0": marker_x, "x1": marker_x, "yref": "paper", "y0": 0, "y1": 1, "line": {"color": theme.INK, "width": 2}})
        annotations.append({"x": marker_x, "yref": "paper", "y": 1.02, "text": "votre résultat", "showarrow": False, "font": {"color": theme.INK}})
    layout = base_layout(title, height=340, shapes=shapes, annotations=annotations, xaxis={"title": {"text": unit}}, yaxis={"title": {"text": "sessions"}})
    return {"data": data, "layout": layout}


def fan_chart(x: Sequence[int], bands: Mapping[str, Sequence[float]], color: str, title: str, samples: Sequence[Sequence[float]] = ()) -> dict:
    """Percentiles de trajectoires (5-95, 25-75, médiane)."""
    data: list[dict] = []
    for path in samples:
        data.append({"type": "scatter", "mode": "lines", "x": list(x), "y": _clean(path), "line": {"color": theme.AXIS, "width": 1},
                     "hoverinfo": "skip", "showlegend": False})
    for lo, hi, alpha, name in (("p5", "p95", "2e", "5e-95e centile"), ("p25", "p75", "5c", "25e-75e centile")):
        data.append({"type": "scatter", "mode": "lines", "x": list(x), "y": _clean(bands[hi]), "line": {"width": 0, "color": color}, "showlegend": False, "hoverinfo": "skip"})
        data.append({"type": "scatter", "mode": "lines", "x": list(x), "y": _clean(bands[lo]), "line": {"width": 0, "color": color}, "fill": "tonexty",
                     "fillcolor": color + alpha, "name": name, "hoverinfo": "skip"})
    data.append({"type": "scatter", "mode": "lines", "x": list(x), "y": _clean(bands["p50"]), "line": {"color": color, "width": 2}, "name": "médiane",
                 "hovertemplate": "session %{x}<br>médiane : %{y:.0f} bb<extra></extra>"})
    layout = base_layout(title, height=380, xaxis={"title": {"text": "sessions"}}, yaxis={"title": {"text": "bankroll (bb)"}})
    return {"data": data, "layout": layout}


def line_with_ci(x: Sequence[Any], y: Sequence[float], ci: Sequence[float], color: str, title: str, xlabel: str, unit: str = "bb/100") -> dict:
    data = [{
        "type": "scatter",
        "mode": "lines+markers",
        "x": list(x),
        "y": _clean(y),
        "error_y": {"type": "data", "array": _clean(ci), "color": theme.INK_2, "thickness": 1.5, "width": 4},
        "line": {"color": color, "width": 2},
        "marker": {"color": color, "size": 9, "line": {"color": theme.SURFACE, "width": 2}},
        "hovertemplate": xlabel + " = %{x}<br>%{y:.1f} ± %{error_y.array:.1f} " + unit + "<extra></extra>",
        "name": unit,
    }]
    layout = base_layout(title, height=360, showlegend=False, xaxis={"title": {"text": xlabel}}, yaxis={"title": {"text": unit}})
    return {"data": data, "layout": layout}


def bars_with_ci(categories: Sequence[str], values: Sequence[float], ci: Sequence[float], color: str, title: str, unit: str, horizontal: bool = True) -> dict:
    trace: dict[str, Any] = {
        "type": "bar",
        "marker": {"color": color, "line": {"color": theme.SURFACE, "width": 2}},
        "name": unit,
    }
    if horizontal:
        trace.update(orientation="h", y=list(categories), x=_clean(values),
                     error_x={"type": "data", "array": _clean(ci), "color": theme.INK_2, "thickness": 1.5, "width": 4},
                     hovertemplate="%{y}<br>%{x:.2f} " + unit + "<extra></extra>")
        layout = base_layout(title, height=max(300, 34 * len(categories) + 120), showlegend=False, yaxis={"autorange": "reversed"}, xaxis={"title": {"text": unit}})
    else:
        trace.update(x=list(categories), y=_clean(values),
                     error_y={"type": "data", "array": _clean(ci), "color": theme.INK_2, "thickness": 1.5, "width": 4},
                     hovertemplate="%{x}<br>%{y:.2f} " + unit + "<extra></extra>")
        layout = base_layout(title, height=360, showlegend=False, yaxis={"title": {"text": unit}})
    return {"data": [trace], "layout": layout}


def train_validation(points: Sequence[dict[str, Any]], color: str, title: str) -> dict:
    """Score entraînement vs validation des candidats (la diagonale = pas de sur-ajustement)."""
    xs = [p["train"] for p in points]
    ys = [p["validation"] for p in points]
    lo = min(xs + ys + [0.0]) if xs else 0.0
    hi = max(xs + ys + [1.0]) if xs else 1.0
    data = [
        {"type": "scatter", "mode": "lines", "x": [lo, hi], "y": [lo, hi], "line": {"color": theme.AXIS, "width": 1}, "name": "entraînement = validation", "hoverinfo": "skip"},
        {"type": "scatter", "mode": "markers", "x": xs, "y": ys, "name": "candidats",
         "marker": {"color": color, "size": 10, "line": {"color": theme.SURFACE, "width": 2}},
         "text": [p["label"] for p in points],
         "hovertemplate": "%{text}<br>entraînement %{x:.1f} · validation %{y:.1f} bb/100<extra></extra>"},
    ]
    layout = base_layout(title, height=380, xaxis={"title": {"text": "bb/100 entraînement"}}, yaxis={"title": {"text": "bb/100 validation"}})
    return {"data": data, "layout": layout}


def sequential_heatmap(z: np.ndarray, rows: Sequence[str], cols: Sequence[str], title: str, unit: str, text: np.ndarray | None = None) -> dict:
    scale = [[i / (len(theme.SEQ) - 1), c] for i, c in enumerate(theme.SEQ)]
    txt = text if text is not None else np.vectorize(lambda v: "" if not np.isfinite(v) else f"{v:.2g}")(z)
    data = [{
        "type": "heatmap",
        "z": [[None if not np.isfinite(v) else float(v) for v in row] for row in z],
        "x": list(cols),
        "y": list(rows),
        "colorscale": scale,
        "reversescale": True,
        "zmin": 0,
        "zmax": 1,
        "text": txt.tolist(),
        "texttemplate": "%{text}",
        "textfont": {"size": 10},
        "xgap": 2,
        "ygap": 2,
        "colorbar": {"title": {"text": unit}, "thickness": 10},
        "hovertemplate": "%{y} vs %{x}<br>" + unit + " = %{z:.3g}<extra></extra>",
    }]
    layout = base_layout(title, height=max(320, 40 * len(rows) + 150), yaxis={"autorange": "reversed", "gridcolor": theme.SURFACE}, xaxis={"gridcolor": theme.SURFACE})
    return {"data": data, "layout": layout}
