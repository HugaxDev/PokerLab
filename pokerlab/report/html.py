"""Rapport HTML autonome : plotly embarqué (aucun CDN), lisible sur mobile, en français."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import pandas as pd

from pokerlab.report import theme

_FIG_COUNTER = [0]

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10)}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5rem;margin:.2rem 0}h2{font-size:1.2rem;margin:1.6rem 0 .4rem}h3{font-size:1rem;margin:1rem 0 .3rem}
p,li{color:var(--ink2)}.sub{color:var(--muted);font-size:.85rem}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px;margin:12px 0}
.verdict ol{margin:.2rem 0;padding-left:1.2rem}.verdict li{color:var(--ink);margin:.25rem 0}
.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:var(--surface)}
th{color:var(--ink2);font-weight:600}
td.best{font-weight:700}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.fig{width:100%;min-height:340px}
details{margin:.4rem 0}summary{cursor:pointer;color:var(--ink)}
pre{white-space:pre;overflow-x:auto;font-size:.75rem;background:var(--page);padding:10px;border-radius:8px;color:var(--ink)}
nav a{color:var(--ink2);margin-right:10px;font-size:.85rem}
button.theme{float:right;background:var(--surface);color:var(--ink);border:1px solid var(--ring);border-radius:8px;padding:4px 10px}
.legend-note{font-size:.8rem;color:var(--muted)}
"""

JS_RENDER = """
const DARK_MAP = %s;
function isDark(){const t=document.documentElement.getAttribute('data-theme');
 if(t==='dark')return true; if(t==='light')return false; return window.matchMedia('(prefers-color-scheme: dark)').matches;}
function themed(txt){ if(!isDark()) return txt; return txt.replace(/#[0-9a-fA-F]{6}/g, m => DARK_MAP[m.toLowerCase()] || m); }
function renderAll(){ document.querySelectorAll('script.figdata').forEach(s => {
  const fig = JSON.parse(themed(s.textContent)); const el = document.getElementById(s.dataset.target);
  Plotly.react(el, fig.data, fig.layout, {responsive:true, displaylogo:false}); }); }
function toggleTheme(){ const r=document.documentElement; r.setAttribute('data-theme', isDark()?'light':'dark'); renderAll(); }
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
document.addEventListener('DOMContentLoaded', renderAll);
"""


def base_layout(title: str = "", height: int = 380, **extra: Any) -> dict[str, Any]:
    """Mise en page sobre : grille fine, axes discrets, légende en bas (mobile)."""
    layout: dict[str, Any] = {
        "template": "none",
        "height": height,
        "margin": {"l": 56, "r": 16, "t": 40 if title else 16, "b": 56},
        "paper_bgcolor": theme.SURFACE,
        "plot_bgcolor": theme.SURFACE,
        "font": {"family": theme.FONT, "color": theme.INK_2, "size": 12},
        "title": {"text": title, "font": {"color": theme.INK, "size": 14}, "x": 0.01} if title else None,
        "xaxis": {"gridcolor": theme.GRID, "linecolor": theme.AXIS, "zerolinecolor": theme.AXIS, "automargin": True},
        "yaxis": {"gridcolor": theme.GRID, "linecolor": theme.AXIS, "zerolinecolor": theme.AXIS, "automargin": True},
        "legend": {"orientation": "h", "y": -0.2, "x": 0, "font": {"color": theme.INK_2}},
        "hoverlabel": {"bgcolor": theme.SURFACE, "font": {"color": theme.INK}, "bordercolor": theme.AXIS},
        "hovermode": "closest",
    }
    if layout["title"] is None:
        del layout["title"]
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            layout[key] = {**layout[key], **value}
        else:
            layout[key] = value
    return layout


def esc(text: Any) -> str:
    return html.escape(str(text))


def fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if value != value:
            return "–"
        if value in (float("inf"), float("-inf")):
            return "∞"
        return f"{value:,.{digits}f}".replace(",", " ")
    if isinstance(value, int):
        return f"{value:,}".replace(",", " ")
    return esc(value)


@dataclass
class Section:
    title: str
    anchor: str
    parts: list[str] = field(default_factory=list)
    figures: list[tuple[str, dict]] = field(default_factory=list)

    def html(self, text: str) -> None:
        self.parts.append(text)

    def paragraph(self, text: str) -> None:
        self.parts.append(f"<p>{esc(text)}</p>")

    def figure(self, fig: dict[str, Any], caption: str = "") -> None:
        _FIG_COUNTER[0] += 1
        fid = f"fig{_FIG_COUNTER[0]}"
        self.figures.append((fid, fig))
        cap = f'<p class="legend-note">{esc(caption)}</p>' if caption else ""
        payload = json.dumps(fig, separators=(",", ":"), default=_json_default).replace("</", "<\\/")
        self.parts.append(
            f'<div class="card"><div id="{fid}" class="fig"></div>{cap}'
            f'<script type="application/json" class="figdata" data-target="{fid}">{payload}</script></div>'
        )

    def table(
        self,
        df: pd.DataFrame,
        caption: str = "",
        digits: int = 1,
        cell_style: Callable[[int, str, Any], str] | None = None,
        first_col_html: Sequence[str] | None = None,
    ) -> None:
        head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            cells = []
            for j, col in enumerate(df.columns):
                value = row[col]
                content = first_col_html[i] if (j == 0 and first_col_html is not None) else fmt(value, digits)
                attrs = cell_style(i, col, value) if cell_style else ""
                cells.append(f"<td{attrs}>{content}</td>")
            rows.append("<tr>" + "".join(cells) + "</tr>")
        cap = f'<p class="legend-note">{esc(caption)}</p>' if caption else ""
        self.parts.append(f'<div class="card tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>{cap}</div>')

    def details(self, summary: str, body_html: str) -> None:
        self.parts.append(f"<details><summary>{esc(summary)}</summary>{body_html}</details>")


def _json_default(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if obj != obj else float(obj)
        if isinstance(obj, np.ndarray):
            return [_json_default(x) if isinstance(x, (np.generic,)) else x for x in obj.tolist()]
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"Non sérialisable : {type(obj)}")


@dataclass
class Report:
    """Rapport d'une campagne : verdict, tableau comparatif, graphiques, fuites."""

    title: str
    subtitle: str = ""
    verdict: list[str] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def section(self, title: str) -> Section:
        anchor = f"s{len(self.sections)}"
        sec = Section(title, anchor)
        self.sections.append(sec)
        return sec

    def absorb(self, other: "Report", prefix: str = "") -> None:
        """Ajoute les sections d'un autre rapport (rapport combiné)."""
        for sec in other.sections:
            new = self.section(f"{prefix}{sec.title}")
            new.parts = list(sec.parts)
            new.figures = list(sec.figures)

    def figure_count(self) -> int:
        return sum(len(s.figures) for s in self.sections)

    def render(self) -> str:
        from plotly.offline import get_plotlyjs

        nav = " ".join(f'<a href="#{s.anchor}">{esc(s.title)}</a>' for s in self.sections)
        verdict = "".join(f"<li>{esc(v)}</li>" for v in self.verdict[:5])
        meta = " · ".join(f"{esc(k)} : {esc(v)}" for k, v in self.meta.items())
        body = []
        for s in self.sections:
            body.append(f'<section id="{s.anchor}"><h2>{esc(s.title)}</h2>{"".join(s.parts)}</section>')
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        dark_map = json.dumps({k.lower(): v for k, v in theme.DARK_MAP.items()})
        return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(self.title)}</title>
<style>{CSS}</style>
<script>{get_plotlyjs()}</script>
<script>{JS_RENDER % dark_map}</script>
</head><body><main>
<button class="theme" onclick="toggleTheme()" aria-label="Basculer clair/sombre">clair / sombre</button>
<h1>{esc(self.title)}</h1>
<p class="sub">{esc(self.subtitle)} — généré le {stamp}</p>
<p class="sub">{meta}</p>
<div class="card verdict"><h3>Verdict</h3><ol>{verdict}</ol></div>
<nav>{nav}</nav>
{"".join(body)}
<p class="sub">PokerLab — simulation 100 % locale et déterministe. Les archétypes adverses ne sont pas de vrais humains.</p>
</main></body></html>"""

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path
