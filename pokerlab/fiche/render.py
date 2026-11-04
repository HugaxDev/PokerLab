"""Rendu HTML autonome de la fiche (aucune dépendance externe, lisible sur téléphone)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np

from pokerlab.cards.ranges import CLASS_NAMES
from pokerlab.fiche.content import Expected, SizeRow, money
from pokerlab.fiche.preflop import CALL, POSITIONS, RAISE, SITUATION_LABELS, SITUATIONS, chart_shares, raiser_for
from pokerlab.report.html import CSS, esc

FICHE_CSS = """
:root{--raise:#2a78d6;--raise-ink:#ffffff;--call:#eda100;--call-ink:#0b0b0b;--fold:#f0efec;--fold-ink:#898781}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])){--raise:#3987e5;--call:#c98500;--fold:#2c2c2a;--fold-ink:#898781}}
:root[data-theme="dark"]{--raise:#3987e5;--call:#c98500;--fold:#2c2c2a;--fold-ink:#898781}
main{max-width:980px}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.gridbox h3{margin:.1rem 0 .2rem}.gridbox p{margin:.1rem 0 .4rem;font-size:.8rem}
table.grid{table-layout:fixed;width:100%;border-collapse:separate;border-spacing:2px;font-variant-numeric:normal}
table.grid td{padding:0;height:24px;font-size:9.5px;line-height:24px;text-align:center;border:0;border-radius:3px;
white-space:nowrap;overflow:hidden;position:static}
table.grid td.r{background:var(--raise);color:var(--raise-ink);font-weight:600}
table.grid td.c{background:var(--call);color:var(--call-ink);font-weight:600}
table.grid td.f{background:var(--fold);color:var(--fold-ink)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:.85rem;color:var(--ink2);margin:.4rem 0}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{display:inline-block;width:14px;height:14px;border-radius:3px}
ol.rules li,ul.forbid li{color:var(--ink);margin:.35rem 0}
.todo{border-left:3px solid var(--axis);padding-left:10px;color:var(--ink2)}
code{font-size:.85em;background:var(--page);padding:1px 4px;border-radius:4px;white-space:normal;word-break:break-word}
.tag{font-size:.75rem;color:var(--muted)}
td.l{text-align:left;white-space:normal}
table.exp th{white-space:normal}table.exp td,table.exp th{padding:6px 4px}
"""

THEME_JS = """function toggleTheme(){const r=document.documentElement;const dark=r.getAttribute('data-theme')==='dark'||
(!r.getAttribute('data-theme')&&window.matchMedia('(prefers-color-scheme: dark)').matches);r.setAttribute('data-theme',dark?'light':'dark');}"""

CELL_CLASS = {RAISE: "r", CALL: "c"}


def grid_html(codes: np.ndarray) -> str:
    rows = []
    for r in range(13):
        cells = []
        for c in range(13):
            idx = r * 13 + c
            cls = CELL_CLASS.get(int(codes[idx]), "f")
            cells.append(f'<td class="{cls}">{CLASS_NAMES[idx]}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return '<table class="grid" aria-label="grille 13x13">' + "".join(rows) + "</table>"


def preflop_section(chart: dict[tuple[str, str], Any], raise_bb: float) -> str:
    legend = ('<div class="legend"><span><i style="background:var(--raise)"></i>relancer</span>'
              '<span><i style="background:var(--call)"></i>payer (ou check gratuit en BB)</span>'
              '<span><i style="background:var(--fold);border:1px solid var(--axis)"></i>se coucher</span></div>')
    parts = [f'<p>Grilles calculées en interrogeant la stratégie elle-même. « Face à une relance » = une relance à {raise_bb:g} bb '
             "juste devant toi (du BTN si tu es en blinds). En BB « personne n'a relancé » = la SB a complété. "
             "Au-dessus de la diagonale : assorties ; en dessous : dépareillées.</p>", legend]
    for position in POSITIONS:
        boxes = []
        for situation in SITUATIONS:
            codes = chart.get((position, situation))
            title = f"{position} — {SITUATION_LABELS[situation]}"
            if codes is None:
                boxes.append(f'<div class="gridbox"><h3>{esc(title)}</h3><p class="todo">UTG parle en premier avant le flop : '
                             "personne ne peut avoir relancé devant toi.</p></div>")
                continue
            raise_pct, call_pct = chart_shares(codes)
            who = f" (relance de {raiser_for(position)})" if situation == "vs_raise" else ""
            boxes.append(f'<div class="gridbox"><h3>{esc(title)}{esc(who)}</h3>'
                         f"<p>relance {fr(raise_pct)} % · paie {fr(call_pct)} % des mains</p>{grid_html(codes)}</div>")
        parts.append(f'<div class="card"><div class="pair">{"".join(boxes)}</div></div>')
    return "".join(parts)


def fr(value: float, digits: int = 1) -> str:
    """Nombre à la française (virgule décimale, signe moins typographique)."""
    return f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def frs(value: float, digits: int = 1) -> str:
    return fr(value, digits) if value < 0 else "+" + fr(value, digits)


def sizes_section(rows: list[SizeRow], stakes: list[dict[str, Any]]) -> str:
    if not rows:
        return '<p class="todo">Tailles non décrites pour ce type de stratégie.</p>'
    head = "<th>situation</th><th>bb</th>" + "".join(f"<th>{esc(s['label'])}</th>" for s in stakes)
    body = []
    for r in rows:
        if r.bb is None:
            cells = '<td>–</td>' + "".join("<td>–</td>" for _ in stakes)
        else:
            cells = f"<td>{fr(r.bb)}</td>" + "".join(f"<td>{money(r.bb, float(s['big_blind']))}</td>" for s in stakes)
        body.append(f'<tr><td class="l">{esc(r.label)}<br><span class="tag">{esc(r.detail)}</span></td>{cells}</tr>')
    return (f'<div class="card tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
            '<p class="legend-note">Montants en jetons de table, arrondis au centime. Tapis de départ : 100 bb '
            + " / ".join(f"{money(100, float(s['big_blind']))} en {esc(s['label'])}" for s in stakes) + ".</p></div>")


def todo_block(strategy: str, why: str) -> str:
    return (f'<div class="card todo"><p><strong>Pas encore de chiffres</strong> ({esc(why)}). Lance compare puis audit :</p>'
            f"<p><code>python -m pokerlab compare</code><br><code>python -m pokerlab audit --set strategy={esc(strategy)}</code><br>"
            f"<code>python -m pokerlab fiche --strategy {esc(strategy)} --campaign out</code></p></div>")


def leaks_section(leaks: list[dict[str, Any]], source: str, strategy: str) -> str:
    if not leaks:
        return todo_block(strategy, source)
    items = []
    for lk in leaks:
        flag = "" if lk["fiable"] else ' <span class="tag">(à confirmer : peu de cas ou intervalle qui touche 0)</span>'
        items.append(f"<li>{esc(lk['texte'])} <span class=\"tag\">Coût : {fr(lk['mean'], 2)} bb par décision, {lk['count']} cas, "
                     f"{fr(lk['total'])} bb au total.</span>{flag}</li>")
    return f'<div class="card"><ul class="forbid">{"".join(items)}</ul><p class="legend-note">Source : {esc(source)}.</p></div>'


def expected_section(exp: Expected | None, source: str, strategy: str) -> str:
    if exp is None:
        return todo_block(strategy, source)
    warn = "" if exp.reliable else '<p class="tag">Échantillon faible : fourchettes indicatives, relance compare pour plus de mains.</p>'
    head = (f"<p>bb/100 all-in adjusted : <strong>{frs(exp.bb100_ev)}</strong> ± {fr(exp.ci_half)} (IC 95 %) · réel : {frs(exp.bb100_real)} · "
            f"écart type : {fr(exp.sd100, 0)} bb/100</p>")
    rows = "".join(f"<tr><td>{int(r['mains'])}</td><td>{frs(r['moyenne'], 0)}</td><td>{frs(r['p5'], 0)} à {frs(r['p95'], 0)}</td></tr>" for r in exp.ranges)
    table = ('<div class="tbl"><table class="exp"><thead><tr><th>mains</th><th>moyenne (bb)</th><th>9 fois sur 10 (bb)</th></tr></thead>'
             f"<tbody>{rows}</tbody></table></div>")
    return (f'<div class="card">{head}{table}<p class="legend-note">9 fois sur 10, ton résultat sur ce nombre de mains tombe dans la fourchette '
            f"(5e et 95e centiles, tirage de mains simulées avec les résultats réels). Source : {esc(source)}.</p>{warn}</div>")


def render(meta: dict[str, Any], chart, sizes: list[SizeRow], stakes, rules: list[str], leaks_html: str, expected_html: str, raise_bb: float) -> str:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    swatch = f'<span class="sw" style="background:{esc(meta.get("color") or "#2a78d6")}"></span>'
    rules_html = "".join(f"<li>{esc(r)}</li>" for r in rules)
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fiche {esc(meta['label'])}</title>
<style>{CSS}{FICHE_CSS}</style>
<script>{THEME_JS}</script>
</head><body><main>
<button class="theme" onclick="toggleTheme()" aria-label="Basculer clair/sombre">clair / sombre</button>
<h1>{swatch}Fiche — {esc(meta['label'])}</h1>
<p class="sub">{esc(meta.get('description', ''))}</p>
<p class="sub">Générée le {stamp} · config : {esc(meta['source'])}{(' · campagne : ' + esc(meta['campaign'])) if meta.get('campaign') else ''}</p>
<h2>1. Avant le flop</h2>
{preflop_section(chart, raise_bb)}
<h2>2. Tailles de mise</h2>
{sizes_section(sizes, stakes)}
<h2>3. Après le flop</h2>
<div class="card"><ol class="rules">{rules_html}</ol></div>
<h2>4. Interdits (audit)</h2>
{leaks_html}
<h2>5. Résultat attendu</h2>
{expected_html}
<p class="sub">PokerLab — fiche générée localement depuis la config et, si fournie, une campagne simulée. Les adversaires simulés ne sont pas de vrais humains.</p>
</main></body></html>"""
