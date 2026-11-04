"""Pages HTML du tableau de bord (générées par Python)."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from pokerlab.report import charts, theme
from pokerlab.report.html import _json_default, base_layout, esc, fmt
from pokerlab.site.data import Campaign, Played, Reference, campaign_params, money_label, verdict_vs_cone

NAV = (("index.html", "Mon jeu"), ("labo.html", "Labo"), ("mains.html", "Mains"), ("fiche.html", "Fiche"))
_FIG = [0]


def nav(active: str) -> str:
    links = "".join(f'<a href="{href}" class="{"on" if href == active else ""}">{label}</a>' for href, label in NAV)
    return (f'<header class="site"><div class="in"><b>PokerLab</b>{links}'
            '<button class="theme" onclick="toggleTheme()" aria-label="Basculer clair/sombre">clair / sombre</button></div></header>')


def page(title: str, active: str, body: str, version: str, plotly: bool = False, scripts: tuple[str, ...] = ()) -> str:
    extra = "".join(f'<script src="{s}"></script>' for s in scripts)
    plot = '<script src="assets/plotly.js"></script>' if plotly else ""
    return f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<link rel="stylesheet" href="assets/site.css">
{plot}<script>window.SITE_VERSION = {json.dumps(version)};</script>
<script src="assets/site.js"></script>{extra}
</head><body>{nav(active)}<main>
{body}
<p class="sub">PokerLab — tableau de bord local, aucune connexion réseau. Généré le {esc(version[:16].replace("T", " "))}.</p>
</main></body></html>"""


def figure(fig: dict[str, Any], caption: str = "") -> str:
    _FIG[0] += 1
    fid = f"f{_FIG[0]}"
    payload = json.dumps(fig, separators=(",", ":"), default=_json_default).replace("</", "<\\/")
    cap = f'<p class="legend-note">{esc(caption)}</p>' if caption else ""
    return (f'<div class="card"><div id="{fid}" class="fig"></div>{cap}'
            f'<script type="application/json" class="figdata" data-target="{fid}">{payload}</script></div>')


def table(df: pd.DataFrame, caption: str = "", digits: int = 1, raw_cols: tuple[str, ...] = ()) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, row in df.iterrows():
        cells = "".join(f"<td>{row[c] if c in raw_cols else fmt(row[c], digits)}</td>" for c in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    cap = f'<p class="legend-note">{esc(caption)}</p>' if caption else ""
    return f'<div class="card tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>{cap}</div>'


def num(v: float, digits: int = 1, sign: bool = False) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "–"
    s = f"{v:+,.{digits}f}" if sign else f"{v:,.{digits}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def count(n: float) -> str:
    """Effectif compact : 1,2 M, 199 k, 850."""
    n = float(n)
    if n >= 1e6:
        return num(n / 1e6, 1) + " M"
    if n >= 1e4:
        return num(n / 1e3, 0) + " k"
    return num(n, 0)


# ------------------------------------------------------------------------------------ Mon jeu
def empty_state(cfg: dict[str, Any], has_example: bool) -> str:
    ex = '<p><a href="exemple.html">Voir « Mon jeu » sur les données d\'exemple</a> (mains simulées, marquées « exemple »).</p>' if has_example else ""
    return f"""<h1>Mon jeu</h1>
<div class="banner">Aucune main réelle importée pour l'instant.</div>
<div class="card"><h3>Importer tes historiques PokerStars</h3>
<ol>
<li>Dans PokerStars : Paramètres → Historique des mains : note le dossier d'enregistrement (fichiers .txt, format anglais).</li>
<li>Copie ces fichiers .txt dans <code>{esc(cfg.get('dossier_import', 'data/a_importer'))}</code> : avec <code>python -m pokerlab serve</code> lancé,
l'import et la mise à jour du site sont automatiques.</li>
<li>Ou en une commande : <code>python -m pokerlab import chemin/vers/historiques</code>, puis <code>python -m pokerlab site</code>.</li>
</ol>
<p>Stratégie suivie : <b>{esc(cfg.get('strategie_suivie'))}</b> (à changer dans <code>configs/site.yaml</code>). Ton pseudo : <code>hero</code> dans la même config
(sinon le joueur qui reçoit « Dealt to »).</p></div>
{ex}"""


def mon_jeu(p: Played, ref: Reference | None, cone: dict[str, Any] | None, color: str, label: str, cfg: dict[str, Any]) -> str:
    h = p.hands
    bb100, ci = p.bb100()
    respect = p.respect_rate()
    cur = money_label(p.currency)
    parts = ["<h1>Mon jeu" + (' <span class="tag-ex">EXEMPLE</span>' if p.example else "") + "</h1>"]
    if p.example:
        parts.append('<div class="banner warn"><b>Exemple :</b> mains simulées par PokerLab (fichier journal/exemple_pokerstars.txt), '
                     "pas de vraies données. Importe tes historiques PokerStars pour voir ton propre jeu.</div>")
    parts.append(f'<p class="sub">Stratégie suivie : <b>{esc(label)}</b>'
                 + (f" · référence simulée : {esc(ref.composition)} ({num(ref.bb100, 1, True)} ± {num(ref.ci)} bb/100)" if ref else "") + "</p>")
    kpis = [
        (f"{p.n:,}".replace(",", " "), "mains réelles jouées"),
        (f"{num(bb100, 1, True)} <span style='font-size:.9rem'>± {num(ci)}</span>", "bb/100 réel (IC 95 %)"),
        (f"{num(respect, 0)} %", "décisions conformes à la fiche"),
        (f"{num(p.result_money(), 2, True)}", f"résultat en {cur}"),
    ]
    parts.append('<div class="kpis">' + "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{esc(l)}</div></div>' for v, l in kpis) + "</div>")
    cum = np.cumsum(h["result"].to_numpy(dtype=float))
    data = []
    if cone is not None:
        xs = cone["x"]
        for lo, hi, alpha, name in (("p5", "p95", "26", "attendu 5e–95e centile"), ("p25", "p75", "4d", "attendu 25e–75e centile")):
            data.append({"type": "scatter", "mode": "lines", "x": xs, "y": cone[hi], "line": {"width": 0, "color": theme.MUTED}, "showlegend": False, "hoverinfo": "skip"})
            data.append({"type": "scatter", "mode": "lines", "x": xs, "y": cone[lo], "line": {"width": 0, "color": theme.MUTED}, "fill": "tonexty",
                         "fillcolor": theme.MUTED + alpha, "name": name, "hoverinfo": "skip"})
        data.append({"type": "scatter", "mode": "lines", "x": xs, "y": cone["p50"], "line": {"width": 1.5, "color": theme.INK_2},
                     "name": "médiane attendue", "hovertemplate": "après %{x} mains<br>médiane attendue %{y:.0f} bb<extra></extra>"})
    custom = [[str(r.hand_id), str(r.date)[:16], r.position, f"{fmt_cards(r)}", line_label(r), f"{r.result:+.1f}"] for r in h.itertuples()]
    data.append({"type": "scatter", "mode": "lines", "x": list(range(1, len(cum) + 1)), "y": cum.tolist(), "name": "mes gains réels",
                 "line": {"color": color, "width": 2}, "customdata": custom,
                 "hovertemplate": "main %{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} · %{customdata[3]}<br>"
                                  "action : %{customdata[4]}<br>montant : %{customdata[5]} bb<br>cumul : %{y:.1f} bb<extra></extra>"})
    layout = base_layout("Mes gains réels cumulés et cône attendu", height=420, xaxis={"title": {"text": "mains"}}, yaxis={"title": {"text": "bb"}})
    parts.append("<h2>Mes gains face à la variance attendue</h2>")
    if cone is not None:
        msg, level = verdict_vs_cone(float(cum[-1]) if cum.size else 0.0, cone)
        parts.append(f'<div class="banner {level}">{esc(msg)}</div>')
    else:
        parts.append('<div class="banner">Pas de référence simulée : lance <code>python -m pokerlab compare</code> pour afficher le cône attendu.</div>')
    parts.append(figure({"data": data, "layout": layout},
                        f"Cône : {cone.get('methode', '')} à partir des mains simulées de la stratégie ({ref.composition if ref else ''})." if cone else ""))
    if ref is not None and math.isfinite(ref.hands_needed):
        need = ref.hands_needed
        pct = min(100.0, 100.0 * p.n / need)
        parts.append(f'<div class="card"><p>Mains nécessaires pour distinguer le taux de gain de zéro à 95 % (si tu joues comme la stratégie) : '
                     f"<b>{num(need, 0)}</b>. Tu en as joué {p.n:,} ({num(pct, 0)} %).</p>".replace(",", " ")
                     + f'<div class="progress" role="progressbar" aria-valuenow="{pct:.0f}" aria-valuemin="0" aria-valuemax="100"><div style="width:{pct:.1f}%"></div></div></div>')
    parts.append("<h2>Dernières sessions</h2>")
    parts.append(table(p.sessions().head(15), "Résultat en bb et en " + money_label(p.currency) + ". Une session = mains sans pause de plus de "
                       + str(cfg.get("pause_session_minutes", 30)) + " min. « hors fiche » = mains avec au moins une décision différente de la fiche.", 1))
    parts.append("<h2>Par position et par main</h2>")
    pos = p.by_position()
    parts.append(figure(charts.bars_with_ci(list(pos["position"]), list(pos["bb100"]), [0.0] * len(pos), color, "bb/100 réel par position", "bb/100", horizontal=False),
                        "Nombre de mains par position dans le tableau ci-dessous."))
    parts.append(table(pd.DataFrame({"position": pos["position"], "mains": pos["count"], "bb/100": pos["bb100"], "total (bb)": pos["sum"]}), digits=1))
    parts.append(figure(class_grid(*p.by_class(), "Mes bb gagnées par main (grille 13×13)"),
                        "Au-dessus de la diagonale : assorties ; en dessous : dépareillées. Survol (ou toucher) : bb par main et nombre de mains."))
    parts.append("<h2>Mes stats et celles de la stratégie simulée</h2>")
    mine = p.stats()
    sim = ref.stats if ref else {}
    rows = [{"stat": k, "moi": mine.get(k, float("nan")), f"{label} simulé": sim.get(k, float("nan")),
             "écart": mine.get(k, float("nan")) - sim.get(k, float("nan")) if k in sim else float("nan")} for k in mine]
    parts.append(table(pd.DataFrame(rows), "VPIP, PFR, 3-bet en % ; AF = (mises + relances) / paies ; WTSD = % des flops vus allant à l'abattage."))
    parts.append("<h2>Mes 5 situations les plus perdantes</h2>")
    losing = p.losing_situations()
    if not losing:
        parts.append('<p>Aucune situation perdante récurrente sur ces mains.</p>')
    else:
        items = []
        for s in losing:
            links = " ".join(f'<a href="mains.html#{esc(hid)}">{esc(hid)}</a>' for hid in s["hands"])
            items.append(f"<li><b>{esc(s['situation'])}</b> : {num(s['total'], 1, True)} bb sur {s['count']} décision(s) "
                         f"({num(s['mean'], 1, True)} bb en moyenne) — mains : {links}</li>")
        parts.append('<div class="card"><ol>' + "".join(items) + '</ol><p class="legend-note">Gain à partir de la décision '
                     "(résultat de la main + mise déjà investie) : négatif = continuer a coûté plus qu'abandonner.</p></div>")
    return "\n".join(parts)


def class_grid(means: np.ndarray, counts: np.ndarray, title: str) -> dict[str, Any]:
    """Grille 13x13 compacte pour mobile : pas d'échelle latérale, valeur et effectif au survol."""
    from pokerlab.cards.ranges import CLASS_NAMES

    finite = means[np.isfinite(means)]
    lim = float(np.quantile(np.abs(finite), 0.95)) if finite.size else 1.0
    lim = lim if lim > 0 else 1.0
    z = [[None if not np.isfinite(means[r * 13 + c]) else float(means[r * 13 + c]) for c in range(13)] for r in range(13)]
    custom = [[[CLASS_NAMES[r * 13 + c], int(counts[r * 13 + c])] for c in range(13)] for r in range(13)]
    data = [{"type": "heatmap", "z": z, "x": list(range(13)), "y": list(range(13)), "zmin": -lim, "zmax": lim,
             "colorscale": charts.diverging_scale(), "customdata": custom, "xgap": 1, "ygap": 1, "showscale": False,
             "hovertemplate": "%{customdata[0]}<br>%{z:+.2f} bb par main<br>%{customdata[1]} main(s)<extra></extra>"}]
    hidden = {"showticklabels": False, "showgrid": False, "zeroline": False, "fixedrange": True}
    layout = base_layout(title, height=400, xaxis=hidden, yaxis={**hidden, "autorange": "reversed"},
                         annotations=[{"x": c, "y": r, "text": CLASS_NAMES[r * 13 + c], "showarrow": False,
                                       "font": {"size": 8, "color": theme.INK_2}} for r in range(13) for c in range(13)])
    layout["margin"] = {"l": 6, "r": 6, "t": 36, "b": 6}
    return {"data": data, "layout": layout}


def fmt_cards(r: Any) -> str:
    from pokerlab.cards.cards import CARD_STR

    return CARD_STR[int(r.c1)] + CARD_STR[int(r.c2)]


def line_label(r: Any) -> str:
    from pokerlab.engine.flags import LINE_LABELS, STREET_END_LABELS

    return f"{LINE_LABELS[int(r.line)]} ({STREET_END_LABELS[int(r.street_end)]})"


# --------------------------------------------------------------------------------------- Labo
def _best_style(color: str) -> str:
    return f' class="best" style="background:{color}2e;box-shadow:inset 3px 0 0 {color}"'


def labo(cells: pd.DataFrame, rob: pd.DataFrame, colors: dict[str, str], labels: dict[str, str], campaigns: list[Campaign],
         report_links: dict[str, str], sweeps: list[Campaign], ablations: list[Campaign]) -> str:
    parts = ["<h1>Labo</h1>"]
    sw = lambda s: f'<span class="sw" style="background:{colors.get(s, theme.MUTED)}"></span>{esc(labels.get(s, s))}'  # noqa: E731
    if cells.empty:
        parts.append('<div class="banner">Aucune campagne compare avec résultats : lance <code>python -m pokerlab compare</code>.</div>')
    else:
        comps = sorted(cells["composition"].unique())
        parts.append("<h2>Classement par composition</h2>")
        boxes = []
        for comp in comps:
            sub = cells[cells["composition"] == comp].sort_values("bb100", ascending=False)
            rows = []
            for rank, (_, r) in enumerate(sub.iterrows(), 1):
                style = _best_style(colors.get(r["strategy"], theme.MUTED)) if rank == 1 else ""
                rows.append(f"<tr><td>{rank}</td><td class=\"l\">{sw(r['strategy'])}</td><td{style}>{num(r['bb100'], 1, True)} ± {num(r['ci'])}</td>"
                            f"<td>{count(r['mains'])}</td></tr>")
            boxes.append(f'<div class="card tbl"><h3>{esc(comp)}</h3><table><thead><tr><th>#</th><th class="l">stratégie</th><th>bb/100 ± IC</th><th>mains</th></tr></thead>'
                         f'<tbody>{"".join(rows)}</tbody></table></div>')
        parts.append(f'<div class="cols">{"".join(boxes)}</div>')
        parts.append('<p class="legend-note">bb/100 all-in adjusted ± IC 95 % (duplicate). La meilleure valeur de chaque composition est colorée '
                     "de la couleur de sa stratégie.</p>")
        if not rob.empty:
            n_comp = len(comps)
            rows = []
            for k, (_, r) in enumerate(rob.iterrows()):
                complete = int(r["compositions"]) == n_comp
                style = _best_style(colors.get(r["strategy"], theme.MUTED)) if k == 0 and complete else ""
                note = "" if complete else f" <span class=\"tag\">({int(r['compositions'])}/{n_comp} compositions)</span>"
                rows.append(f"<tr><td class=\"l\">{sw(r['strategy'])}{note}</td><td{style}>{num(r['score_robustesse'], 1, True)}</td>"
                            f"<td>{esc(r['pire_composition'])}</td><td>{num(r['rang_moyen'], 1)}</td></tr>")
            parts.append("<h2>Score de robustesse</h2>")
            parts.append('<div class="card tbl"><table><thead><tr><th class="l">stratégie</th><th>score</th><th>pire table</th><th>rang moyen</th></tr></thead>'
                         f'<tbody>{"".join(rows)}</tbody></table><p class="legend-note">Score = pire borne basse de l\'IC 95 % sur les {n_comp} compositions '
                         "mesurées : le gain qu'on peut garantir quelle que soit la table. Un score n'est comparable qu'entre stratégies mesurées sur "
                         "toutes les compositions.</p></div>")
        strats = [s for s in labels if s in set(cells["strategy"]) and s != "random_legal"]
        z = np.array([[float(cells[(cells["strategy"] == s) & (cells["composition"] == c)]["bb100"].iloc[0])
                       if ((cells["strategy"] == s) & (cells["composition"] == c)).any() else np.nan for c in comps] for s in strats])
        parts.append(figure(charts.matrix_heatmap(z, [labels.get(s, s) for s in strats], comps, "Stratégie × composition (bb/100)"),
                            "Le témoin aléatoire est retiré de la carte (il écraserait l'échelle de couleurs) ; il reste dans les classements."))
    for c in sweeps:
        grid = c.meta.get("grille") or {}
        pts = grid.get("points", [])
        prm = grid.get("params", [])
        if not pts:
            continue
        title = f"Sweep {labels.get(str(c.meta.get('strategie')), c.meta.get('strategie', ''))} ({c.meta.get('composition', '')})"
        color = colors.get(str(c.meta.get("strategie")), theme.CATEGORICAL_LIGHT[0])
        parts.append(f"<h2>{esc(title)}</h2>")
        if len(prm) == 1:
            parts.append(figure(charts.line_with_ci([p["values"][0] for p in pts], [p["bb100"] for p in pts], [p["ci"] for p in pts], color,
                                                    title, prm[0]["label"])))
        else:
            v0, v1 = prm[0]["values"], prm[1]["values"]
            zz = np.full((len(v0), len(v1)), np.nan)
            text = np.full((len(v0), len(v1)), "", dtype=object)
            for p in pts:
                a, b = v0.index(p["values"][0]), v1.index(p["values"][1])
                zz[a, b] = p["bb100"]
                text[a, b] = f"{p['bb100']:+.1f} ± {p['ci']:.1f}"
            parts.append(figure(charts.matrix_heatmap(zz, [str(v) for v in v0], [str(v) for v in v1],
                                                      f"{prm[0]['label']} (lignes) × {prm[1]['label']} (colonnes)", text=text),
                                "bb/100 ± IC 95 % de chaque réglage (mêmes donnes pour tous : comparaisons appariées)."))
    for c in ablations:
        res = c.meta.get("resultats") or []
        if not res:
            continue
        title = f"Ablation {labels.get(str(c.meta.get('strategie')), c.meta.get('strategie', ''))} ({c.meta.get('composition', '')})"
        color = colors.get(str(c.meta.get("strategie")), theme.CATEGORICAL_LIGHT[0])
        parts.append(f"<h2>{esc(title)}</h2>")
        parts.append(figure(charts.bars_with_ci([r["regle"] for r in res], [r["apport_bb100"] for r in res], [r["ci"] for r in res], color,
                                                "Ce que rapporte chaque règle (complet − sans la règle)", "bb/100"),
                            " ; ".join(f"{r['regle']} : {'significatif' if r['significatif'] else 'non significatif'} (p Holm {r['p_holm']:.3g})" for r in res)))
    parts.append("<h2>Campagnes</h2>")
    items = []
    for c in campaigns:
        link = report_links.get(str(c.folder))
        name = f'<a href="{link}">{esc(c.name)}</a>' if link else esc(c.name)
        items.append(f'<li><b>{name}</b> · {esc(c.experience)} · {c.date:%d/%m/%Y %H:%M}<br><span class="tag">{esc(campaign_params(c))}</span></li>')
    parts.append(f'<div class="card"><ul class="camps">{"".join(items)}</ul><p class="legend-note">Le lien ouvre le rapport détaillé de la campagne.</p></div>')
    return "\n".join(parts)


# -------------------------------------------------------------------------------------- Mains
def mains(example: bool, sessions: list[int], positions: list[str]) -> str:
    banner = ('<div class="banner warn"><b>Exemple :</b> aucune main réelle importée, voici les mains simulées d\'exemple.</div>' if example else "")
    opts_s = "".join(f'<option value="{s}">session {s}</option>' for s in sessions)
    opts_p = "".join(f'<option value="{p}">{p}</option>' for p in positions)
    return f"""<h1>Mains{' <span class="tag-ex">EXEMPLE</span>' if example else ''}</h1>
{banner}
<div class="filters" role="search">
<label class="sr" for="q">Recherche</label><input id="q" type="search" placeholder="cartes (AsKd, AKo), numéro, tableau">
<label class="sr" for="f-session">Session</label><select id="f-session"><option value="">toutes les sessions</option>{opts_s}</select>
<label class="sr" for="f-pos">Position</label><select id="f-pos"><option value="">toutes positions</option>{opts_p}</select>
<label class="sr" for="f-res">Résultat</label><select id="f-res"><option value="">gagnées et perdues</option><option value="g">gagnées</option><option value="p">perdues</option></select>
<label class="sr" for="f-hf">Hors fiche</label><select id="f-hf"><option value="">fiche : toutes</option><option value="true">hors fiche</option><option value="false">conformes</option></select>
</div>
<p class="sub" id="count"></p>
<div id="replay" class="card" style="display:none">
<h3 id="rp-title"></h3>
<div id="felt" class="felt"></div>
<div class="controls"><button id="prev" type="button">◀ précédent</button><span id="step" class="sub"></span><button id="next" type="button">suivant ▶</button></div>
<ol id="steps" class="steps"></ol>
</div>
<div class="card tbl"><table id="hands" class="hands"><thead><tr><th>date</th><th>session</th><th>position</th><th>cartes</th><th>tableau</th><th>résultat (bb)</th><th>hors fiche</th></tr></thead><tbody></tbody></table>
<p><button id="more" type="button" class="theme" style="float:none">afficher plus</button></p></div>"""
