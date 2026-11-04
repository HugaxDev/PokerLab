"""Sections de rapport réutilisées par les expériences (stratégies comparées)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from pokerlab.analysis import decomposition as deco
from pokerlab.analysis.leaks import find_leaks, format_ci, leak_examples, leak_title, situation_labels
from pokerlab.analysis.player_stats import STAT_LABELS, stats_from_hands
from pokerlab.analysis.stats import WinrateSummary, max_drawdown, summarize, z_value
from pokerlab.cards.cards import CARD_STR
from pokerlab.engine.flags import LINE_LABELS, STREET_END_LABELS
from pokerlab.report import charts, theme
from pokerlab.report.html import Section, esc, fmt
from pokerlab.strategies.registry import strategy_meta


def colors_labels(specs: Sequence[Any]) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Noms, couleurs fixes et libellés des stratégies (ordre de la config)."""
    metas = [strategy_meta(s) for s in specs]
    names = [m["name"] for m in metas]
    colors = theme.strategy_colors(names, {m["name"]: m.get("color") for m in metas})
    labels = {m["name"]: m["label"] for m in metas}
    return names, colors, labels


@dataclass
class StrategySummary:
    """Résumé chiffré d'une stratégie dans une cellule (composition)."""

    name: str
    ev: WinrateSummary
    real: WinrateSummary
    stats: dict[str, float]
    drawdown: float
    longest_below_peak: int


def summarize_strategy(hands: pd.DataFrame, name: str, confidence: float, duplicate: bool = True) -> StrategySummary:
    """bb/100 (deck = unité appariée en duplicate), σ, mains nécessaires, stats, drawdown."""
    if duplicate and "deck" in hands:
        per = hands.groupby("deck")[["result_ev", "result"]].mean()
        hpu = len(hands) / max(len(per), 1)
        ev = summarize(per["result_ev"].to_numpy(), hands["result_ev"].to_numpy(), hpu, confidence)
        real = summarize(per["result"].to_numpy(), hands["result"].to_numpy(), hpu, confidence)
    else:
        ev = summarize(hands["result_ev"].to_numpy(), hands["result_ev"].to_numpy(), 1, confidence)
        real = summarize(hands["result"].to_numpy(), hands["result"].to_numpy(), 1, confidence)
    dd, longest = max_drawdown(np.cumsum(hands["result"].to_numpy(dtype=float)))
    return StrategySummary(name, ev, real, stats_from_hands(hands), dd, longest)


TABLE_ROWS: list[tuple[str, Callable[[StrategySummary], float], str, int]] = [
    ("bb/100 all-in adjusted", lambda s: s.ev.bb100, "max", 1),
    ("IC 95 % (±)", lambda s: s.ev.ci_half, "min", 1),
    ("bb/100 réel", lambda s: s.real.bb100, "max", 1),
    ("σ brut (bb/100)", lambda s: s.real.sd_hand_bb100, "min", 1),
    ("σ effectif duplicate+EV (bb/100)", lambda s: s.ev.sd_effective_bb100, "min", 1),
    ("mains jouées", lambda s: float(s.ev.hands), "", 0),
    ("mains pour être significatif", lambda s: s.ev.hands_needed, "min", 0),
    ("drawdown max (bb)", lambda s: s.drawdown, "min", 1),
    ("plus longue série perdante (mains sous le sommet)", lambda s: float(s.longest_below_peak), "min", 0),
    ("VPIP %", lambda s: s.stats["VPIP"], "", 1),
    ("PFR %", lambda s: s.stats["PFR"], "", 1),
    ("3-bet %", lambda s: s.stats["3bet"], "", 1),
    ("AF", lambda s: s.stats["AF"], "", 2),
    ("WTSD %", lambda s: s.stats["WTSD"], "", 1),
    ("W$SD %", lambda s: s.stats["W$SD"], "", 1),
    ("C-bet %", lambda s: s.stats["cbet"], "", 1),
    ("Fold to c-bet %", lambda s: s.stats["fold_cbet"], "", 1),
]


def _tint(color: str) -> str:
    return color + "2e"


def comparison_table(section: Section, summaries: Mapping[str, StrategySummary], colors: Mapping[str, str], labels: Mapping[str, str], caption: str = "") -> None:
    """Lignes = métriques, colonnes = stratégies ; la meilleure valeur prend la couleur de sa stratégie."""
    names = list(summaries)
    head = "<th>métrique</th>" + "".join(
        f'<th><span class="sw" style="background:{colors[n]}"></span>{esc(labels.get(n, n))}</th>' for n in names
    )
    rows = []
    for label, getter, direction, digits in TABLE_ROWS:
        values = {n: getter(summaries[n]) for n in names}
        finite = {n: v for n, v in values.items() if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))}
        best = None
        if direction and len(finite) > 1:
            best = (max if direction == "max" else min)(finite, key=finite.get)
        cells = [f"<td>{esc(label)}</td>"]
        for n in names:
            v = values[n]
            text = fmt(float(v), digits) if v is not None else "–"
            if n == best:
                cells.append(f'<td class="best" style="background:{_tint(colors[n])};box-shadow:inset 3px 0 0 {colors[n]}">{text}</td>')
            else:
                cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    cap = f'<p class="legend-note">{esc(caption)}</p>' if caption else ""
    section.html(f'<div class="card tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>{cap}</div>')


def winrate_figure(summaries: Mapping[str, StrategySummary], colors: Mapping[str, str], labels: Mapping[str, str]) -> dict:
    names = list(summaries)
    return charts.winrate_bars(
        names,
        [summaries[n].ev.bb100 for n in names],
        [summaries[n].ev.ci_half for n in names],
        [summaries[n].real.bb100 for n in names],
        colors,
        labels,
    )


def curve_payload(hands: pd.DataFrame, column: str, max_points: int) -> dict[str, Any]:
    """Courbe cumulée sous-échantillonnée (les cumuls restent exacts aux points gardés)."""
    values = hands[column].to_numpy(dtype=float)
    cum = np.cumsum(values)
    n = len(values)
    step = max(1, math.ceil(n / max_points))
    idx = np.arange(step - 1, n, step) if n else np.array([], dtype=int)
    if n and idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    sub = hands.iloc[idx]
    custom = [
        [int(h), p, CARD_STR[int(a)] + CARD_STR[int(b)], f"{LINE_LABELS[int(li)]} ({STREET_END_LABELS[int(se)]})", f"{v:+.2f}"]
        for h, p, a, b, li, se, v in zip(sub["hand_id"], sub["position"], sub["c1"], sub["c2"], sub["line"], sub["street_end"], sub[column])
    ]
    return {"x": (idx + 1).tolist(), "y": cum[idx].tolist(), "custom": custom}


def add_curves(section: Section, by_name: Mapping[str, pd.DataFrame], colors: Mapping[str, str], labels: Mapping[str, str], max_points: int) -> None:
    curves = {n: curve_payload(h, "result_ev", max_points) for n, h in by_name.items()}
    section.figure(charts.cumulative_lines(curves, colors, labels, "Gains cumulés all-in adjusted"),
                   "Survol : main, position, cartes, ligne jouée, montant, cumul. Courbe sous-échantillonnée si besoin.")
    real = {n: curve_payload(h, "result", max_points) for n, h in by_name.items()}
    section.figure(charts.cumulative_lines(real, colors, labels, "Gains cumulés réels"), "Mêmes mains, résultats réels (avec la chance des tapis).")


def add_showdown_lines(section: Section, by_name: Mapping[str, pd.DataFrame], labels: Mapping[str, str], max_points: int) -> None:
    for n, h in by_name.items():
        total = np.cumsum(h["result"].to_numpy(dtype=float))
        sd, nsd = deco.showdown_split(h, "result")
        k = max(1, math.ceil(len(total) / max_points))
        idx = np.arange(0, len(total), k)
        section.figure(charts.showdown_lines((idx + 1).tolist(), total[idx].tolist(), sd[idx].tolist(), nsd[idx].tolist(),
                                             f"{labels.get(n, n)} : avec / sans showdown"))


def add_decomposition(section: Section, by_name: Mapping[str, pd.DataFrame], colors: Mapping[str, str], labels: Mapping[str, str], n_players: int) -> None:
    names = list(by_name)
    pos = {n: deco.by_position(by_name[n], n_players) for n in names}
    categories = list(pos[names[0]]["position"]) if names else []
    section.figure(charts.grouped_bars(categories, {n: pos[n].set_index("position").reindex(categories)["bb100"].tolist() for n in names},
                                       colors, labels, "bb/100 par position", "bb/100"))
    street = {n: deco.by_street(by_name[n]) for n in names}
    section.figure(charts.grouped_bars(list(STREET_END_LABELS), {n: street[n]["bb_per_100_total"].tolist() for n in names}, colors, labels,
                                       "Contribution au bb/100 selon la street où la main se termine", "bb/100"))
    lines = {n: deco.by_line(by_name[n]) for n in names}
    section.figure(charts.grouped_bars(list(LINE_LABELS), {n: lines[n]["bb_per_100_total"].tolist() for n in names}, colors, labels,
                                       "Value bet / bluff / semi-bluff / paie : contribution au bb/100", "bb/100"),
                   "Ligne = dernière action volontaire postflop. « value bet » : mise avec paire moyenne ou mieux.")
    groups = {n: deco.by_group(by_name[n]) for n in names}
    first = groups[names[0]] if names else pd.DataFrame()
    section.figure(charts.grouped_bars(list(first.get("groupe", [])), {n: groups[n]["bb_per_100_total"].tolist() for n in names}, colors, labels,
                                       "Contribution au bb/100 par catégorie de main", "bb/100"))
    grids = {n: deco.by_class(by_name[n]) for n in names}
    section.figure(charts.class_heatmaps(grids, labels, "Grille 13x13 : bb gagnées par main (all-in adjusted)"),
                   "Au-dessus de la diagonale : assorties ; en dessous : dépareillées. Menu : choisir la stratégie.")
    table = pd.DataFrame({"position": categories})
    for n in names:
        table[labels.get(n, n)] = pos[n].set_index("position").reindex(categories)["bb100"].to_numpy()
    section.table(table, "Table de lecture : bb/100 par position.")


def add_player_stats(section: Section, summaries: Mapping[str, StrategySummary], labels: Mapping[str, str]) -> None:
    rows = []
    for n, s in summaries.items():
        row = {"stratégie": labels.get(n, n)}
        row.update({STAT_LABELS[k]: s.stats[k] for k in STAT_LABELS})
        rows.append(row)
    section.table(pd.DataFrame(rows), "Statistiques mesurées sur les mains simulées (pas déclarées).")


def add_archetype_stats(section: Section, flags_by_name: Mapping[str, Sequence[int]], archetypes: Mapping[str, Mapping[str, Any]], exclude: Sequence[str] = ()) -> None:
    """Stats réelles des archétypes à côté des valeurs annoncées dans leur config."""
    from pokerlab.analysis.player_stats import stats_from_flags

    rows = []
    for name, flags in flags_by_name.items():
        if name in exclude:
            continue
        st = stats_from_flags(flags)
        cfg = archetypes.get(name, {})
        def target(key: str) -> float:
            return 100 * float(cfg[key]) if key in cfg else float("nan")

        entry_target = target("vpip") if cfg.get("limp") else target("pfr")
        rows.append({
            "archétype": cfg.get("label", name),
            "mains": st["hands"],
            "entrée pot non ouvert : cible": entry_target,
            "mesurée": st["unopened_entry"],
            "relance pot non ouvert : cible": target("pfr"),
            "mesurée ": st["unopened_raise"],
            "3-bet : cible": target("three_bet"),
            "mesuré": st["3bet"],
            "VPIP global": st["VPIP"],
            "PFR global": st["PFR"],
            "AF": st["AF"],
            "c-bet : param.": target("cbet"),
            "c-bet mesuré": st["cbet"],
            "fold c-bet : param.": target("fold_to_cbet"),
            "fold c-bet mesuré": st["fold_cbet"],
            "WTSD": st["WTSD"],
        })
    if rows:
        section.table(pd.DataFrame(rows), "Cibles = percentiles de mains de la config (comparables à la mesure en pot non ouvert). "
                      "VPIP/PFR globaux dépendent de la table : face à beaucoup de relances, on joue moins de pots non ouverts. "
                      "« param. » = probabilité de la règle, la fréquence mesurée inclut aussi les mises de value.")


def add_leaks(
    section: Section,
    decisions: Mapping[str, pd.DataFrame],
    labels: Mapping[str, str],
    replay: Callable[[str, pd.Series], str] | None,
    min_count: int,
    examples: int,
    top: int = 6,
    confidence: float = 0.95,
) -> int:
    """Fuites par stratégie, avec exemples de mains rejouées. Renvoie le nombre de fuites listées."""
    total = 0
    for name, dec in decisions.items():
        dec = situation_labels(dec) if not dec.empty else dec
        leaks = find_leaks(dec, min_count=min_count, confidence_z=z_value(confidence), top=top)
        section.html(f"<h3>{esc(labels.get(name, name))}</h3>")
        if leaks.empty:
            section.paragraph(f"Aucune situation récurrente (≥ {min_count} décisions) à EV négative détectée sur cet échantillon.")
            continue
        total += len(leaks)
        table = pd.DataFrame({
            "situation": [leak_title(r) for _, r in leaks.iterrows()],
            "décisions": leaks["count"].astype(int),
            "EV après décision (bb)": [format_ci(m, h) for m, h in zip(leaks["mean"], leaks["ci_half"])],
            "perte totale (bb)": leaks["sum"],
            "significatif": ["oui" if s else "non" for s in leaks["significant"]],
        })
        section.table(table, "EV après décision = gain all-in adjusted à partir de la décision (0 = abandonner la main à ce moment).")
        if replay is not None:
            for _, leak in leaks.iterrows():
                ex = leak_examples(dec, leak, examples)
                texts = []
                for _, row in ex.iterrows():
                    try:
                        texts.append(replay(name, row))
                    except Exception as exc:  # pragma: no cover - rejeu défensif
                        texts.append(f"(rejeu impossible : {exc})")
                body = "".join(f"<pre>{esc(t)}</pre>" for t in texts)
                section.details(f"Exemples : {leak_title(leak)}", body)
    return total
