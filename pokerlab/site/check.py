"""Vérifications du site généré : liens cassés, références réseau, rendu Chromium à 390 px."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

PAGES = ("index.html", "exemple.html", "labo.html", "mains.html", "fiche.html")


class _Refs(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for key, value in attrs:
            if key in ("href", "src") and value:
                self.refs.append(value)


def check_links(site: Path) -> dict[str, list[str]]:
    """Liens relatifs inexistants et références externes (http, https, //) dans les pages du site."""
    broken, external = [], []
    for page in sorted(site.rglob("*.html")):
        if "rapports" in page.relative_to(site).parts:
            continue
        parser = _Refs()
        parser.feed(page.read_text(encoding="utf-8"))
        for ref in parser.refs:
            parts = urlsplit(ref)
            if parts.scheme in ("http", "https") or ref.startswith("//"):
                external.append(f"{page.name} -> {ref}")
                continue
            if parts.scheme or not parts.path:
                continue
            target = (page.parent / unquote(parts.path)).resolve()
            if not target.exists():
                broken.append(f"{page.name} -> {ref}")
    for css in site.rglob("*.css"):
        for ref in re.findall(r"url\(([^)]+)\)", css.read_text(encoding="utf-8")):
            if "http" in ref:
                external.append(f"{css.name} -> {ref}")
    return {"liens_casses": broken, "references_externes": external}


def chromium_available() -> bool:
    if shutil.which("node") is None:
        return False
    try:
        subprocess.run(["node", "-e", "require('playwright')"], check=True, capture_output=True, timeout=30)
    except (subprocess.SubprocessError, OSError):
        return False
    return True


def chromium_check(base_url: str, pages: list[str], shots: Path | None = None, timeout: float = 240) -> list[dict[str, Any]] | None:
    """Lance le script Playwright ; None si Node ou Playwright manquent."""
    if not chromium_available():
        return None
    script = Path(__file__).with_name("chromium_check.js")
    cmd = ["node", str(script), base_url, ",".join(pages)] + ([str(shots)] if shots else [])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    return json.loads(out.stdout.strip().splitlines()[-1])


def problems(results: list[dict[str, Any]]) -> list[str]:
    """Erreurs console, requêtes externes, débordement horizontal, replay inopérant."""
    out = []
    for r in results:
        tag = f"{r['page']} ({r['scheme']})"
        out += [f"{tag} : erreur {e}" for e in r["errors"]]
        out += [f"{tag} : requête externe {u}" for u in r["external"]]
        if r["overflow"] > 0:
            out.append(f"{tag} : débordement horizontal de {r['overflow']} px à 390 px")
        if r["figures"] != r["rendered"]:
            out.append(f"{tag} : {r['figures'] - r['rendered']} graphique(s) non rendu(s)")
        if r.get("rows") and not (r.get("replay") and r.get("next")):
            out.append(f"{tag} : replay inopérant")
    return out


def full_check(site: Path, shots: Path | None = None) -> dict[str, Any]:
    """Liens + Chromium (si disponible) sur les pages présentes."""
    res: dict[str, Any] = check_links(site)
    pages = [p for p in PAGES if (site / p).exists()]
    hands = site / "data" / "mains.js"
    if hands.exists():
        txt = hands.read_text(encoding="utf-8")
        m = re.search(r'"id":"([^"]+)"', txt)
        if m:
            pages.append(f"mains.html#{m.group(1)}")
    chrome = chromium_check(site.resolve().as_uri(), pages, shots)
    res["chromium"] = "indisponible (Node + Playwright requis)" if chrome is None else "ok"
    res["problemes_chromium"] = [] if chrome is None else problems(chrome)
    res["ok"] = not (res["liens_casses"] or res["references_externes"] or res["problemes_chromium"])
    return res
