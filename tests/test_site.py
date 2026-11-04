"""Tableau de bord : génération (smoke + exemple), liens, aucune requête réseau, 390 px dans Chromium, serveur."""

from __future__ import annotations

import json
import shutil
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from pokerlab.config import PROJECT_ROOT, LabConfig
from pokerlab.journal.store import import_paths
from pokerlab.site.build import build_site, site_config
from pokerlab.site.check import check_links, chromium_available, chromium_check, problems
from pokerlab.site.data import Reference, expected_cone, verdict_vs_cone

SMOKE = PROJECT_ROOT / "out" / "smoke"
EXAMPLE = PROJECT_ROOT / "journal" / "exemple_pokerstars.txt"


@pytest.fixture(scope="module")
def lab() -> LabConfig:
    return LabConfig.load()


def _cfg(tmp: Path, campaigns: Path, real: Path | None = None) -> dict:
    cfg = site_config()
    cfg.update({"campagnes": str(campaigns), "sortie": str(tmp / "site"), "exclure": [], "cone_tirages": 300,
                "donnees_reelles": str(real or tmp / "rien"), "exemple": str(tmp / "exemple"), "fiche_campagne": None,
                "dossier_import": str(tmp / "depot")})
    return cfg


@pytest.fixture(scope="module")
def smoke_site(tmp_path_factory, lab) -> Path:
    if not (SMOKE / "compare" / "resume.json").exists():
        pytest.skip("out/smoke absent : lance python -m pokerlab smoke")
    tmp = tmp_path_factory.mktemp("site")
    return build_site(_cfg(tmp, SMOKE), lab=lab)


def test_site_pages_and_links(smoke_site: Path) -> None:
    for page in ("index.html", "exemple.html", "labo.html", "mains.html", "fiche.html", "version.json",
                 "assets/plotly.js", "assets/site.css", "assets/site.js", "assets/mains.js", "data/mains.js"):
        assert (smoke_site / page).exists(), page
    res = check_links(smoke_site)
    assert res["liens_casses"] == []
    assert res["references_externes"] == []
    # plotly embarqué une seule fois, dans le fichier partagé
    for page in smoke_site.glob("*.html"):
        assert page.stat().st_size < 1_000_000, page.name
    index = (smoke_site / "index.html").read_text(encoding="utf-8")
    assert "Aucune main réelle importée" in index and "exemple.html" in index
    ex = (smoke_site / "exemple.html").read_text(encoding="utf-8")
    assert "EXEMPLE" in ex and "Mes gains face à la variance attendue" in ex
    labo = (smoke_site / "labo.html").read_text(encoding="utf-8")
    has_cells = "resultats" in json.loads((SMOKE / "compare" / "resume.json").read_text(encoding="utf-8"))
    assert ("Classement par composition" if has_cells else "Aucune campagne compare") in labo
    assert "rapports/" in labo  # rapports hors de la racine servie : copiés dans le site


def test_site_with_real_hands(tmp_path: Path, lab) -> None:
    real = tmp_path / "reel"
    import_paths([EXAMPLE], lab, "joueur", real, None, 30.0)
    campaigns = SMOKE if (SMOKE / "compare" / "resume.json").exists() else tmp_path / "vide"
    site = build_site(_cfg(tmp_path, campaigns, real), lab=lab)
    index = (site / "index.html").read_text(encoding="utf-8")
    assert "Mes gains face à la variance attendue" in index and "EXEMPLE" not in index
    assert "Mes 5 situations les plus perdantes" in index
    data = (site / "data" / "mains.js").read_text(encoding="utf-8")
    hands = json.loads(data.split("window.HANDS = ")[1].split(";\n")[0])
    assert len(hands) == 120 and {"id", "session", "position", "cards", "result", "hors_fiche"} <= set(hands[0])
    assert check_links(site)["liens_casses"] == []


@pytest.mark.skipif(not chromium_available(), reason="Node + Playwright indisponibles")
def test_site_chromium_390(smoke_site: Path) -> None:
    data = (smoke_site / "data" / "mains.js").read_text(encoding="utf-8")
    first = json.loads(data.split("window.HANDS = ")[1].split(";\n")[0])[0]["id"]
    pages = ["index.html", "exemple.html", "labo.html", "mains.html", f"mains.html#{first}", "fiche.html"]
    res = chromium_check(smoke_site.resolve().as_uri(), pages)
    assert res is not None
    assert problems(res) == []
    replay = [r for r in res if r["page"].startswith("mains.html#")]
    assert replay and all(r["replay"] and r["next"] and r["seats"] == 6 for r in replay)


def test_serve_redirect_and_auto_rebuild(tmp_path: Path, lab) -> None:
    from pokerlab.site.serve import Watcher, make_server

    cfg = _cfg(tmp_path, tmp_path / "campagnes", tmp_path / "reel")
    (tmp_path / "campagnes").mkdir()
    site = build_site(cfg, lab=lab)
    server = make_server(site, 0)
    th = threading.Thread(target=server.serve_forever, daemon=True)
    th.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as r:  # suit la redirection vers /site/
            assert r.status == 200 and "Mon jeu" in r.read().decode("utf-8")
        watcher = Watcher(cfg, site, lab, 1.0)
        assert watcher.step() is False
        before = json.loads((site / "version.json").read_text())["version"]
        depot = tmp_path / "depot"
        depot.mkdir(exist_ok=True)
        shutil.copy(EXAMPLE, depot / "historique.txt")
        assert watcher.step() is True
        assert (depot / "importes" / "historique.txt").exists()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/site/index.html") as r:
            assert "Mes gains face à la variance attendue" in r.read().decode("utf-8")
        camp = tmp_path / "campagnes" / "nouvelle"
        camp.mkdir()
        (camp / "resume.json").write_text(json.dumps({"experience": "audit", "config": {"strategy": "joueur"}}))
        assert watcher.step() is True
        assert "nouvelle" in (site / "labo.html").read_text(encoding="utf-8")
        assert json.loads((site / "version.json").read_text())["version"] >= before
    finally:
        server.shutdown()
        server.server_close()


def test_cone_and_verdict() -> None:
    rng = np.random.default_rng(1)
    ref = Reference("joueur", "x", 0.5, 10.0, 50.0, 3.0, {}, rng.normal(0.5, 10.0, 5000), "")
    cone = expected_cone(ref, 400, 500, 7)
    assert cone["methode"] == "simulation"
    assert cone["p5"][-1] < cone["p50"][-1] < cone["p95"][-1]
    assert abs(cone["p50"][-1] - 400 * ref.sample.mean()) < 30  # médiane ≈ n × moyenne de l'échantillon
    assert verdict_vs_cone(cone["p5"][-1] - 1, cone)[1] == "warn"
    assert verdict_vs_cone(cone["p50"][-1], cone)[1] == "ok"
    assert ref.hands_needed == pytest.approx((1.959964 * 10 / 0.5) ** 2, rel=1e-3)
