"""Serveur local du tableau de bord (``python -m pokerlab serve``) avec régénération automatique.

Serveur HTTP de la bibliothèque standard, lié à 127.0.0.1 uniquement. Un fil surveille les
résultats de campagne (resume.json, rapport.html), les mains réelles importées et le dossier
de dépôt des historiques PokerStars ; tout changement déclenche l'import puis la régénération.
"""

from __future__ import annotations

import functools
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from pokerlab.config import LabConfig
from pokerlab.journal.store import import_drop_folder, resolve
from pokerlab.site.build import build_site, site_config


def fingerprint(cfg: dict[str, Any], site: Path) -> tuple:
    """Empreinte (chemin, taille, date) des fichiers dont dépend le site."""
    root = resolve(cfg.get("campagnes", "out"))
    files: list[Path] = []
    for pattern in ("**/resume.json", "**/rapport.html"):
        files += [p for p in root.glob(pattern) if site not in p.parents]
    real = resolve(cfg.get("donnees_reelles", "data/reel")) / "mains.parquet"
    if real.exists():
        files.append(real)
    out = []
    for p in sorted(files):
        try:
            st = p.stat()
        except OSError:
            continue
        out.append((str(p), st.st_size, st.st_mtime_ns))
    return tuple(out)


class Watcher(threading.Thread):
    def __init__(self, cfg: dict[str, Any], site: Path, lab: LabConfig, interval: float) -> None:
        super().__init__(daemon=True)
        self.cfg, self.site, self.lab, self.interval = cfg, site, lab, interval
        self.last = fingerprint(cfg, site)
        self.stop = threading.Event()

    def step(self) -> bool:
        """Un tour de surveillance ; True si le site a été régénéré."""
        res = import_drop_folder(self.cfg.get("dossier_import", "data/a_importer"), self.lab, str(self.cfg.get("strategie_suivie", "joueur")),
                                 self.cfg.get("donnees_reelles", "data/reel"), self.cfg.get("hero"), float(self.cfg.get("pause_session_minutes", 30)))
        if res is not None:
            print(f"Import : {res['lues']} mains lues, {res['ignorees']} ignorées, {res['total']} au total", flush=True)
        now = fingerprint(self.cfg, self.site)
        if now == self.last and res is None:
            return False
        self.last = now
        t0 = time.time()
        build_site(self.cfg, lab=self.lab)
        print(f"Site régénéré ({time.time() - t0:.1f} s)", flush=True)
        return True

    def run(self) -> None:
        while not self.stop.wait(self.interval):
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001 - le serveur doit survivre à un fichier en cours d'écriture
                print(f"Régénération impossible pour l'instant : {exc}", flush=True)


class Handler(SimpleHTTPRequestHandler):
    site_prefix = "/site/"

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("", "/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", self.site_prefix)
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return


def make_server(site: Path, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    root = site.parent
    handler = type("SiteHandler", (Handler,), {"site_prefix": f"/{site.name}/"})
    return ThreadingHTTPServer((host, port), functools.partial(handler, directory=str(root)))


def serve(port: int | None = None, cfg: dict[str, Any] | None = None) -> int:
    cfg = dict(cfg or site_config())
    port = int(port or cfg.get("port", 8765))
    site = resolve(cfg.get("sortie", "out/site"))
    site.mkdir(parents=True, exist_ok=True)
    try:
        server = make_server(site, port)
    except OSError as exc:
        print(f"Port {port} indisponible ({exc.strerror}) : un autre serveur tourne déjà ? Essaie --port {port + 1}.")
        return 1
    lab = LabConfig.load()
    drop = resolve(cfg.get("dossier_import", "data/a_importer"))
    drop.mkdir(parents=True, exist_ok=True)
    import_drop_folder(drop, lab, str(cfg.get("strategie_suivie", "joueur")), cfg.get("donnees_reelles", "data/reel"), cfg.get("hero"),
                       float(cfg.get("pause_session_minutes", 30)))
    build_site(cfg, lab=lab)
    watcher = Watcher(cfg, site, lab, float(cfg.get("intervalle_surveillance_s", 3)))
    watcher.start()
    print(f"Tableau de bord : http://localhost:{port}/  (Ctrl+C pour arrêter)", flush=True)
    print(f"Dépose tes historiques PokerStars dans {drop} : import automatique.", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        print("Arrêt du serveur.")
    finally:
        watcher.stop.set()
        server.server_close()
    return 0
