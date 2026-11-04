"""Comportements « humains » des archétypes, chacun désactivable dans la config (bloc ``humain``).

- ``tirage`` : chaque joueur tire ses paramètres autour de ceux de l'archétype (écart réglable,
  loi log-normale de moyenne 1) au début de chaque session ;
- ``mixte`` : décisions en fréquences (pas de seuil déterministe) : probabilité logistique autour du
  seuil de range préflop, et une part d'écarts autour du seuil de « tapis » postflop ;
- ``tilt`` : après une perte de plus de ``perte_bb`` sur une main, plus large et plus agressif pendant
  ``mains`` mains ;
- ``adaptation`` : observe le VPIP, le PFR et les bluffs montrés du héros sur une fenêtre glissante et
  ajuste ses abandons / paies face à lui.

L'état (tilt, fenêtre d'observation) ne change qu'en fin de main (``end_hand``) : une main rejouée
avec le même état donne les mêmes décisions (audit, historiques)."""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any

from pokerlab.config import deep_merge

HUMAN_DEFAULTS: dict[str, Any] = {
    "tirage": {"enabled": False, "ecart": 0.25},
    "mixte": {"enabled": False, "douceur": 0.04, "ecart_postflop": 0.10},
    "tilt": {"enabled": False, "perte_bb": 30.0, "mains": 12, "elargissement": 1.5, "agressivite": 1.6},
    "adaptation": {"enabled": False, "fenetre": 60, "min_mains": 15, "force": 1.0,
                   "vpip_ref": 0.25, "pfr_ref": 0.18, "bluff_ref": 0.25},
}

# Paramètres tirés par joueur ; les fréquences restent dans [0, 0.98].
DRAWN = ("vpip", "pfr", "three_bet", "four_bet", "cold_call_share", "fold_to_3bet", "aggression", "cbet", "fold_to_cbet",
         "bluff", "semi_bluff", "thin_value", "call_weak", "float", "raise_value", "check_raise", "bluff_raise",
         "bet_size_pot", "call_cap_bb", "bb_defend_cap_bb")
NOT_PROBA = {"bet_size_pot", "call_cap_bb", "bb_defend_cap_bb"}
LOOSER = ("vpip", "pfr", "three_bet")
MORE_AGGRESSIVE = ("aggression", "cbet", "bluff", "semi_bluff", "bluff_raise", "thin_value")


def human_config(config: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(HUMAN_DEFAULTS, config.get("humain") or {})


def is_stateful(config: dict[str, Any]) -> bool:
    """Vrai si l'archétype a besoin d'une instance par siège (paramètres tirés, tilt ou adaptation)."""
    h = config.get("humain") or {}
    return any(bool((h.get(k) or {}).get("enabled")) for k in ("tirage", "tilt", "adaptation"))


def clip(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def draw_params(base: dict[str, Any], ecart: float, seed: int) -> dict[str, Any]:
    """Paramètres d'un joueur : valeur × exp(écart × N(0,1) − écart²/2) (moyenne conservée)."""
    rng = random.Random(seed)
    out = dict(base)
    for key in DRAWN:
        v = float(base[key])
        if v <= 0:
            continue
        x = v * math.exp(ecart * rng.gauss(0.0, 1.0) - ecart * ecart / 2)
        out[key] = x if key in NOT_PROBA else clip(x, 0.0, 0.98)
    out["pfr"] = min(out["pfr"], out["vpip"])
    return out


class HeroWindow:
    """Ce qu'un adversaire a vu du héros sur les ``size`` dernières mains."""

    def __init__(self, size: int) -> None:
        self.hands: deque[tuple[int, int, int, int]] = deque(maxlen=max(1, size))

    def add(self, vpip: int, pfr: int, shown_aggr: int, shown_bluff: int) -> None:
        self.hands.append((vpip, pfr, shown_aggr, shown_bluff))

    def __len__(self) -> int:
        return len(self.hands)

    def rates(self, bluff_prior: float, prior_weight: float = 4.0) -> tuple[float, float, float]:
        n = len(self.hands)
        vpip = sum(h[0] for h in self.hands) / n
        pfr = sum(h[1] for h in self.hands) / n
        shown = sum(h[2] for h in self.hands)
        bluffs = sum(h[3] for h in self.hands)
        bluff = (bluffs + bluff_prior * prior_weight) / (shown + prior_weight)   # rétréci vers la référence : peu de mains montrées
        return vpip, pfr, bluff


class HumanState:
    """État d'un joueur humain simulé pendant une session."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.tilt_left = 0
        self.window = HeroWindow(int(cfg["adaptation"]["fenetre"]))
        self.my_seat: int | None = None
        self.hero_seat: int | None = None

    # ------------------------------------------------------------------ cycle de vie
    def end_hand(self, my_result_bb: float, hero_obs: tuple[int, int, int, int] | None) -> None:
        t = self.cfg["tilt"]
        if t["enabled"]:
            if self.tilt_left > 0:
                self.tilt_left -= 1
            if my_result_bb <= -float(t["perte_bb"]):
                self.tilt_left = int(t["mains"])
        if self.cfg["adaptation"]["enabled"] and hero_obs is not None:
            self.window.add(*hero_obs)

    # ------------------------------------------------------------------ effets
    @property
    def tilted(self) -> bool:
        return self.cfg["tilt"]["enabled"] and self.tilt_left > 0

    def apply_tilt(self, p: dict[str, Any]) -> dict[str, Any]:
        if not self.tilted:
            return p
        t = self.cfg["tilt"]
        q = dict(p)
        for k in LOOSER:
            q[k] = clip(float(p[k]) * float(t["elargissement"]), 0.0, 0.95)
        for k in MORE_AGGRESSIVE:
            q[k] = clip(float(p[k]) * float(t["agressivite"]), 0.0, 0.95)
        q["fold_to_cbet"] = clip(float(p["fold_to_cbet"]) / float(t["agressivite"]), 0.0, 0.95)
        return q

    def adaptation_factors(self) -> tuple[float, float] | None:
        """(préflop, postflop) : multiplicateurs de défense face au héros ; None si pas assez observé."""
        a = self.cfg["adaptation"]
        if not a["enabled"] or len(self.window) < int(a["min_mains"]):
            return None
        vpip, pfr, bluff = self.window.rates(float(a["bluff_ref"]))
        force = float(a["force"])
        pre = clip(1 + force * 0.5 * (pfr - float(a["pfr_ref"])) / float(a["pfr_ref"]), 0.6, 1.6)
        post = clip(1 + force * 0.5 * (bluff - float(a["bluff_ref"])) / float(a["bluff_ref"]), 0.5, 1.8)
        post *= clip(1 + force * 0.25 * (vpip - float(a["vpip_ref"])) / float(a["vpip_ref"]), 0.8, 1.3)
        return pre, post

    def mixed_below(self, pct: float, threshold: float, rng: random.Random) -> bool:
        """Seuil de range « mixte » : probabilité logistique au lieu de pct <= seuil."""
        s = float(self.cfg["mixte"]["douceur"])
        x = (threshold - pct) / s
        if x > 30:
            return True
        if x < -30:
            return False
        return rng.random() < 1.0 / (1.0 + math.exp(-x))
