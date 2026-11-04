"""Graines déterministes : dérivation stable (indépendante du processus et du hash Python)."""

from __future__ import annotations

import hashlib
import random
from typing import Iterable

_MASK64 = (1 << 64) - 1
_MASK63 = (1 << 63) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & _MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _MASK64
    return x ^ (x >> 31)


def _as_int(part: int | str) -> int:
    if isinstance(part, int):
        return part & _MASK64
    digest = hashlib.blake2b(str(part).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def derive_seed(*parts: int | str) -> int:
    """Dérive une graine 64 bits stable à partir d'entiers et de chaînes."""
    acc = 0x2545F4914F6CDD1D
    for part in parts:
        acc = _splitmix64(acc ^ _as_int(part))
    return acc & _MASK63


def stable_hash(obj: object) -> int:
    """Hash stable (blake2b) de la représentation ``repr`` d'un objet."""
    digest = hashlib.blake2b(repr(obj).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def shuffled_deck(seed: int) -> list[int]:
    """Paquet de 52 cartes (entiers 0..51) mélangé de façon reproductible."""
    deck = list(range(52))
    random.Random(seed).shuffle(deck)
    return deck


def uniform(seed: int) -> float:
    """Tirage uniforme [0, 1) déterministe à partir d'une graine."""
    return (_splitmix64(seed) >> 11) * (1.0 / (1 << 53))


def seeds(base: int, items: Iterable[int | str]) -> list[int]:
    """Liste de graines dérivées d'une base."""
    return [derive_seed(base, item) for item in items]
