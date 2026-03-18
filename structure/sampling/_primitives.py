"""Low-level unit-hypercube → value helpers used by the map functions."""

from __future__ import annotations
import math
from typing import Any, Sequence


def _cont(u: float, lo: float, hi: float, *, log: bool = False) -> float:
    """Map u ∈ [0,1] to [lo, hi], optionally in log-space."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        return lo
    if log:
        return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    return lo + u * (hi - lo)


def _cat(u: float, choices: Sequence[Any]) -> Any:
    """Map u ∈ [0,1] to one element of choices (uniform)."""
    idx = min(int(u * len(choices)), len(choices) - 1)
    return choices[idx]


def _bool(u: float, p_true: float = 0.5) -> bool:
    return u < p_true


def _int(u: float, lo: int, hi: int) -> int:
    """Map u ∈ [0,1] to an integer in [lo, hi] inclusive."""
    return min(hi, max(lo, int(lo + u * (hi - lo + 1))))
