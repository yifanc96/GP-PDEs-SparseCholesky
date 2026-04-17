"""PDE definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple


@dataclass
class NonlinElliptic2d:
    """-Δu + α u^m = f on Ω, u = bdy on ∂Ω.

    Matches the Julia struct of the same name in main_NonLinElliptic2d.jl.
    """

    alpha: float
    m: int
    domain: Tuple[Tuple[float, float], Tuple[float, float]]  # ((x1l, x1r), (x2l, x2r))
    bdy: Callable[[Sequence[float]], float]
    rhs: Callable[[Sequence[float]], float]

    @property
    def d(self) -> int:
        return 2
