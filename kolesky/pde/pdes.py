"""PDE definitions.

`NonlinElliptic` and `VarLinElliptic` are dimension-agnostic — the
solvers for them work in any `d`. `MongeAmpere2d` and `Burgers1d` are
intrinsically tied to their dimension (the first needs a 2-D Hessian,
the second is a 1-D time-stepping problem), so they keep the `2d`/`1d`
suffix. The historical `NonlinElliptic2d` / `VarLinElliptic2d` names
remain as aliases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple


# N-dim axis-aligned bounding box: a tuple of (lo, hi) pairs, one per axis.
DomainBBox = Tuple[Tuple[float, float], ...]


@dataclass
class NonlinElliptic:
    """-Δu + α u^m = f  on Ω,  u = bdy on ∂Ω.

    Works for any spatial dimension `d` — the `domain` field is a
    length-`d` tuple of `(lo, hi)` pairs and is *optional*: the solver
    doesn't look at it (it only sees the `X_domain` / `X_boundary`
    point clouds). The 2-D grid-sampling helper `sample_points_grid_2d`
    does use it.
    """

    alpha: float
    m: int
    domain: DomainBBox
    bdy: Callable[[Sequence[float]], float]
    rhs: Callable[[Sequence[float]], float]

    @property
    def d(self) -> int:
        return len(self.domain)


# Backwards-compatible alias (the original name mirrored the Julia
# main_NonLinElliptic2d.jl); kept so callers upgrading from the first
# release don't have to rename.
NonlinElliptic2d = NonlinElliptic
