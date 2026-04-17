"""PDE solvers built on the sparse Cholesky factorization in ``kolesky``.

Port of https://github.com/yifanc96/PDEs-GP-KoleskySolver (Chen, Owhadi,
Schäfer, *Math. Comp.* 2025; arXiv:2304.01294).

Each PDE exposes a dataclass describing the equation and a `solve_*`
function that takes sample points, an initial iterate, and factorization
hyper-parameters. The nonlinear elliptic and variable-coefficient
elliptic solvers are **dimension-agnostic**; `Burgers1d` and
`MongeAmpere2d` are tied to their respective dimension.

    from kolesky.pde import NonlinElliptic, solve_nonlin_elliptic
    eqn = NonlinElliptic(alpha=1.0, m=3, domain=((0, 1), (0, 1)),
                         bdy=..., rhs=...)
    sol = solve_nonlin_elliptic(eqn, kernel, X_domain, X_boundary,
                                sol_init, rho_big=3, rho_small=3,
                                k_neighbors=3, backend='auto')

Supported backends: 'cpu' (scipy / numpy), 'jax' (JAX, auto-picks CPU
or CUDA per ``jax.default_backend()``), or 'auto'.
"""

from .pdes import NonlinElliptic, NonlinElliptic2d
from .sampling import sample_points_grid_2d, sample_points_rdm_2d
from .nonlin_elliptic import solve_nonlin_elliptic, iterGPR_exact
# Historical aliases: the original release used `_2d`-suffixed names.
solve_nonlin_elliptic_2d = solve_nonlin_elliptic
from .varlin_elliptic import VarLinElliptic, solve_var_lin_elliptic
VarLinElliptic2d = VarLinElliptic
solve_var_lin_elliptic_2d = solve_var_lin_elliptic
from .burgers import Burgers1d, solve_burgers_1d, sample_points_grid_1d
from .monge_ampere import MongeAmpere2d, solve_monge_ampere_2d
from .pcg_ops import (
    BigFactorOperator,
    LiftedThetaTrainMatVec,
    SmallPrecond,
)

__all__ = [
    # equations (d-agnostic)
    "NonlinElliptic",
    "VarLinElliptic",
    # equations (dimension-specific)
    "Burgers1d",
    "MongeAmpere2d",
    # solvers
    "solve_nonlin_elliptic",
    "solve_var_lin_elliptic",
    "solve_burgers_1d",
    "solve_monge_ampere_2d",
    "iterGPR_exact",
    # sample-point helpers
    "sample_points_grid_2d",
    "sample_points_rdm_2d",
    "sample_points_grid_1d",
    # reusable operators
    "BigFactorOperator",
    "LiftedThetaTrainMatVec",
    "SmallPrecond",
    # historical aliases (still exported)
    "NonlinElliptic2d",
    "VarLinElliptic2d",
    "solve_nonlin_elliptic_2d",
    "solve_var_lin_elliptic_2d",
]
