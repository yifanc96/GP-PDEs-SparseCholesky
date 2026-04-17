"""kolesky — sparse approximate-Cholesky factorization for GP-based PDE solvers.

This is the Python port of KoLesky.jl (Schäfer, Katzfuss, Owhadi,
[arXiv:2004.14455](https://arxiv.org/abs/2004.14455)) combined with the
Gauss-Newton / pCG PDE solvers from Chen, Owhadi, Schäfer,
[arXiv:2304.01294](https://arxiv.org/abs/2304.01294).

Top-level API (fast sparse Cholesky of kernel matrices):

    from kolesky import (
        MaternCovariance5_2, GaussianCovariance,
        PointMeasurement, LaplaceDiracPointMeasurement,
        LaplaceGradDiracPointMeasurement, HessianDiracPointMeasurement,
        point_measurements,
        ImplicitKLFactorization, ExplicitKLFactorization,
        assemble_covariance,
    )

PDE solvers (Gauss-Newton + sparse Cholesky preconditioned CG):

    from kolesky.pde import (
        NonlinElliptic2d, solve_nonlin_elliptic_2d,
        VarLinElliptic2d, solve_var_lin_elliptic_2d,
        Burgers1d, solve_burgers_1d,
        MongeAmpere2d, solve_monge_ampere_2d,
    )
"""

from .measurements import (
    PointMeasurement,
    DeltaDiracPointMeasurement,
    LaplaceDiracPointMeasurement,
    LaplaceGradDiracPointMeasurement,
    PartialPartialPointMeasurement,
    HessianDiracPointMeasurement,
    point_measurements,
    stack_measurements,
    get_coordinates,
)
from .covariance import (
    AbstractCovarianceFunction,
    MaternCovariance1_2,
    MaternCovariance3_2,
    MaternCovariance5_2,
    MaternCovariance7_2,
    MaternCovariance9_2,
    MaternCovariance11_2,
    GaussianCovariance,
)
from .ordering import maximin_ordering
from .supernodes import (
    IndexSuperNode,
    supernodal_reverse_maximin_sparsity_pattern,
    ordering_and_sparsity_pattern,
)
from .factorization import (
    ImplicitKLFactorization,
    ExplicitKLFactorization,
    assemble_covariance,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # measurements
    "PointMeasurement",
    "DeltaDiracPointMeasurement",
    "LaplaceDiracPointMeasurement",
    "LaplaceGradDiracPointMeasurement",
    "PartialPartialPointMeasurement",
    "HessianDiracPointMeasurement",
    "point_measurements",
    "stack_measurements",
    "get_coordinates",
    # kernels
    "AbstractCovarianceFunction",
    "MaternCovariance1_2",
    "MaternCovariance3_2",
    "MaternCovariance5_2",
    "MaternCovariance7_2",
    "MaternCovariance9_2",
    "MaternCovariance11_2",
    "GaussianCovariance",
    # ordering / sparsity pattern
    "maximin_ordering",
    "IndexSuperNode",
    "supernodal_reverse_maximin_sparsity_pattern",
    "ordering_and_sparsity_pattern",
    # factorization
    "ImplicitKLFactorization",
    "ExplicitKLFactorization",
    "assemble_covariance",
]
