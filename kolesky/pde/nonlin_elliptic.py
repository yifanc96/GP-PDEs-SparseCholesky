"""Nonlinear elliptic PDE solver via iterative GPR + sparse Cholesky + pCG.

Solves  -Δu + α u^m = f  with Dirichlet data in any spatial dimension.
Mirrors the Julia `iterGPR_fast_pcg` routine from main_NonLinElliptic2d.jl
but generalized: the Δδ measurement kernel takes `d` as a parameter and
the maximin ordering / sparsity pattern are dimension-agnostic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .. import (
    AbstractCovarianceFunction,
    ExplicitKLFactorization,
    ImplicitKLFactorization,
    measurements as _m,
)
from ..measurements import LaplaceDiracPointMeasurement, PointMeasurement

from .pdes import NonlinElliptic, NonlinElliptic2d  # NonlinElliptic2d is alias
from .pcg_ops import BigFactorOperator, SmallPrecond, ThetaTrainMatVec, predict_via_big


def _make_measurements_big(
    X_boundary: np.ndarray, X_domain: np.ndarray
):
    """3-set measurement list used for the 'big' factorization:
    (δ_boundary, δ_interior, -Δ_interior), all promoted to LaplaceDiracPointMeasurement.
    """
    N_bdy = X_boundary.shape[0]
    N_dom = X_domain.shape[0]
    bdy = LaplaceDiracPointMeasurement(
        coordinate=X_boundary,
        weight_laplace=np.zeros(N_bdy),
        weight_delta=np.ones(N_bdy),
    )
    d_int = LaplaceDiracPointMeasurement(
        coordinate=X_domain,
        weight_laplace=np.zeros(N_dom),
        weight_delta=np.ones(N_dom),
    )
    minus_laplace_int = LaplaceDiracPointMeasurement(
        coordinate=X_domain,
        weight_laplace=-np.ones(N_dom),
        weight_delta=np.zeros(N_dom),
    )
    return [bdy, d_int, minus_laplace_int]


def _make_measurements_small(
    X_boundary: np.ndarray, X_domain: np.ndarray, delta_coefs: np.ndarray
):
    """2-set measurement list for the 'small' factorization at the current GN step:
    (δ_boundary, (-Δ + c δ)_interior), c = eqn.α * eqn.m * sol_now^(m-1).
    """
    N_bdy = X_boundary.shape[0]
    N_dom = X_domain.shape[0]
    bdy = LaplaceDiracPointMeasurement(
        coordinate=X_boundary,
        weight_laplace=np.zeros(N_bdy),
        weight_delta=np.ones(N_bdy),
    )
    linearized = LaplaceDiracPointMeasurement(
        coordinate=X_domain,
        weight_laplace=-np.ones(N_dom),
        weight_delta=np.asarray(delta_coefs, dtype=np.float64),
    )
    return [bdy, linearized]


def _apply_vector_rhs(eqn, X):
    """Evaluate rhs or boundary function on rows of X."""
    return np.array([eqn(X[i]) for i in range(X.shape[0])], dtype=np.float64)


def solve_nonlin_elliptic(
    eqn: NonlinElliptic,
    kernel: AbstractCovarianceFunction,
    X_domain: np.ndarray,
    X_boundary: np.ndarray,
    sol_init: np.ndarray,
    nugget: float = 1e-10,
    GN_steps: int = 3,
    rho_big: float = 4.0,
    rho_small: float = 6.0,
    k_neighbors: int = 4,
    lambda_: float = 1.5,
    alpha: float = 1.0,
    backend: str = 'auto',
    pcg_tol: float = 1e-6,
    pcg_maxiter: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """Iterative GPR + fast sparse Cholesky + pCG for -Δu + eqn.α * u^m = f.

    Parameters
    ----------
    backend : {'auto', 'cpu', 'jax'}
        Which factorization path kolesky uses. 'auto' = 'jax' if a GPU
        backend is available.
    """
    N_dom = X_domain.shape[0]
    N_bdy = X_boundary.shape[0]
    d = X_domain.shape[1]
    # Dimension-agnostic: the Δδ kernel takes d as a parameter, and the
    # maximin ordering / sparsity pattern use Euclidean distance in any
    # dimension. We sanity-check 1 ≤ d ≤ 3; higher dims should in principle
    # also work but have never been tested here.
    if d not in (1, 2, 3):
        raise ValueError(f'untested dimension d={d} (tested: 1, 2, 3)')

    rhs_values = _apply_vector_rhs(eqn.rhs, X_domain)
    bdy_values = _apply_vector_rhs(eqn.bdy, X_boundary)

    sol_now = np.asarray(sol_init, dtype=np.float64).copy()

    def _log(msg):
        if verbose:
            print(msg)

    # --- big factor (built ONCE outside the Gauss-Newton loop) ---
    _log('[big factor] FollowDiracs ordering + sparsity …')
    t0 = time.perf_counter()
    meas_big = _make_measurements_big(X_boundary, X_domain)
    implicit_big = ImplicitKLFactorization.build_follow_diracs(
        kernel, meas_big, rho_big, k_neighbors=k_neighbors,
        lambda_=lambda_, alpha=alpha,
    )
    t1 = time.perf_counter()
    _log(f'[big factor] implicit: {t1 - t0:.3f} s '
         f'({len(implicit_big.supernodes.supernodes)} supernodes)')

    explicit_big = ExplicitKLFactorization(
        implicit_big, nugget=nugget, backend=backend,
    )
    t2 = time.perf_counter()
    _log(f'[big factor] explicit: {t2 - t1:.3f} s '
         f'(U.nnz = {explicit_big.U.nnz:,})')

    big_op = BigFactorOperator(explicit_big.U, explicit_big.P)

    # --- Gauss-Newton iteration ---
    implicit_small = None
    theta_train_op = ThetaTrainMatVec(big_op, N_bdy, N_dom)

    for step in range(GN_steps):
        _log(f'[GN step {step + 1}/{GN_steps}]')
        delta_coefs_int = eqn.alpha * eqn.m * sol_now ** (eqn.m - 1)
        theta_train_op.set_delta_coefs(delta_coefs_int)

        # small factor: rebuild or update measurements in-place
        t_s0 = time.perf_counter()
        meas_small = _make_measurements_small(X_boundary, X_domain, delta_coefs_int)
        if implicit_small is None:
            implicit_small = ImplicitKLFactorization.build(
                kernel, meas_small, rho_small, k_neighbors=k_neighbors,
                lambda_=lambda_, alpha=alpha,
            )
        else:
            # The Julia code reuses the ordering and sparsity pattern; only
            # the measurements (i.e. the δ_coefs) change. We do the same by
            # stacking the new measurements and permuting by the cached P.
            from .. import measurements as _m
            merged = _m.stack_measurements(meas_small)
            implicit_small.supernodes.measurements = _m.select(merged, implicit_small.P)
        t_s1 = time.perf_counter()
        _log(f'  small implicit : {t_s1 - t_s0:.3f} s')

        explicit_small = ExplicitKLFactorization(
            implicit_small, nugget=nugget, backend=backend,
        )
        t_s2 = time.perf_counter()
        _log(f'  small explicit : {t_s2 - t_s1:.3f} s')

        precond = SmallPrecond(explicit_small.U, explicit_small.P)

        rhs_now = np.concatenate([
            bdy_values,
            rhs_values + eqn.alpha * (eqn.m - 1) * sol_now ** eqn.m,
        ])

        # Initial guess: preconditioner applied to rhs.
        x0 = precond.matvec(rhs_now)

        A_op = theta_train_op.as_linear_operator()
        M_op = precond.as_linear_operator()

        t_p0 = time.perf_counter()
        iter_count = [0]

        def _cb(_xk):
            iter_count[0] += 1

        theta_inv_rhs, info = scipy.sparse.linalg.cg(
            A_op, rhs_now, x0=x0, M=M_op, rtol=pcg_tol, maxiter=pcg_maxiter,
            callback=_cb,
        )
        t_p1 = time.perf_counter()
        _log(f'  pCG: {iter_count[0]} iters, {t_p1 - t_p0:.3f} s (info={info})')

        sol_now = predict_via_big(
            big_op, theta_inv_rhs, delta_coefs_int, N_bdy, N_dom
        )

    return sol_now


# ---------------------------------------------------------------------------
# Exact (dense) variant — for validation only
# ---------------------------------------------------------------------------


def iterGPR_exact(
    eqn: NonlinElliptic,
    kernel: AbstractCovarianceFunction,
    X_domain: np.ndarray,
    X_boundary: np.ndarray,
    sol_init: np.ndarray,
    nugget: float = 1e-10,
    GN_steps: int = 3,
) -> np.ndarray:
    """Exact version via dense covariance matrices (small N only)."""
    from .. import measurements as _m
    N_dom = X_domain.shape[0]
    N_bdy = X_boundary.shape[0]
    d = 2

    rhs_values = _apply_vector_rhs(eqn.rhs, X_domain)
    bdy_values = _apply_vector_rhs(eqn.bdy, X_boundary)

    sol_now = np.asarray(sol_init, dtype=np.float64).copy()

    for _step in range(GN_steps):
        delta_coefs_int = eqn.alpha * eqn.m * sol_now ** (eqn.m - 1)
        meas_delta_bdy = LaplaceDiracPointMeasurement(
            coordinate=X_boundary,
            weight_laplace=np.zeros(N_bdy),
            weight_delta=np.ones(N_bdy),
        )
        meas_linearized = LaplaceDiracPointMeasurement(
            coordinate=X_domain,
            weight_laplace=-np.ones(N_dom),
            weight_delta=delta_coefs_int,
        )
        meas_test = LaplaceDiracPointMeasurement(
            coordinate=X_domain,
            weight_laplace=np.zeros(N_dom),
            weight_delta=np.ones(N_dom),
        )
        all_train = _m.stack_measurements([meas_delta_bdy, meas_linearized])
        Theta_train = kernel(all_train, all_train)
        Theta_test = np.concatenate([
            kernel(meas_test, meas_delta_bdy),
            kernel(meas_test, meas_linearized),
        ], axis=1)

        rhs_now = np.concatenate([
            bdy_values,
            rhs_values + eqn.alpha * (eqn.m - 1) * sol_now ** eqn.m,
        ])

        d_idx = np.arange(Theta_train.shape[0])
        Theta_train_reg = Theta_train.copy()
        Theta_train_reg[d_idx, d_idx] *= (1.0 + nugget)
        sol_now = Theta_test @ np.linalg.solve(Theta_train_reg, rhs_now)

    return sol_now
