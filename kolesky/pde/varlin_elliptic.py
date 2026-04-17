"""Variable-coefficient nonlinear elliptic PDE solver.

-∇·(a(x) ∇u(x)) + α u(x)^m = f(x)  on Ω
u = bdy  on ∂Ω

Mirrors main_VarLinElliptic2d.jl. Measurements are Δ∇δ (Laplacian +
gradient + Dirac); the big factor uses the DiracsFirstThenUnifScale
ordering rather than FollowDiracs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .. import (
    AbstractCovarianceFunction,
    ExplicitKLFactorization,
    ImplicitKLFactorization,
    measurements as _m,
)
from ..measurements import LaplaceGradDiracPointMeasurement

from .pcg_ops import (
    BigFactorOperator,
    LiftedThetaTrainMatVec,
    SmallPrecond,
)


@dataclass
class VarLinElliptic2d:
    alpha: float
    m: int
    domain: Tuple[Tuple[float, float], Tuple[float, float]]
    a: Callable          # a(x) -> float
    grad_a: Callable     # grad_a(x) -> (d,) array
    bdy: Callable
    rhs: Callable

    @property
    def d(self) -> int:
        return 2


def _make_measurements_big(
    X_boundary: np.ndarray, X_domain: np.ndarray,
    lap_coefs: np.ndarray, grad_coefs: np.ndarray,
):
    """3-set measurement list, all promoted to LaplaceGradDiracPointMeasurement:
        set 0: δ_boundary
        set 1: δ_interior
        set 2: ( -a·Δ - ∇a·∇ )_interior   (no Dirac term)
    """
    N_bdy = X_boundary.shape[0]
    N_dom = X_domain.shape[0]
    d = X_domain.shape[1]
    bdy = LaplaceGradDiracPointMeasurement(
        coordinate=X_boundary,
        weight_laplace=np.zeros(N_bdy),
        weight_grad=np.zeros((N_bdy, d)),
        weight_delta=np.ones(N_bdy),
    )
    d_int = LaplaceGradDiracPointMeasurement(
        coordinate=X_domain,
        weight_laplace=np.zeros(N_dom),
        weight_grad=np.zeros((N_dom, d)),
        weight_delta=np.ones(N_dom),
    )
    spatial = LaplaceGradDiracPointMeasurement(
        coordinate=X_domain,
        weight_laplace=np.asarray(lap_coefs, dtype=np.float64),
        weight_grad=np.asarray(grad_coefs, dtype=np.float64),
        weight_delta=np.zeros(N_dom),
    )
    return [bdy, d_int, spatial]


def _make_measurements_small(
    X_boundary: np.ndarray, X_domain: np.ndarray,
    lap_coefs: np.ndarray, grad_coefs: np.ndarray, delta_coefs: np.ndarray,
):
    """2-set measurement list at the current GN iterate:
        set 0: δ_boundary
        set 1: ( -a·Δ - ∇a·∇ + c·δ )_interior    with c = α m v^(m-1)
    """
    N_bdy = X_boundary.shape[0]
    N_dom = X_domain.shape[0]
    d = X_domain.shape[1]
    bdy = LaplaceGradDiracPointMeasurement(
        coordinate=X_boundary,
        weight_laplace=np.zeros(N_bdy),
        weight_grad=np.zeros((N_bdy, d)),
        weight_delta=np.ones(N_bdy),
    )
    linearized = LaplaceGradDiracPointMeasurement(
        coordinate=X_domain,
        weight_laplace=np.asarray(lap_coefs, dtype=np.float64),
        weight_grad=np.asarray(grad_coefs, dtype=np.float64),
        weight_delta=np.asarray(delta_coefs, dtype=np.float64),
    )
    return [bdy, linearized]


def _apply_vector_rhs(fn, X):
    return np.array([fn(X[i]) for i in range(X.shape[0])], dtype=np.float64)


def _apply_grad(fn, X):
    return np.stack([np.asarray(fn(X[i]), dtype=np.float64) for i in range(X.shape[0])], axis=0)


def solve_var_lin_elliptic_2d(
    eqn: VarLinElliptic2d,
    kernel: AbstractCovarianceFunction,
    X_domain: np.ndarray,
    X_boundary: np.ndarray,
    sol_init: np.ndarray,
    nugget: float = 1e-15,
    GN_steps: int = 3,
    rho_big: float = 3.0,
    rho_small: float = 3.0,
    k_neighbors: int = 3,
    lambda_: float = 1.5,
    alpha: float = 1.0,
    backend: str = 'auto',
    pcg_tol: float = 1e-6,
    pcg_maxiter: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    N_dom = X_domain.shape[0]
    N_bdy = X_boundary.shape[0]

    rhs_values = _apply_vector_rhs(eqn.rhs, X_domain)
    bdy_values = _apply_vector_rhs(eqn.bdy, X_boundary)
    lap_coefs = -_apply_vector_rhs(eqn.a, X_domain)           # (N_dom,)
    grad_coefs = -_apply_grad(eqn.grad_a, X_domain)           # (N_dom, 2)

    sol_now = np.asarray(sol_init, dtype=np.float64).copy()

    def log(msg):
        if verbose:
            print(msg)

    # --- big factor (fixed spatial operator -∇·(a∇), no reaction term) ---
    log('[big factor] DiracsFirstThenUnifScale ordering + sparsity …')
    t0 = time.perf_counter()
    meas_big = _make_measurements_big(X_boundary, X_domain, lap_coefs, grad_coefs)
    implicit_big = ImplicitKLFactorization.build_diracs_first_then_unif_scale(
        kernel, meas_big, rho_big, k_neighbors=k_neighbors,
        lambda_=lambda_, alpha=alpha,
    )
    t1 = time.perf_counter()
    log(f'[big factor] implicit: {t1 - t0:.3f} s  ({len(implicit_big.supernodes.supernodes)} supernodes)')

    explicit_big = ExplicitKLFactorization(implicit_big, nugget=nugget, backend=backend)
    t2 = time.perf_counter()
    log(f'[big factor] explicit: {t2 - t1:.3f} s  (U.nnz = {explicit_big.U.nnz:,})')

    big_op = BigFactorOperator(explicit_big.U, explicit_big.P)
    theta_train_op = LiftedThetaTrainMatVec(big_op, N_bdy, N_dom, n_dom_sets=2)

    implicit_small = None

    for step in range(GN_steps):
        log(f'[GN step {step + 1}/{GN_steps}]')
        delta_coefs_int = eqn.alpha * eqn.m * sol_now ** (eqn.m - 1)
        # weights for lift/extract:
        #   set 0 (δ_int):       w = c
        #   set 1 (spatial):     w = 1
        theta_train_op.set_weights([delta_coefs_int, 1.0])

        t_s0 = time.perf_counter()
        meas_small = _make_measurements_small(
            X_boundary, X_domain, lap_coefs, grad_coefs, delta_coefs_int,
        )
        if implicit_small is None:
            implicit_small = ImplicitKLFactorization.build(
                kernel, meas_small, rho_small, k_neighbors=k_neighbors,
                lambda_=lambda_, alpha=alpha,
            )
        else:
            from .. import measurements as _m
            merged = _m.stack_measurements(meas_small)
            implicit_small.supernodes.measurements = _m.select(merged, implicit_small.P)
        t_s1 = time.perf_counter()
        log(f'  small implicit : {t_s1 - t_s0:.3f} s')

        explicit_small = ExplicitKLFactorization(
            implicit_small, nugget=nugget, backend=backend,
        )
        t_s2 = time.perf_counter()
        log(f'  small explicit : {t_s2 - t_s1:.3f} s')

        precond = SmallPrecond(explicit_small.U, explicit_small.P)

        rhs_now = np.concatenate([
            bdy_values,
            rhs_values + eqn.alpha * (eqn.m - 1) * sol_now ** eqn.m,
        ])

        x0 = precond.matvec(rhs_now)
        A_op = theta_train_op.as_linear_operator()
        M_op = precond.as_linear_operator()

        t_p0 = time.perf_counter()
        it_count = [0]
        theta_inv_rhs, info = scipy.sparse.linalg.cg(
            A_op, rhs_now, x0=x0, M=M_op, rtol=pcg_tol, maxiter=pcg_maxiter,
            callback=lambda _xk: it_count.__setitem__(0, it_count[0] + 1),
        )
        t_p1 = time.perf_counter()
        log(f'  pCG: {it_count[0]} iters, {t_p1 - t_p0:.3f} s (info={info})')

        # Predict: sol_now = δ_int row of Θ_big @ lift(theta_inv_rhs)
        t = theta_train_op.predict_blocks(theta_inv_rhs)
        sol_now = t[N_bdy:N_bdy + N_dom]

    return sol_now
