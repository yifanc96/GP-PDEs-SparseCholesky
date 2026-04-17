"""Monge-Ampere equation solver:  det(∇² u) = f  on Ω, u = bdy on ∂Ω.

Mirrors main_MongeAmpere2d.jl. Linearizing around current iterate
(v_xx, v_xy, v_yy) the measurement is

    L_i(u) = v_yy_i · u_xx(x_i) + v_xx_i · u_yy(x_i) - 2 v_xy_i · u_xy(x_i)

i.e. the HessianDiracPointMeasurement with w_δ=0, w11=v_yy, w12=-2 v_xy, w22=v_xx.

The big factor is a 5-set KL factorization over (δ_bdy, δ_int, ∂11_int,
∂22_int, ∂12_int) via DiracsFirstThenUnifScale ordering.
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
from ..measurements import HessianDiracPointMeasurement

from .pcg_ops import (
    BigFactorOperator,
    LiftedThetaTrainMatVec,
    SmallPrecond,
)


@dataclass
class MongeAmpere2d:
    domain: Tuple[Tuple[float, float], Tuple[float, float]]
    bdy: Callable
    rhs: Callable


def _hd(coord: np.ndarray, w11, w12, w22, wd) -> HessianDiracPointMeasurement:
    """Build a (possibly batched) HessianDiracPointMeasurement.
    Each w_* may be a scalar or an (N,) array.
    """
    coord = np.atleast_2d(coord).astype(np.float64)
    N = coord.shape[0]

    def _vec(x):
        if np.isscalar(x):
            return np.full(N, float(x), dtype=np.float64)
        return np.asarray(x, dtype=np.float64).reshape(N)

    return HessianDiracPointMeasurement(
        coordinate=coord,
        weight_11=_vec(w11),
        weight_12=_vec(w12),
        weight_22=_vec(w22),
        weight_delta=_vec(wd),
    )


def _apply_fn(fn, X):
    return np.array([fn(X[i]) for i in range(X.shape[0])], dtype=np.float64)


def solve_monge_ampere_2d(
    eqn: MongeAmpere2d,
    kernel: AbstractCovarianceFunction,
    X_domain: np.ndarray,
    X_boundary: np.ndarray,
    sol_init: np.ndarray,
    sol_init_xx: np.ndarray,
    sol_init_xy: np.ndarray,
    sol_init_yy: np.ndarray,
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

    rhs_values = _apply_fn(eqn.rhs, X_domain)
    bdy_values = _apply_fn(eqn.bdy, X_boundary)

    v = np.asarray(sol_init, dtype=np.float64).copy()
    v_xx = np.asarray(sol_init_xx, dtype=np.float64).copy()
    v_xy = np.asarray(sol_init_xy, dtype=np.float64).copy()
    v_yy = np.asarray(sol_init_yy, dtype=np.float64).copy()

    def log(msg):
        if verbose:
            print(msg)

    # ---- 5-set big factor: (δ_bdy, δ_int, ∂11_int, ∂22_int, ∂12_int) ----
    meas_big = [
        _hd(X_boundary, 0.0, 0.0, 0.0, 1.0),   # δ_bdy
        _hd(X_domain,   0.0, 0.0, 0.0, 1.0),   # δ_int
        _hd(X_domain,   1.0, 0.0, 0.0, 0.0),   # ∂11_int
        _hd(X_domain,   0.0, 0.0, 1.0, 0.0),   # ∂22_int
        _hd(X_domain,   0.0, 1.0, 0.0, 0.0),   # ∂12_int
    ]

    log('[big factor] DiracsFirstThenUnifScale ordering + sparsity …')
    t0 = time.perf_counter()
    implicit_big = ImplicitKLFactorization.build_diracs_first_then_unif_scale(
        kernel, meas_big, rho_big, k_neighbors=k_neighbors, lambda_=lambda_, alpha=alpha,
    )
    t1 = time.perf_counter()
    log(f'[big factor] implicit: {t1 - t0:.3f} s  ({len(implicit_big.supernodes.supernodes)} supernodes)')

    explicit_big = ExplicitKLFactorization(implicit_big, nugget=nugget, backend=backend)
    t2 = time.perf_counter()
    log(f'[big factor] explicit: {t2 - t1:.3f} s  (U.nnz = {explicit_big.U.nnz:,})')

    big_op = BigFactorOperator(explicit_big.U, explicit_big.P)
    theta_train_op = LiftedThetaTrainMatVec(big_op, N_bdy, N_dom, n_dom_sets=4)

    implicit_small = None

    for step in range(GN_steps):
        log(f'[GN step {step + 1}/{GN_steps}]')
        # Lift/extract weights for Θ_train (over δ_int, ∂11, ∂22, ∂12):
        # δ_int → 0 (no u term in Monge-Ampere); ∂11 → v_yy; ∂22 → v_xx; ∂12 → -2 v_xy
        theta_train_op.set_weights([0.0, v_yy, v_xx, -2.0 * v_xy])

        # Small factor: 2-set (δ_bdy, ∂∂_linearized)
        meas_small = [
            _hd(X_boundary, 0.0, 0.0, 0.0, 1.0),
            _hd(X_domain, v_yy, -2.0 * v_xy, v_xx, 0.0),
        ]

        t_s0 = time.perf_counter()
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
            rhs_values + v_xx * v_yy - v_xy ** 2,
        ])

        x0 = precond.matvec(rhs_now)
        A_op = theta_train_op.as_linear_operator()
        M_op = precond.as_linear_operator()

        it_count = [0]
        t_p0 = time.perf_counter()
        theta_inv_rhs, info = scipy.sparse.linalg.cg(
            A_op, rhs_now, x0=x0, M=M_op, rtol=pcg_tol, maxiter=pcg_maxiter,
            callback=lambda _xk: it_count.__setitem__(0, it_count[0] + 1),
        )
        t_p1 = time.perf_counter()
        log(f'  pCG: {it_count[0]} iters, {t_p1 - t_p0:.3f} s (info={info})')

        # Predict u, u_xx, u_yy, u_xy from Θ_big @ lift(theta_inv_rhs).
        # Blocks in order: bdy, δ_int, ∂11, ∂22, ∂12.
        t = theta_train_op.predict_blocks(theta_inv_rhs)
        v = t[N_bdy:N_bdy + N_dom]
        v_xx = t[N_bdy + N_dom:N_bdy + 2 * N_dom]
        v_yy = t[N_bdy + 2 * N_dom:N_bdy + 3 * N_dom]
        v_xy = t[N_bdy + 3 * N_dom:N_bdy + 4 * N_dom]

    return v
