"""1D Burgers equation solver: u_t + u u_x - ν u_xx = 0 on [-1, 1] × [0, T].

Mirrors main_Burgers1d.jl:

* Crank-Nicolson time-stepping. At each time step, Gauss-Newton iterations
  linearize u u_x around the current estimate `u_now`, and the resulting
  linearized PDE has Δ∇δ measurement with:
      w_Δ = -ν    (scalar)
      w_∇ = u_now (scalar vector for d=1)
      w_δ = 2/dt + u_{x,now}

* **Big factor** built once: 4-set (δ_bdy, δ_int, ∇_int, Δ_int) via
  FollowDiracs — so the matvec can compose arbitrary w_Δ, w_∇, w_δ at
  each GN step without rebuilding the big factor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

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
class Burgers1d:
    nu: float
    bdy: Callable
    rhs: Callable
    init: Callable
    init_dx: Callable
    init_dxx: Callable


def _lgd_1d(coord_1d: np.ndarray, w_lap, w_grad, w_delta) -> LaplaceGradDiracPointMeasurement:
    """Build a 1D Δ∇δ measurement.  `coord_1d` is (N,) or (N,1). Weights may
    be scalars or (N,) arrays; gradient weight is (N,1) after broadcasting."""
    coord = np.atleast_2d(coord_1d)
    if coord.shape[0] == 1:
        coord = coord.T
    if coord.ndim == 1:
        coord = coord[:, None]
    N = coord.shape[0]

    def _vec(x, shape):
        if np.isscalar(x):
            return np.full(shape, float(x), dtype=np.float64)
        return np.asarray(x, dtype=np.float64).reshape(shape)

    wl = _vec(w_lap, (N,))
    wd = _vec(w_delta, (N,))
    if np.isscalar(w_grad):
        wg = np.full((N, 1), float(w_grad), dtype=np.float64)
    else:
        wg = np.asarray(w_grad, dtype=np.float64).reshape(N, 1)
    return LaplaceGradDiracPointMeasurement(
        coordinate=coord, weight_laplace=wl, weight_grad=wg, weight_delta=wd,
    )


def sample_points_grid_1d(h_in: float):
    """Interior grid (-1+h ... 1-h) and boundary {-1, 1}."""
    X_domain = np.arange(-1.0 + h_in, 1.0 - h_in + 1e-12, h_in)
    X_boundary = np.array([-1.0, 1.0])
    return X_domain, X_boundary


def solve_burgers_1d(
    eqn: Burgers1d,
    kernel: AbstractCovarianceFunction,
    X_domain: np.ndarray,
    X_boundary: np.ndarray,
    dt: float,
    T: float,
    nugget: float = 1e-10,
    GN_steps: int = 2,
    rho_big: float = 3.0,
    rho_small: float = 3.0,
    k_neighbors: int = 1,
    lambda_: float = 1.5,
    alpha: float = 1.0,
    backend: str = 'auto',
    pcg_tol: float = 1e-6,
    pcg_maxiter: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    N_dom = X_domain.shape[0]
    N_bdy = X_boundary.shape[0]
    Nt = int(round(T / dt))

    sol_u = np.array([eqn.init(X_domain[i]) for i in range(N_dom)], dtype=np.float64)
    sol_ux = np.array([eqn.init_dx(X_domain[i]) for i in range(N_dom)], dtype=np.float64)
    sol_uxx = np.array([eqn.init_dxx(X_domain[i]) for i in range(N_dom)], dtype=np.float64)
    bdy_values = np.array([eqn.bdy(X_boundary[i]) for i in range(N_bdy)], dtype=np.float64)
    rhs_values = np.array([eqn.rhs(X_domain[i]) for i in range(N_dom)], dtype=np.float64)

    rhs_CN = np.empty(N_bdy + N_dom, dtype=np.float64)
    rhs_CN[:N_bdy] = bdy_values

    def log(msg):
        if verbose:
            print(msg)

    # --- big factor (fixed) ---
    # 4 sets: δ_bdy, δ_int, ∇_int, Δ_int — all Δ∇δ type (promoted)
    meas_big = [
        _lgd_1d(X_boundary, 0.0, 0.0, 1.0),                     # δ_bdy
        _lgd_1d(X_domain, 0.0, 0.0, 1.0),                       # δ_int
        _lgd_1d(X_domain, 0.0, 1.0, 0.0),                       # ∇_int (pure gradient)
        _lgd_1d(X_domain, 1.0, 0.0, 0.0),                       # Δ_int (pure Laplacian)
    ]
    log('[big factor] FollowDiracs ordering + sparsity …')
    t0 = time.perf_counter()
    implicit_big = ImplicitKLFactorization.build_follow_diracs(
        kernel, meas_big, rho_big, k_neighbors=k_neighbors, lambda_=lambda_, alpha=alpha,
    )
    t1 = time.perf_counter()
    log(f'[big factor] implicit: {t1 - t0:.3f} s  ({len(implicit_big.supernodes.supernodes)} supernodes)')

    explicit_big = ExplicitKLFactorization(implicit_big, nugget=nugget, backend=backend)
    t2 = time.perf_counter()
    log(f'[big factor] explicit: {t2 - t1:.3f} s  (U.nnz = {explicit_big.U.nnz:,})')

    big_op = BigFactorOperator(explicit_big.U, explicit_big.P)
    theta_train_op = LiftedThetaTrainMatVec(big_op, N_bdy, N_dom, n_dom_sets=3)

    implicit_small = None

    for it in range(Nt):
        t_now = (it + 1) * dt
        log(f'[time] t = {t_now:.4f}')
        # Crank-Nicolson RHS:  rhs + 2/dt u^n + ν u_xx^n - u^n u_x^n
        rhs_CN[N_bdy:] = rhs_values + 2.0 / dt * sol_u + eqn.nu * sol_uxx - sol_u * sol_ux

        for step in range(GN_steps):
            # linearized operator at current sol_u, sol_ux:
            w_lap = -eqn.nu                 # scalar
            w_grad = sol_u                  # (N_dom,)
            w_delta = 2.0 / dt + sol_ux     # (N_dom,)

            # weights for lift/extract over (δ_int, ∇_int, Δ_int):
            theta_train_op.set_weights([w_delta, w_grad, w_lap])

            meas_small = [
                _lgd_1d(X_boundary, 0.0, 0.0, 1.0),
                _lgd_1d(X_domain, w_lap, w_grad, w_delta),
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

            explicit_small = ExplicitKLFactorization(
                implicit_small, nugget=nugget, backend=backend,
            )
            t_s2 = time.perf_counter()
            log(f'  GN {step + 1}: small implicit {t_s1 - t_s0:.3f} s, explicit {t_s2 - t_s1:.3f} s')

            precond = SmallPrecond(explicit_small.U, explicit_small.P)

            cur_rhs_CN = rhs_CN.copy()
            cur_rhs_CN[N_bdy:] += sol_u * sol_ux

            x0 = precond.matvec(cur_rhs_CN)
            A_op = theta_train_op.as_linear_operator()
            M_op = precond.as_linear_operator()
            it_count = [0]
            t_p0 = time.perf_counter()
            theta_inv_rhs, info = scipy.sparse.linalg.cg(
                A_op, cur_rhs_CN, x0=x0, M=M_op, rtol=pcg_tol, maxiter=pcg_maxiter,
                callback=lambda _xk: it_count.__setitem__(0, it_count[0] + 1),
            )
            t_p1 = time.perf_counter()
            log(f'           pCG {it_count[0]} iters, {t_p1 - t_p0:.3f} s (info={info})')

            # Predict u, u_x, u_xx:  Θ_big @ lift(theta_inv_rhs), extract 3 blocks.
            t = theta_train_op.predict_blocks(theta_inv_rhs)
            sol_u = t[N_bdy:N_bdy + N_dom]
            sol_ux = t[N_bdy + N_dom:N_bdy + 2 * N_dom]
            sol_uxx = t[N_bdy + 2 * N_dom:N_bdy + 3 * N_dom]

    return sol_u
