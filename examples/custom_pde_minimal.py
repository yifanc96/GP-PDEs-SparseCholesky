"""Template: solving your own *linear* PDE with the low-level API.

Problem:  −Δu(x) + c(x)·u(x) = f(x)   on  Ω = [0, 1]²
          u(x) = 0                     on  ∂Ω

This is a linear reaction-diffusion equation. No Gauss-Newton needed —
a single GP regression does the whole solve. The same pattern extends
to any linear PDE whose operator is a *weighted sum of Δ, ∇, and δ at
each interior point* — just change the measurement weights below.

For a nonlinear PDE you would wrap this single solve in a Gauss-Newton
loop that updates the `weight_delta` field on each iteration — see
`kolesky/pde/nonlin_elliptic.py` for the canonical example.

Four-step recipe (same as every PDE in `kolesky.pde`):
    1. Describe your measurements (boundary δ, interior δ, interior L).
    2. Build the "big" sparse Cholesky factor once.
    3. Build a "small" preconditioner factor for the *linearized*
       operator (here: the full operator, since the PDE is already linear).
    4. Solve the linear GP regression via preconditioned CG.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg

import kolesky as kl
from kolesky.measurements import LaplaceDiracPointMeasurement
from kolesky.pde.pcg_ops import (
    BigFactorOperator, LiftedThetaTrainMatVec, SmallPrecond,
)


# ---------------------------------------------------------------------------
# User input: the PDE
# ---------------------------------------------------------------------------


def c(x: np.ndarray) -> float:
    """Reaction coefficient c(x). Must be non-negative for ellipticity."""
    return 1.0 + 10.0 * x[0] * x[1]


def u_exact(x: np.ndarray) -> float:
    return float(np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


def f(x: np.ndarray) -> float:
    """rhs = -Δu_exact + c(x)·u_exact  (for a manufactured solution)."""
    u = u_exact(x)
    lap_u = -2.0 * np.pi ** 2 * u
    return -lap_u + c(x) * u


def u_bdy(x: np.ndarray) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# User input: sample points
# ---------------------------------------------------------------------------


def sample_points(h: float = 0.03):
    """Tensor-product grid inside [0,1]^2 + a grid of boundary points."""
    xs = np.arange(h, 1.0 - h + 1e-12, h)
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    X_dom = np.stack([XX.ravel(), YY.ravel()], axis=1)

    t = np.arange(0, 1 + 1e-12, h)
    zeros, ones = np.zeros_like(t), np.ones_like(t)
    X_bdy = np.concatenate([
        np.stack([t,     zeros], axis=1),   # bottom
        np.stack([t,     ones ], axis=1),   # top
        np.stack([zeros, t    ], axis=1),   # left
        np.stack([ones,  t    ], axis=1),   # right
    ], axis=0)
    X_bdy = np.unique(np.round(X_bdy, 8), axis=0)
    return X_dom, X_bdy


# ---------------------------------------------------------------------------
# The four-step recipe
# ---------------------------------------------------------------------------


def solve(X_dom, X_bdy, kernel, nugget=1e-10, rho=3.0, k_neighbors=3,
          pcg_tol=1e-7, pcg_maxiter=200, backend='cpu', verbose=True):
    N_dom = X_dom.shape[0]
    N_bdy = X_bdy.shape[0]

    # Evaluate the PDE's spatially varying coefficients on the grid.
    c_vals = np.array([c(X_dom[i]) for i in range(N_dom)])
    rhs    = np.array([f(X_dom[i]) for i in range(N_dom)])
    bdy    = np.array([u_bdy(X_bdy[i]) for i in range(N_bdy)])

    # -------------------------------------------------------------------
    # Step 1: define 3 measurement sets for the "big" factor:
    #           set 0 — δ_bdy   (boundary)
    #           set 1 — δ_int   (for prediction + δ-part of the operator)
    #           set 2 — (−Δ)_int  (Laplacian-part of the operator)
    # -------------------------------------------------------------------
    m_bdy = LaplaceDiracPointMeasurement(
        coordinate=X_bdy,
        weight_laplace=np.zeros(N_bdy),
        weight_delta=np.ones(N_bdy),
    )
    m_d_int = LaplaceDiracPointMeasurement(
        coordinate=X_dom,
        weight_laplace=np.zeros(N_dom),
        weight_delta=np.ones(N_dom),
    )
    m_lap_int = LaplaceDiracPointMeasurement(
        coordinate=X_dom,
        weight_laplace=-np.ones(N_dom),       # operator has −Δ
        weight_delta=np.zeros(N_dom),
    )

    if verbose:
        print('[big factor]    FollowDiracs ordering + sparsity …')
    t0 = time.perf_counter()
    impl_big = kl.ImplicitKLFactorization.build_follow_diracs(
        kernel, [m_bdy, m_d_int, m_lap_int],
        rho=rho, k_neighbors=k_neighbors,
    )
    expl_big = kl.ExplicitKLFactorization(impl_big, nugget=nugget, backend=backend)
    big = BigFactorOperator(expl_big.U, expl_big.P)
    if verbose:
        print(f'[big factor]    {time.perf_counter()-t0:.2f} s  (U.nnz = {expl_big.U.nnz:,})')

    # -------------------------------------------------------------------
    # Step 2: the train operator is (δ + (−Δ + c·δ))_int = δ_bdy-row, plus
    # at interior: −Δu + c·u. Lift weights pull the interior δ (set 1)
    # with weight `c(x_i)` and the Laplacian block (set 2) with weight 1.
    # -------------------------------------------------------------------
    theta_train = LiftedThetaTrainMatVec(big, N_bdy, N_dom, n_dom_sets=2)
    theta_train.set_weights([c_vals, 1.0])

    # -------------------------------------------------------------------
    # Step 3: small "preconditioner" factor. 2 measurement sets: boundary
    # δ and the full linear operator (−Δ + c·δ) at each interior point.
    # -------------------------------------------------------------------
    m_train_int = LaplaceDiracPointMeasurement(
        coordinate=X_dom,
        weight_laplace=-np.ones(N_dom),
        weight_delta=c_vals,
    )
    t0 = time.perf_counter()
    impl_small = kl.ImplicitKLFactorization.build(
        kernel, [m_bdy, m_train_int],
        rho=rho, k_neighbors=k_neighbors,
    )
    expl_small = kl.ExplicitKLFactorization(impl_small, nugget=nugget, backend=backend)
    precond = SmallPrecond(expl_small.U, expl_small.P)
    if verbose:
        print(f'[small factor]  {time.perf_counter()-t0:.2f} s  (U.nnz = {expl_small.U.nnz:,})')

    # -------------------------------------------------------------------
    # Step 4: pCG on the GP regression linear system
    #            Θ_train · α  =  [ u_bdy ;  f ]
    # Then the solution at interior points is  Θ(δ_int, train) · α, which
    # `theta_train.predict_blocks` computes for us via the big factor.
    # -------------------------------------------------------------------
    rhs_vec = np.concatenate([bdy, rhs])
    A = theta_train.as_linear_operator()
    M = precond.as_linear_operator()
    x0 = precond.matvec(rhs_vec)

    it = [0]
    t0 = time.perf_counter()
    alpha, info = scipy.sparse.linalg.cg(
        A, rhs_vec, x0=x0, M=M, rtol=pcg_tol, maxiter=pcg_maxiter,
        callback=lambda _x: it.__setitem__(0, it[0] + 1),
    )
    if verbose:
        print(f'[pCG]           {it[0]} iters, {time.perf_counter()-t0:.2f} s (info={info})')

    t = theta_train.predict_blocks(alpha)
    return t[N_bdy : N_bdy + N_dom]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--h', type=float, default=0.03,
                   help='grid spacing (default 0.03 → ~33x33 interior)')
    p.add_argument('--sigma', type=float, default=0.3)
    p.add_argument('--rho', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--nugget', type=float, default=1e-10)
    p.add_argument('--backend', choices=['cpu', 'jax', 'auto'], default='cpu')
    args = p.parse_args()

    X_dom, X_bdy = sample_points(h=args.h)
    kernel = kl.MaternCovariance7_2(args.sigma)
    print(f'[points]        interior = {X_dom.shape[0]}, boundary = {X_bdy.shape[0]}')
    print(f'[kernel]        Matern 7/2, length_scale = {args.sigma}')

    sol = solve(
        X_dom, X_bdy, kernel,
        nugget=args.nugget, rho=args.rho, k_neighbors=args.k_neighbors,
        backend=args.backend,
    )
    truth = np.array([u_exact(X_dom[i]) for i in range(X_dom.shape[0])])
    L2   = float(np.sqrt(np.mean((truth - sol) ** 2)))
    Linf = float(np.max(np.abs(truth - sol)))
    print(f'[accuracy]      L² = {L2:.2e},  L∞ = {Linf:.2e}')


if __name__ == '__main__':
    main()
