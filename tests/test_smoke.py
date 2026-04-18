"""End-to-end smoke tests.

Small grids, tight accuracy bounds, CPU-only so tests can run anywhere.
"""

from __future__ import annotations

import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import pytest

import kolesky as kl
from kolesky.pde import (
    Burgers1d, solve_burgers_1d, sample_points_grid_1d,
    MongeAmpere2d, solve_monge_ampere_2d,
    NonlinElliptic, solve_nonlin_elliptic,
    NonlinElliptic2d, solve_nonlin_elliptic_2d,
    VarLinElliptic, solve_var_lin_elliptic,
    VarLinElliptic2d, solve_var_lin_elliptic_2d,
    sample_points_grid_2d,
)


def _sample_cube_grid_3d(h_in=0.2, h_bd=0.2):
    """Uniform grid on [0,1]³ interior + its 6 faces (no edge duplicates)."""
    xs = np.arange(h_in, 1.0 - h_in + 1e-12, h_in)
    XX, YY, ZZ = np.meshgrid(xs, xs, xs, indexing='ij')
    X_dom = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=1)

    # Boundary: grid on each of the 6 faces at x/y/z = 0 or 1.
    bs = np.arange(0.0, 1.0 + 1e-12, h_bd)
    B1, B2 = np.meshgrid(bs, bs, indexing='ij')
    b1, b2 = B1.ravel(), B2.ravel()
    zeros = np.zeros_like(b1); ones = np.ones_like(b1)
    faces = np.concatenate([
        np.stack([zeros, b1, b2], axis=1),   # x = 0
        np.stack([ones,  b1, b2], axis=1),   # x = 1
        np.stack([b1, zeros, b2], axis=1),   # y = 0
        np.stack([b1, ones,  b2], axis=1),   # y = 1
        np.stack([b1, b2, zeros], axis=1),   # z = 0
        np.stack([b1, b2, ones ], axis=1),   # z = 1
    ], axis=0)
    # Deduplicate edge/corner overlaps.
    keys = np.round(faces, 6)
    _, uniq = np.unique(keys.view([('', keys.dtype)] * 3), return_index=True)
    return X_dom, faces[np.sort(uniq)]


def test_kolesky_factorization_roundtrip():
    """Sanity-check the sparse Cholesky factor: valid sparsity pattern,
    positive diagonal, approximates K well when we only check K @ b against
    Θ_approx @ b via the factor (no dense inversion)."""
    import scipy.sparse.linalg as spla
    xs = np.linspace(0.05, 0.95, 20)
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    x = np.stack([XX.flatten(), YY.flatten()], axis=1)
    m = kl.point_measurements(x, dims=2)
    kernel = kl.MaternCovariance5_2(0.2)

    implicit = kl.ImplicitKLFactorization.build(
        kernel, m, rho=6.0, k_neighbors=1,
    )
    explicit = kl.ExplicitKLFactorization(implicit, nugget=1e-8, backend='cpu')

    # All diagonal entries of U are strictly positive by construction.
    diag = explicit.U.diagonal()
    assert np.all(diag > 0), f'min diag = {diag.min()}'
    assert explicit.U.nnz > explicit.U.shape[0]   # non-trivial fill

    # Check K @ b ≈ Θ_approx @ b via two triangular solves.
    rng = np.random.default_rng(0)
    b = rng.standard_normal(x.shape[0])
    U_csr = explicit.U.tocsr()
    L_csr = explicit.U.T.tocsr()
    bp = b[explicit.P]
    y = spla.spsolve_triangular(U_csr, bp, lower=False)
    z = spla.spsolve_triangular(L_csr, y, lower=True)
    Theta_b = np.empty_like(b)
    Theta_b[explicit.P] = z

    K_true_b = kernel(m) @ b
    rel_err = np.linalg.norm(Theta_b - K_true_b) / np.linalg.norm(K_true_b)
    assert rel_err < 1e-2, f'rel_err = {rel_err}'


def _nonlin_elliptic_ground_truth():
    freq, s = 200, 6
    ks = np.arange(1, freq + 1)

    def u_exact(x):
        return float(np.sum(np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))

    def rhs(x):
        lap = float(np.sum(2 * ks ** 2 * np.pi ** 2
                           * np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1])
                           / ks ** s))
        return lap + u_exact(x) ** 3

    return u_exact, rhs


def test_nonlin_elliptic_2d_solves():
    u_exact, rhs = _nonlin_elliptic_ground_truth()
    eqn = NonlinElliptic2d(alpha=1.0, m=3, domain=((0, 1), (0, 1)),
                            bdy=u_exact, rhs=rhs)
    X_dom, X_bdy = sample_points_grid_2d(eqn.domain, 0.1, 0.1)
    kernel = kl.MaternCovariance7_2(0.3)
    sol = solve_nonlin_elliptic_2d(
        eqn, kernel, X_dom, X_bdy, np.zeros(X_dom.shape[0]),
        GN_steps=2, rho_big=3, rho_small=3, k_neighbors=3,
        backend='cpu', verbose=False,
    )
    truth = np.array([u_exact(X_dom[i]) for i in range(X_dom.shape[0])])
    L2 = float(np.sqrt(np.mean((truth - sol) ** 2)))
    assert L2 < 5e-2, f'NonLinElliptic L2 error too large: {L2}'


def test_varlin_elliptic_2d_solves():
    freq, s, k_var = 50, 3, 5
    ks = np.arange(1, freq + 1)

    def u(x):
        return float(np.sum(np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))

    def a(x):
        return float(np.exp(np.sin(k_var * np.pi * x[0] * x[1])))

    def grad_a(x):
        arg = k_var * np.pi * x[0] * x[1]
        c = np.cos(arg) * k_var * np.pi * np.exp(np.sin(arg))
        return np.array([c * x[1], c * x[0]])

    def grad_u(x):
        dx = float(np.sum(np.pi * ks * np.cos(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))
        dy = float(np.sum(np.pi * ks * np.sin(np.pi * ks * x[0]) * np.cos(np.pi * ks * x[1]) / ks ** s))
        return np.array([dx, dy])

    def lap_u(x):
        return float(np.sum(2 * ks ** 2 * np.pi ** 2
                            * np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1])
                            / ks ** s))

    def rhs(x):
        return float(-np.dot(grad_a(x), grad_u(x)) + a(x) * lap_u(x) + u(x) ** 3)

    eqn = VarLinElliptic2d(
        alpha=1.0, m=3, domain=((0, 1), (0, 1)),
        a=a, grad_a=grad_a, bdy=u, rhs=rhs,
    )
    X_dom, X_bdy = sample_points_grid_2d(eqn.domain, 0.1, 0.1)
    kernel = kl.MaternCovariance7_2(0.3)
    sol = solve_var_lin_elliptic_2d(
        eqn, kernel, X_dom, X_bdy, np.zeros(X_dom.shape[0]),
        GN_steps=2, rho_big=3, rho_small=3, k_neighbors=3,
        backend='cpu', verbose=False,
    )
    truth = np.array([u(X_dom[i]) for i in range(X_dom.shape[0])])
    L2 = float(np.sqrt(np.mean((truth - sol) ** 2)))
    assert L2 < 2e-1, f'VarLinElliptic L2 error: {L2}'


def test_nonlin_elliptic_3d_solves():
    """-Δu + u³ = f on [0,1]³ with u(x,y,z) = sin(πx) sin(πy) sin(πz)."""
    def u_exact(x):
        return float(np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2]))

    def rhs(x):
        return 3 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3

    eqn = NonlinElliptic(alpha=1.0, m=3, domain=((0, 1), (0, 1), (0, 1)),
                         bdy=u_exact, rhs=rhs)
    assert eqn.d == 3
    X_dom, X_bdy = _sample_cube_grid_3d(h_in=0.2, h_bd=0.2)
    kernel = kl.MaternCovariance7_2(0.3)
    sol = solve_nonlin_elliptic(
        eqn, kernel, X_dom, X_bdy, np.zeros(X_dom.shape[0]),
        GN_steps=2, rho_big=3, rho_small=3, k_neighbors=3,
        backend='cpu', verbose=False,
    )
    truth = np.array([u_exact(X_dom[i]) for i in range(X_dom.shape[0])])
    L2 = float(np.sqrt(np.mean((truth - sol) ** 2)))
    assert L2 < 2e-1, f'NonlinElliptic 3D L2 error: {L2}'


def test_varlin_elliptic_3d_solves():
    """-∇·(a∇u) + u³ = f on [0,1]³ with a(x,y,z) = 1 + 0.5·x and
    u(x,y,z) = sin(πx) sin(πy) sin(πz)."""
    pi = np.pi

    def u(x):
        return float(np.sin(pi * x[0]) * np.sin(pi * x[1]) * np.sin(pi * x[2]))

    def grad_u(x):
        return np.array([
            pi * np.cos(pi * x[0]) * np.sin(pi * x[1]) * np.sin(pi * x[2]),
            pi * np.sin(pi * x[0]) * np.cos(pi * x[1]) * np.sin(pi * x[2]),
            pi * np.sin(pi * x[0]) * np.sin(pi * x[1]) * np.cos(pi * x[2]),
        ])

    def lap_u(x):
        return -3 * pi ** 2 * u(x)

    def a(x):
        return 1.0 + 0.5 * x[0]

    def grad_a(x):
        return np.array([0.5, 0.0, 0.0])

    def rhs(x):
        # -∇·(a∇u) = -a·Δu - ∇a·∇u
        return float(-a(x) * lap_u(x) - np.dot(grad_a(x), grad_u(x)) + u(x) ** 3)

    eqn = VarLinElliptic(
        alpha=1.0, m=3, domain=((0, 1), (0, 1), (0, 1)),
        a=a, grad_a=grad_a, bdy=u, rhs=rhs,
    )
    assert eqn.d == 3
    X_dom, X_bdy = _sample_cube_grid_3d(h_in=0.2, h_bd=0.2)
    kernel = kl.MaternCovariance7_2(0.3)
    sol = solve_var_lin_elliptic(
        eqn, kernel, X_dom, X_bdy, np.zeros(X_dom.shape[0]),
        GN_steps=2, rho_big=3, rho_small=3, k_neighbors=3,
        backend='cpu', verbose=False,
    )
    truth = np.array([u(X_dom[i]) for i in range(X_dom.shape[0])])
    L2 = float(np.sqrt(np.mean((truth - sol) ** 2)))
    assert L2 < 3e-1, f'VarLinElliptic 3D L2 error: {L2}'


def test_burgers_1d_solves():
    def u0(x):
        return float(-np.sin(np.pi * x))

    def dxu0(x):
        return float(-np.pi * np.cos(np.pi * x))

    def dxxu0(x):
        return float(np.pi ** 2 * np.sin(np.pi * x))

    eqn = Burgers1d(
        nu=0.05, bdy=lambda _x: 0.0, rhs=lambda _x: 0.0,
        init=u0, init_dx=dxu0, init_dxx=dxxu0,
    )
    X_dom, X_bdy = sample_points_grid_1d(0.05)
    kernel = kl.MaternCovariance7_2(0.1)
    sol = solve_burgers_1d(
        eqn, kernel, X_dom, X_bdy, dt=0.1, T=0.1,
        GN_steps=2, rho_big=3, rho_small=3, k_neighbors=1,
        backend='cpu', verbose=False,
    )
    # Just check it ran and returns a sensible-shape vector with no NaNs.
    assert sol.shape == X_dom.shape
    assert np.all(np.isfinite(sol))


def test_monge_ampere_2d_solves():
    def u(x):
        return float(np.exp(0.5 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)))

    def rhs(x):
        return float((1.0 + (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) * u(x) ** 2)

    eqn = MongeAmpere2d(domain=((0, 1), (0, 1)), bdy=u, rhs=rhs)
    X_dom, X_bdy = sample_points_grid_2d(eqn.domain, 0.125, 0.125)
    kernel = kl.MaternCovariance5_2(0.3)
    N = X_dom.shape[0]
    sol = solve_monge_ampere_2d(
        eqn, kernel, X_dom, X_bdy,
        sol_init=np.zeros(N),
        sol_init_xx=np.ones(N), sol_init_xy=np.zeros(N), sol_init_yy=np.ones(N),
        GN_steps=2, rho_big=3, rho_small=3, k_neighbors=3,
        backend='jax', verbose=False,
    )
    truth = np.array([u(X_dom[i]) for i in range(N)])
    L2 = float(np.sqrt(np.mean((truth - sol) ** 2)))
    assert L2 < 5e-2, f'MongeAmpere L2 error: {L2}'
