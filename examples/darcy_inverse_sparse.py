"""2-D Darcy inverse problem — sparse Cholesky scaffold (work in progress).

This file is a *scaffold* for scaling ``examples/darcy_inverse.py`` (which
uses dense Cholesky, ~200 interior points) up to N_domain ≫ 1000 via
the sparse infrastructure in ``kolesky``. The pieces all build cleanly:

    * 5-set big factor for Theta_u over { δ_bdy, δ_int, ∂₁_int, ∂₂_int,
      Δ_int } via ``ImplicitKLFactorization.build_diracs_first_then_unif_scale``;
    * 3-set big factor for Theta_w over { δ_int, ∂₁_int, ∂₂_int };
    * matrix-free GN-Hessian apply with inner pCG for the Θ_{u,w}⁻¹
      block applies;
    * diagonal Jacobi preconditioner for the outer pCG on the joint
      (6 N_domain) GN-Hessian system.

**Honest status.** With the bare-Hessian formulation here, GN does
*not* converge at N_dom ≳ 300 — the joint Hessian is badly conditioned
and the matrix-free application via the multi-set big factor amplifies
numerical noise too much for outer-pCG to drive a useful step. The
correct path is to mirror ``kolesky.pde``'s pattern: at each GN step,
formulate a *single* combined linearized-PDE measurement (same idea as
the ``LaplaceGradDiracPointMeasurement`` train-row in
``solve_var_lin_elliptic``), assemble a small 2-set ``Theta_train``
factor for the linearized constraint + boundary/data, and run pCG on
that. Algorithm 4.1's noisy ichol then adds the (1/σ²) data-term
diagonal to the small factor — exactly the combination the user asked
for. That restructuring is left as the next iteration.

The ``examples/darcy_inverse.py`` (dense) version solves the problem
end-to-end at N_dom ≈ 200 with L²(a) ≈ 26 %, L²(u) ≈ 5 %; the next-
iteration restructuring of *this* file is the path to N_dom ≫ 1000.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import scipy.sparse
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Reference forward FD solver — same as in examples/darcy_inverse.py
# ---------------------------------------------------------------------------


def fd_darcy_forward(N: int, fun_a, f) -> np.ndarray:
    hg = 1.0 / (N + 1)
    x_mid = (np.arange(0, N + 1) + 0.5) * hg
    x_grid = np.arange(1, N + 1) * hg
    mid, grid = np.meshgrid(x_mid, x_grid)
    a1 = fun_a(mid.flatten(), grid.flatten()).reshape(N, N + 1)
    a2 = fun_a(grid.flatten(), mid.flatten()).reshape(N, N + 1).T

    a_diag = (a1[:, :N] + a1[:, 1:] + a2[:N, :] + a2[1:, :]).reshape(-1)
    a_super1 = np.append(a1[:, 1:N], np.zeros((N, 1)), axis=1).reshape(-1)
    a_super2 = a2[1:N, :].reshape(-1)
    A = scipy.sparse.diags(
        [-a_super2, -a_super1, a_diag, -a_super1, -a_super2],
        offsets=[-N, -1, 0, 1, N], shape=(N * N, N * N),
    ).tocsr() / (hg * hg)

    XX, YY = np.meshgrid(x_grid, x_grid)
    fv = f(XX.flatten(), YY.flatten())
    sol = scipy.sparse.linalg.spsolve(A, fv).reshape(N, N)
    out = np.zeros((N + 2, N + 2))
    out[1:N + 1, 1:N + 1] = sol
    return out


# ---------------------------------------------------------------------------
# Sparse big factors for Theta_u  (5-set) and Theta_w  (3-set).
# ---------------------------------------------------------------------------


def _lgd(coord, wL, wG, wD):
    """Helper: build a LaplaceGradDiracPointMeasurement."""
    from kolesky.measurements import LaplaceGradDiracPointMeasurement
    return LaplaceGradDiracPointMeasurement(
        coordinate=np.asarray(coord, dtype=np.float64),
        weight_laplace=np.asarray(wL, dtype=np.float64),
        weight_grad=np.asarray(wG, dtype=np.float64),
        weight_delta=np.asarray(wD, dtype=np.float64),
    )


def _theta_u_groups(X_dom: np.ndarray, X_bdy: np.ndarray):
    """5-set list:  [δ_bdy, δ_int, ∂₁_int, ∂₂_int, Δ_int]."""
    N = X_dom.shape[0]; Nb = X_bdy.shape[0]
    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    return [
        _lgd(X_bdy, np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
        _lgd(X_dom, np.zeros(N),  np.zeros((N, 2)),  np.ones(N)),
        _lgd(X_dom, np.zeros(N),  e1,                 np.zeros(N)),
        _lgd(X_dom, np.zeros(N),  e2,                 np.zeros(N)),
        _lgd(X_dom, np.ones(N),   np.zeros((N, 2)),  np.zeros(N)),
    ]


def _theta_w_groups(X_dom: np.ndarray):
    """For Theta_w we use the same machinery but with a *single* dummy
    boundary point far outside the domain. ``build_diracs_first_then_unif_scale``
    requires N_bdy ≥ 1; the dummy point gets ordered first (its length
    scale is huge), all the *real* w-measurements come after, and we
    just zero its row/col when applying the factor."""
    N = X_dom.shape[0]
    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    dummy = np.array([[10.0, 10.0]])    # far outside [0,1]²
    return [
        _lgd(dummy, np.zeros(1), np.zeros((1, 2)), np.ones(1)),
        _lgd(X_dom, np.zeros(N), np.zeros((N, 2)), np.ones(N)),
        _lgd(X_dom, np.zeros(N), e1,               np.zeros(N)),
        _lgd(X_dom, np.zeros(N), e2,               np.zeros(N)),
    ]


# ---------------------------------------------------------------------------
# Theta forward / inverse apply via the big factor U  (Θ ≈ Pᵀ (UᵀU)⁻¹ P).
# ---------------------------------------------------------------------------


class _BigFactor:
    """Apply Θ ≈ Pᵀ (UᵀU)⁻¹ P  (forward) or  Θ⁻¹ ≈ Pᵀ UᵀU P  (inverse)."""
    __slots__ = ('U_csr', 'UT_csr', 'P')

    def __init__(self, explicit):
        self.U_csr = explicit.U.tocsr()
        self.UT_csr = explicit.U.T.tocsr()
        self.P = np.asarray(explicit.P, dtype=np.int64)

    def apply(self, v: np.ndarray) -> np.ndarray:
        vp = v[self.P]
        y = spla.spsolve_triangular(self.U_csr,  vp, lower=False)
        z = spla.spsolve_triangular(self.UT_csr, y,  lower=True)
        out = np.empty_like(v); out[self.P] = z
        return out

    def apply_inv(self, b: np.ndarray) -> np.ndarray:
        bp = b[self.P]
        z = self.UT_csr @ (self.U_csr @ bp)
        out = np.empty_like(b); out[self.P] = z
        return out


# ---------------------------------------------------------------------------
# z ↔ v_all / w_all packing  (with linearized v3 around z_old).
# ---------------------------------------------------------------------------


def _coefs(z, rhs_f, N):
    """Jacobian entries for the (linearized at z) PDE constraint
        Δu = c_w0 w0 + c_w1 w1 + c_w2 w2 + c_v1 v1 + c_v2 v2,
    matching ∂v3_lin/∂z at z."""
    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    return (rhs_f * np.exp(-w0), -v1, -v2, -w1, -w2)


def _v3_nonlinear(z, rhs_f, N):
    """Exact (nonlinear) v3 = -v1·w1 - v2·w2 - f·exp(-w0)  at z."""
    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    return -v1 * w1 - v2 * w2 - rhs_f * np.exp(-w0)


def _pack_v_all(z, v3_vals, bdy_g, N, Nb):
    """[δ_bdy, δ_int, ∂₁_int, ∂₂_int, Δ_int]  in *natural* order."""
    v0 = z[3*N:4*N]; v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    out = np.empty(Nb + 4*N)
    out[0:Nb]                = bdy_g
    out[Nb     : Nb +   N]   = v0
    out[Nb +   N : Nb + 2*N] = v1
    out[Nb + 2*N : Nb + 3*N] = v2
    out[Nb + 3*N : Nb + 4*N] = v3_vals
    return out


def _pack_w_all(z, N):
    """[dummy(0), δ_int, ∂₁_int, ∂₂_int]  in *natural* order (dummy=0)."""
    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    out = np.zeros(1 + 3*N)
    out[1     : 1 +   N] = w0
    out[1 +   N : 1 + 2*N] = w1
    out[1 + 2*N : 1 + 3*N] = w2
    return out


def _unpack_v_grad(y_u, coefs, N, Nb):
    """Apply Jᵀ_v to y_u (a length-(Nb + 4N) vector in natural order)
    using GN linearization coefs at the current z. Returns 6N-vector."""
    c_w0, c_w1, c_w2, c_v1, c_v2 = coefs
    yv0 = y_u[Nb     : Nb +   N]
    yv1 = y_u[Nb +   N : Nb + 2*N]
    yv2 = y_u[Nb + 2*N : Nb + 3*N]
    yv3 = y_u[Nb + 3*N : Nb + 4*N]
    out = np.empty(6 * N)
    out[0    : 1*N] = c_w0 * yv3
    out[1*N  : 2*N] = c_w1 * yv3
    out[2*N  : 3*N] = c_w2 * yv3
    out[3*N  : 4*N] = yv0
    out[4*N  : 5*N] = yv1 + c_v1 * yv3
    out[5*N  : 6*N] = yv2 + c_v2 * yv3
    return out


def _unpack_w_grad(y_w, N):
    """Apply Jᵀ_w to y_w (length 1 + 3N, dummy in slot 0)."""
    out = np.empty(6 * N)
    out[0    : 1*N] = y_w[1     : 1 +   N]
    out[1*N  : 2*N] = y_w[1 +   N : 1 + 2*N]
    out[2*N  : 3*N] = y_w[1 + 2*N : 1 + 3*N]
    out[3*N  : 6*N] = 0.0     # v-block
    return out


# ---------------------------------------------------------------------------
# GN Hessian matvec (matrix-free) and full-loss gradient.
# ---------------------------------------------------------------------------


def _full_grad(z, big_u, big_w, rhs_f, bdy_g, data, N, Nb, N_data, sigma):
    """Gradient of the *nonlinear* loss at z — uses exact v3."""
    v3 = _v3_nonlinear(z, rhs_f, N)
    v_all = _pack_v_all(z, v3, bdy_g, N, Nb)
    w_all = _pack_w_all(z, N)
    y_u = big_u.apply_inv(v_all)
    y_w = big_w.apply_inv(w_all)
    coefs = _coefs(z, rhs_f, N)
    g = 2.0 * _unpack_v_grad(y_u, coefs, N, Nb)
    g += 2.0 * _unpack_w_grad(y_w, N)
    v0 = z[3*N:4*N]
    g[3*N : 3*N + N_data] += (2.0 / sigma**2) * (v0[:N_data] - data)
    return g


def _gn_hess_matvec(q, z, big_u, big_w, rhs_f, N, Nb, N_data, sigma):
    """GN Hessian (linearized v3 at z) applied to q.

    Uses the bare big-factor inverse apply (Pᵀ UᵀU P) for Θ_u⁻¹, Θ_w⁻¹.
    NOTE: at modest ρ this matvec is too inaccurate to drive the outer
    pCG to convergence on this badly-conditioned joint Hessian; see the
    module docstring for the path forward (mirror kolesky.pde's
    train-op + preconditioner pattern).
    """
    coefs = _coefs(z, rhs_f, N)
    c_w0, c_w1, c_w2, c_v1, c_v2 = coefs
    qw0 = q[0:N]; qw1 = q[N:2*N]; qw2 = q[2*N:3*N]
    qv0 = q[3*N:4*N]; qv1 = q[4*N:5*N]; qv2 = q[5*N:6*N]
    qv3 = c_w0*qw0 + c_w1*qw1 + c_w2*qw2 + c_v1*qv1 + c_v2*qv2

    Jvq = _pack_v_all(q, qv3, np.zeros(Nb), N, Nb)
    Jwq = _pack_w_all(q, N)
    yu = big_u.apply_inv(Jvq)
    yw = big_w.apply_inv(Jwq)
    out = 2.0 * _unpack_v_grad(yu, coefs, N, Nb)
    out += 2.0 * _unpack_w_grad(yw, N)
    out[3*N : 3*N + N_data] += (2.0 / sigma**2) * qv0[:N_data]
    return out


# ---------------------------------------------------------------------------
# Block-diagonal Jacobi preconditioner for pCG.
#
# The GN Hessian's diagonal can be approximated cheaply by considering
# the pure-derivative diagonal part of Theta_u⁻¹ ≈ (UᵀU). Combined with
# coefs (a², ...) and the data term, we get a strong diagonal scaling.
# ---------------------------------------------------------------------------


def _build_diag_precond(big_u, big_w, z, rhs_f, N, Nb, N_data, sigma):
    """Approximate diag(GN-Hessian) for use as a Jacobi preconditioner.

    Each entry of diag(H) involves Θ_u⁻¹ on a basis vector, which we
    approximate by the *diagonal* of UᵀU. That's a single matvec on a
    diagonal mask via the sparse factor's structure: we form
    `(U^T U)_{ii} = ‖U[:, i]‖²`. Same trick on U_w.
    """
    # diag of UᵀU per natural-order index, then permute back.
    U_u = big_u.U_csr
    diag_U_perm = np.asarray(U_u.multiply(U_u).sum(axis=0)).ravel()
    P_u_inv = np.empty_like(big_u.P); P_u_inv[big_u.P] = np.arange(big_u.P.size)
    diag_u_natural = diag_U_perm[P_u_inv]   # length Nb + 4N
    # Slices in natural order
    diag_v0 = diag_u_natural[Nb     : Nb +   N]
    diag_v1 = diag_u_natural[Nb +   N : Nb + 2*N]
    diag_v2 = diag_u_natural[Nb + 2*N : Nb + 3*N]
    diag_v3 = diag_u_natural[Nb + 3*N : Nb + 4*N]

    U_w = big_w.U_csr
    diag_w_perm = np.asarray(U_w.multiply(U_w).sum(axis=0)).ravel()
    P_w_inv = np.empty_like(big_w.P); P_w_inv[big_w.P] = np.arange(big_w.P.size)
    diag_w_natural = diag_w_perm[P_w_inv]   # length 1 + 3N
    diag_w0 = diag_w_natural[1     : 1 +   N]
    diag_w1 = diag_w_natural[1 +   N : 1 + 2*N]
    diag_w2 = diag_w_natural[1 + 2*N : 1 + 3*N]

    c_w0, c_w1, c_w2, c_v1, c_v2 = _coefs(z, rhs_f, N)
    H_diag = np.empty(6 * N)
    # H = 2 (Jᵥᵀ Θ_u⁻¹ Jᵥ + Jᵥᵥᵀ Θ_w⁻¹ Jw) + (2/σ²) E_data
    H_diag[0:N]      = 2.0 * (c_w0**2 * diag_v3 + diag_w0)
    H_diag[N:2*N]    = 2.0 * (c_w1**2 * diag_v3 + diag_w1)
    H_diag[2*N:3*N]  = 2.0 * (c_w2**2 * diag_v3 + diag_w2)
    H_diag[3*N:4*N]  = 2.0 * diag_v0
    H_diag[3*N : 3*N + N_data] += 2.0 / sigma**2
    H_diag[4*N:5*N]  = 2.0 * (diag_v1 + c_v1**2 * diag_v3)
    H_diag[5*N:6*N]  = 2.0 * (diag_v2 + c_v2**2 * diag_v3)
    H_diag = np.maximum(H_diag, 1e-12)
    return 1.0 / H_diag


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--N-domain', type=int, default=2500)
    p.add_argument('--N-boundary', type=int, default=200)
    p.add_argument('--N-data', type=int, default=200)
    p.add_argument('--noise', type=float, default=1e-3)
    p.add_argument('--N-fd', type=int, default=120)
    p.add_argument('--kernel', default='Matern7half',
                   choices=['Gaussian', 'Matern5half', 'Matern7half', 'Matern9half'])
    p.add_argument('--kernel-sigma', type=float, default=0.2)
    p.add_argument('--rho', type=float, default=4.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--nugget', type=float, default=1e-8)
    p.add_argument('--GN-steps', type=int, default=6)
    p.add_argument('--pcg-rtol', type=float, default=1e-4)
    p.add_argument('--pcg-maxiter', type=int, default=400)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='docs/darcy_inverse_sparse.png')
    p.add_argument('--verbose', action='store_true', default=True)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import kolesky as kl

    kernels = {
        'Gaussian':    kl.GaussianCovariance,
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
    }
    kernel = kernels[args.kernel](args.kernel_sigma)

    print(f'[setup]  kernel = {args.kernel}, σ_kernel = {args.kernel_sigma},  '
          f'σ_noise = {args.noise},  ρ = {args.rho}')

    # ----- ground truth via FD -----
    c = 1.0
    def a_truth(x1, x2):
        return (np.exp(c * np.sin(2 * np.pi * x1) + c * np.sin(2 * np.pi * x2))
                + np.exp(-c * np.sin(2 * np.pi * x1) - c * np.sin(2 * np.pi * x2)))
    def f_rhs(x1, x2):
        return np.ones_like(x1)

    print(f'[truth]  FD solve at {args.N_fd}² grid …')
    t0 = time.perf_counter()
    u_truth = fd_darcy_forward(args.N_fd - 2, a_truth, f_rhs)
    print(f'[truth]  FD wall: {time.perf_counter()-t0:.2f} s')

    # ----- sample points -----
    X_dom = rng.uniform(0, 1, (args.N_domain, 2))
    nb = args.N_boundary // 4
    t = np.linspace(0, 1, nb + 1)[:-1]
    o = np.ones_like(t); z0 = np.zeros_like(t)
    X_bdy = np.concatenate([
        np.stack([t, z0], 1), np.stack([o, t], 1),
        np.stack([t[::-1], o], 1), np.stack([z0, t[::-1]], 1),
    ], axis=0)[:args.N_boundary]
    X_data = X_dom[:args.N_data]
    N = args.N_domain; Nb = X_bdy.shape[0]; Nd = args.N_data
    print(f'[sample] N_dom = {N},  N_bdy = {Nb},  N_data = {Nd}')

    # ----- noisy observations -----
    from scipy.interpolate import RegularGridInterpolator
    grid_xs = np.linspace(0, 1, args.N_fd)
    interp = RegularGridInterpolator((grid_xs, grid_xs), u_truth.T)
    u_at_data = interp(X_data)
    data_noisy = u_at_data + args.noise * rng.standard_normal(Nd)

    # ----- build big factors  U_u, U_w  -----
    print('[gram]   building sparse big factor U_u (5-set DiracsFirstThenUnifScale) …')
    t0 = time.perf_counter()
    impl_u = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
        kernel, _theta_u_groups(X_dom, X_bdy),
        rho=args.rho, k_neighbors=args.k_neighbors,
    )
    expl_u = kl.ExplicitKLFactorization(impl_u, nugget=args.nugget, backend='cpu')
    print(f'[gram]   U_u: shape = {expl_u.U.shape}, nnz = {expl_u.U.nnz:,}, '
          f'wall = {time.perf_counter()-t0:.2f} s')
    big_u = _BigFactor(expl_u)

    print('[gram]   building sparse big factor U_w (3-set + dummy bdy) …')
    t0 = time.perf_counter()
    impl_w = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
        kernel, _theta_w_groups(X_dom),
        rho=args.rho, k_neighbors=args.k_neighbors,
    )
    expl_w = kl.ExplicitKLFactorization(impl_w, nugget=args.nugget, backend='cpu')
    print(f'[gram]   U_w: shape = {expl_w.U.shape}, nnz = {expl_w.U.nnz:,}, '
          f'wall = {time.perf_counter()-t0:.2f} s')
    big_w = _BigFactor(expl_w)

    # ----- pCG operator + preconditioner -----
    rhs_f = np.ones(N, dtype=np.float64)
    bdy_g = np.zeros(Nb, dtype=np.float64)

    # ----- GN iteration -----
    z = rng.standard_normal(6 * N)
    print(f'\n[GN]     {args.GN_steps} steps,  pCG rtol = {args.pcg_rtol}, '
          f'maxiter = {args.pcg_maxiter}')

    def _loss_value(z_):
        v3 = _v3_nonlinear(z_, rhs_f, N)
        v_all = _pack_v_all(z_, v3, bdy_g, N, Nb)
        w_all = _pack_w_all(z_, N)
        Lu_v = big_u.U_csr @ v_all[big_u.P]
        Lw_w = big_w.U_csr @ w_all[big_w.P]
        v0 = z_[3*N:4*N]
        return (Lu_v @ Lu_v + Lw_w @ Lw_w
                + (1.0 / args.noise**2) * np.sum((v0[:Nd] - data_noisy)**2))

    print(f'         iter  0:  loss = {_loss_value(z):.4e}')
    t_loop = time.perf_counter()
    diverged = False
    for step in range(1, args.GN_steps + 1):
        g = _full_grad(z, big_u, big_w, rhs_f, bdy_g, data_noisy, N, Nb, Nd, args.noise)

        # pCG: solve  H · Δ = g   with diagonal preconditioner
        H_op = spla.LinearOperator(
            (6*N, 6*N),
            matvec=lambda q: _gn_hess_matvec(q, z, big_u, big_w, rhs_f, N, Nb, Nd, args.noise),
            dtype=np.float64,
        )
        diag_inv = _build_diag_precond(big_u, big_w, z, rhs_f, N, Nb, Nd, args.noise)
        M_op = spla.LinearOperator((6*N, 6*N), matvec=lambda q: diag_inv * q, dtype=np.float64)
        it = [0]
        t_pcg = time.perf_counter()
        delta, info = spla.cg(H_op, g, M=M_op, rtol=args.pcg_rtol,
                               maxiter=args.pcg_maxiter,
                               callback=lambda _x: it.__setitem__(0, it[0] + 1))
        if not np.all(np.isfinite(delta)):
            print(f'         iter {step:2d}:  pCG produced NaN/inf — outer GN diverged.')
            diverged = True
            break
        z = z - delta
        loss_now = _loss_value(z)
        print(f'         iter {step:2d}:  loss = {loss_now:.4e}   '
              f'pCG: {it[0]:>3d} iters, {time.perf_counter()-t_pcg:.2f} s')
        if not np.isfinite(loss_now):
            print(f'         iter {step:2d}:  loss is non-finite — outer GN diverged.')
            diverged = True
            break
    print(f'[GN]     wall: {time.perf_counter()-t_loop:.2f} s')
    if diverged:
        print('[note]   outer pCG with the bare big-factor matvec for Θ⁻¹ does '
              'not give an accurate enough Hessian apply to drive GN convergence '
              'on the joint inverse problem at this scale. See module docstring '
              'for the path forward (kolesky.pde-style train-op + small factor + '
              'noisy-ichol).')
        return

    # ----- extend solution to a test grid -----
    print('\n[extend] GP regression onto an 80² test grid …')
    N_test = 80
    xs = np.linspace(0, 1, N_test)
    XX, YY = np.meshgrid(xs, xs)
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Nt = X_test.shape[0]

    # Coefficients α_u = Θ_u⁻¹ v_all, α_w = Θ_w⁻¹ w_all
    v3_final = _v3_nonlinear(z, rhs_f, N)
    v_all = _pack_v_all(z, v3_final, bdy_g, N, Nb)
    w_all = _pack_w_all(z, N)
    alpha_u = big_u.apply_inv(v_all)    # length Nb + 4N
    alpha_w = big_u.apply_inv(w_all) if False else big_w.apply_inv(w_all)

    # K(δ_test, train) for u and w
    from kolesky.measurements import LaplaceGradDiracPointMeasurement, stack_measurements
    test_meas = LaplaceGradDiracPointMeasurement(
        coordinate=X_test, weight_laplace=np.zeros(Nt),
        weight_grad=np.zeros((Nt, 2)), weight_delta=np.ones(Nt),
    )
    train_meas_u = stack_measurements(_theta_u_groups(X_dom, X_bdy))
    train_meas_w = stack_measurements(_theta_w_groups(X_dom))
    Theta_u_test = np.asarray(kernel(test_meas, train_meas_u), dtype=np.float64)
    Theta_w_test = np.asarray(kernel(test_meas, train_meas_w), dtype=np.float64)

    u_recover = Theta_u_test @ alpha_u
    w_recover = Theta_w_test @ alpha_w
    a_recover = np.exp(w_recover)

    a_truth_grid = a_truth(XX.ravel(), YY.ravel())
    u_truth_test = interp(X_test)
    L2_u = float(np.sqrt(np.mean((u_recover - u_truth_test) ** 2)))
    L2_a = float(np.sqrt(np.mean((a_recover - a_truth_grid) ** 2)))
    rel_a = L2_a / float(np.sqrt(np.mean(a_truth_grid ** 2)))
    rel_u = L2_u / max(1e-12, float(np.sqrt(np.mean(u_truth_test ** 2))))
    print(f'\n[error]  L²(u)         = {L2_u:.3e}   (rel {rel_u:.2%})')
    print(f'[error]  L²(a-recover) = {L2_a:.3e}   (rel {rel_a:.2%})')

    # ----- render -----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    a_true_2d = a_truth_grid.reshape(N_test, N_test)
    a_rec_2d  = a_recover.reshape(N_test, N_test)
    u_true_2d = u_truth_test.reshape(N_test, N_test)
    u_rec_2d  = u_recover.reshape(N_test, N_test)
    a_lim = (float(min(a_true_2d.min(), a_rec_2d.min())),
             float(max(a_true_2d.max(), a_rec_2d.max())))
    u_lim = (float(min(u_true_2d.min(), u_rec_2d.min())),
             float(max(u_true_2d.max(), u_rec_2d.max())))

    im = axes[0, 0].contourf(XX, YY, a_true_2d, 40, cmap='coolwarm', vmin=a_lim[0], vmax=a_lim[1])
    axes[0, 0].set_title('truth $a(x)$'); plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].contourf(XX, YY, a_rec_2d, 40, cmap='coolwarm', vmin=a_lim[0], vmax=a_lim[1])
    axes[0, 1].set_title(f'recovered $a(x)$  (rel {rel_a:.1%})')
    plt.colorbar(im, ax=axes[0, 1])
    im = axes[1, 0].contourf(XX, YY, u_true_2d, 40, cmap='coolwarm', vmin=u_lim[0], vmax=u_lim[1])
    axes[1, 0].set_title('truth $u(x)$'); plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].contourf(XX, YY, u_rec_2d, 40, cmap='coolwarm', vmin=u_lim[0], vmax=u_lim[1])
    axes[1, 1].set_title(f'recovered $u(x)$  (rel {rel_u:.1%})')
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].plot(X_data[:, 0], X_data[:, 1], 'k.', ms=2.0, alpha=0.5)
    for ax in axes.flat:
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_aspect('equal')
    fig.suptitle(
        f'Darcy inverse — N_dom={N}, N_data={Nd}, σ={args.noise},  '
        f'sparse Cholesky (ρ={args.rho}) + GN+pCG',
        fontsize=12,
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[fig]    {out_path}')


if __name__ == '__main__':
    main()
