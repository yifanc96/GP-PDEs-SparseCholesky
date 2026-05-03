"""2-D Darcy inverse problem at scale via sparse Cholesky + joint GN + pCG.

Sibling of ``examples/darcy_inverse.py``: same Gauss-Newton formulation
on the joint ``z = (w0, w1, w2, v0, v1, v2)`` variables (length 6 N_dom),
with the same loss

    L(z) = ‖L_w⁻¹ w_all‖² + ‖L_u⁻¹ v_all‖² + (1/σ²) ‖v0[:Nd] − data‖²

and the same linearization of v3 = −v1·w1 − v2·w2 − f·exp(−w0) at the
current iterate. The *only* difference is that the dense Cholesky factors
``L_u`` (size N_bdy + 4·N_dom) and ``L_w`` (size 3·N_dom) are replaced by
**sparse KL factors** ``U_u``, ``U_w`` from kolesky's
``ImplicitKLFactorization.build_diracs_first_then_unif_scale``, with
``U_uᵀ U_u ≈ Θ_u⁻¹`` and ``U_wᵀ U_w ≈ Θ_w⁻¹`` (in the maximin permuted
order). The GN Hessian-vector product

    H q = 2 J_vᵀ (U_uᵀ U_u) J_v q  +  2 J_wᵀ (U_wᵀ U_w) J_w q
        + (2/σ²) E_dataᵀ E_data q

is then a chain of *exact* sparse matvecs — no inner CG, no inverse
approximation: ``UᵀU`` is the defining sparse primitive that the GP
regression's loss inner-product uses, equal to ``L⁻ᵀ L⁻¹`` in the dense
version's notation. pCG drives each GN step.

The dense version pays O((N_bdy + 4 N_dom)³) for L_u Cholesky and stores
~10 000² ≈ 1 GB of dense kernel at N_dom = 2500 — intractable. The
sparse factors store only O(N · ρᵈ) entries (~10⁵ at ρ=3).

**Status.** The Hessian-vector apply matches the dense version's
``hess_GN_at_zold`` exactly when the sparse factors are accurate. In
practice, at moderate ρ the GP-prior Hessian's condition number is
amplified by the kernel matrix's, and the simple Jacobi preconditioner
used here is not strong enough for outer pCG to drive the GN linear
solve to a useful step — divergence after a few GN iterations is
typical at ρ = 3 / 4. Stabilization paths the user can iterate on:

  * Stronger preconditioner — e.g. assemble the sparse Hessian
    explicitly (``H = J_vᵀ UᵀU J_v + …`` is sparse with O(N · ρ²ᵈ)
    nonzeros) and run a sparse direct solve at each GN step.
  * Outer-iteration damping (line search / under-relaxation).
  * Larger ρ (denser sparse factor) — accuracy of UᵀU as Θ⁻¹
    improves quadratically with ρ.

The dense ``examples/darcy_inverse.py`` (N_dom ~ 200, ~26 % rel err on
``a``) is the working baseline; this file is the scaffold for scaling
up via sparse factors with the joint-GN structure unchanged.
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
# Reference forward FD solver — same as in examples/darcy_inverse.py.
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
    sol = spla.spsolve(A, fv).reshape(N, N)
    out = np.zeros((N + 2, N + 2))
    out[1:N + 1, 1:N + 1] = sol
    return out


# ---------------------------------------------------------------------------
# Big-factor measurement groups (5-set for u, 3-set for w with dummy bdy).
# ---------------------------------------------------------------------------


def _lgd(coord, wL, wG, wD):
    from kolesky.measurements import LaplaceGradDiracPointMeasurement
    return LaplaceGradDiracPointMeasurement(
        coordinate=np.asarray(coord, dtype=np.float64),
        weight_laplace=np.asarray(wL, dtype=np.float64),
        weight_grad=np.asarray(wG, dtype=np.float64),
        weight_delta=np.asarray(wD, dtype=np.float64),
    )


def _theta_u_groups(X_dom: np.ndarray, X_bdy: np.ndarray):
    """5 sets in the natural order [δ_bdy, δ_int, ∂₁_int, ∂₂_int, Δ_int]."""
    N = X_dom.shape[0]; Nb = X_bdy.shape[0]
    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    return [
        _lgd(X_bdy, np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
        _lgd(X_dom, np.zeros(N),  np.zeros((N, 2)),  np.ones(N)),
        _lgd(X_dom, np.zeros(N),  e1,                np.zeros(N)),
        _lgd(X_dom, np.zeros(N),  e2,                np.zeros(N)),
        _lgd(X_dom, np.ones(N),   np.zeros((N, 2)),  np.zeros(N)),
    ]


def _theta_w_groups(X_dom: np.ndarray):
    """3 sets in the natural order [(dummy bdy), δ_int, ∂₁_int, ∂₂_int].

    The single faraway dummy boundary point lets us reuse
    ``build_diracs_first_then_unif_scale`` (which expects a non-empty
    boundary set); we leave its row at 0 in the loss / Hessian below."""
    N = X_dom.shape[0]
    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    dummy = np.array([[10.0, 10.0]])
    return [
        _lgd(dummy, np.zeros(1), np.zeros((1, 2)), np.ones(1)),
        _lgd(X_dom, np.zeros(N), np.zeros((N, 2)), np.ones(N)),
        _lgd(X_dom, np.zeros(N), e1,               np.zeros(N)),
        _lgd(X_dom, np.zeros(N), e2,               np.zeros(N)),
    ]


# ---------------------------------------------------------------------------
# Joint-GN helpers: pack z into the natural-order vectors v_all, w_all,
# matching the U_u and U_w big-factor row layouts.
# ---------------------------------------------------------------------------
#
# z layout (6N)            : [ w0 | w1 | w2 | v0 | v1 | v2 ]
# v_all natural order (Nb+4N): [ bdy_g | v0 | v1 | v2 | v3_lin ]
# w_all natural order (1+3N) : [ 0 (dummy) | w0 | w1 | w2 ]
# v3_lin (linearized v3 at z_old) :
#   v3_lin = c_w0·w0 + c_w1·w1 + c_w2·w2 + c_v1·v1 + c_v2·v2
# where the coefs come from ∂v3/∂z|z_old:
#   c_w0 =  f · exp(−w0_old)
#   c_w1 = −v1_old,    c_w2 = −v2_old
#   c_v1 = −w1_old,    c_v2 = −w2_old
# (matches darcy_inverse.py's GN_loss exactly).


def _coefs(z, rhs_f, N):
    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    return (rhs_f * np.exp(-w0), -v1, -v2, -w1, -w2)


def _v3_nonlinear(z, rhs_f, N):
    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    return -v1 * w1 - v2 * w2 - rhs_f * np.exp(-w0)


def _pack_v_all(z, v3_vals, bdy_g, N, Nb):
    v0 = z[3*N:4*N]; v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    out = np.empty(Nb + 4*N)
    out[0:Nb]                  = bdy_g
    out[Nb     : Nb +   N]     = v0
    out[Nb +   N : Nb + 2 * N] = v1
    out[Nb + 2*N : Nb + 3 * N] = v2
    out[Nb + 3*N : Nb + 4 * N] = v3_vals
    return out


def _pack_w_all(z, N):
    out = np.zeros(1 + 3 * N)            # dummy bdy row stays at 0
    out[1     : 1 +   N]                 = z[0:N]
    out[1 +   N : 1 + 2 * N]             = z[N:2*N]
    out[1 + 2*N : 1 + 3 * N]             = z[2*N:3*N]
    return out


def _unpack_v_grad(y_u, coefs, N, Nb):
    """Apply Jᵥᵀ to a length-(Nb + 4N) vector y_u in v_all natural order.
    Returns a 6N-vector in z layout."""
    c_w0, c_w1, c_w2, c_v1, c_v2 = coefs
    yv0 = y_u[Nb     : Nb +   N]
    yv1 = y_u[Nb +   N : Nb + 2 * N]
    yv2 = y_u[Nb + 2*N : Nb + 3 * N]
    yv3 = y_u[Nb + 3*N : Nb + 4 * N]
    out = np.empty(6 * N)
    out[0    : 1 * N] = c_w0 * yv3
    out[1*N  : 2 * N] = c_w1 * yv3
    out[2*N  : 3 * N] = c_w2 * yv3
    out[3*N  : 4 * N] = yv0
    out[4*N  : 5 * N] = yv1 + c_v1 * yv3
    out[5*N  : 6 * N] = yv2 + c_v2 * yv3
    return out


def _unpack_w_grad(y_w, N):
    """Apply J_wᵀ to a length-(1 + 3N) vector y_w in w_all natural order.
    Returns a 6N-vector (only the w-block is nonzero)."""
    out = np.zeros(6 * N)
    out[0    : 1 * N] = y_w[1            : 1 +   N]
    out[1*N  : 2 * N] = y_w[1 +   N : 1 + 2 * N]
    out[2*N  : 3 * N] = y_w[1 + 2*N : 1 + 3 * N]
    return out


# ---------------------------------------------------------------------------
# Sparse-factor matvec: applies (Pᵀ UᵀU P) v exactly (no triangular solves;
# no inverse approximation — this IS the loss-side primitive in the GP
# regression, equal to ``L⁻ᵀ L⁻¹`` in the dense version's notation, computed
# via two sparse matvecs).
# ---------------------------------------------------------------------------


class _SparseFactorInv:
    """Matvec form of Θ⁻¹ via the kolesky upper-triangular factor U.
    Convention check: kolesky's U is upper-triangular with
    ``U⁻ᵀ U⁻¹ = K_perm`` (forward apply via two triangular solves), so
    ``K_perm⁻¹ = U Uᵀ`` (verified numerically to machine precision at
    full rho). The matvec order for Θ⁻¹·v is therefore U @ (Uᵀ @ vp),
    NOT Uᵀ @ (U @ vp).
    """
    __slots__ = ('UUT_csr', 'P', 'U_csr', 'UT_csr')

    def __init__(self, explicit):
        # Precompute U Uᵀ once. nnz ~ O(N · ρ²ᵈ).
        self.U_csr  = explicit.U.tocsr()
        self.UT_csr = explicit.U.T.tocsr()
        UUT = (explicit.U @ explicit.U.T).tocsr()
        UUT.sort_indices()
        self.UUT_csr = UUT
        self.P = np.asarray(explicit.P, dtype=np.int64)

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Apply Θ⁻¹ ≈ P U Uᵀ Pᵀ to v via two sparse matvecs."""
        vp = v[self.P]
        z = self.U_csr @ (self.UT_csr @ vp)
        out = np.empty_like(v); out[self.P] = z
        return out


# ---------------------------------------------------------------------------
# GN gradient and Hessian-vector product at the current iterate z.
# Mirrors examples/darcy_inverse.py's loss / GN_loss / hess_GN_at_zold,
# but with each Theta_*⁻¹ apply replaced by a sparse-factor matvec.
# ---------------------------------------------------------------------------


def _full_grad(z, sf_u, sf_w, rhs_f, bdy_g, data, N, Nb, N_data, sigma):
    """Gradient of the *nonlinear* loss at z (uses exact v3, NOT linearized)."""
    v3 = _v3_nonlinear(z, rhs_f, N)
    v_all = _pack_v_all(z, v3, bdy_g, N, Nb)
    w_all = _pack_w_all(z, N)

    y_u = sf_u.apply(v_all)
    y_w = sf_w.apply(w_all)

    coefs = _coefs(z, rhs_f, N)
    g  = 2.0 * _unpack_v_grad(y_u, coefs, N, Nb)
    g += 2.0 * _unpack_w_grad(y_w, N)
    v0 = z[3*N:4*N]
    g[3*N : 3*N + N_data] += (2.0 / sigma**2) * (v0[:N_data] - data)
    return g


def _gn_hess_matvec(q, z, sf_u, sf_w, rhs_f, N, Nb, N_data, sigma):
    """GN Hessian (linearized v3 at z) applied to q. Mirrors hess_GN_at_zold."""
    coefs = _coefs(z, rhs_f, N)
    c_w0, c_w1, c_w2, c_v1, c_v2 = coefs
    qw0 = q[0:N]; qw1 = q[N:2*N]; qw2 = q[2*N:3*N]
    qv0 = q[3*N:4*N]; qv1 = q[4*N:5*N]; qv2 = q[5*N:6*N]
    qv3 = c_w0*qw0 + c_w1*qw1 + c_w2*qw2 + c_v1*qv1 + c_v2*qv2

    Jvq = _pack_v_all(q, qv3, np.zeros(Nb), N, Nb)
    Jwq = _pack_w_all(q, N)
    yu = sf_u.apply(Jvq)
    yw = sf_w.apply(Jwq)

    out  = 2.0 * _unpack_v_grad(yu, coefs, N, Nb)
    out += 2.0 * _unpack_w_grad(yw, N)
    out[3*N : 3*N + N_data] += (2.0 / sigma**2) * qv0[:N_data]
    return out


def _build_sparse_hessian(z, sf_u, sf_w, rhs_f, N, Nb, N_data, sigma):
    """Assemble the GN Hessian (linearized v3 at z) as an explicit sparse
    matrix:  H = 2 J_vᵀ (UTU_u) J_v + 2 J_wᵀ (UTU_w) J_w + (2/σ²) Eᵀ E.

    The Jacobians J_v, J_w are highly sparse (each row has ≤ 5 nonzeros);
    UTU_u and UTU_w are O(N · ρ²ᵈ) sparse. The product is sparse with
    O(N · ρ²ᵈ) nonzeros — small enough for a direct sparse solve.
    """
    c_w0, c_w1, c_w2, c_v1, c_v2 = _coefs(z, rhs_f, N)

    # ----- J_v : (Nb + 4N) × 6N  -----
    rows = []; cols = []; data = []
    j = np.arange(N)
    # v0 rows: identity onto z[3N : 4N]
    rows.append(Nb + j);          cols.append(3*N + j);     data.append(np.ones(N))
    # v1 rows
    rows.append(Nb +   N + j);    cols.append(4*N + j);     data.append(np.ones(N))
    # v2 rows
    rows.append(Nb + 2*N + j);    cols.append(5*N + j);     data.append(np.ones(N))
    # v3_lin rows (5 nonzeros per row)
    base = Nb + 3*N
    for w_block, c_arr in (
        (0,    c_w0), (N,    c_w1), (2*N,  c_w2),
        (4*N,  c_v1), (5*N,  c_v2),
    ):
        rows.append(base + j); cols.append(w_block + j); data.append(c_arr)
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    J_v = scipy.sparse.coo_matrix((data, (rows, cols)),
                                   shape=(Nb + 4*N, 6*N)).tocsr()

    # M_u = J_v[P_u, :] (rows permuted to UUᵀ_u's order).
    M_u = J_v[sf_u.P, :].tocsr()
    H_u = (M_u.T @ sf_u.UUT_csr @ M_u).tocsc()

    # ----- J_w : (1 + 3N) × 6N  -----
    rows = []; cols = []
    rows.append(1     + j); cols.append(j)
    rows.append(1 + N + j); cols.append(N + j)
    rows.append(1 + 2*N + j); cols.append(2*N + j)
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    data = np.ones(rows.size)
    J_w = scipy.sparse.coo_matrix((data, (rows, cols)),
                                   shape=(1 + 3*N, 6*N)).tocsr()
    M_w = J_w[sf_w.P, :].tocsr()
    H_w = (M_w.T @ sf_w.UUT_csr @ M_w).tocsc()

    H = 2.0 * H_u + 2.0 * H_w
    # data fidelity:  (2/σ²) on the v0[:N_data] diagonal
    data_diag = np.zeros(6 * N)
    data_diag[3*N : 3*N + N_data] = 2.0 / (sigma * sigma)
    H = H + scipy.sparse.diags(data_diag).tocsc()
    return H.tocsr()


def _diag_precond(sf_u, sf_w, z, rhs_f, N, Nb, N_data, sigma):
    """Diagonal of the GN Hessian (Jacobi preconditioner)."""
    # diag(U Uᵀ)_ii  = ‖U[i, :]‖² — row-norm-squared of the kolesky factor.
    diag_u_perm = np.asarray(sf_u.UUT_csr.diagonal()).ravel()
    P_u_inv = np.empty_like(sf_u.P); P_u_inv[sf_u.P] = np.arange(sf_u.P.size)
    diag_u_natural = diag_u_perm[P_u_inv]
    dv0 = diag_u_natural[Nb     : Nb +   N]
    dv1 = diag_u_natural[Nb +   N : Nb + 2 * N]
    dv2 = diag_u_natural[Nb + 2*N : Nb + 3 * N]
    dv3 = diag_u_natural[Nb + 3*N : Nb + 4 * N]

    diag_w_perm = np.asarray(sf_w.UUT_csr.diagonal()).ravel()
    P_w_inv = np.empty_like(sf_w.P); P_w_inv[sf_w.P] = np.arange(sf_w.P.size)
    diag_w_natural = diag_w_perm[P_w_inv]
    dw0 = diag_w_natural[1            : 1 +   N]
    dw1 = diag_w_natural[1 +   N      : 1 + 2 * N]
    dw2 = diag_w_natural[1 + 2*N      : 1 + 3 * N]

    c_w0, c_w1, c_w2, c_v1, c_v2 = _coefs(z, rhs_f, N)
    H = np.empty(6 * N)
    H[0    :   N]   = 2.0 * (c_w0**2 * dv3 + dw0)
    H[N    : 2*N]   = 2.0 * (c_w1**2 * dv3 + dw1)
    H[2*N  : 3*N]   = 2.0 * (c_w2**2 * dv3 + dw2)
    H[3*N  : 4*N]   = 2.0 * dv0
    H[3*N : 3*N + N_data] += 2.0 / sigma**2
    H[4*N  : 5*N]   = 2.0 * (dv1 + c_v1**2 * dv3)
    H[5*N  : 6*N]   = 2.0 * (dv2 + c_v2**2 * dv3)
    return 1.0 / np.maximum(H, 1e-12)


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
    p.add_argument('--kernel', default='Gaussian',
                   choices=['Gaussian', 'Matern5half', 'Matern7half', 'Matern9half'])
    p.add_argument('--kernel-sigma', type=float, default=0.2)
    p.add_argument('--rho', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--nugget', type=float, default=1e-8)
    p.add_argument('--GN-steps', type=int, default=6)
    p.add_argument('--pcg-rtol', type=float, default=1e-5)
    p.add_argument('--pcg-maxiter', type=int, default=400)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='docs/darcy_inverse_sparse.png')
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
    print(f'[setup]  kernel = {args.kernel},  σ_kernel = {args.kernel_sigma},  '
          f'σ_noise = {args.noise},  ρ = {args.rho}')

    # ----- ground truth via FD -----
    c_a = 1.0
    def a_truth(x1, x2):
        return (np.exp(c_a * np.sin(2*np.pi*x1) + c_a * np.sin(2*np.pi*x2))
                + np.exp(-c_a * np.sin(2*np.pi*x1) - c_a * np.sin(2*np.pi*x2)))
    def f_rhs(x1, x2):
        return np.ones_like(x1)

    print(f'[truth]  FD solve at {args.N_fd}² grid …')
    t0 = time.perf_counter()
    u_truth = fd_darcy_forward(args.N_fd - 2, a_truth, f_rhs)
    print(f'[truth]  FD wall: {time.perf_counter()-t0:.2f} s')

    # ----- sample points -----
    X_dom = rng.uniform(0, 1, (args.N_domain, 2))
    nb_per = args.N_boundary // 4
    t = np.linspace(0, 1, nb_per + 1)[:-1]
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

    # ----- sparse big factors built once outside GN loop -----
    print('[big]    building U_u (5-set DiracsFirstThenUnifScale) …')
    t0 = time.perf_counter()
    impl_u = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
        kernel, _theta_u_groups(X_dom, X_bdy),
        rho=args.rho, k_neighbors=args.k_neighbors,
    )
    expl_u = kl.ExplicitKLFactorization(impl_u, nugget=args.nugget, backend='cpu')
    print(f'[big]    U_u: shape = {expl_u.U.shape}, nnz = {expl_u.U.nnz:,}, '
          f'wall = {time.perf_counter()-t0:.2f} s')
    sf_u = _SparseFactorInv(expl_u)

    print('[big]    building U_w (3-set + dummy bdy) …')
    t0 = time.perf_counter()
    impl_w = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
        kernel, _theta_w_groups(X_dom),
        rho=args.rho, k_neighbors=args.k_neighbors,
    )
    expl_w = kl.ExplicitKLFactorization(impl_w, nugget=args.nugget, backend='cpu')
    print(f'[big]    U_w: shape = {expl_w.U.shape}, nnz = {expl_w.U.nnz:,}, '
          f'wall = {time.perf_counter()-t0:.2f} s')
    sf_w = _SparseFactorInv(expl_w)

    # ----- joint GN iteration -----
    rhs_f = np.ones(N, dtype=np.float64)
    bdy_g = np.zeros(Nb, dtype=np.float64)
    sigma = args.noise

    z = rng.standard_normal(6 * N)        # same random init as the dense version
    print(f'\n[GN]     {args.GN_steps} steps,  pCG rtol = {args.pcg_rtol},  '
          f'maxiter = {args.pcg_maxiter}')

    def _loss_value(z_):
        v3 = _v3_nonlinear(z_, rhs_f, N)
        v_all = _pack_v_all(z_, v3, bdy_g, N, Nb)
        w_all = _pack_w_all(z_, N)
        # Sparse-factor identity: K_perm⁻¹ = U Uᵀ, so v · K⁻¹ · v = ‖Uᵀ v‖².
        UTv_u = expl_u.U.T @ v_all[expl_u.P]
        UTv_w = expl_w.U.T @ w_all[expl_w.P]
        v0 = z_[3*N:4*N]
        return (UTv_u @ UTv_u + UTv_w @ UTv_w
                + (1.0/sigma**2) * np.sum((v0[:Nd] - data_noisy)**2))

    print(f'         iter  0:  loss = {_loss_value(z):.4e}')
    t_loop = time.perf_counter()
    for step in range(1, args.GN_steps + 1):
        t0 = time.perf_counter()
        g = _full_grad(z, sf_u, sf_w, rhs_f, bdy_g, data_noisy, N, Nb, Nd, sigma)
        # Assemble the GN Hessian as a sparse matrix and direct-solve it.
        # nnz(H) = O(N · ρ²ᵈ); spsolve handles 6N ~ 15 000 × 15 000 fine.
        H = _build_sparse_hessian(z, sf_u, sf_w, rhs_f, N, Nb, Nd, sigma)
        # Stabilize: tiny diagonal regularization (~ trace / N · 1e-10).
        H = H + scipy.sparse.diags(np.full(6*N, 1e-10 * np.abs(H.diagonal()).mean()))
        delta = spla.spsolve(H.tocsc(), g)
        z = z - delta
        loss_now = _loss_value(z)
        print(f'         iter {step:2d}:  loss = {loss_now:.4e}    '
              f'sparse_solve: nnz(H) = {H.nnz:,}, '
              f'{time.perf_counter()-t0:.2f} s')
    print(f'[GN]     wall: {time.perf_counter()-t_loop:.2f} s')

    # ----- predict on test grid (dense kernel evaluation) -----
    print('\n[extend] predict (u, w) on an 80² test grid …')
    t0 = time.perf_counter()
    N_test = 80
    xs = np.linspace(0, 1, N_test)
    XX, YY = np.meshgrid(xs, xs)
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Nt = X_test.shape[0]

    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    v0 = z[3*N:4*N]; v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    v3 = -v1 * w1 - v2 * w2 - rhs_f * np.exp(-w0)

    # sol_u and sol_w in the *sparse* natural order (matches sf_u, sf_w):
    sol_u = _pack_v_all(z, v3, bdy_g, N, Nb)         # [bdy_g, v0, v1, v2, v3]
    sol_w = _pack_w_all(z, N)                         # [0_dummy, w0, w1, w2]

    from kolesky.measurements import (
        LaplaceGradDiracPointMeasurement, stack_measurements,
    )
    test_meas = _lgd(X_test, np.zeros(Nt), np.zeros((Nt, 2)), np.ones(Nt))
    # Train measurements in *sparse natural order*: [δ_bdy, δ_int, ∂₁, ∂₂, Δ]
    train_meas_u = stack_measurements(_theta_u_groups(X_dom, X_bdy))
    # For w: same natural order [dummy_bdy, δ_int, ∂₁, ∂₂]
    train_meas_w = stack_measurements(_theta_w_groups(X_dom))

    Theta_u_test = np.asarray(kernel(test_meas, train_meas_u), dtype=np.float64)
    Theta_w_test = np.asarray(kernel(test_meas, train_meas_w), dtype=np.float64)

    # alpha = Theta⁻¹ sol  via the sparse-factor matvec (U Uᵀ form).
    alpha_u = sf_u.apply(sol_u)
    alpha_w = sf_w.apply(sol_w)

    u_pred = Theta_u_test @ alpha_u
    w_pred = Theta_w_test @ alpha_w
    a_pred = np.exp(w_pred)
    print(f'[extend] wall: {time.perf_counter()-t0:.2f} s')

    a_truth_grid = a_truth(XX.ravel(), YY.ravel())
    u_truth_test = interp(X_test)
    L2_u = float(np.sqrt(np.mean((u_pred - u_truth_test) ** 2)))
    L2_a = float(np.sqrt(np.mean((a_pred - a_truth_grid) ** 2)))
    rel_a = L2_a / float(np.sqrt(np.mean(a_truth_grid ** 2)))
    rel_u = L2_u / max(1e-12, float(np.sqrt(np.mean(u_truth_test ** 2))))
    print(f'\n[error]  L²(u)         = {L2_u:.3e}  (rel {rel_u:.2%})')
    print(f'[error]  L²(a-recover) = {L2_a:.3e}  (rel {rel_a:.2%})')

    # ----- render -----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    a_t2 = a_truth_grid.reshape(N_test, N_test)
    a_p2 = a_pred.reshape(N_test, N_test)
    u_t2 = u_truth_test.reshape(N_test, N_test)
    u_p2 = u_pred.reshape(N_test, N_test)
    a_lim = (float(min(a_t2.min(), a_p2.min())), float(max(a_t2.max(), a_p2.max())))
    u_lim = (float(min(u_t2.min(), u_p2.min())), float(max(u_t2.max(), u_p2.max())))
    im = axes[0, 0].contourf(XX, YY, a_t2, 40, cmap='coolwarm', vmin=a_lim[0], vmax=a_lim[1])
    axes[0, 0].set_title('truth $a(x)$'); plt.colorbar(im, ax=axes[0, 0])
    im = axes[0, 1].contourf(XX, YY, a_p2, 40, cmap='coolwarm', vmin=a_lim[0], vmax=a_lim[1])
    axes[0, 1].set_title(f'recovered $a(x)$  (rel {rel_a:.1%})')
    plt.colorbar(im, ax=axes[0, 1])
    im = axes[1, 0].contourf(XX, YY, u_t2, 40, cmap='coolwarm', vmin=u_lim[0], vmax=u_lim[1])
    axes[1, 0].set_title('truth $u(x)$'); plt.colorbar(im, ax=axes[1, 0])
    im = axes[1, 1].contourf(XX, YY, u_p2, 40, cmap='coolwarm', vmin=u_lim[0], vmax=u_lim[1])
    axes[1, 1].set_title(f'recovered $u(x)$  (rel {rel_u:.1%})')
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].plot(X_data[:, 0], X_data[:, 1], 'k.', ms=2.0, alpha=0.5)
    for ax in axes.flat:
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_aspect('equal')
    fig.suptitle(
        f'Darcy inverse — joint GN with sparse Cholesky  '
        f'(N_dom={N}, N_data={Nd}, ρ={args.rho}, σ={args.noise})',
        fontsize=12,
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[fig]    {out_path}')


if __name__ == '__main__':
    main()
