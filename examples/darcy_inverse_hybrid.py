"""Darcy inverse via joint-GN warm-up + alternating refinement (sparse Cholesky pCG).

Sibling of ``examples/darcy_inverse_pcg.py`` (joint GN with two-block
sparse-LU preconditioner). The user's mental model: alternating solve
*is* "forward PDE solve (u | a) → inverse coefficient solve (a | u)",
which is the natural problem split. Pure alternating from ``w = 0``
gets stuck at the trivial fixed point because the u-step at ``w = 0``
forces ``v3 = −f`` exactly, killing the w-step's signal. Joint GN does
not stall but its block-Jacobi pCG preconditioner ignores the v3
coupling and needs hundreds of iters at large N.

The hybrid: a few joint-GN steps to escape the trivial basin, then
switch to alternating where each subproblem is a single-GP regression
(``nonlin_elliptic`` template — small-factor sparse Cholesky as
preconditioner, ``O(10–30)`` pCG iters per subproblem).

Stage 1 (joint warm-up): ``warmup_steps`` joint-GN steps — same code
as ``darcy_inverse_pcg.py``. Output: non-trivial ``z̄ = (w̄, ū)``.

Stage 2 (alternating refinement): ``alt_steps`` outer iterations.
  * **u | w step**: With ``w̄`` fixed, solve ``L_u u = f exp(−w̄₀)``
    with boundary ``u = g``, noisy data ``u(x_i) ≈ y_i + N(0, σ²)``,
    where ``L_u = −Δ − w̄₁ ∂₁ − w̄₂ ∂₂``. Constraint set
    ``[δ_bdy, δ_data, L_u_int]``; α has size ``Nb + Nd + N``. Big
    factor ``U_big_u`` (5-set, built once); small factor on the
    constraint set (rebuilt per step, weights depend on w̄).
  * **w | u step**: With ``ū`` fixed, solve ``L_w w = v̄₃ +
    f exp(−w̄₀)(1 + w̄₀)`` where ``L_w = f exp(−w̄₀) δ + (−v̄₁) ∂₁ +
    (−v̄₂) ∂₂`` (linearized at w̄). Constraint set ``[L_w_int]``;
    α has size ``N``. Big factor ``U_big_w`` (3-set + dummy bdy);
    small factor on constraint set.

After Stage 2, the predicted ``u(x*) = k(δ_x*, train_u) · α_u`` and
``a(x*) = exp(k(δ_x*, train_w) · α_w)``.

Key cost difference from ``darcy_inverse_pcg.py``:
  joint-pcg: ~30 s × 6 GN steps for block-LU on H_uu, H_ww at N=2500.
  hybrid   : ~30 s × ``warmup_steps`` (joint), plus ~5 s per alternating
             step (small-factor cholesky + ~20 pCG iters at α-space size).
For ``warmup_steps = 2, alt_steps = 6`` at N=2500: ~90 s total vs ~3 min.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg as spla


# ===========================================================================
# Shared helpers (FD reference, kernel groups, packers, sparse-factor matvec).
# ===========================================================================


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


def _lgd(coord, wL, wG, wD):
    from kolesky.measurements import LaplaceGradDiracPointMeasurement
    return LaplaceGradDiracPointMeasurement(
        coordinate=np.asarray(coord, dtype=np.float64),
        weight_laplace=np.asarray(wL, dtype=np.float64),
        weight_grad=np.asarray(wG, dtype=np.float64),
        weight_delta=np.asarray(wD, dtype=np.float64),
    )


def _theta_u_groups(X_dom, X_bdy):
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


def _theta_w_groups(X_dom):
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
    out = np.zeros(1 + 3 * N)
    out[1     : 1 +   N]                 = z[0:N]
    out[1 +   N : 1 + 2 * N]             = z[N:2*N]
    out[1 + 2*N : 1 + 3 * N]             = z[2*N:3*N]
    return out


class _SparseFactorInv:
    __slots__ = ('UUT_csr', 'P', 'U_csr', 'UT_csr')
    def __init__(self, explicit):
        self.U_csr  = explicit.U.tocsr()
        self.UT_csr = explicit.U.T.tocsr()
        UUT = (explicit.U @ explicit.U.T).tocsr()
        UUT.sort_indices()
        self.UUT_csr = UUT
        self.P = np.asarray(explicit.P, dtype=np.int64)
    def apply(self, v):
        vp = v[self.P]
        z = self.U_csr @ (self.UT_csr @ vp)
        out = np.empty_like(v); out[self.P] = z
        return out


# ===========================================================================
# Stage 1 — joint-GN warm-up (compressed copy of darcy_inverse_pcg.py).
# ===========================================================================


def _build_jacobians(z, rhs_f, N, Nb):
    c_w0, c_w1, c_w2, c_v1, c_v2 = _coefs(z, rhs_f, N)
    j = np.arange(N, dtype=np.int64)
    rows_v = []; cols_v = []; data_v = []
    rows_v.append(Nb + j);          cols_v.append(3*N + j);     data_v.append(np.ones(N))
    rows_v.append(Nb +   N + j);    cols_v.append(4*N + j);     data_v.append(np.ones(N))
    rows_v.append(Nb + 2*N + j);    cols_v.append(5*N + j);     data_v.append(np.ones(N))
    base = Nb + 3*N
    for w_block, c_arr in (
        (0,    c_w0), (N,    c_w1), (2*N,  c_w2),
        (4*N,  c_v1), (5*N,  c_v2),
    ):
        rows_v.append(base + j); cols_v.append(w_block + j); data_v.append(c_arr)
    J_v = scipy.sparse.coo_matrix(
        (np.concatenate(data_v),
         (np.concatenate(rows_v), np.concatenate(cols_v))),
        shape=(Nb + 4*N, 6*N),
    ).tocsr()
    rows_w = []; cols_w = []
    rows_w.append(1     + j); cols_w.append(j)
    rows_w.append(1 + N + j); cols_w.append(N + j)
    rows_w.append(1 + 2*N + j); cols_w.append(2*N + j)
    J_w = scipy.sparse.coo_matrix(
        (np.ones(3*N), (np.concatenate(rows_w), np.concatenate(cols_w))),
        shape=(1 + 3*N, 6*N),
    ).tocsr()
    return J_v, J_w


def _build_full_hessian(z, sf_u, sf_w, rhs_f, N, Nb, Nd, sigma):
    J_v, J_w = _build_jacobians(z, rhs_f, N, Nb)
    M_u = J_v[sf_u.P, :].tocsr()
    H_u_full = (M_u.T @ sf_u.UUT_csr @ M_u).tocsr()
    M_w = J_w[sf_w.P, :].tocsr()
    H_w_full = (M_w.T @ sf_w.UUT_csr @ M_w).tocsr()
    H = (2.0 * H_u_full + 2.0 * H_w_full).tocsr()
    data_diag = np.zeros(6 * N)
    data_diag[3*N : 3*N + Nd] = 2.0 / (sigma * sigma)
    H = (H + scipy.sparse.diags(data_diag, format='csr')).tocsr()
    H = (0.5 * (H + H.T)).tocsr()
    return H


def _full_grad(z, sf_u, sf_w, rhs_f, bdy_g, data, N, Nb, Nd, sigma):
    v3 = _v3_nonlinear(z, rhs_f, N)
    v_all = _pack_v_all(z, v3, bdy_g, N, Nb)
    w_all = _pack_w_all(z, N)
    y_u = sf_u.apply(v_all); y_w = sf_w.apply(w_all)
    c_w0, c_w1, c_w2, c_v1, c_v2 = _coefs(z, rhs_f, N)
    yv0 = y_u[Nb : Nb +   N]
    yv1 = y_u[Nb +   N : Nb + 2 * N]
    yv2 = y_u[Nb + 2*N : Nb + 3 * N]
    yv3 = y_u[Nb + 3*N : Nb + 4 * N]
    g = np.empty(6 * N)
    g[0    : 1 * N] = 2.0 * c_w0 * yv3 + 2.0 * y_w[1     : 1 +   N]
    g[1*N  : 2 * N] = 2.0 * c_w1 * yv3 + 2.0 * y_w[1 +   N : 1 + 2 * N]
    g[2*N  : 3 * N] = 2.0 * c_w2 * yv3 + 2.0 * y_w[1 + 2*N : 1 + 3 * N]
    g[3*N  : 4 * N] = 2.0 * yv0
    g[4*N  : 5 * N] = 2.0 * yv1 + 2.0 * c_v1 * yv3
    g[5*N  : 6 * N] = 2.0 * yv2 + 2.0 * c_v2 * yv3
    v0 = z[3*N:4*N]
    g[3*N : 3*N + Nd] += (2.0 / sigma**2) * (v0[:Nd] - data)
    return g


class _BlockJacobiPrecond:
    __slots__ = ('lu_w', 'lu_u', 'N', 'shape')
    def __init__(self, H_csr, N, reg=1e-8):
        self.N = N
        H_ww = H_csr[:3*N, :3*N].tocsc()
        H_uu = H_csr[3*N:, 3*N:].tocsc()
        if reg > 0:
            tr_w = H_ww.diagonal().mean(); tr_u = H_uu.diagonal().mean()
            H_ww = H_ww + scipy.sparse.eye(3*N, format='csc') * (reg * abs(tr_w))
            H_uu = H_uu + scipy.sparse.eye(3*N, format='csc') * (reg * abs(tr_u))
        # Exact block solve (splu) factorises a 2D-PDE-like sparse matrix:
        # heavy fill-in -> O(N^~2) factorisation + superlinear triangular
        # solves, the dominant cost at large N. spilu (incomplete LU with a
        # drop tolerance) caps fill -> near-linear build/apply. Since this is
        # an inexact-Newton warm-up, an approximate block solve is enough.
        mode = os.environ.get('BJ_PREC', 'splu')
        if mode == 'spilu':
            drop = float(os.environ.get('BJ_DROP', '1e-4'))
            fill = float(os.environ.get('BJ_FILL', '10'))
            self.lu_w = spla.spilu(H_ww, drop_tol=drop, fill_factor=fill,
                                   permc_spec='MMD_AT_PLUS_A')
            self.lu_u = spla.spilu(H_uu, drop_tol=drop, fill_factor=fill,
                                   permc_spec='MMD_AT_PLUS_A')
        else:
            self.lu_w = spla.splu(H_ww, permc_spec='MMD_AT_PLUS_A')
            self.lu_u = spla.splu(H_uu, permc_spec='MMD_AT_PLUS_A')
        self.shape = (6*N, 6*N)
    def matvec(self, v):
        N = self.N
        out = np.empty(6 * N)
        out[:3*N] = self.lu_w.solve(v[:3*N])
        out[3*N:] = self.lu_u.solve(v[3*N:])
        return out
    def as_linop(self):
        return spla.LinearOperator(self.shape, matvec=self.matvec, dtype=np.float64)


def joint_gn_warmup(z, sf_u, sf_w, expl_u, expl_w, rhs_f, bdy_g, data,
                    N, Nb, Nd, sigma, n_steps, pcg_rtol, pcg_maxiter, verbose=True):
    """Run ``n_steps`` joint-GN steps with two-block sparse-LU pCG. Returns z."""
    def _loss_value(z_):
        v3 = _v3_nonlinear(z_, rhs_f, N)
        UTv_u = expl_u.U.T @ _pack_v_all(z_, v3, bdy_g, N, Nb)[expl_u.P]
        UTv_w = expl_w.U.T @ _pack_w_all(z_, N)[expl_w.P]
        v0 = z_[3*N:4*N]
        return (UTv_u @ UTv_u + UTv_w @ UTv_w
                + (1.0/sigma**2) * np.sum((v0[:Nd] - data)**2))
    if verbose:
        print(f'         iter  0:  loss = {_loss_value(z):.4e}')
    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        g = _full_grad(z, sf_u, sf_w, rhs_f, bdy_g, data, N, Nb, Nd, sigma)
        H = _build_full_hessian(z, sf_u, sf_w, rhs_f, N, Nb, Nd, sigma)
        precond = _BlockJacobiPrecond(H, N, reg=1e-8)
        H_op = spla.aslinearoperator(H); M_op = precond.as_linop()
        x0 = precond.matvec(g)
        n_it = [0]
        def _cb(_): n_it[0] += 1
        delta, info = spla.cg(H_op, g, x0=x0, M=M_op, rtol=pcg_rtol,
                               maxiter=pcg_maxiter, callback=_cb)
        z = z - delta
        if verbose:
            print(f'         iter {step:2d}:  loss = {_loss_value(z):.4e}    '
                  f'pCG: {n_it[0]} iters, info = {info},  '
                  f'wall = {time.perf_counter()-t0:.2f} s')
    return z


def _krige_w_to_fine(kl, kernel_w, alpha_wc, X_dom_c, X_fine):
    """Evaluate (w0, ∂₁w, ∂₂w) at X_fine from coarse dual weights alpha_wc.

    alpha_wc are the dual weights of the coarse Θ_w measurement set
    (_theta_w_groups(X_dom_c)); the fine w-field is the GP posterior mean
    k(·, train_w_c) · alpha_wc evaluated for δ / ∂₁ / ∂₂ measurements.
    """
    from kolesky.measurements import stack_measurements
    Nf = X_fine.shape[0]
    train_wc = stack_measurements(_theta_w_groups(X_dom_c))
    e1 = np.tile(np.array([1.0, 0.0]), (Nf, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (Nf, 1))
    m_val = _lgd(X_fine, np.zeros(Nf), np.zeros((Nf, 2)), np.ones(Nf))
    m_d1  = _lgd(X_fine, np.zeros(Nf), e1,                np.zeros(Nf))
    m_d2  = _lgd(X_fine, np.zeros(Nf), e2,                np.zeros(Nf))
    w0 = np.asarray(kernel_w(m_val, train_wc), dtype=np.float64) @ alpha_wc
    w1 = np.asarray(kernel_w(m_d1,  train_wc), dtype=np.float64) @ alpha_wc
    w2 = np.asarray(kernel_w(m_d2,  train_wc), dtype=np.float64) @ alpha_wc
    return w0, w1, w2


def coarse_warmup(kl, kernel_u, kernel_w, X_dom, X_bdy, interp, bdy_g,
                  N, Nb, Nd, sigma, args, verbose=True):
    """Approach A: joint-GN on a coarse subset, then krige w to all N points.

    Cost is O(1) in the fine N — the coarse joint-GN runs at a fixed budget
    and the only fine-N work is one Θ_w cross-covariance eval (kriging). Not
    multigrid: a single coarse solve + a kriging eval, no hierarchy/cycles.
    """
    Nc  = min(args.warmup_coarse_N, N)
    Ndc = min(args.warmup_coarse_Nd, Nd, Nc)
    Xc  = X_dom[:Nc]
    Xdc = Xc[:Ndc]
    data_c = interp(Xdc)  # clean truth values suffice to find the basin
    if verbose:
        print(f'         [coarse] Nc={Nc}, Ndc={Ndc}: building coarse big factors …')
    t0 = time.perf_counter()
    impl_uc = kl.ImplicitKLFactorization.build_follow_diracs(
        kernel_u, _theta_u_groups(Xc, X_bdy), rho=args.rho, k_neighbors=args.k_neighbors)
    expl_uc = kl.ExplicitKLFactorization(impl_uc, nugget=args.nugget, backend='cpu')
    sf_uc = _SparseFactorInv(expl_uc)
    impl_wc = kl.ImplicitKLFactorization.build_follow_diracs(
        kernel_w, _theta_w_groups(Xc), rho=args.rho, k_neighbors=args.k_neighbors)
    expl_wc = kl.ExplicitKLFactorization(impl_wc, nugget=args.nugget, backend='cpu')
    sf_wc = _SparseFactorInv(expl_wc)
    if verbose:
        print(f'         [coarse] factors built ({time.perf_counter()-t0:.2f} s); '
              f'joint-GN {args.warmup_steps} steps …')
    rhs_fc = np.ones(Nc, dtype=np.float64)
    z_c = np.random.default_rng(args.seed + 1).standard_normal(6 * Nc)
    z_c = joint_gn_warmup(
        z_c, sf_uc, sf_wc, expl_uc, expl_wc, rhs_fc, bdy_g, data_c,
        Nc, Nb, Ndc, sigma, args.warmup_steps, args.pcg_rtol, args.pcg_maxiter,
        verbose=verbose)
    # Krige the coarse w-field to all fine points.
    alpha_wc = sf_wc.apply(_pack_w_all(z_c, Nc))
    t0 = time.perf_counter()
    w0, w1, w2 = _krige_w_to_fine(kl, kernel_w, alpha_wc, Xc, X_dom)
    if verbose:
        print(f'         [coarse] kriged w-field to N={N} ({time.perf_counter()-t0:.2f} s)')
    z = np.zeros(6 * N)
    z[0:N] = w0; z[N:2*N] = w1; z[2*N:3*N] = w2
    return z


def data_first_warmup(kl, kernel_u, kernel_w, expl_w, sf_w, X_dom, X_bdy, X_data,
                      data_noisy, bdy_g, rhs_f, N, Nb, Nd, sigma, args, verbose=True):
    """Approach B: GP-regress u from {bdy, data} alone (no PDE), then one w|u
    solve to seed the w-field. No joint-GN.

    The data-only u fit gives gradients/Laplacian (v1, v2, v3) carrying the
    measurement signal; the single linearized w|u solve (at w0=0) turns that
    into a non-trivial w-field that escapes the trivial fixed point.
    """
    from kolesky.measurements import stack_measurements
    from kolesky.pde.pcg_ops import SmallPrecond
    # ---- u | (data + bdy), no PDE constraint ----
    meas_reg = [_lgd(X_bdy,  np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
                _lgd(X_data, np.zeros(Nd), np.zeros((Nd, 2)), np.ones(Nd))]
    impl_r = kl.ImplicitKLFactorization.build(
        kernel_u, meas_reg, rho=args.rho_small, k_neighbors=args.k_neighbors)
    expl_r = kl.ExplicitKLFactorization(impl_r, nugget=args.nugget, backend='cpu')
    sf_r = _SparseFactorInv(expl_r)
    nF = Nb + Nd
    sigma2 = float(sigma * sigma)
    # Solve (Θ_train + σ² on data rows) α = [g; data] via dense-free CG with
    # the noiseless small factor as preconditioner.
    train_r = stack_measurements(meas_reg)
    U_csr = sf_r.U_csr; UT_csr = sf_r.UT_csr; P = sf_r.P
    def _Theta_apply(a):
        v = a[P]
        y = spla.spsolve_triangular(U_csr, v, lower=False)
        zz = spla.spsolve_triangular(UT_csr, y, lower=True)
        out = np.empty_like(a); out[P] = zz
        return out
    def _A(a):
        out = _Theta_apply(a)
        out[Nb:] += sigma2 * a[Nb:]
        return out
    A_op = spla.LinearOperator((nF, nF), matvec=_A, dtype=np.float64)
    M_op = SmallPrecond(expl_r.U, expl_r.P).as_linear_operator()
    rhs = np.empty(nF); rhs[:Nb] = bdy_g; rhs[Nb:] = data_noisy
    alpha_r, _ = spla.cg(A_op, rhs, x0=M_op @ rhs, M=M_op,
                         rtol=args.pcg_rtol, maxiter=args.pcg_maxiter)
    # Evaluate v1=∂₁u, v2=∂₂u, v3=Δu at X_dom.
    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    m_d1 = _lgd(X_dom, np.zeros(N), e1, np.zeros(N))
    m_d2 = _lgd(X_dom, np.zeros(N), e2, np.zeros(N))
    m_lap = _lgd(X_dom, np.ones(N), np.zeros((N, 2)), np.zeros(N))
    v1 = np.asarray(kernel_u(m_d1,  train_r), dtype=np.float64) @ alpha_r
    v2 = np.asarray(kernel_u(m_d2,  train_r), dtype=np.float64) @ alpha_r
    v3 = np.asarray(kernel_u(m_lap, train_r), dtype=np.float64) @ alpha_r
    if verbose:
        print(f'         [data-first] u regressed from {nF} measurements; one w|u solve …')
    # ---- one w | u solve at w0 = 0 (so c0 = f) ----
    w0 = np.zeros(N)
    c0 = rhs_f * np.exp(-w0)   # = f
    c1 = -v1; c2 = -v2
    op_w = ThetaTrainOpW(expl_w, sf_w, N)
    op_w.set_weights(c0, c1, c2)
    meas_sw = _meas_constraint_w(X_dom, c0, c1, c2)
    impl_sw = kl.ImplicitKLFactorization.build(
        kernel_w, meas_sw[0], rho=args.rho_small, k_neighbors=args.k_neighbors)
    expl_sw = kl.ExplicitKLFactorization(impl_sw, nugget=args.nugget, backend='cpu')
    precond_w = SmallPrecond(expl_sw.U, expl_sw.P)
    rhs_w = v3 + c0 * (1.0 + w0)
    alpha_w, _ = spla.cg(op_w.as_linop(), rhs_w, x0=precond_w.matvec(rhs_w),
                         M=precond_w.as_linear_operator(),
                         rtol=args.pcg_rtol, maxiter=args.pcg_maxiter)
    y_full_w = op_w.predict_at(alpha_w)
    z = np.zeros(6 * N)
    z[0:N]     = y_full_w[1         : 1 +     N]
    z[N:2*N]   = y_full_w[1 +     N : 1 + 2 * N]
    z[2*N:3*N] = y_full_w[1 + 2 * N : 1 + 3 * N]
    return z


# ===========================================================================
# Stage 2 — alternating refinement.
#
# Each subproblem is a single-GP regression with the ``nonlin_elliptic``
# template:
#   * BigFactor matvec for ``Θ_train · α``  (lift / Θ_big_apply / extract).
#   * Small-factor sparse Cholesky of ``Θ_train`` as preconditioner.
#   * Outer pCG.
# ===========================================================================


# ---- u | w  subproblem -----------------------------------------------------
#
# Constraints (size Nb + N + Nd) — DATA LAST so PDE Diracs (more abundant)
# screen them in the maximin ordering:
#   bdy:  u(x_bdy) = g                                          (Nb)
#   pde:  L_u u = f exp(-w̄₀),   L_u = -Δ - w̄₁ ∂₁ - w̄₂ ∂₂       (N)
#   data: u(x_data) ≈ y_data,   noise σ                          (Nd, last)
#
# Big-factor lift α (Nb + N + Nd) → multi-feature (Nb + 4N):
#   α_bdy[i]  →  feat_δ_bdy[i]                       weight 1
#   α_pde[k]  →  feat_∂₁_int[k] / feat_∂₂_int[k] / feat_Δ_int[k]
#                with weights (-w̄₁[k], -w̄₂[k], -1.0)
#   α_data[k] →  feat_δ_int[k]   for k < Nd          weight 1
#
# Two noise modes:
#   * 'data': only data points have noise (σ²). Operator =
#     Θ_train + R_data with R_data = σ² on data block, 0 elsewhere.
#     Preconditioner = noiseless small-factor Cholesky of Θ_train
#     (no ichol noise-correction). With data ordered LAST, the
#     screening effect of PDE Diracs keeps the noiseless factor
#     accurate where it matters.
#   * 'uniform': treat ALL collocations as having σ²-noise (homoscedastic).
#     Operator = Θ_train + σ² I. Preconditioner = Algorithm 4.1 noisy
#     ichol of Θ_train⁻¹ + (1/σ²) I — exactly the regime ichol was
#     designed for. Slight model mismatch (PDE constraint becomes
#     "soft" with σ² slack) but very clean preconditioning.


class ThetaTrainOpU:
    """Lifted Θ_train operator for the u | w subproblem.

    α layout: [α_bdy (Nb), α_pde (N), α_data (Nd)] — data last.

    The matvec adds the noise term to the operator according to the
    selected noise_mode:
      * 'data'    : σ² · α_data on the data block only.
      * 'uniform' : σ² · α everywhere (homoscedastic).
      * 'none'    : no noise term — pure K matvec. Used by the SMW
        preconditioner's inner-pCG K-solve (where the operator IS K
        alone, separately from the σ²-data correction).
    """
    def __init__(self, expl_u, sf_u, Nb, N, Nd, sigma, noise_mode='data'):
        self.U_csr  = sf_u.U_csr
        self.UT_csr = sf_u.UT_csr
        self.P      = sf_u.P
        self.Nb = Nb; self.N = N; self.Nd = Nd
        self.sigma2 = float(sigma * sigma)
        self.noise_mode = noise_mode
        self.w1 = np.zeros(N); self.w2 = np.zeros(N)

    def set_weights(self, w1, w2):
        self.w1 = np.asarray(w1, dtype=np.float64)
        self.w2 = np.asarray(w2, dtype=np.float64)

    def _lift(self, alpha):
        Nb, N, Nd = self.Nb, self.N, self.Nd
        out = np.zeros(Nb + 4 * N)
        # α_bdy → δ_bdy block
        out[:Nb] = alpha[:Nb]
        # α_pde → derivative blocks (∂₁, ∂₂, Δ) at X_dom
        ap = alpha[Nb : Nb + N]
        out[Nb +     N : Nb + 2 * N] = -self.w1 * ap
        out[Nb + 2 * N : Nb + 3 * N] = -self.w2 * ap
        out[Nb + 3 * N : Nb + 4 * N] = -ap
        # α_data → δ_int block at first Nd of X_dom
        out[Nb : Nb + Nd] = alpha[Nb + N :]
        return out

    def _extract(self, y_full):
        Nb, N, Nd = self.Nb, self.N, self.Nd
        out = np.empty(Nb + N + Nd)
        out[:Nb]      = y_full[:Nb]
        d1 = y_full[Nb +     N : Nb + 2 * N]
        d2 = y_full[Nb + 2 * N : Nb + 3 * N]
        Dl = y_full[Nb + 3 * N : Nb + 4 * N]
        out[Nb : Nb + N]    = -self.w1 * d1 - self.w2 * d2 - Dl
        out[Nb + N :]       = y_full[Nb : Nb + Nd]
        return out

    def matvec(self, alpha):
        v = self._lift(alpha)
        vp = v[self.P]
        y = spla.spsolve_triangular(self.U_csr, vp, lower=False)
        z = spla.spsolve_triangular(self.UT_csr, y, lower=True)
        Theta_v = np.empty_like(v); Theta_v[self.P] = z
        Aalpha = self._extract(Theta_v)
        if self.noise_mode == 'data':
            Aalpha[self.Nb + self.N :] += self.sigma2 * alpha[self.Nb + self.N :]
        elif self.noise_mode == 'uniform':
            Aalpha += self.sigma2 * alpha
        elif self.noise_mode == 'none':
            pass  # pure K matvec, no noise term
        else:
            raise ValueError(f'unknown noise_mode={self.noise_mode!r}')
        return Aalpha

    def predict_at(self, alpha):
        v = self._lift(alpha)
        vp = v[self.P]
        y = spla.spsolve_triangular(self.U_csr, vp, lower=False)
        z = spla.spsolve_triangular(self.UT_csr, y, lower=True)
        Theta_v = np.empty_like(v); Theta_v[self.P] = z
        return Theta_v

    def as_linop(self):
        n = self.Nb + self.N + self.Nd
        return spla.LinearOperator((n, n), matvec=self.matvec, dtype=np.float64)


def _meas_constraint_u(X_bdy, X_dom, X_data, w1, w2, data_order='last'):
    """Constraint measurements for u | w step.

    α-vector layout depends on ``data_order``:

      * 'last' (default): [bdy, pde, data] — data LAST in the input list,
        so the maximin ordering's "later indices" tend to land on data
        points. Pairs with the screening intuition: PDE Diracs
        (more abundant, dense) are coarse-scale; data points get
        screened by them.
      * 'before': [bdy, data, pde] — data BEFORE the PDE constraint
        block. Useful for comparison; the small-factor's *coordinate
        set* is identical in both cases (since X_data ⊂ X_dom), but
        the index labelling and the resulting α-block layout differ.

    NOTE: ``ImplicitKLFactorization.build`` computes maximin on the
    UNION of all coordinates, so pure group-order changes here only
    relabel rows/cols of the small factor; they don't change which
    geometric point goes early vs. late in maximin. To force data
    points to truly come "later" in maximin, a separate conditioned-
    ordering API would be needed. For now the two ``data_order``
    choices test whether the labelling alone affects pCG.
    """
    Nb = X_bdy.shape[0]; N = X_dom.shape[0]; Nd = X_data.shape[0]
    grad = np.stack([-w1, -w2], axis=1)
    bdy = _lgd(X_bdy,  np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb))
    pde = _lgd(X_dom,  -np.ones(N),  grad,              np.zeros(N))
    dat = _lgd(X_data, np.zeros(Nd), np.zeros((Nd, 2)), np.ones(Nd))
    if data_order == 'last':
        return [bdy, pde, dat]
    elif data_order == 'before':
        return [bdy, dat, pde]
    raise ValueError(f'data_order must be "last" or "before", got {data_order!r}')


def _meas_constraint_u_fd(X_bdy, X_dom, Nd, w1, w2):
    """FollowDiracs-grouped constraint measurements for the u | w step.

    The plain ``build`` path runs maximin on the *union* of coordinates,
    and since the data δ-points are co-located with PDE collocation
    points (``X_data = X_dom[:Nd] ⊂ X_dom``), the union has duplicate
    coordinates → degenerate maximin / poor screening (needs huge ρ).

    Mirror how the big factor and the forward solver treat co-located
    functionals: group the data δ with the PDE L_w measurement *at the
    same point* into a single supernode via ``build_follow_diracs``.
    Layout (n_dom_sets = 2 over the Nd data points):

        group[0] = [bdy-δ (Nb)] ⊕ [non-data PDE L_w (N-Nd)]   (maximin set 0)
        group[1] = [data PDE L_w (Nd)]                          (maximin set 1, δ-ref)
        group[2] = [data δ (Nd)]                                (follows group[1])

    All coordinates are now *distinct* across the two maximin sets (data
    points appear only in group[1]/group[2], non-data only in group[0]),
    so maximin is non-degenerate; multi-set ordering forces the data
    points to the finest scale (screened by the abundant PDE points).

    Returns ``(measurements, src)`` where ``src`` maps the small factor's
    natural merged measurement order to the Θ_train α-layout
    ``[bdy(Nb), pde(N), data(Nd)]``:  ``merged_vec = alpha[src]``.
    """
    Nb = X_bdy.shape[0]; N = X_dom.shape[0]
    grad = np.stack([-w1, -w2], axis=1)
    X_data = X_dom[:Nd]; X_rest = X_dom[Nd:]
    bdy      = _lgd(X_bdy,  np.zeros(Nb),     np.zeros((Nb, 2)),     np.ones(Nb))
    pde_rest = _lgd(X_rest, -np.ones(N - Nd), grad[Nd:],             np.zeros(N - Nd))
    pde_data = _lgd(X_data, -np.ones(Nd),     grad[:Nd],             np.zeros(Nd))
    dat      = _lgd(X_data, np.zeros(Nd),     np.zeros((Nd, 2)),     np.ones(Nd))
    measurements = [[bdy, pde_rest], pde_data, dat]
    src = np.concatenate([
        np.arange(Nb),                 # bdy        -> α bdy block
        Nb + np.arange(Nd, N),         # non-data pde -> α pde block (tail)
        Nb + np.arange(Nd),            # data pde   -> α pde block (head)
        Nb + N + np.arange(Nd),        # data δ     -> α data block
    ]).astype(np.int64)
    return measurements, src


def _build_small_precond_alpha(args, kernel, X_bdy, X_dom, Nd, w1, w2, N, Nb):
    """Build the noiseless small-factor inverse M_0 ≈ Θ_train⁻¹ as a matvec
    in the Θ_train α-layout [bdy(Nb), pde(N), data(Nd)], for the
    parameterized ``--noise-mode small`` experiments.

    Controlled by:
      * ``--small-algo``: 'build' (plain maximin on the coordinate union)
        or 'follow-diracs' (group each data point's {L_w, δ} into one
        supernode — robust to the co-located data/PDE coordinates).
      * ``--data-order`` (plain build only): 'last' → [bdy, pde, data];
        'before' → [bdy, data, pde]. Changes the multi-set maximin
        sequence (which block is coarse vs. fine).

    Returns ``(M0, src)`` where ``M0(v)`` applies the small-factor inverse
    in α-layout and ``src`` maps α-layout → the factor's merged order
    (``merged = v[src]``).
    """
    import kolesky as kl
    from kolesky.pde.pcg_ops import SmallPrecond
    if args.small_algo == 'follow-diracs':
        meas, src = _meas_constraint_u_fd(X_bdy, X_dom, Nd, w1, w2)
        impl = kl.ImplicitKLFactorization.build_follow_diracs(
            kernel, meas, rho=args.rho_small, k_neighbors=args.k_neighbors)
    else:  # plain build
        X_data = X_dom[:Nd]
        meas = _meas_constraint_u(X_bdy, X_dom, X_data, w1, w2,
                                  data_order=args.data_order)
        impl = kl.ImplicitKLFactorization.build(
            kernel, meas, rho=args.rho_small, k_neighbors=args.k_neighbors)
        if args.data_order == 'last':           # merged [bdy, pde, data] == α
            src = np.arange(Nb + N + Nd, dtype=np.int64)
        else:                                   # 'before': merged [bdy, data, pde]
            src = np.concatenate([
                np.arange(Nb),
                Nb + N + np.arange(Nd),
                Nb + np.arange(N),
            ]).astype(np.int64)
    nugget = (args.pc_nugget if (args.pc_nugget is not None and args.pc_nugget >= 0.0)
              else args.nugget)
    expl = kl.ExplicitKLFactorization(impl, nugget=nugget, backend='cpu')
    base = SmallPrecond(expl.U, expl.P)

    def M0(v, base=base, src=src):
        out = np.empty_like(v)
        out[src] = base.matvec(v[src])
        return out

    return M0, src


# ---- w | u  subproblem -----------------------------------------------------
#
# Constraint (size N):
#   pde:  L_w w = v̄₃ + f exp(-w̄₀)(1 + w̄₀)
#         L_w = (f exp(-w̄₀)) δ + (-v̄₁) ∂₁ + (-v̄₂) ∂₂


class ThetaTrainOpW:
    def __init__(self, expl_w, sf_w, N):
        self.U_csr  = sf_w.U_csr
        self.UT_csr = sf_w.UT_csr
        self.P      = sf_w.P
        self.N = N
        self.c0 = np.zeros(N); self.c1 = np.zeros(N); self.c2 = np.zeros(N)

    def set_weights(self, c0, c1, c2):
        self.c0 = np.asarray(c0, dtype=np.float64)
        self.c1 = np.asarray(c1, dtype=np.float64)
        self.c2 = np.asarray(c2, dtype=np.float64)

    def _lift(self, alpha):
        N = self.N
        out = np.zeros(1 + 3 * N)
        out[1         : 1 +     N] = self.c0 * alpha
        out[1 +     N : 1 + 2 * N] = self.c1 * alpha
        out[1 + 2 * N : 1 + 3 * N] = self.c2 * alpha
        return out

    def _extract(self, y_full):
        N = self.N
        d  = y_full[1         : 1 +     N]
        d1 = y_full[1 +     N : 1 + 2 * N]
        d2 = y_full[1 + 2 * N : 1 + 3 * N]
        return self.c0 * d + self.c1 * d1 + self.c2 * d2

    def matvec(self, alpha):
        v = self._lift(alpha)
        vp = v[self.P]
        y = spla.spsolve_triangular(self.U_csr, vp, lower=False)
        z = spla.spsolve_triangular(self.UT_csr, y, lower=True)
        Theta_v = np.empty_like(v); Theta_v[self.P] = z
        return self._extract(Theta_v)

    def predict_at(self, alpha):
        v = self._lift(alpha)
        vp = v[self.P]
        y = spla.spsolve_triangular(self.U_csr, vp, lower=False)
        z = spla.spsolve_triangular(self.UT_csr, y, lower=True)
        Theta_v = np.empty_like(v); Theta_v[self.P] = z
        return Theta_v

    def as_linop(self):
        return spla.LinearOperator((self.N, self.N), matvec=self.matvec, dtype=np.float64)


def _meas_constraint_w(X_dom, c0, c1, c2):
    N = X_dom.shape[0]
    grad = np.stack([c1, c2], axis=1)
    return [_lgd(X_dom, np.zeros(N), grad, c0)]


def _z_to_state(z, rhs_f, bdy_g, N, Nb):
    """Extract (w0, w1, w2, v0, v1, v2, v3_nonlinear) at iterate z."""
    w0 = z[0:N]; w1 = z[N:2*N]; w2 = z[2*N:3*N]
    v0 = z[3*N:4*N]; v1 = z[4*N:5*N]; v2 = z[5*N:6*N]
    v3 = -v1 * w1 - v2 * w2 - rhs_f * np.exp(-w0)
    return w0, w1, w2, v0, v1, v2, v3


# ===========================================================================
# Driver
# ===========================================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--N-domain', type=int, default=2500)
    p.add_argument('--N-boundary', type=int, default=200)
    p.add_argument('--N-data', type=int, default=200)
    p.add_argument('--noise', type=float, default=1e-3)
    p.add_argument('--N-fd', type=int, default=120)
    p.add_argument('--kernel', default='Matern7half',
                   choices=['Gaussian', 'Matern5half', 'Matern7half'])
    p.add_argument('--kernel-sigma', type=float, default=0.2)
    p.add_argument('--kernel-sigma-u', type=float, default=None,
                   help="lengthscale for the Θ_u (PDE/u-field) prior. Defaults "
                        "to --kernel-sigma. Recovering the coefficient's spatial "
                        "structure needs a LONG ℓ_u (≈1.0): u solves an elliptic "
                        "PDE so it is smooth, and a short ℓ_u injects spurious "
                        "short-scale wiggles whose Laplacian (which pins a through "
                        "the PDE) is garbage. See --kernel-sigma-w.")
    p.add_argument('--kernel-sigma-w', type=float, default=None,
                   help="lengthscale for the Θ_w (coefficient) prior. Defaults to "
                        "--kernel-sigma. Keep SHORT (≈0.2) so a can have sharp "
                        "peaks and the w-factor stays local/sparse.")
    p.add_argument('--rho', type=float, default=3.0)
    p.add_argument('--rho-small', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=4)
    p.add_argument('--nugget', type=float, default=1e-8)
    p.add_argument('--pc-nugget', type=float, default=-1.0,
                   help="uniform-floor nugget for the data-smw preconditioner "
                        "base factor M_0 ≈ (Θ_train + ε²I)⁻¹, decoupled from the "
                        "operator nugget. <0 means reuse --nugget. The two-valued "
                        "diagonal R is split as ε²I (uniform, Schäfer-factorizable) "
                        "+ rank-Nd data correction (lifted to σ² by SMW).")
    p.add_argument('--warmup-steps', type=int, default=2)
    p.add_argument('--warmup-mode', default='coarse',
                   choices=['joint-gn', 'coarse', 'data-first'],
                   help="Stage-1 basin-finder. 'joint-gn' (default): joint-GN on "
                        "all N points (~N^2 splu/step). 'coarse': run joint-GN on a "
                        "fixed coarse subset (--warmup-coarse-N pts) then krige the "
                        "recovered w-field to all N via the Θ_w cross-cov — cost O(1) "
                        "in fine N. 'data-first': GP-regress u from data+bdy alone "
                        "(no PDE), then one w|u solve to seed w — no joint-GN at all.")
    p.add_argument('--warmup-coarse-N', type=int, default=400,
                   help="interior point budget for --warmup-mode coarse.")
    p.add_argument('--warmup-coarse-Nd', type=int, default=150,
                   help="data point budget for --warmup-mode coarse.")
    p.add_argument('--alt-steps', type=int, default=6)
    p.add_argument('--pcg-rtol', type=float, default=1e-6)
    p.add_argument('--pcg-maxiter', type=int, default=200)
    p.add_argument('--eps-pde', type=float, default=1e-8,
                   help="floor ε² on the boundary/PDE rows for the two-valued "
                        "Schäfer preconditioner ('schafer-2val' mode). The true "
                        "operator has ~0 there; ε²>0 makes R invertible so the "
                        "ichol of Θ⁻¹+R⁻¹ exists. ε²→0 recovers the exact "
                        "constraints; sweep to find where Schäfer accuracy breaks.")
    p.add_argument('--noise-mode', default='data',
                   choices=['data', 'small', 'data-fd', 'data-fd-smw', 'data-smw', 'data-schur', 'data-bordered', 'uniform', 'schafer-2val'],
                   help="'data' (default): noise σ² only on data points; "
                        "preconditioner = noiseless small-factor Cholesky of Θ_train. "
                        "'uniform': treat ALL collocations as σ²-noisy "
                        "(homoscedastic); preconditioner = Algorithm 4.1 noisy "
                        "ichol of Θ_train⁻¹ + (1/σ²) I.")
    p.add_argument('--small-algo', default='build', choices=['build', 'follow-diracs'],
                   help="small-Θ factorization algorithm for --noise-mode small: "
                        "'build' (plain maximin on coord union) or 'follow-diracs' "
                        "(group co-located data δ + PDE L_w per point).")
    p.add_argument('--small-smw', action='store_true',
                   help="for --noise-mode small: add exact rank-Nd SMW data-noise "
                        "correction on top of the noiseless small factor.")
    p.add_argument('--data-order', default='last', choices=['last', 'before'],
                   help="small-factor (plain build) group order for data δ vs "
                        "PDE block: 'last' = [bdy, pde, data] (data finest in "
                        "multi-set maximin); 'before' = [bdy, data, pde].")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='docs/darcy_inverse_hybrid.png')
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
    }
    ell_u = args.kernel_sigma_u if args.kernel_sigma_u is not None else args.kernel_sigma
    ell_w = args.kernel_sigma_w if args.kernel_sigma_w is not None else args.kernel_sigma
    kernel_u = kernels[args.kernel](ell_u)   # Θ_u (PDE/u-field) prior
    kernel_w = kernels[args.kernel](ell_w)   # Θ_w (coefficient) prior
    kernel = kernel_u                        # back-compat default for shared call sites
    print(f'[setup]  kernel = {args.kernel},  ℓ_u = {ell_u},  ℓ_w = {ell_w},  '
          f'σ_noise = {args.noise},  ρ = {args.rho},  ρ_small = {args.rho_small}')

    # Truth via FD
    c_a = 1.0
    def a_truth(x1, x2):
        return (np.exp(c_a * np.sin(2*np.pi*x1) + c_a * np.sin(2*np.pi*x2))
                + np.exp(-c_a * np.sin(2*np.pi*x1) - c_a * np.sin(2*np.pi*x2)))
    def f_rhs_fn(x1, x2):
        return np.ones_like(x1)
    print(f'[truth]  FD solve at {args.N_fd}² grid …')
    t0 = time.perf_counter()
    u_truth = fd_darcy_forward(args.N_fd - 2, a_truth, f_rhs_fn)
    print(f'[truth]  FD wall: {time.perf_counter()-t0:.2f} s')

    # Sample points
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

    from scipy.interpolate import RegularGridInterpolator
    grid_xs = np.linspace(0, 1, args.N_fd)
    interp = RegularGridInterpolator((grid_xs, grid_xs), u_truth.T)
    u_at_data = interp(X_data)
    data_noisy = u_at_data + args.noise * rng.standard_normal(Nd)

    # Big factors built once.
    print('\n[big]    building U_u (5-set FollowDiracs) …')
    t0 = time.perf_counter()
    impl_u = kl.ImplicitKLFactorization.build_follow_diracs(
        kernel_u, _theta_u_groups(X_dom, X_bdy),
        rho=args.rho, k_neighbors=args.k_neighbors,
    )
    expl_u = kl.ExplicitKLFactorization(impl_u, nugget=args.nugget, backend='cpu')
    sf_u = _SparseFactorInv(expl_u)
    print(f'[big]    U_u: shape = {expl_u.U.shape}, nnz = {expl_u.U.nnz:,}, '
          f'wall = {time.perf_counter()-t0:.2f} s')

    print('[big]    building U_w (3-set + dummy bdy) …')
    t0 = time.perf_counter()
    impl_w = kl.ImplicitKLFactorization.build_follow_diracs(
        kernel_w, _theta_w_groups(X_dom),
        rho=args.rho, k_neighbors=args.k_neighbors,
    )
    expl_w = kl.ExplicitKLFactorization(impl_w, nugget=args.nugget, backend='cpu')
    sf_w = _SparseFactorInv(expl_w)
    print(f'[big]    U_w: shape = {expl_w.U.shape}, nnz = {expl_w.U.nnz:,}, '
          f'wall = {time.perf_counter()-t0:.2f} s')

    rhs_f = np.ones(N, dtype=np.float64)
    bdy_g = np.zeros(Nb, dtype=np.float64)
    sigma = args.noise

    # ----- Stage 1: warm-up (basin finder) -----
    z = rng.standard_normal(6 * N)
    print(f'\n[stage 1] warm-up mode={args.warmup_mode}, steps={args.warmup_steps}')
    t_warmup = time.perf_counter()
    if args.warmup_mode == 'coarse':
        z = coarse_warmup(
            kl, kernel_u, kernel_w, X_dom, X_bdy, interp, bdy_g,
            N, Nb, Nd, sigma, args, verbose=True)
    elif args.warmup_mode == 'data-first':
        z = data_first_warmup(
            kl, kernel_u, kernel_w, expl_w, sf_w, X_dom, X_bdy, X_data,
            data_noisy, bdy_g, rhs_f, N, Nb, Nd, sigma, args, verbose=True)
    elif args.warmup_steps > 0:
        z = joint_gn_warmup(
            z, sf_u, sf_w, expl_u, expl_w, rhs_f, bdy_g, data_noisy,
            N, Nb, Nd, sigma, args.warmup_steps, args.pcg_rtol, args.pcg_maxiter,
            verbose=True,
        )
    else:
        z = np.zeros(6 * N)  # physical constant-coef start: w=0 -> a=1
        print('         (skipped: warmup_steps=0, z=0 -> w=0, a=1)')
    print(f'[stage 1] wall: {time.perf_counter()-t_warmup:.2f} s')

    # ----- Stage 2: alternating refinement -----
    print(f'\n[stage 2] alternating refinement: {args.alt_steps} outer steps  '
          f'(noise_mode={args.noise_mode})')
    # 'data-smw' and 'schafer-2val' both precondition the TRUE two-valued
    # operator (σ² on data rows only), so the operator stays in 'data' mode.
    op_noise_mode = 'data' if args.noise_mode in ('small', 'data-smw', 'data-fd', 'data-fd-smw', 'data-schur', 'data-bordered', 'schafer-2val') else args.noise_mode
    op_u = ThetaTrainOpU(expl_u, sf_u, Nb, N, Nd, sigma, noise_mode=op_noise_mode)
    op_w = ThetaTrainOpW(expl_w, sf_w, N)

    alpha_u = None; alpha_w = None
    t_stage2 = time.perf_counter()
    for outer in range(1, args.alt_steps + 1):
        t_outer = time.perf_counter()
        _prof = {}
        w0, w1, w2, v0, v1, v2, v3 = _z_to_state(z, rhs_f, bdy_g, N, Nb)

        # ---- u | w ----
        op_u.set_weights(w1, w2)
        _tp = time.perf_counter()
        meas_su = _meas_constraint_u(X_bdy, X_dom, X_data, w1, w2,
                                     data_order=args.data_order)
        impl_su = kl.ImplicitKLFactorization.build(
            kernel_u, meas_su, rho=args.rho_small, k_neighbors=args.k_neighbors,
        )
        expl_su = kl.ExplicitKLFactorization(impl_su, nugget=args.nugget, backend='cpu')
        _prof['u_precond_build'] = time.perf_counter() - _tp

        # Preconditioner: depends on noise_mode.
        #   * 'data':    kolesky sparse Cholesky of Θ_train + low-rank
        #     Sherman-Morrison-Woodbury correction for the rank-Nd data
        #     noise. Operator is Θ + σ² E_dᵀ E_d. Apply via
        #         M⁻¹ v = M_sparse v - M_sparse E_dᵀ (σ²I + Θ_dd)⁻¹ E_d M_sparse v
        #     where M_sparse ≈ Θ (= U U^T from the noiseless small factor)
        #     and Θ_dd = E_d Θ E_dᵀ is Nd × Nd dense (cached Cholesky at
        #     setup time). One extra small dense solve and one extra sparse
        #     matvec per pCG iter — turns the precond's effective
        #     condition number from O(σ²·‖Θ⁻¹‖) back to O(1) regardless
        #     of σ. Nd is small (typically Nd ≪ N), so the Nd × Nd
        #     dense block is cheap.
        #   * 'uniform': Algorithm 4.1 noisy ichol of Θ_train⁻¹ + (1/σ²) I
        #     (homoscedastic — Algorithm 4.1's stable regime).
        from kolesky.pde.pcg_ops import SmallPrecond
        if args.noise_mode == 'small':
            # Unified parameterized small-Θ preconditioner for ordering /
            # algorithm experiments. --small-algo {build, follow-diracs},
            # --data-order {last, before} (plain build), --small-smw to add
            # the exact rank-Nd data-noise SMW correction on top.
            _tp = time.perf_counter()
            M0, _src = _build_small_precond_alpha(
                args, kernel_u, X_bdy, X_dom, Nd, w1, w2, N, Nb)
            _prof['u_precond_build'] = _prof.get('u_precond_build', 0.0) + (time.perf_counter() - _tp)
            n_alpha = Nb + N + Nd
            if (not args.small_smw) or Nd == 0:
                M_u_op = spla.LinearOperator((n_alpha, n_alpha), matvec=M0, dtype=np.float64)
            else:
                import scipy.linalg as _sla
                sigma2 = float(sigma * sigma)
                d_rows = Nb + N + np.arange(Nd)
                _tp = time.perf_counter()
                W = np.empty((n_alpha, Nd))
                ej = np.zeros(n_alpha)
                for j in range(Nd):
                    ej[d_rows[j]] = 1.0
                    W[:, j] = M0(ej)
                    ej[d_rows[j]] = 0.0
                _prof['u_smw_Wloop'] = time.perf_counter() - _tp
                B = (1.0 / sigma2) * np.eye(Nd) + W[d_rows, :]
                B = 0.5 * (B + B.T)
                B_chol = _sla.cho_factor(B, lower=True)
                def Minv_small_smw(v, M0=M0, W=W, B_chol=B_chol, d_rows=d_rows):
                    y1 = M0(v)
                    s = _sla.cho_solve(B_chol, y1[d_rows])
                    return y1 - W @ s
                M_u_op = spla.LinearOperator((n_alpha, n_alpha), matvec=Minv_small_smw, dtype=np.float64)
        elif args.noise_mode == 'data':
            precond_u = SmallPrecond(expl_su.U, expl_su.P)
            M_u_op = precond_u.as_linear_operator()
        elif args.noise_mode in ('data-fd', 'data-fd-smw'):
            # FollowDiracs-grouped small factor: data δ co-located with the
            # PDE L_w measurement are kept in a single supernode (no
            # degenerate maximin on duplicate coords). The small factor
            # itself ≈ Θ_train⁻¹ (noiseless). Its natural measurement order
            # differs from the α-layout, so wrap with the `src` permutation.
            meas_fd, src_fd = _meas_constraint_u_fd(X_bdy, X_dom, Nd, w1, w2)
            impl_fd = kl.ImplicitKLFactorization.build_follow_diracs(
                kernel_u, meas_fd, rho=args.rho_small, k_neighbors=args.k_neighbors,
            )
            expl_fd = kl.ExplicitKLFactorization(impl_fd, nugget=args.nugget, backend='cpu')
            base_fd = SmallPrecond(expl_fd.U, expl_fd.P)
            n_alpha = Nb + N + Nd
            if args.noise_mode == 'data-fd' or Nd == 0:
                def Minv_fd(v, base=base_fd, src=src_fd):
                    out = np.empty_like(v)
                    out[src] = base.matvec(v[src])
                    return out
                M_u_op = spla.LinearOperator((n_alpha, n_alpha), matvec=Minv_fd, dtype=np.float64)
            else:
                # data-fd-smw: lift the data rows from the noiseless base to
                # σ² via exact rank-Nd Sherman-Morrison-Woodbury.
                #   M = (Θ + σ² E_dᵀ E_d)⁻¹,  M_0 ≈ Θ⁻¹ (the follow-diracs base)
                #   M v = M_0 v - W (σ⁻²I + E_d W)⁻¹ E_d M_0 v,  W = M_0 E_dᵀ
                import scipy.linalg as _sla
                sigma2 = float(sigma * sigma)
                d_rows = Nb + N + np.arange(Nd)  # data block in α-layout
                def M0(v, base=base_fd, src=src_fd):
                    out = np.empty_like(v)
                    out[src] = base.matvec(v[src])
                    return out
                W = np.empty((n_alpha, Nd))
                ej = np.zeros(n_alpha)
                for j in range(Nd):
                    ej[d_rows[j]] = 1.0
                    W[:, j] = M0(ej)
                    ej[d_rows[j]] = 0.0
                B = (1.0 / sigma2) * np.eye(Nd) + W[d_rows, :]
                B = 0.5 * (B + B.T)
                B_chol = _sla.cho_factor(B, lower=True)
                def Minv_fd_smw(v, M0=M0, W=W, B_chol=B_chol, d_rows=d_rows):
                    y1 = M0(v)
                    s = _sla.cho_solve(B_chol, y1[d_rows])
                    return y1 - W @ s
                M_u_op = spla.LinearOperator((n_alpha, n_alpha), matvec=Minv_fd_smw, dtype=np.float64)
        elif args.noise_mode == 'data-smw':
            # Precomputed Sherman-Morrison-Woodbury preconditioner for the
            # rank-Nd data noise. M_0 ≈ Θ_train⁻¹ is the small-factor
            # matvec; W := M_0 · E_dᵀ (Nd columns) is precomputed via Nd
            # sparse matvecs; B := σ⁻²I_Nd + E_d·W is dense Nd×Nd; one
            # dense Cholesky in setup. Per apply: y1 = M_0 v;
            # y2 = W·(B⁻¹·(E_d y1)); return y1 - y2. No inner pCG.
            # Base factor M_0 ≈ (Θ_train + ε²I)⁻¹. With --pc-nugget the floor
            # ε² (relative) is raised ABOVE the operator's nugget, so the
            # factorized operator is well-conditioned and screening-accurate
            # even when the bare Θ_train is near-singular at large N. The
            # rank-Nd SMW step below then lifts only the data rows to σ².
            if args.pc_nugget is not None and args.pc_nugget >= 0.0:
                expl_su_pc = kl.ExplicitKLFactorization(
                    impl_su, nugget=args.pc_nugget, backend='cpu')
                base = SmallPrecond(expl_su_pc.U, expl_su_pc.P)
            else:
                base = SmallPrecond(expl_su.U, expl_su.P)
            n_alpha = Nb + N + Nd
            sigma2 = float(sigma * sigma)
            # W = M_0 · E_dᵀ: Nd columns, one per data row in α-layout.
            W = np.empty((n_alpha, Nd), dtype=np.float64)
            ej = np.zeros(n_alpha)
            for j in range(Nd):
                ej[Nb + N + j] = 1.0
                W[:, j] = base.matvec(ej)
                ej[Nb + N + j] = 0.0
            # B = σ⁻²·I + E_d · W = σ⁻²·I + W[data rows]
            B = (1.0 / sigma2) * np.eye(Nd) + W[Nb + N :, :]
            B = 0.5 * (B + B.T)
            B_chol = np.linalg.cholesky(B)  # lower-triangular L: L L^T = B
            def Minv_smw(v, base=base, W=W, B_chol=B_chol, Nb=Nb, N=N):
                y1 = base.matvec(v)
                r  = y1[Nb + N :]
                t  = np.linalg.solve(B_chol, r)
                z  = np.linalg.solve(B_chol.T, t)
                y2 = W @ z
                return y1 - y2
            M_u_op = spla.LinearOperator(
                (n_alpha, n_alpha), matvec=Minv_smw, dtype=np.float64,
            )
        elif args.noise_mode == 'data-schur':
            # Data-free base factor + exact rank-Nd Schur border.
            #
            # The cost of the bare 'data' preconditioner is the data
            # δ-measurements co-located on the PDE Δδ-collocation points
            # (X_data ⊂ X_dom): near-duplicate rows that the screened factor
            # only resolves at large ρ. Instead, factor the *data-free*
            # forward operator Θ_FF over {bdy, PDE} only — a clean,
            # perfectly-screened system that inherits the forward solver's
            # O(1) pCG behaviour — and bring the Nd data rows in as an exact
            # bordered (Schur-complement) correction.
            #
            #   A = [[Θ_FF , Θ_Fd        ],   M_F ≈ Θ_FF⁻¹ (data-free factor)
            #        [Θ_dF , Θ_dd + σ²I ]]    G   = Θ_Fd  (Nd cross-cov cols)
            #   S = (Θ_dd + σ²I) − Gᵀ M_F G            (Nd×Nd Schur complement)
            #   A⁻¹ via the standard 2×2 block-inverse using M_F and S⁻¹.
            import scipy.linalg as _sla
            nF = Nb + N
            n_alpha = nF + Nd
            sigma2 = float(sigma * sigma)
            # Data-free forward factor: {δ_bdy, (−Δ + grad·∇)_pde}, NO data.
            grad_F = np.stack([-op_u.w1, -op_u.w2], axis=1)
            meas_F = [
                _lgd(X_bdy, np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
                _lgd(X_dom, -np.ones(N),  grad_F,            np.zeros(N)),
            ]
            impl_F = kl.ImplicitKLFactorization.build(
                kernel_u, meas_F, rho=args.rho_small, k_neighbors=args.k_neighbors,
            )
            expl_F = kl.ExplicitKLFactorization(impl_F, nugget=args.nugget, backend='cpu')
            M_F = SmallPrecond(expl_F.U, expl_F.P)
            if Nd == 0:
                M_u_op = spla.LinearOperator(
                    (n_alpha, n_alpha), matvec=M_F.matvec, dtype=np.float64)
            else:
                # Cross-covariance G = Θ_Fd and Θ_dd from the noiseless big-factor
                # operator: column j is Θ_train · e_j for the j-th data row.
                op_nl = ThetaTrainOpU(expl_u, sf_u, Nb, N, Nd, sigma, noise_mode='none')
                op_nl.set_weights(op_u.w1, op_u.w2)
                G = np.empty((nF, Nd)); Theta_dd = np.empty((Nd, Nd))
                ej = np.zeros(n_alpha)
                for j in range(Nd):
                    ej[nF + j] = 1.0
                    col = op_nl.matvec(ej)
                    ej[nF + j] = 0.0
                    G[:, j] = col[:nF]
                    Theta_dd[:, j] = col[nF:]
                MG = np.empty((nF, Nd))
                for j in range(Nd):
                    MG[:, j] = M_F.matvec(G[:, j])
                S = Theta_dd + sigma2 * np.eye(Nd) - G.T @ MG
                S = 0.5 * (S + S.T)
                S_chol = _sla.cho_factor(S, lower=True)
                def Minv_schur(v, M_F=M_F, G=G, S_chol=S_chol, nF=nF):
                    vF = v[:nF]; vd = v[nF:]
                    q  = M_F.matvec(vF)
                    s  = _sla.cho_solve(S_chol, G.T @ q - vd)
                    out = np.empty_like(v)
                    out[:nF] = q + M_F.matvec(G @ s)
                    out[nF:] = -s
                    return out
                M_u_op = spla.LinearOperator(
                    (n_alpha, n_alpha), matvec=Minv_schur, dtype=np.float64)
        elif args.noise_mode == 'data-bordered':
            # NEAR-LINEAR data u-step: DIRECT bordered/Schur solve, no outer CG.
            #   F = {bdy, PDE} forward block (screens, M_F + accurate CG);
            #   d = {Nd noisy interior δ}.  Solve in closed form:
            #     yF0 = Θ_FF⁻¹ rhs_F ;  x_d = S⁻¹(rhs_d − Θ_dF yF0)
            #     x_F = Θ_FF⁻¹(rhs_F − Θ_Fd x_d)
            #   with S = S0 + σ²I and S0 = Θ_dd − Θ_dF Θ_FF⁻¹ Θ_Fd the data-block
            #   posterior covariance.  KEY: S0 is obtained WITHOUT any forward
            #   solves via the block-inverse identity (Θ_joint⁻¹)_dd = S0⁻¹ — read
            #   from the joint screened precision factor expl_su (built above from
            #   meas_su=[bdy,pde,data]).  Cost = 2 forward solves + Nd sparse
            #   matvecs + one tiny Nd×Nd dense solve.
            import scipy.linalg as _sla
            n_alpha = Nb + N + Nd
            nF = Nb + N
            sigma2 = float(sigma * sigma)
            # data-free forward factor M_F + forward-only operator
            meas_F = [_lgd(X_bdy, np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
                      _lgd(X_dom, -np.ones(N), np.stack([-w1, -w2], 1), np.zeros(N))]
            impl_F = kl.ImplicitKLFactorization.build(
                kernel_u, meas_F, rho=args.rho_small, k_neighbors=args.k_neighbors)
            expl_F = kl.ExplicitKLFactorization(impl_F, nugget=args.nugget, backend='cpu')
            M_F = SmallPrecond(expl_F.U, expl_F.P)
            MF_lin = M_F.as_linear_operator()
            op_F = ThetaTrainOpU(expl_u, sf_u, Nb, N, 0, sigma, noise_mode='none')
            op_F.set_weights(w1, w2)
            opF_lin = op_F.as_linop()
            op_nl = ThetaTrainOpU(expl_u, sf_u, Nb, N, Nd, sigma, noise_mode='none')
            op_nl.set_weights(w1, w2)
            if Nd > 0:
                # S0⁻¹ = (Θ_joint⁻¹)_dd via the joint screened precision factor.
                # NEAR-LINEAR extraction: SmallPrecond applies Θ⁻¹ = U Uᵀ in
                # permuted coordinates (out[P] = U Uᵀ b[P]), so
                #   (Θ⁻¹)[d_i, d_j] = (U Uᵀ)[a_i, a_j],  a = invP[d],
                # i.e. the dd-block is the dense Gram of the Nd permuted-data ROWS
                # of U. This is O(nnz in those rows), NOT Nd full Θ⁻¹-matvecs
                # (which cost O(Nd·nnz(U)) = O(N²)).
                d_rows = nF + np.arange(Nd)
                P_su = np.asarray(expl_su.P, dtype=np.int64)
                invP = np.empty(P_su.shape[0], dtype=np.int64)
                invP[P_su] = np.arange(P_su.shape[0])
                U_su = expl_su.U.tocsr()
                Usub = U_su[invP[d_rows], :]
                Pdd = np.asarray((Usub @ Usub.T).todense(), dtype=np.float64)
                Pdd = 0.5 * (Pdd + Pdd.T)
                S = np.linalg.inv(Pdd); S = 0.5 * (S + S.T) + sigma2 * np.eye(Nd)
                Sc = _sla.cho_factor(S, lower=True)
            def _solveFF(b):
                nj = [0]
                y, _ = spla.cg(opF_lin, b, x0=MF_lin @ b, M=MF_lin,
                               rtol=1e-10, maxiter=200,
                               callback=lambda _: nj.__setitem__(0, nj[0] + 1))
                return y, nj[0]
            def _Theta_dF(yF):
                v = np.zeros(n_alpha); v[:nF] = yF; return op_nl.matvec(v)[nF:]
            def _Theta_Fd(xd_):
                v = np.zeros(n_alpha); v[nF:] = xd_; return op_nl.matvec(v)[:nF]
            def bordered_solve(rhs, nF=nF):
                yF0, k0 = _solveFF(rhs[:nF])
                if Nd == 0:
                    a = np.empty(n_alpha); a[:nF] = yF0; return a, 0, k0
                xd = _sla.cho_solve(Sc, rhs[nF:] - _Theta_dF(yF0))
                xF, k1 = _solveFF(rhs[:nF] - _Theta_Fd(xd))
                a = np.empty(n_alpha); a[:nF] = xF; a[nF:] = xd
                return a, 0, k0 + k1
            M_u_op = None
        elif args.noise_mode in ('uniform', 'schafer-2val'):
            # Schäfer Algorithm 4.1 preconditioner for Σ = Θ_train + R.
            #   * 'uniform':     R = σ²·I on ALL rows (homoscedastic; matched to
            #     the 'uniform' operator). Validates that the trick gives O(1).
            #   * 'schafer-2val': R = diag(ε² on the Nb+N bdy/PDE rows,
            #     σ² on the Nd data rows) — the TRUE two-valued nugget. The
            #     ichol of Θ⁻¹+R⁻¹ supports a per-row R⁻¹ vector directly, so
            #     the screening pattern is untouched; the only question is
            #     whether accuracy survives the ε²↔σ² spread.
            # Correct symmetric preconditioner (this is the piece the old
            # 'uniform' code was missing — it applied only Ũ⁻ᵀŨ⁻¹ = (Θ⁻¹+R⁻¹)⁻¹,
            # which is NOT ≈(Θ+R)⁻¹):
            #     M = R⁻¹ − R⁻¹ Ũ⁻ᵀ Ũ⁻¹ R⁻¹  ≈  (Θ + R)⁻¹.
            n_alpha = Nb + N + Nd
            if args.noise_mode == 'uniform':
                R_vec = np.full(n_alpha, sigma * sigma, dtype=np.float64)
            else:
                R_vec = np.full(n_alpha, float(args.eps_pde), dtype=np.float64)
                R_vec[Nb + N :] = sigma * sigma
            noisy_su = kl.NoisyExplicitKLFactorization.build(expl_su, R=R_vec)
            Ut_csr   = noisy_su.U_tilde.T.tocsr()
            Utop_csr = noisy_su.U_tilde.tocsr()
            P_su     = np.asarray(noisy_su.P, dtype=np.int64)
            Rinv     = 1.0 / R_vec
            def Minv_u(v, P_su=P_su, Ut=Ut_csr, Utop=Utop_csr, Rinv=Rinv):
                t  = Rinv * v                       # R⁻¹ v
                tp = t[P_su]
                yy = spla.spsolve_triangular(Utop, tp, lower=False)   # Ũ⁻¹
                zz = spla.spsolve_triangular(Ut,   yy, lower=True)    # Ũ⁻ᵀ
                u  = np.empty_like(v); u[P_su] = zz                   # A⁻¹ R⁻¹ v
                return t - Rinv * u                                   # R⁻¹v − R⁻¹A⁻¹R⁻¹v
            M_u_op = spla.LinearOperator((n_alpha, n_alpha), matvec=Minv_u, dtype=np.float64)
        else:
            raise ValueError(f'unknown noise_mode {args.noise_mode!r}')

        # rhs in α-layout [bdy, pde, data]:
        rhs_u = np.empty(Nb + N + Nd)
        rhs_u[:Nb]               = bdy_g
        rhs_u[Nb : Nb + N]       = rhs_f * np.exp(-w0)
        rhs_u[Nb + N :]          = data_noisy
        n_it_u = [0]
        def _cb_u(_): n_it_u[0] += 1
        _tp = time.perf_counter()
        if args.noise_mode == 'data-bordered':
            alpha_u, info_u, n_it_u[0] = bordered_solve(rhs_u)
        else:
            x0u = M_u_op @ rhs_u
            alpha_u, info_u = spla.cg(
                op_u.as_linop(), rhs_u, x0=x0u, M=M_u_op,
                rtol=args.pcg_rtol, maxiter=args.pcg_maxiter, callback=_cb_u,
            )
        _prof['u_pcg'] = time.perf_counter() - _tp
        # Recover (v0, v1, v2, v3) at X_dom from α_u via the big-factor forward apply.
        y_full_u = op_u.predict_at(alpha_u)
        v0_new = y_full_u[Nb            : Nb +     N]
        v1_new = y_full_u[Nb +     N    : Nb + 2 * N]
        v2_new = y_full_u[Nb + 2 * N    : Nb + 3 * N]
        v3_new = y_full_u[Nb + 3 * N    : Nb + 4 * N]
        z[3*N:4*N] = v0_new
        z[4*N:5*N] = v1_new
        z[5*N:6*N] = v2_new
        # also store v3 for the w-step: held in v3_new
        res_u = float(np.linalg.norm(op_u.matvec(alpha_u) - rhs_u) / max(np.linalg.norm(rhs_u), 1e-300))

        # ---- w | u ----
        c0 = rhs_f * np.exp(-w0)
        c1 = -v1_new
        c2 = -v2_new
        op_w.set_weights(c0, c1, c2)
        _tp = time.perf_counter()
        meas_sw = _meas_constraint_w(X_dom, c0, c1, c2)
        impl_sw = kl.ImplicitKLFactorization.build(
            kernel_w, meas_sw[0], rho=args.rho_small, k_neighbors=args.k_neighbors,
        )
        expl_sw = kl.ExplicitKLFactorization(impl_sw, nugget=args.nugget, backend='cpu')
        precond_w = SmallPrecond(expl_sw.U, expl_sw.P)
        _prof['w_precond_build'] = time.perf_counter() - _tp
        rhs_w = v3_new + c0 * (1.0 + w0)
        n_it_w = [0]
        def _cb_w(_): n_it_w[0] += 1
        _tp = time.perf_counter()
        x0w = precond_w.matvec(rhs_w)
        alpha_w, info_w = spla.cg(
            op_w.as_linop(), rhs_w, x0=x0w, M=precond_w.as_linear_operator(),
            rtol=args.pcg_rtol, maxiter=args.pcg_maxiter, callback=_cb_w,
        )
        _prof['w_pcg'] = time.perf_counter() - _tp
        y_full_w = op_w.predict_at(alpha_w)
        z[0:N]       = y_full_w[1         : 1 +     N]
        z[N:2*N]     = y_full_w[1 +     N : 1 + 2 * N]
        z[2*N:3*N]   = y_full_w[1 + 2 * N : 1 + 3 * N]
        res_w = float(np.linalg.norm(op_w.matvec(alpha_w) - rhs_w) / max(np.linalg.norm(rhs_w), 1e-300))

        # Diagnostic at every alt step.
        u_truth_at_X_dom = interp(X_dom)
        err_v0 = np.linalg.norm(z[3*N:4*N] - u_truth_at_X_dom) / max(np.linalg.norm(u_truth_at_X_dom), 1e-12)
        print(f'  alt {outer:2d}: u-pCG = {n_it_u[0]} (info {info_u}, res {res_u:.1e}),  '
              f'w-pCG = {n_it_w[0]} (info {info_w}, res {res_w:.1e}),  '
              f'‖v0 - u_truth‖ rel = {err_v0:.2%},  '
              f'wall = {time.perf_counter()-t_outer:.2f} s')
        print('    [prof] ' + ',  '.join(f'{k}={v:.2f}s' for k, v in _prof.items()))
    print(f'[stage 2] wall: {time.perf_counter()-t_stage2:.2f} s')

    # ----- predict on test grid -----
    print('\n[extend] predict (u, w) on an 80² test grid …')
    t0 = time.perf_counter()
    N_test = 80
    xs = np.linspace(0, 1, N_test)
    XX, YY = np.meshgrid(xs, xs)
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Nt = X_test.shape[0]

    from kolesky.measurements import stack_measurements
    test_meas = _lgd(X_test, np.zeros(Nt), np.zeros((Nt, 2)), np.ones(Nt))

    w0, w1, w2, v0, v1, v2, v3 = _z_to_state(z, rhs_f, bdy_g, N, Nb)
    if alpha_u is None or alpha_w is None:
        # Stage-1-only (no alternating refinement ran): extend the converged
        # joint-GN state z to the test grid directly from the *big* factors.
        # α = Θ⁻¹ v with Θ⁻¹ ≈ U Uᵀ (sf.apply), v = measurement values implied
        # by z on the big-factor measurement ordering.
        v3_vals = _v3_nonlinear(z, rhs_f, N)
        alpha_u_b = sf_u.apply(_pack_v_all(z, v3_vals, bdy_g, N, Nb))
        alpha_w_b = sf_w.apply(_pack_w_all(z, N))
        train_u = stack_measurements(_theta_u_groups(X_dom, X_bdy))
        train_w = stack_measurements(_theta_w_groups(X_dom))
        Theta_u_test = np.asarray(kernel_u(test_meas, train_u), dtype=np.float64)
        Theta_w_test = np.asarray(kernel_w(test_meas, train_w), dtype=np.float64)
        u_pred = Theta_u_test @ alpha_u_b
        w_pred = Theta_w_test @ alpha_w_b
    else:
        # Use the *last* alternating subproblem's small-factor representation:
        # k(δ_test, train_u) · α_u where train_u = constraint set last used.
        train_u = stack_measurements(_meas_constraint_u(X_bdy, X_dom, X_data, w1, w2))
        train_w = stack_measurements(_meas_constraint_w(X_dom, rhs_f * np.exp(-w0), -v1, -v2))
        Theta_u_test = np.asarray(kernel_u(test_meas, train_u), dtype=np.float64)
        Theta_w_test = np.asarray(kernel_w(test_meas, train_w), dtype=np.float64)
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
        f'Darcy inverse — joint warm-up + alternating refinement  '
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
