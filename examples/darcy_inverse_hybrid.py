"""Near-linear sparse-Cholesky solver for the Darcy inverse problem.

Recover both the diffusion coefficient ``a = exp(w)`` and the state ``u``
of ``-div(exp(w) grad u) = f`` from a forcing, boundary data, and noisy
interior measurements of ``u``. Total cost grows essentially linearly in
the number of collocation points ``N`` — the same complexity class as a
single forward elliptic solve, up to a constant for the second field.
Companion note: ``docs/darcy_inverse_nearlinear_note.tex``.

Two independent mean-zero GP priors are placed on the fields, ``u ~ GP(0,
K_u)`` and ``w ~ GP(0, K_w)``, both Matérn-7/2 but with separate length
scales (long ``ell_u`` for the smooth state, short ``ell_w`` for the rough
coefficient). The two kernels are NEVER summed: each Gram matrix stays a
single-kernel covariance that the screening (KL) sparse-Cholesky theory
can factor in near-linear time. The MAP loss is the sum of the two prior
RKHS norms plus the ``(1/σ²)``-weighted data misfit.

Stage 1 — warm-up (basin finder), ``--warmup-mode``:
  * 'coarse' (default): run a few joint Gauss-Newton steps on a FIXED
    coarse subset (cost O(1) in fine N), then krige the recovered
    w-field to all N points via the Θ_w cross-covariance (O(N)). This
    sidesteps ever assembling/factoring the dense, biharmonic-like joint
    GN Hessian on the fine grid.
  * 'joint-gn': joint-GN on all N points (~N² per step; for comparison).
  * 'data-first': seed w by regressing u from data alone (stalls; kept
    for reference).

Stage 2 — alternating refinement, ``--alt-steps`` outer iterations. With
one field frozen the loss is a linear GP regression in the other:
  * u | w step (the hard half): solve the posterior mean of u under
    ``[δ_bdy = g, L_w u = -f exp(-w̄₀) on the interior, u(x_d) ≈ y]``,
    where ``L_w u = Δu + w̄₁ ∂₁u + w̄₂ ∂₂u``. The noisy data δ are
    co-located with the PDE points and σ is tiny, so a screened factor of
    the full augmented operator cannot resolve the near-null data modes.
    The 'data-bordered' u-step instead does an EXACT bordered/Schur solve:
    the forward block F={bdy, PDE} is applied by the screened forward pCG
    (N-independent iters), and the Nd data rows are peeled into a small
    dense Nd×Nd system whose posterior covariance S0⁻¹ = (Θ_joint⁻¹)_dd is
    read off the joint precision factor as a Gram of Nd sparse rows of U —
    no per-data forward solves.
  * w | u step (easy): a single-GP regression on Θ_w, ~O(10) pCG iters.

After Stage 2: ``u(x*) = k(δ_x*, train_u) · α_u`` and
``a(x*) = exp(k(δ_x*, train_w) · α_w)``.

Running with no flags uses the note's configuration (coarse warm-up,
data-bordered u-step, ell_u=1.0 / ell_w=0.2, rho=3, 6 warm-up + 3
alternating steps). Reproduce the scaling tables by sweeping the size::

    python examples/darcy_inverse_hybrid.py --N-domain 800 --N-data 300
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
# Operator: Θ_train + R_data with R_data = σ² on the data block, 0
# elsewhere (the true two-valued nugget — exact bdy/PDE rows, noisy data
# rows). The u-step solves this by the exact bordered/Schur split rather
# than preconditioning through the near-null data modes.


class ThetaTrainOpU:
    """Lifted Θ_train operator for the u | w subproblem.

    α layout: [α_bdy (Nb), α_pde (N), α_data (Nd)] — data last.

    The matvec adds the noise term to the operator according to the
    selected noise_mode:
      * 'data' : σ² · α_data on the data block only (the true two-valued
        operator: σ² on data rows, exact elsewhere).
      * 'none' : no noise term — pure K matvec. Used by the bordered-solve
        forward-block operators (the σ²-data block is handled separately
        by the dense Schur complement).
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
    p.add_argument('--kernel-sigma', type=float, default=None,
                   help="shared lengthscale override applied to BOTH fields when "
                        "the per-field flags below are unset. Leaving everything "
                        "unset uses the recommended split (ℓ_u=1.0, ℓ_w=0.2).")
    p.add_argument('--kernel-sigma-u', type=float, default=None,
                   help="lengthscale for the Θ_u (PDE/u-field) prior (default 1.0). "
                        "Recovering the coefficient's spatial structure needs a "
                        "LONG ℓ_u (≈1.0): u solves an elliptic PDE so it is smooth, "
                        "and a short ℓ_u injects spurious short-scale wiggles whose "
                        "Laplacian (which pins a through the PDE) is garbage. See "
                        "--kernel-sigma-w.")
    p.add_argument('--kernel-sigma-w', type=float, default=None,
                   help="lengthscale for the Θ_w (coefficient) prior (default 0.2). "
                        "Keep SHORT (≈0.2) so a can have sharp peaks and the "
                        "w-factor stays local/sparse.")
    p.add_argument('--rho', type=float, default=3.0)
    p.add_argument('--rho-small', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=4)
    p.add_argument('--nugget', type=float, default=1e-8)
    p.add_argument('--warmup-steps', type=int, default=6)
    p.add_argument('--warmup-mode', default='coarse',
                   choices=['coarse', 'joint-gn', 'data-first'],
                   help="Stage-1 basin-finder. 'coarse' (default): run joint-GN on "
                        "a fixed coarse subset (--warmup-coarse-N pts) then krige the "
                        "recovered w-field to all N via the Θ_w cross-cov — cost O(1) "
                        "in fine N. 'joint-gn': joint-GN on all N points "
                        "(~N^2 per step; not near-linear). 'data-first': GP-regress u "
                        "from data+bdy alone (no PDE), then one w|u solve to seed w.")
    p.add_argument('--warmup-coarse-N', type=int, default=400,
                   help="interior point budget for --warmup-mode coarse.")
    p.add_argument('--warmup-coarse-Nd', type=int, default=150,
                   help="data point budget for --warmup-mode coarse.")
    p.add_argument('--alt-steps', type=int, default=3)
    p.add_argument('--pcg-rtol', type=float, default=1e-6)
    p.add_argument('--pcg-maxiter', type=int, default=200)
    p.add_argument('--noise-mode', default='data-bordered',
                   choices=['data-bordered'],
                   help="u-step solver. 'data-bordered' (default): exact "
                        "bordered/Schur solve — the forward block {bdy, PDE} is "
                        "applied by the screened forward pCG, the Nd noisy data rows "
                        "are peeled into a small dense Nd×Nd block whose posterior "
                        "covariance is read off the joint precision factor "
                        "(near-linear).")
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
    shared = args.kernel_sigma
    ell_u = args.kernel_sigma_u if args.kernel_sigma_u is not None else (shared if shared is not None else 1.0)
    ell_w = args.kernel_sigma_w if args.kernel_sigma_w is not None else (shared if shared is not None else 0.2)
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
    op_u = ThetaTrainOpU(expl_u, sf_u, Nb, N, Nd, sigma, noise_mode='data')
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

        # ---- u | w solve: near-linear DIRECT bordered/Schur step, no outer CG.
        from kolesky.pde.pcg_ops import SmallPrecond
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

        # rhs in α-layout [bdy, pde, data]:
        rhs_u = np.empty(Nb + N + Nd)
        rhs_u[:Nb]               = bdy_g
        rhs_u[Nb : Nb + N]       = rhs_f * np.exp(-w0)
        rhs_u[Nb + N :]          = data_noisy
        n_it_u = [0]
        _tp = time.perf_counter()
        alpha_u, info_u, n_it_u[0] = bordered_solve(rhs_u)
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
