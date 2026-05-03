"""2-D Darcy inverse problem via alternating sparse Cholesky + Algorithm 4.1.

Sibling of ``examples/darcy_inverse.py``: the dense version pays O(N³)
for Cholesky and stores Theta_u of size (N_bdy + 4·N_dom)² ~ 1 GB at
N_dom = 2500. This file uses an *alternating* (Gauss-Seidel) scheme
where each subproblem is a single-kernel GP regression — making the
Algorithm 4.1 noisy ichol of Schäfer-Katzfuss-Owhadi (2020) directly
applicable:

    u-subproblem (w fixed at w_old):
        −Δu − ∇w_old · ∇u = f · exp(−w_old)        (interior)
        u = 0                                       (boundary)
        u(x_data_k) ≈ data_k        with σ² noise   (likelihood)

      → noisy GP regression, single u-field, partial-diagonal R on data
        rows. Apply Algorithm 4.1 verbatim: build sparse Cholesky of
        K_u_train, ichol on top to add R, solve_Sigma for outer pCG.

    w-subproblem (u fixed at u_new):
        −∇u_new · ∇w + f · exp(−w_old) · w
              = f · exp(−w_old) · (1 + w_old) + Δu_new          (interior)

      → *noiseless* GP regression, single w-field. Sparse Cholesky alone.

Iterate u-, then-w-, then-u-, … until both converge.

**Status / known issues.**

  * The Algorithm 4.1 ichol step is correctly applied to the u-subproblem
    — sparse big factor + ichol with R = σ² on data rows + ε on others.
    `solve_Sigma` drives outer pCG to dense-equivalent accuracy on the
    u-step linear system. (Verified end-to-end against
    ``tests/test_smoke.py::test_noisy_ichol_factorization``.)
  * The alternating outer iteration in this file is *unstable* at w=0
    initialization for the standard Darcy test (fluctuating signs, NaNs
    after a few iterates). Two failure modes contributing:

    1. The w-subproblem's linearized PDE
       ``-∇u·∇w + f·exp(-w_old)·w = ...``
       is *first-order* in w (no Δw term), so it isn't a standard
       elliptic operator and the maximin-based KL factor is not the
       natural support for its inverse. A standard fix is to add a
       small Δw regularization (or use a stronger w-prior).
    2. The Δu_new term in the w-subproblem rhs amplifies u-step
       errors, especially at trivial w=0 init where the PDE constraint
       is not a useful prior on w.

  * To stabilize: (i) better init (e.g. solve a single forward Darcy
    with assumed a≡1 to seed u, then refine w); (ii) damping in the
    outer iteration (w_new := w_old + α(w_solve − w_old) for α<1);
    (iii) regularize w-subproblem with a tiny Δw term.

For a *working* end-to-end inverse-problem demo, see
``examples/darcy_inverse.py`` (dense, N_dom ~ 200, ~26% rel err on a).
This sparse alternating file is the scaffold for scaling up; the
noisy-ichol piece works correctly, the outer alternation needs the
stabilization fixes above.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.linalg
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


def _lgd(coord, wL, wG, wD):
    from kolesky.measurements import LaplaceGradDiracPointMeasurement
    return LaplaceGradDiracPointMeasurement(
        coordinate=np.asarray(coord, dtype=np.float64),
        weight_laplace=np.asarray(wL, dtype=np.float64),
        weight_grad=np.asarray(wG, dtype=np.float64),
        weight_delta=np.asarray(wD, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Indexed-measurement / dense-lookup-kernel pair.
# Lets kolesky run its sparse Cholesky pipeline on a precomputed dense
# kernel matrix without us having to introduce a new measurement type
# at the kolesky core.
# ---------------------------------------------------------------------------


@dataclass
class _IndexedMeas:
    coordinate: np.ndarray
    indices: np.ndarray

    @property
    def d(self):
        return self.coordinate.shape[-1]

    def is_batched(self):
        return self.coordinate.ndim == 2


def _patch_kolesky_select_for_indexed_meas():
    """Idempotently monkey-patch ``kolesky.measurements.select`` so it
    knows how to subset our ``_IndexedMeas`` (subset both the
    coordinates *and* the integer indices into the precomputed K matrix)."""
    import kolesky.measurements as _km
    if getattr(_km.select, '_patched_for_indexed', False):
        return
    _orig = _km.select
    def _patched(meas, idx):
        if isinstance(meas, _IndexedMeas):
            idx = np.asarray(idx, dtype=np.int64)
            return _IndexedMeas(coordinate=meas.coordinate[idx],
                                indices=meas.indices[idx])
        return _orig(meas, idx)
    _patched._patched_for_indexed = True
    _km.select = _patched


class _DenseLookupKernel:
    """Kernel callable that slices a precomputed dense kernel matrix.
    Plays the role of an ``AbstractCovarianceFunction`` for kolesky's
    sparse-Cholesky pipeline."""
    def __init__(self, K: np.ndarray):
        self.K = K

    def __call__(self, meas_a, meas_b=None):
        idx_a = meas_a.indices
        idx_b = meas_b.indices if meas_b is not None else idx_a
        return self.K[np.ix_(idx_a, idx_b)]


# ---------------------------------------------------------------------------
# Sparse Cholesky + Algorithm 4.1 ichol on an arbitrary dense kernel matrix.
# ---------------------------------------------------------------------------


def _sparse_factor_with_noise(K_dense, train_coord, R_diag,
                              rho=3.0, k_neighbors=3, nugget=1e-10,
                              verbose=True, tag=''):
    """Given a dense kernel matrix ``K_dense`` (size N_train × N_train)
    over points ``train_coord``, plus a diagonal-noise vector ``R_diag``,
    build a ``NoisyExplicitKLFactorization`` whose
    ``solve_Sigma(b) ≈ (K + R)⁻¹ b`` to dense-equivalent accuracy."""
    import kolesky as kl
    _patch_kolesky_select_for_indexed_meas()
    Ntot = K_dense.shape[0]
    if nugget > 0:
        diag_mean = float(np.diag(K_dense).mean())
        K_dense = K_dense.copy()
        K_dense[np.arange(Ntot), np.arange(Ntot)] += nugget * diag_mean
    train_meas = _IndexedMeas(coordinate=np.asarray(train_coord, dtype=np.float64),
                              indices=np.arange(Ntot, dtype=np.int64))
    kernel_lookup = _DenseLookupKernel(K_dense)

    t0 = time.perf_counter()
    impl = kl.ImplicitKLFactorization.build(
        kernel_lookup, train_meas, rho=rho, k_neighbors=k_neighbors,
    )
    expl = kl.ExplicitKLFactorization(impl, nugget=0.0, backend='cpu')
    t_chol = time.perf_counter() - t0
    if verbose:
        print(f'    [{tag}] sparse Chol: {expl.U.shape}, nnz = {expl.U.nnz:,}, '
              f'{t_chol:.2f} s')

    if R_diag is None:
        return expl, None     # no noise needed

    t0 = time.perf_counter()
    noisy = kl.NoisyExplicitKLFactorization.build(expl, R=R_diag)
    t_ichol = time.perf_counter() - t0
    if verbose:
        print(f'    [{tag}] ichol (Alg 4.1): nnz = {noisy.U_tilde.nnz:,}, '
              f'{t_ichol:.2f} s')
    return expl, noisy


# ---------------------------------------------------------------------------
# u-subproblem: solve for u given w_old (and ∇w_old).
# ---------------------------------------------------------------------------


def _solve_u_subproblem(w_old, gw_old, X_dom, X_bdy, Nd, rhs_f, data_noisy,
                        sigma2, kernel, rho, k_neighbors, nugget,
                        eps_off_data, pcg_rtol, pcg_maxiter, verbose,
                        return_lap_u=False):
    """Linear elliptic in u:
        L_u(u) := -Δu - ∇w_old · ∇u = f · exp(-w_old)        on interior
        u = 0                                                  on ∂Ω
        u(x_data_k) ≈ data_k                  (noisy, σ² var)

    Train measurements are 3 groups: [bdy_δ, data_δ, PDE_u_int]. We
    factor the dense ``K_u_train`` via kolesky + ichol (Algorithm 4.1)
    with R = σ² · E_data + ε · I (tiny ε on non-data rows so R⁻¹ is
    well-defined; effect on the answer is negligible)."""
    Nb = X_bdy.shape[0]; N = X_dom.shape[0]
    Ntot = Nb + Nd + N

    # Train measurements (u-side only).
    c_old = rhs_f * np.exp(-w_old)
    train_u = [
        _lgd(X_bdy,        np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
        _lgd(X_dom[:Nd],   np.zeros(Nd), np.zeros((Nd, 2)), np.ones(Nd)),
        _lgd(X_dom,        -np.ones(N),  -gw_old,            np.zeros(N)),
    ]
    import kolesky as kl
    train_u_stack = kl.stack_measurements(train_u)

    K_u = np.asarray(kernel(train_u_stack, train_u_stack), dtype=np.float64)
    K_u = 0.5 * (K_u + K_u.T)

    # Train coordinates for the sparse-Cholesky ordering (matches K_u rows).
    train_coord = np.vstack([X_bdy, X_dom[:Nd], X_dom])

    # Noise: σ² on data rows, tiny ε on others.
    R_diag = np.full(Ntot, eps_off_data, dtype=np.float64)
    R_diag[Nb:Nb + Nd] = sigma2

    expl_u, noisy_u = _sparse_factor_with_noise(
        K_u, train_coord, R_diag,
        rho=rho, k_neighbors=k_neighbors, nugget=nugget,
        verbose=verbose, tag='u-subproblem',
    )

    # RHS for u-subproblem.
    rhs_pde_u = c_old.copy()                 # f · exp(-w_old)
    y_train = np.concatenate([np.zeros(Nb), data_noisy, rhs_pde_u])

    # Solve (K_u + R) α = y via paper §4.1 outer CG.
    t0 = time.perf_counter()
    alpha_u = noisy_u.solve_Sigma(y_train, rtol=pcg_rtol, maxiter=pcg_maxiter)
    if verbose:
        print(f'    [u-subproblem] solve_Sigma: {time.perf_counter()-t0:.2f} s')

    # Predict u, ∇₁u, ∇₂u at interior.
    test_u_d  = _lgd(X_dom, np.zeros(N), np.zeros((N, 2)), np.ones(N))
    test_u_d1 = _lgd(X_dom, np.zeros(N), np.tile([1.0, 0.0], (N, 1)), np.zeros(N))
    test_u_d2 = _lgd(X_dom, np.zeros(N), np.tile([0.0, 1.0], (N, 1)), np.zeros(N))
    K_test_d  = np.asarray(kernel(test_u_d,  train_u_stack), dtype=np.float64)
    K_test_d1 = np.asarray(kernel(test_u_d1, train_u_stack), dtype=np.float64)
    K_test_d2 = np.asarray(kernel(test_u_d2, train_u_stack), dtype=np.float64)
    u_new   = K_test_d  @ alpha_u
    gu_new  = np.stack([K_test_d1 @ alpha_u, K_test_d2 @ alpha_u], axis=1)

    if return_lap_u:
        # Δu at interior points — needed by the w-subproblem rhs because the
        # u-subproblem's PDE constraint is *softly* satisfied (data fidelity
        # can override it at data points), so we cannot eliminate Δu_new via
        # the PDE identity.
        test_u_lap = _lgd(X_dom, np.ones(N), np.zeros((N, 2)), np.zeros(N))
        K_test_lap = np.asarray(kernel(test_u_lap, train_u_stack), dtype=np.float64)
        lap_u_new = K_test_lap @ alpha_u
        return u_new, gu_new, lap_u_new, alpha_u, train_u_stack
    return u_new, gu_new, alpha_u, train_u_stack


# ---------------------------------------------------------------------------
# w-subproblem: solve for w given u_new (no noise).
# ---------------------------------------------------------------------------


def _solve_w_subproblem(u_new, gu_new, lap_u_new, w_old, gw_old, X_dom, rhs_f, kernel,
                        rho, k_neighbors, nugget, eps_off_data,
                        pcg_rtol, pcg_maxiter, verbose):
    """Linear in w (linearizing exp(-w) around w_old):
        L_w(w) := -∇u_new · ∇w + f · c_old · w
               = f · c_old · (1 + w_old) + Δu_new       on interior

    Train: PDE_w at all interior points only. No boundary, no data, no
    noise — pure GP regression with the prior on w. Sparse Cholesky
    suffices (no ichol needed)."""
    N = X_dom.shape[0]
    c_old = rhs_f * np.exp(-w_old)
    train_w = _lgd(X_dom, np.zeros(N), -gu_new, c_old)

    import kolesky as kl
    K_w = np.asarray(kernel(train_w, train_w), dtype=np.float64)
    K_w = 0.5 * (K_w + K_w.T)

    expl_w, _ = _sparse_factor_with_noise(
        K_w, X_dom, R_diag=None,
        rho=rho, k_neighbors=k_neighbors, nugget=nugget,
        verbose=verbose, tag='w-subproblem',
    )

    # rhs at PDE rows.  The GN linearization of exp(-w) at w_old gives
    # the linear-in-w PDE  -∇u_new·∇w + f·c_old·w = f·c_old·(1+w_old) + Δu_new,
    # where Δu_new is what u_new actually has (NOT the PDE-identity value;
    # data fidelity can pull u_new off the PDE at observation points, so
    # we evaluate Δu via the u-subproblem's predictor).
    y_train = c_old * (1.0 + w_old) + lap_u_new

    # Apply Theta_w⁻¹ via the sparse forward apply:  α = Pᵀ (UᵀU)⁻¹ P · ... no,
    # we want (K_w)⁻¹ y; but we only have UᵀU ≈ K_w⁻¹ (matvec form, less
    # accurate). Use outer pCG on K_w x = y with the noisy ichol's symmetric
    # SMW-form preconditioner — but we have no R, so just dense Cholesky here
    # if N is moderate, else sparse + outer pCG with U-fwd preconditioner.
    # Simplest: Cholesky via U-fwd-apply matvec + Jacobi precond CG.
    U_csr  = expl_w.U.tocsr()
    UT_csr = expl_w.U.T.tocsr()
    P_w = expl_w.P
    def K_w_apply(v):
        vp = v[P_w]
        y = spla.spsolve_triangular(U_csr, vp, lower=False)
        z = spla.spsolve_triangular(UT_csr, y, lower=True)
        out = np.empty_like(v); out[P_w] = z
        return out
    def K_w_inv_apply(v):     # cheap one-shot preconditioner
        vp = v[P_w]
        z = expl_w.U.T @ (expl_w.U @ vp)
        out = np.empty_like(v); out[P_w] = z
        return out
    A_op = spla.LinearOperator((N, N), matvec=K_w_apply, dtype=np.float64)
    M_op = spla.LinearOperator((N, N), matvec=K_w_inv_apply, dtype=np.float64)
    t0 = time.perf_counter()
    x0 = K_w_inv_apply(y_train)
    alpha_w, _info = spla.cg(A_op, y_train, x0=x0, M=M_op,
                              rtol=pcg_rtol, maxiter=pcg_maxiter)
    if verbose:
        print(f'    [w-subproblem] solve K_w α = y: {time.perf_counter()-t0:.2f} s')

    # Predict w, ∇₁w, ∇₂w at interior.
    test_w_d  = _lgd(X_dom, np.zeros(N), np.zeros((N, 2)), np.ones(N))
    test_w_d1 = _lgd(X_dom, np.zeros(N), np.tile([1.0, 0.0], (N, 1)), np.zeros(N))
    test_w_d2 = _lgd(X_dom, np.zeros(N), np.tile([0.0, 1.0], (N, 1)), np.zeros(N))
    K_test_d  = np.asarray(kernel(test_w_d,  train_w), dtype=np.float64)
    K_test_d1 = np.asarray(kernel(test_w_d1, train_w), dtype=np.float64)
    K_test_d2 = np.asarray(kernel(test_w_d2, train_w), dtype=np.float64)
    w_new  = K_test_d  @ alpha_w
    gw_new = np.stack([K_test_d1 @ alpha_w, K_test_d2 @ alpha_w], axis=1)

    return w_new, gw_new, alpha_w, train_w


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
    p.add_argument('--eps-off-data', type=float, default=1e-10,
                   help='tiny noise ε on non-data train rows so the ichol R⁻¹ is finite')
    p.add_argument('--outer-iters', type=int, default=8,
                   help='number of alternating u/w sweeps')
    p.add_argument('--pcg-rtol', type=float, default=1e-6)
    p.add_argument('--pcg-maxiter', type=int, default=80)
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
    print(f'[setup]  kernel = {args.kernel}, σ_kernel = {args.kernel_sigma}, '
          f'σ_noise = {args.noise}, ρ = {args.rho}')

    # ----- ground truth via FD -----
    c_a = 1.0
    def a_truth(x1, x2):
        return (np.exp(c_a * np.sin(2 * np.pi * x1) + c_a * np.sin(2 * np.pi * x2))
                + np.exp(-c_a * np.sin(2 * np.pi * x1) - c_a * np.sin(2 * np.pi * x2)))
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
    print(f'[sample] N_dom = {N}, N_bdy = {Nb}, N_data = {Nd}')

    # ----- noisy observations of u_truth at X_data -----
    from scipy.interpolate import RegularGridInterpolator
    grid_xs = np.linspace(0, 1, args.N_fd)
    interp = RegularGridInterpolator((grid_xs, grid_xs), u_truth.T)
    u_at_data = interp(X_data)
    data_noisy = u_at_data + args.noise * rng.standard_normal(Nd)
    sigma2 = args.noise ** 2

    rhs_f = np.ones(N, dtype=np.float64)

    # ----- alternating GS iteration -----
    u_old  = np.zeros(N); gu_old = np.zeros((N, 2))
    w_old  = np.zeros(N); gw_old = np.zeros((N, 2))
    last_alpha_u = None; last_train_u = None
    last_alpha_w = None; last_train_w = None

    print(f'\n[GS]  {args.outer_iters} alternating u/w sweeps')
    t_loop = time.perf_counter()
    for step in range(1, args.outer_iters + 1):
        print(f'  step {step}:')
        # u-subproblem (noisy, Algorithm 4.1)
        u_new, gu_new, lap_u_new, alpha_u, train_u = _solve_u_subproblem(
            w_old, gw_old, X_dom, X_bdy, Nd, rhs_f, data_noisy, sigma2,
            kernel, args.rho, args.k_neighbors, args.nugget,
            args.eps_off_data, args.pcg_rtol, args.pcg_maxiter, verbose=True,
            return_lap_u=True,
        )
        last_alpha_u = alpha_u; last_train_u = train_u

        # w-subproblem (noiseless)
        w_new, gw_new, alpha_w, train_w = _solve_w_subproblem(
            u_new, gu_new, lap_u_new, w_old, gw_old, X_dom, rhs_f, kernel,
            args.rho, args.k_neighbors, args.nugget,
            args.eps_off_data, args.pcg_rtol, args.pcg_maxiter, verbose=True,
        )
        last_alpha_w = alpha_w; last_train_w = train_w

        delta_u = np.linalg.norm(u_new - u_old) / (np.linalg.norm(u_new) + 1e-12)
        delta_w = np.linalg.norm(w_new - w_old) / (np.linalg.norm(w_new) + 1e-12)
        print(f'    Δu_rel = {delta_u:.2e},  Δw_rel = {delta_w:.2e}')
        print(f'    u range = [{u_new.min():+.3e}, {u_new.max():+.3e}],  '
              f'w range = [{w_new.min():+.3e}, {w_new.max():+.3e}]')
        u_old, gu_old, w_old, gw_old = u_new, gu_new, w_new, gw_new
    print(f'[GS]  wall: {time.perf_counter()-t_loop:.2f} s')

    # ----- predict on test grid -----
    print('\n[extend] predict (u, w) on an 80² test grid …')
    t0 = time.perf_counter()
    N_test = 80
    xs = np.linspace(0, 1, N_test)
    XX, YY = np.meshgrid(xs, xs)
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Nt = X_test.shape[0]
    test_meas = _lgd(X_test, np.zeros(Nt), np.zeros((Nt, 2)), np.ones(Nt))
    K_u_test = np.asarray(kernel(test_meas, last_train_u), dtype=np.float64)
    K_w_test = np.asarray(kernel(test_meas, last_train_w), dtype=np.float64)
    u_pred = K_u_test @ last_alpha_u
    w_pred = K_w_test @ last_alpha_w
    a_pred = np.exp(w_pred)
    print(f'[extend] wall: {time.perf_counter()-t0:.2f} s')

    # ----- accuracy -----
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
        f'Darcy inverse — alternating u/w with Algorithm 4.1 noisy ichol on u-step  '
        f'(N_dom={N}, N_data={Nd}, σ={args.noise}, ρ={args.rho})',
        fontsize=11,
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[fig]    {out_path}')


if __name__ == '__main__':
    main()
