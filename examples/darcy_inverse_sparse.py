"""2-D Darcy inverse problem at scale via sparse Cholesky + GN + pCG.

Sibling of ``examples/darcy_inverse.py``, which uses the dense
formulation with separate `Theta_u` (size  Nb + 4·N_dom) and
`Theta_w` (size  3·N_dom) and dense Cholesky on each. This file
follows the ``kolesky.pde`` pattern instead:

  * One *combined* GN-linearized PDE measurement per interior point.
  * Joint kernel  K = K_u + K_w  thanks to independent priors on u and w.
  * Big sparse factors for `Theta_u` and `Theta_w` (built once outside
    the GN loop, used for predictions on the test grid in O(N · ρᵈ)).
  * Train system at each GN step is built over only `Nb + Nd + N` joint
    measurements — much smaller than the dense version's `Nb + 4N` —
    and the data-noise term enters as a partial diagonal on the data
    rows of `Theta_train` (Algorithm 4.1's "additive R" structure).

Forward Darcy:  −∇·(a∇u) = f   on Ω,   u = 0 on ∂Ω,  with `a = exp(w)`.
Multiplied through by `e⁻ʷ`,   F̃ = −Δu − ∇w·∇u − f e⁻ʷ = 0.
Linearizing F̃ at (u_old, w_old) gives, at every interior point x_j,
the *single* linear functional of (δu, δw)

   L_j(δu, δw) = −Δ(δu) − ∇w_old·∇(δu) − ∇u_old·∇(δw) + f e⁻ʷ⁰ˡᵈ · δw
              = RHS_j  =  f·e⁻ʷ⁰ˡᵈ · (1 + w_old)  −  ∇u_old · ∇w_old.

So the GN regression has three measurement groups:

   bdy_i  : u(x_bdy_i)        (δ on u, target 0)
   data_k : u(x_data_k)       (δ on u, noisy target = data_k, var σ²)
   PDE_j  : L_j(δu, δw)       (joint u+w functional, target RHS_j)

and the train kernel splits as `K_u + K_w` because the priors on u and
w are independent. We assemble Theta_train *densely* (size
Nb + Nd + N — small even at large N) and add `σ² Eᵀ E` on the data
rows. Per GN step is one Cholesky + one back-solve. The big factors
are used only for the post-GN predictions on the visualization grid.

Run:
    python examples/darcy_inverse_sparse.py --N-domain 2500 --rho 4.0
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
    sol = scipy.sparse.linalg.spsolve(A, fv).reshape(N, N)
    out = np.zeros((N + 2, N + 2))
    out[1:N + 1, 1:N + 1] = sol
    return out


# ---------------------------------------------------------------------------
# Helpers — measurement constructors and big-factor wrapper.
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
    """5-set list  [δ_bdy, δ_int, ∂₁_int, ∂₂_int, Δ_int]  for the u big factor."""
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
    """3-set list (with single dummy-bdy point) for the w big factor."""
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


class _BigFactor:
    """Forward apply  Θ v ≈ Pᵀ (UᵀU)⁻¹ P v  via two triangular solves."""
    __slots__ = ('U_csr', 'UT_csr', 'P', 'shape')

    def __init__(self, explicit):
        self.U_csr = explicit.U.tocsr()
        self.UT_csr = explicit.U.T.tocsr()
        self.P = np.asarray(explicit.P, dtype=np.int64)
        self.shape = explicit.U.shape

    def apply(self, v: np.ndarray) -> np.ndarray:
        vp = v[self.P]
        y = spla.spsolve_triangular(self.U_csr,  vp, lower=False)
        z = spla.spsolve_triangular(self.UT_csr, y,  lower=True)
        out = np.empty_like(v); out[self.P] = z
        return out


# ---------------------------------------------------------------------------
# One GN step.
#
# At each step we
#   (a) build the joint train measurements (current PDE coefs from old iterate),
#   (b) assemble Theta_train = K_u + K_w  (dense, size Nb + Nd + N) + R diag,
#   (c) Cholesky-solve to get α,
#   (d) predict (u, ∇u, w, ∇w) at the interior points for the next GN step.
# ---------------------------------------------------------------------------


def _gn_step(
    u_old: np.ndarray, gu_old: np.ndarray,
    w_old: np.ndarray, gw_old: np.ndarray,
    kernel,
    X_dom: np.ndarray, X_bdy: np.ndarray, Nd: int,
    rhs_f: np.ndarray, data_noisy: np.ndarray, sigma2: float,
    nugget: float, verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    Nb = X_bdy.shape[0]; N = X_dom.shape[0]
    Ntot = Nb + Nd + N

    # ---- (a) linearization coefs at (u_old, w_old) ----
    c_old = rhs_f * np.exp(-w_old)              # c_old(x_j) = f · exp(-w_old)
    rhs_pde = c_old * (1.0 + w_old) - (gu_old * gw_old).sum(axis=1)
    y_train = np.concatenate([np.zeros(Nb), data_noisy, rhs_pde])

    # ---- (b) train measurements (joint u-side + w-side) ----
    # u-side at every train row, as a single LaplaceGradDirac
    # (bdy: pure δ; data: pure δ; PDE: (-Δ - ∇w_old·∇)).
    from kolesky.measurements import (
        LaplaceGradDiracPointMeasurement, stack_measurements,
    )
    train_u = stack_measurements([
        _lgd(X_bdy,        np.zeros(Nb), np.zeros((Nb, 2)), np.ones(Nb)),
        _lgd(X_dom[:Nd],   np.zeros(Nd), np.zeros((Nd, 2)), np.ones(Nd)),
        _lgd(X_dom,        -np.ones(N),  -gw_old,            np.zeros(N)),
    ])
    # w-side: only PDE rows have nonzero w-side.
    train_w_pde = _lgd(X_dom, np.zeros(N), -gu_old, c_old)

    # Theta_train = K_u(train_u, train_u) + K_w(train_w, train_w),
    # with K_w nonzero only on the PDE×PDE block.
    t_assemble = time.perf_counter()
    K_u_train = np.asarray(kernel(train_u, train_u), dtype=np.float64)
    K_u_train = np.array(K_u_train, copy=True)        # ensure writeable
    K_u_train = 0.5 * (K_u_train + K_u_train.T)
    K_w_pde_pde = np.asarray(kernel(train_w_pde, train_w_pde), dtype=np.float64)
    K_w_pde_pde = 0.5 * (K_w_pde_pde + K_w_pde_pde.T)
    Theta_train = K_u_train.copy()
    Theta_train[Nb + Nd:, Nb + Nd:] += K_w_pde_pde
    # data-noise on the data rows
    idx_data = np.arange(Nb, Nb + Nd)
    Theta_train[idx_data, idx_data] += sigma2
    # diagonal nugget for stability
    if nugget > 0:
        Theta_train[np.arange(Ntot), np.arange(Ntot)] += nugget * np.diag(Theta_train).mean()
    t_assemble = time.perf_counter() - t_assemble

    # ---- (c) Cholesky solve ----
    t_solve = time.perf_counter()
    L = np.linalg.cholesky(Theta_train)
    alpha = scipy.linalg.cho_solve((L, True), y_train)
    t_solve = time.perf_counter() - t_solve

    if verbose:
        print(f'    assemble = {t_assemble:.2f} s,  cholesky+solve = {t_solve:.2f} s')

    # ---- (d) predict (u, ∇u, w, ∇w) at the interior points ----
    # At each interior point x_j we want u(x_j), ∂₁u(x_j), ∂₂u(x_j), and
    # similarly for w. Each is one row of K(test_meas, train_u/w) · α.
    t_pred = time.perf_counter()
    test_u_d   = _lgd(X_dom, np.zeros(N), np.zeros((N, 2)), np.ones(N))
    test_u_d1  = _lgd(X_dom, np.zeros(N), np.tile([1.0, 0.0], (N, 1)), np.zeros(N))
    test_u_d2  = _lgd(X_dom, np.zeros(N), np.tile([0.0, 1.0], (N, 1)), np.zeros(N))
    K_test_u_d  = np.asarray(kernel(test_u_d,  train_u), dtype=np.float64)
    K_test_u_d1 = np.asarray(kernel(test_u_d1, train_u), dtype=np.float64)
    K_test_u_d2 = np.asarray(kernel(test_u_d2, train_u), dtype=np.float64)
    u_new   = K_test_u_d  @ alpha
    gu_new  = np.stack([K_test_u_d1 @ alpha, K_test_u_d2 @ alpha], axis=1)

    K_test_w_d  = np.asarray(kernel(test_u_d,  train_w_pde), dtype=np.float64)
    K_test_w_d1 = np.asarray(kernel(test_u_d1, train_w_pde), dtype=np.float64)
    K_test_w_d2 = np.asarray(kernel(test_u_d2, train_w_pde), dtype=np.float64)
    alpha_pde = alpha[Nb + Nd:]
    w_new   = K_test_w_d  @ alpha_pde
    gw_new  = np.stack([K_test_w_d1 @ alpha_pde, K_test_w_d2 @ alpha_pde], axis=1)
    t_pred = time.perf_counter() - t_pred
    if verbose:
        print(f'    predict (u,∇u,w,∇w) at interior = {t_pred:.2f} s')

    info = dict(alpha=alpha, gw_old=gw_old, gu_old=gu_old, c_old=c_old, train_u=train_u, train_w_pde=train_w_pde)
    return u_new, gu_new, w_new, gw_new, info


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
    p.add_argument('--rho', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--nugget', type=float, default=1e-8)
    p.add_argument('--GN-steps', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='docs/darcy_inverse_sparse.png')
    p.add_argument('--build-big-factors', action='store_true', default=True,
                   help="build sparse big factors for U_u, U_w (used for cheap "
                        "test-grid prediction at the end)")
    p.add_argument('--no-big-factors', dest='build_big_factors', action='store_false')
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
    print(f'[sample] N_dom = {N}, N_bdy = {Nb}, N_data = {Nd}, '
          f'N_train = {Nb + Nd + N}')

    # ----- noisy observations of u_truth at X_data -----
    from scipy.interpolate import RegularGridInterpolator
    grid_xs = np.linspace(0, 1, args.N_fd)
    interp = RegularGridInterpolator((grid_xs, grid_xs), u_truth.T)
    u_at_data = interp(X_data)
    data_noisy = u_at_data + args.noise * rng.standard_normal(Nd)
    sigma2 = args.noise ** 2

    # ----- big factors (built once; used for test-grid prediction) -----
    big_u = big_w = None
    if args.build_big_factors:
        print('[big]    building U_u (5-set DiracsFirstThenUnifScale) …')
        t0 = time.perf_counter()
        impl_u = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
            kernel, _theta_u_groups(X_dom, X_bdy),
            rho=args.rho, k_neighbors=args.k_neighbors,
        )
        expl_u = kl.ExplicitKLFactorization(impl_u, nugget=args.nugget, backend='cpu')
        print(f'[big]    U_u: shape = {expl_u.U.shape}, nnz = {expl_u.U.nnz:,}, '
              f'wall = {time.perf_counter()-t0:.2f} s')
        big_u = _BigFactor(expl_u)

        print('[big]    building U_w (3-set + dummy bdy) …')
        t0 = time.perf_counter()
        impl_w = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
            kernel, _theta_w_groups(X_dom),
            rho=args.rho, k_neighbors=args.k_neighbors,
        )
        expl_w = kl.ExplicitKLFactorization(impl_w, nugget=args.nugget, backend='cpu')
        print(f'[big]    U_w: shape = {expl_w.U.shape}, nnz = {expl_w.U.nnz:,}, '
              f'wall = {time.perf_counter()-t0:.2f} s')
        big_w = _BigFactor(expl_w)

    # ----- GN iteration -----
    rhs_f = np.ones(N, dtype=np.float64)
    u_old = np.zeros(N); gu_old = np.zeros((N, 2))
    w_old = np.zeros(N); gw_old = np.zeros((N, 2))
    last_info = None

    print(f'\n[GN]     {args.GN_steps} steps')
    t_loop = time.perf_counter()
    for step in range(1, args.GN_steps + 1):
        print(f'  step {step}:')
        u_new, gu_new, w_new, gw_new, last_info = _gn_step(
            u_old, gu_old, w_old, gw_old,
            kernel, X_dom, X_bdy, Nd,
            rhs_f, data_noisy, sigma2,
            args.nugget, verbose=True,
        )
        delta = np.linalg.norm(u_new - u_old) / (np.linalg.norm(u_new) + 1e-12)
        print(f'    Δu_rel = {delta:.2e},  '
              f'u range = [{u_new.min():+.3e}, {u_new.max():+.3e}],  '
              f'w range = [{w_new.min():+.3e}, {w_new.max():+.3e}]')
        u_old, gu_old, w_old, gw_old = u_new, gu_new, w_new, gw_new
    print(f'[GN]     wall: {time.perf_counter()-t_loop:.2f} s')

    # ----- predict on test grid -----
    print('\n[extend] predict (u, w) on an 80² test grid …')
    t0 = time.perf_counter()
    N_test = 80
    xs = np.linspace(0, 1, N_test)
    XX, YY = np.meshgrid(xs, xs)
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)
    Nt = X_test.shape[0]

    test_meas = _lgd(X_test, np.zeros(Nt), np.zeros((Nt, 2)), np.ones(Nt))
    K_u_test = np.asarray(kernel(test_meas, last_info['train_u']),     dtype=np.float64)
    K_w_test = np.asarray(kernel(test_meas, last_info['train_w_pde']), dtype=np.float64)
    alpha = last_info['alpha']
    u_pred = K_u_test @ alpha
    w_pred = K_w_test @ alpha[Nb + Nd:]
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
        f'Darcy inverse — joint GP-PDE regression  '
        f'(N_dom={N}, N_data={Nd}, σ_noise={args.noise})',
        fontsize=12,
    )
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[fig]    {out_path}')


if __name__ == '__main__':
    main()
