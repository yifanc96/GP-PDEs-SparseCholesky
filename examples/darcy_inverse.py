"""2-D Darcy inverse problem: recover a(x) from noisy observations of u.

Forward PDE:   −∇·(a(x) ∇u(x)) = f(x)   on Ω = [0,1]²,  u = 0 on ∂Ω.
Inverse:       given N_data noisy point measurements `data_i = u(x_i) + η_i`
               with η_i ~ N(0, σ²), recover both u(·) and a(·) jointly.

This is the classical Darcy inverse problem from Chen-Hosseini-Owhadi-
Schäfer (2021); the PDE-direct GP formulation here matches
`yifanc96/NonLinPDEs-GPsolver`'s `main_DarcyFlow2d.py`. We place
independent GP priors on `u` and on `w := log a`, then minimize

    ‖u‖_{H_u}²  +  ‖w‖_{H_w}²  +  (1/σ²) Σᵢ |u(xᵢ) − data_i|²

subject to the PDE constraint, which after taking logs becomes

    Δu  =  −(∂₁w)(∂₁u) − (∂₂w)(∂₂u) − f · exp(−w).      (*)

The constraint is bilinear in the unknowns (u, w), so we run Gauss-
Newton on the joint loss (linearizing (*) around the current iterate).

`kolesky` connection.

* Both Gram matrices `Theta_u` and `Theta_w` are kernel matrices over
  *derivative measurements* of the GP — exactly the
  `LaplaceGradDiracPointMeasurement` class. Theta_u uses 5 measurement
  groups per interior point + boundary δ; Theta_w uses 3 per interior
  point. The full kernel evaluation is one call to
  `kernel(stack_measurements([...]))`.

* The data-fidelity term `(1/σ²) E^T E` in the GN Hessian is *exactly*
  the additive-noise structure that ``NoisyExplicitKLFactorization``
  (Algorithm 4.1) handles. For this demo we use a dense Cholesky of
  the GN Hessian for simplicity (N_domain ~ 250); for larger N you
  would build a sparse `kolesky` factor of ``Theta_u`` and feed it
  + the data noise into ``NoisyExplicitKLFactorization`` to apply
  ``(Theta_u⁻¹ + (1/σ²) E^T E)⁻¹`` — i.e. the GP regression posterior
  over u given the noisy data — at `O(N · ρ²ᵈ)` cost.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Reference FD solver for the forward Darcy PDE.
# Lifted from `yifanc96/NonLinPDEs-GPsolver/reference_solver/FD_for_Darcy_flow.py`.
# ---------------------------------------------------------------------------


def fd_darcy_forward(N: int, fun_a, f) -> np.ndarray:
    """Solve −∇·(a∇u) = f, u=0 on ∂[0,1]² via 5-point cell-centred FD.

    Returns u on an (N+2) × (N+2) grid (with zero boundary)."""
    import scipy.sparse
    import scipy.sparse.linalg

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
    ) / (hg * hg)

    XX, YY = np.meshgrid(x_grid, x_grid)
    fv = f(XX.flatten(), YY.flatten())
    sol = scipy.sparse.linalg.spsolve(A, fv).reshape(N, N)
    out = np.zeros((N + 2, N + 2))
    out[1:N + 1, 1:N + 1] = sol
    return out


# ---------------------------------------------------------------------------
# Build the two kernel Gram matrices via kolesky measurements.
# ---------------------------------------------------------------------------


def build_gram_u(kernel, X_domain: np.ndarray, X_boundary: np.ndarray) -> np.ndarray:
    """Theta_u over [∂₁u_int, ∂₂u_int, Δu_int, u_int, u_bdy] (size 4N+Nb).

    Each row block is a `LaplaceGradDiracPointMeasurement` with the
    appropriate weights (Δ-weight, ∇-weight, δ-weight)."""
    import kolesky as kl
    from kolesky.measurements import (
        LaplaceGradDiracPointMeasurement, stack_measurements,
    )
    N = X_domain.shape[0]
    Nb = X_boundary.shape[0]

    def lgd(coord, wL, wG, wD):
        return LaplaceGradDiracPointMeasurement(
            coordinate=coord, weight_laplace=wL.astype(np.float64),
            weight_grad=wG.astype(np.float64), weight_delta=wD.astype(np.float64),
        )

    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    z2 = np.zeros((N, 2))
    zN = np.zeros(N)
    oN = np.ones(N)
    zNb = np.zeros(Nb); oNb = np.ones(Nb); z2b = np.zeros((Nb, 2))

    m_du1 = lgd(X_domain, zN, e1, zN)
    m_du2 = lgd(X_domain, zN, e2, zN)
    m_lap = lgd(X_domain, oN, z2, zN)
    m_int = lgd(X_domain, zN, z2, oN)
    m_bdy = lgd(X_boundary, zNb, z2b, oNb)

    all_u = stack_measurements([m_du1, m_du2, m_lap, m_int, m_bdy])
    return np.asarray(kernel(all_u), dtype=np.float64)


def build_gram_w(kernel, X_domain: np.ndarray) -> np.ndarray:
    """Theta_w over [∂₁w, ∂₂w, w] (size 3N)."""
    import kolesky as kl
    from kolesky.measurements import (
        LaplaceGradDiracPointMeasurement, stack_measurements,
    )
    N = X_domain.shape[0]

    def lgd(coord, wL, wG, wD):
        return LaplaceGradDiracPointMeasurement(
            coordinate=coord, weight_laplace=wL.astype(np.float64),
            weight_grad=wG.astype(np.float64), weight_delta=wD.astype(np.float64),
        )

    e1 = np.tile(np.array([1.0, 0.0]), (N, 1))
    e2 = np.tile(np.array([0.0, 1.0]), (N, 1))
    z2 = np.zeros((N, 2))
    zN = np.zeros(N); oN = np.ones(N)

    m_dw1 = lgd(X_domain, zN, e1, zN)
    m_dw2 = lgd(X_domain, zN, e2, zN)
    m_w   = lgd(X_domain, zN, z2, oN)

    all_w = stack_measurements([m_dw1, m_dw2, m_w])
    return np.asarray(kernel(all_w), dtype=np.float64)


def adaptive_nugget(Theta: np.ndarray, block_sizes: Tuple[int, ...], nugget: float) -> np.ndarray:
    """Apply per-block adaptive nugget (matches the reference solver).

    Each block's diagonal is rescaled by trace(block) / trace(last block);
    the last block (point-evaluation block) gets nugget*1, others scale up
    to compensate for the larger derivative variances."""
    diag_scales = np.empty(Theta.shape[0])
    last_tr = np.trace(Theta[-block_sizes[-1]:, -block_sizes[-1]:])
    start = 0
    for k, sz in enumerate(block_sizes):
        if k < len(block_sizes) - 1:
            tr = np.trace(Theta[start:start + sz, start:start + sz])
            diag_scales[start:start + sz] = tr / last_tr
        else:
            diag_scales[start:start + sz] = 1.0
        start += sz
    return Theta + nugget * np.diag(diag_scales)


# ---------------------------------------------------------------------------
# Loss and GN linearization (jit-compiled with JAX for autodiff Hessians).
# ---------------------------------------------------------------------------


def _make_loss_fns(N_domain, N_data, N_boundary, L_u, L_w, rhs_f, bdy_g, sigma):
    """Return (loss, GN_loss, grad_loss, hess_GN) jit-compiled JAX fns."""
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)

    L_u_j = jnp.asarray(L_u); L_w_j = jnp.asarray(L_w)
    rhs_f_j = jnp.asarray(rhs_f); bdy_g_j = jnp.asarray(bdy_g)
    inv_sigma2 = 1.0 / (sigma * sigma)

    def split(z):
        # z = [w0, w1, w2, v0, v1, v2]
        N = N_domain
        return (z[0:N], z[N:2*N], z[2*N:3*N],
                z[3*N:4*N], z[4*N:5*N], z[5*N:6*N])

    def stack_w(w0, w1, w2):
        return jnp.concatenate([w1, w2, w0])

    def stack_v(v0, v1, v2, v3):
        return jnp.concatenate([v1, v2, v3, v0, bdy_g_j])

    def loss(z, data_u):
        w0, w1, w2, v0, v1, v2 = split(z)
        v3 = -v1 * w1 - v2 * w2 - rhs_f_j * jnp.exp(-w0)   # PDE constraint
        w_all = stack_w(w0, w1, w2)
        v_all = stack_v(v0, v1, v2, v3)
        ta = jax.scipy.linalg.solve_triangular(L_w_j, w_all, lower=True)
        tu = jax.scipy.linalg.solve_triangular(L_u_j, v_all, lower=True)
        data_term = inv_sigma2 * jnp.sum((v0[:N_data] - data_u) ** 2)
        return jnp.dot(ta, ta) + jnp.dot(tu, tu) + data_term

    def GN_loss(z, z_old, data_u):
        w0_o, w1_o, w2_o, _, v1_o, v2_o = split(z_old)
        w0, w1, w2, v0, v1, v2 = split(z)
        # Linearize v3 around z_old:
        # original: v3 = -v1 w1 - v2 w2 - f exp(-w0)
        # ∂v3/∂w0 = f exp(-w0_o);  ∂v3/∂w_i = -v_{i,o};  ∂v3/∂v_i = -w_{i,o}
        # Linear v3 (Taylor at z_old, dropping constant since GN minimizes a
        # quadratic in z and constants don't affect argmin):
        v3 = (-rhs_f_j) * (-jnp.exp(-w0_o)) * w0 \
             + (-v1_o) * w1 + (-v2_o) * w2 \
             + (-w1_o) * v1 + (-w2_o) * v2
        w_all = stack_w(w0, w1, w2)
        v_all = stack_v(v0, v1, v2, v3)
        ta = jax.scipy.linalg.solve_triangular(L_w_j, w_all, lower=True)
        tu = jax.scipy.linalg.solve_triangular(L_u_j, v_all, lower=True)
        data_term = inv_sigma2 * jnp.sum((v0[:N_data] - data_u) ** 2)
        return jnp.dot(ta, ta) + jnp.dot(tu, tu) + data_term

    grad_loss = jax.jit(jax.grad(loss, argnums=0))
    hess_GN_at_zold = jax.jit(jax.hessian(GN_loss, argnums=0))
    loss_j = jax.jit(loss)
    return loss_j, hess_GN_at_zold, grad_loss


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--N-domain', type=int, default=200)
    p.add_argument('--N-boundary', type=int, default=80)
    p.add_argument('--N-data', type=int, default=60)
    p.add_argument('--noise', type=float, default=1e-3,
                   help='σ for additive Gaussian observation noise')
    p.add_argument('--N-fd', type=int, default=80,
                   help='FD reference grid resolution')
    p.add_argument('--kernel', default='Gaussian',
                   choices=['Gaussian', 'Matern5half', 'Matern7half', 'Matern9half'])
    p.add_argument('--sigma', type=float, default=0.2,
                   help='kernel length scale')
    p.add_argument('--nugget', type=float, default=1e-8)
    p.add_argument('--GN-steps', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out', default='docs/darcy_inverse.png')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    import kolesky as kl
    kernels = {
        'Gaussian':    kl.GaussianCovariance,
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[setup]  kernel = {args.kernel}, σ_kernel = {args.sigma}, '
          f'σ_noise = {args.noise}, nugget = {args.nugget}')

    # ----- step 1: ground truth via FD -----
    c = 1.0
    def a_truth(x1, x2):
        return (np.exp(c * np.sin(2 * np.pi * x1) + c * np.sin(2 * np.pi * x2))
                + np.exp(-c * np.sin(2 * np.pi * x1) - c * np.sin(2 * np.pi * x2)))
    def f_rhs(x1, x2):
        return np.ones_like(x1) if hasattr(x1, 'shape') else 1.0
    print(f'[truth]  FD solve at {args.N_fd}² grid …')
    t0 = time.perf_counter()
    u_truth = fd_darcy_forward(args.N_fd - 2, a_truth, f_rhs)   # (N_fd, N_fd)
    print(f'[truth]  FD wall: {time.perf_counter()-t0:.2f} s')

    # ----- step 2: sample collocation points + observation locations -----
    # Random interior points (first N_data are observation locations).
    X_domain = rng.uniform(0, 1, (args.N_domain, 2))
    # Boundary: equally spaced around the unit square.
    nb_per_side = args.N_boundary // 4
    t = np.linspace(0, 1, nb_per_side + 1)[:-1]
    o = np.ones_like(t); z = np.zeros_like(t)
    X_boundary = np.concatenate([
        np.stack([t, z], 1), np.stack([o, t], 1),
        np.stack([t[::-1], o], 1), np.stack([z, t[::-1]], 1),
    ], axis=0)[:args.N_boundary]
    X_data = X_domain[:args.N_data]
    print(f'[sample] N_domain = {args.N_domain},  '
          f'N_boundary = {X_boundary.shape[0]},  N_data = {args.N_data}')

    # ----- step 3: noisy observations of u_truth at X_data -----
    from scipy.interpolate import RegularGridInterpolator
    grid_xs = np.linspace(0, 1, args.N_fd)
    interp = RegularGridInterpolator((grid_xs, grid_xs), u_truth.T)
    u_at_data = interp(X_data)
    data_noisy = u_at_data + args.noise * rng.standard_normal(args.N_data)
    print(f'[obs]    u_truth range at data pts: [{u_at_data.min():.3e}, {u_at_data.max():.3e}]')

    # ----- step 4: build Theta_u, Theta_w and Cholesky factors -----
    print('[gram]   assembling Theta_u  …')
    t0 = time.perf_counter()
    Theta_u = build_gram_u(kernel, X_domain, X_boundary)
    Theta_u = adaptive_nugget(
        Theta_u,
        block_sizes=(args.N_domain, args.N_domain, args.N_domain,
                     args.N_domain + X_boundary.shape[0]),
        nugget=args.nugget,
    )
    Theta_u = 0.5 * (Theta_u + Theta_u.T)
    L_u = np.linalg.cholesky(Theta_u)
    print(f'[gram]   Theta_u {Theta_u.shape}, chol wall: {time.perf_counter()-t0:.2f} s')

    print('[gram]   assembling Theta_w  …')
    t0 = time.perf_counter()
    Theta_w = build_gram_w(kernel, X_domain)
    Theta_w = adaptive_nugget(
        Theta_w,
        block_sizes=(args.N_domain, args.N_domain, args.N_domain),
        nugget=args.nugget,
    )
    Theta_w = 0.5 * (Theta_w + Theta_w.T)
    L_w = np.linalg.cholesky(Theta_w)
    print(f'[gram]   Theta_w {Theta_w.shape}, chol wall: {time.perf_counter()-t0:.2f} s')

    # ----- step 5: Gauss-Newton iteration on the joint loss -----
    rhs_f = np.ones(args.N_domain, dtype=np.float64)   # f ≡ 1
    bdy_g = np.zeros(X_boundary.shape[0], dtype=np.float64)

    loss_fn, hess_fn, grad_fn = _make_loss_fns(
        args.N_domain, args.N_data, X_boundary.shape[0],
        L_u, L_w, rhs_f, bdy_g, args.noise,
    )

    z = rng.standard_normal(6 * args.N_domain)         # initial iterate
    print(f'\n[GN]     {args.GN_steps} Gauss-Newton steps')
    print(f'         iter  0:  loss = {float(loss_fn(z, data_noisy)):.4e}')
    t_loop = time.perf_counter()
    for step in range(1, args.GN_steps + 1):
        H = np.asarray(hess_fn(z, z, data_noisy))      # GN Hessian at current iterate
        g = np.asarray(grad_fn(z, data_noisy))
        # Stabilize: H is SPD up to round-off; tiny diagonal regularization helps
        H = H + 1e-10 * np.trace(H) / H.shape[0] * np.eye(H.shape[0])
        delta = np.linalg.solve(H, g)
        z = z - delta
        loss_now = float(loss_fn(z, data_noisy))
        print(f'         iter {step:2d}:  loss = {loss_now:.4e}')
    print(f'[GN]     wall: {time.perf_counter()-t_loop:.2f} s')

    # ----- step 6: extend solution to a test grid for plotting / accuracy -----
    N_test = 80
    xs = np.linspace(0, 1, N_test)
    XX, YY = np.meshgrid(xs, xs)
    X_test = np.stack([XX.ravel(), YY.ravel()], axis=1)

    # Recover the observation vectors at the collocation points
    w0 = z[0 * args.N_domain:1 * args.N_domain]
    w1 = z[1 * args.N_domain:2 * args.N_domain]
    w2 = z[2 * args.N_domain:3 * args.N_domain]
    v0 = z[3 * args.N_domain:4 * args.N_domain]
    v1 = z[4 * args.N_domain:5 * args.N_domain]
    v2 = z[5 * args.N_domain:6 * args.N_domain]
    v3 = -v1 * w1 - v2 * w2 - rhs_f * np.exp(-w0)

    # GP regression at X_test: u(x) = Theta_u(x, train) · Theta_u(train,train)⁻¹ · obs_u
    # Likewise for w. We use the same kernel + same measurement order.
    sol_u = np.concatenate([v1, v2, v3, v0, bdy_g])
    sol_w = np.concatenate([w1, w2, w0])

    # Build Theta(test, train) — only the cross-covariance to point-eval test
    from kolesky.measurements import (
        LaplaceGradDiracPointMeasurement, stack_measurements,
    )

    def lgd_pure_delta(coord):
        N = coord.shape[0]
        return LaplaceGradDiracPointMeasurement(
            coordinate=coord, weight_laplace=np.zeros(N),
            weight_grad=np.zeros((N, 2)), weight_delta=np.ones(N),
        )

    test_meas = lgd_pure_delta(X_test)
    train_meas_u = stack_measurements([
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.zeros(args.N_domain),
            weight_grad=np.tile([1.0, 0.0], (args.N_domain, 1)), weight_delta=np.zeros(args.N_domain),
        ),
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.zeros(args.N_domain),
            weight_grad=np.tile([0.0, 1.0], (args.N_domain, 1)), weight_delta=np.zeros(args.N_domain),
        ),
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.ones(args.N_domain),
            weight_grad=np.zeros((args.N_domain, 2)), weight_delta=np.zeros(args.N_domain),
        ),
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.zeros(args.N_domain),
            weight_grad=np.zeros((args.N_domain, 2)), weight_delta=np.ones(args.N_domain),
        ),
        LaplaceGradDiracPointMeasurement(
            coordinate=X_boundary, weight_laplace=np.zeros(X_boundary.shape[0]),
            weight_grad=np.zeros((X_boundary.shape[0], 2)),
            weight_delta=np.ones(X_boundary.shape[0]),
        ),
    ])
    train_meas_w = stack_measurements([
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.zeros(args.N_domain),
            weight_grad=np.tile([1.0, 0.0], (args.N_domain, 1)), weight_delta=np.zeros(args.N_domain),
        ),
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.zeros(args.N_domain),
            weight_grad=np.tile([0.0, 1.0], (args.N_domain, 1)), weight_delta=np.zeros(args.N_domain),
        ),
        LaplaceGradDiracPointMeasurement(
            coordinate=X_domain, weight_laplace=np.zeros(args.N_domain),
            weight_grad=np.zeros((args.N_domain, 2)), weight_delta=np.ones(args.N_domain),
        ),
    ])
    Theta_u_test = np.asarray(kernel(test_meas, train_meas_u), dtype=np.float64)
    Theta_w_test = np.asarray(kernel(test_meas, train_meas_w), dtype=np.float64)

    coef_u = np.linalg.solve(L_u.T, np.linalg.solve(L_u, sol_u))
    coef_w = np.linalg.solve(L_w.T, np.linalg.solve(L_w, sol_w))

    u_recovered = Theta_u_test @ coef_u
    w_recovered = Theta_w_test @ coef_w
    a_recovered = np.exp(w_recovered)

    a_truth_grid = a_truth(XX.ravel(), YY.ravel())
    u_truth_test = interp(X_test)

    L2_u = float(np.sqrt(np.mean((u_recovered - u_truth_test) ** 2)))
    L2_a = float(np.sqrt(np.mean((a_recovered - a_truth_grid) ** 2)))
    rel_a = L2_a / float(np.sqrt(np.mean(a_truth_grid ** 2)))
    rel_u = L2_u / max(1e-12, float(np.sqrt(np.mean(u_truth_test ** 2))))
    print(f'\n[error]  L²(u)        = {L2_u:.3e}   (rel {rel_u:.2%})')
    print(f'[error]  L²(a-recover) = {L2_a:.3e}   (rel {rel_a:.2%})')

    # ----- step 7: render comparison figure -----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    a_truth_2d = a_truth_grid.reshape(N_test, N_test)
    a_rec_2d = a_recovered.reshape(N_test, N_test)
    u_truth_2d = u_truth_test.reshape(N_test, N_test)
    u_rec_2d = u_recovered.reshape(N_test, N_test)

    a_min, a_max = float(min(a_truth_2d.min(), a_rec_2d.min())), float(max(a_truth_2d.max(), a_rec_2d.max()))
    u_min, u_max = float(min(u_truth_2d.min(), u_rec_2d.min())), float(max(u_truth_2d.max(), u_rec_2d.max()))

    im00 = axes[0, 0].contourf(XX, YY, a_truth_2d, 40, cmap='coolwarm', vmin=a_min, vmax=a_max)
    axes[0, 0].set_title('truth  $a(x)$'); plt.colorbar(im00, ax=axes[0, 0])
    im01 = axes[0, 1].contourf(XX, YY, a_rec_2d, 40, cmap='coolwarm', vmin=a_min, vmax=a_max)
    axes[0, 1].set_title(f'recovered  $a(x)$    L² = {L2_a:.2e} ({rel_a:.1%})')
    plt.colorbar(im01, ax=axes[0, 1])

    im10 = axes[1, 0].contourf(XX, YY, u_truth_2d, 40, cmap='coolwarm', vmin=u_min, vmax=u_max)
    axes[1, 0].set_title('truth  $u(x)$'); plt.colorbar(im10, ax=axes[1, 0])
    im11 = axes[1, 1].contourf(XX, YY, u_rec_2d, 40, cmap='coolwarm', vmin=u_min, vmax=u_max)
    axes[1, 1].set_title(f'recovered  $u(x)$    L² = {L2_u:.2e} ({rel_u:.1%})')
    plt.colorbar(im11, ax=axes[1, 1])

    # show observation locations on the recovered-u panel
    axes[1, 1].plot(X_data[:, 0], X_data[:, 1], 'k.', ms=3, alpha=0.6)

    for ax in axes.flat:
        ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$')
        ax.set_aspect('equal')

    fig.suptitle(
        f'Darcy inverse problem — '
        f'N_dom={args.N_domain}, N_data={args.N_data}, σ_noise={args.noise}, '
        f'kernel={args.kernel}@σ={args.sigma}',
        fontsize=12,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'[fig]    {out_path}')


if __name__ == '__main__':
    main()
