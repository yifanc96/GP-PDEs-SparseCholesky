"""Render the figures embedded in README.md.

Run from the repo root:
    JAX_PLATFORMS=cpu python docs/make_figures.py

Outputs:
    docs/nonlin_elliptic_compare.png   (true vs numerical + error contour)
    docs/U_sparsity.png                (sparsity pattern of U)
"""

from __future__ import annotations

import os
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import kolesky as kl
from kolesky.pde import (
    NonlinElliptic2d, solve_nonlin_elliptic_2d, sample_points_grid_2d,
)


OUT = pathlib.Path(__file__).parent
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Figure 1: true vs numerical + contour of pointwise error
# ---------------------------------------------------------------------------


def _ground_truth(freq=200, s=6):
    ks = np.arange(1, freq + 1)

    def u(x):
        return float(np.sum(np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))

    def rhs(x):
        lap = float(np.sum(2 * ks ** 2 * np.pi ** 2
                            * np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1])
                            / ks ** s))
        return lap + u(x) ** 3

    return u, rhs


def nonlin_elliptic_figure(h=0.02):
    print(f'[fig 1] NonLinElliptic, h={h} …')
    u_exact, rhs_fn = _ground_truth()
    eqn = NonlinElliptic2d(alpha=1.0, m=3, domain=((0, 1), (0, 1)),
                            bdy=u_exact, rhs=rhs_fn)
    X_dom, X_bdy = sample_points_grid_2d(eqn.domain, h, h)
    kernel = kl.MaternCovariance7_2(length_scale=0.3)
    sol = solve_nonlin_elliptic_2d(
        eqn, kernel, X_dom, X_bdy, np.zeros(X_dom.shape[0]),
        GN_steps=3, rho_big=3, rho_small=3, k_neighbors=3,
        backend='cpu', verbose=False,
    )
    truth = np.array([u_exact(X_dom[i]) for i in range(X_dom.shape[0])])
    err = np.abs(truth - sol)
    n_side = int(round(1.0 / h)) - 1

    def grid(v):
        return v.reshape(n_side, n_side).T  # (x, y) -> imshow-friendly

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    extent = [h, 1 - h, h, 1 - h]
    kw = dict(origin='lower', extent=extent, aspect='equal')

    lo, hi = truth.min(), truth.max()
    im0 = axes[0].imshow(grid(truth), cmap='viridis', vmin=lo, vmax=hi, **kw)
    axes[0].set_title(f'True $u(x, y)$')
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(grid(sol), cmap='viridis', vmin=lo, vmax=hi, **kw)
    axes[1].set_title('Numerical solution')
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(grid(err), cmap='inferno', **kw)
    axes[2].set_title(f'|error|   (L²={np.sqrt(np.mean(err**2)):.1e})')
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    for ax in axes:
        ax.set_xlabel('x'); ax.set_ylabel('y')

    out = OUT / 'nonlin_elliptic_compare.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig 1] saved → {out}')


# ---------------------------------------------------------------------------
# Figure 2: sparsity pattern of U (in the maximin-permuted ordering)
# ---------------------------------------------------------------------------


def sparsity_figure(N_side=40, rho=3.0):
    print(f'[fig 2] sparsity for {N_side}x{N_side} grid …')
    xs = np.linspace(0.02, 0.98, N_side)
    XX, YY = np.meshgrid(xs, xs, indexing='ij')
    x = np.stack([XX.flatten(), YY.flatten()], axis=1)
    m = kl.point_measurements(x, dims=2)
    kernel = kl.MaternCovariance5_2(0.15)
    imp = kl.ImplicitKLFactorization.build(kernel, m, rho=rho, k_neighbors=1)
    exp = kl.ExplicitKLFactorization(imp, nugget=1e-8, backend='cpu')
    U = exp.U

    N = U.shape[0]
    density = U.nnz / (N * N)
    print(f'        N = {N}, U.nnz = {U.nnz}  ({density:.2%} of dense)')

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), constrained_layout=True)
    ax.spy(U, markersize=0.6, aspect='equal', color='tab:blue')
    ax.set_title(f'Sparsity of $U$ (maximin-permuted)\n'
                 f'N={N}, nnz={U.nnz:,} ({density:.2%})')
    ax.set_xlabel('column (maximin index)')
    ax.set_ylabel('row')

    out = OUT / 'U_sparsity.png'
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'[fig 2] saved → {out}')


if __name__ == '__main__':
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    nonlin_elliptic_figure(h=0.02)
    sparsity_figure(N_side=40, rho=3.0)
