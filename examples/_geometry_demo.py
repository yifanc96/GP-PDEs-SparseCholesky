"""Shared helpers for the geometry showcase examples.

Each example under examples/*_nonlin_elliptic.py describes its domain
(an interior sampler, a boundary-point array, an outline-polyline for
plotting, a manufactured solution, and an optional eqn-specific `rhs`)
and calls `run_and_plot(...)` here. That keeps each example under
~100 lines and focused on the geometry, not on the boilerplate.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time
from dataclasses import dataclass
from typing import Callable, List, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GeometryDemo:
    name: str
    X_domain: np.ndarray             # (N_int, 2)
    X_boundary: np.ndarray           # (N_bdy, 2)
    outline: List[np.ndarray]        # list of (M_k, 2) polylines for plotting
    u_exact: Callable                # u_exact(x) -> float
    rhs: Callable                    # rhs(x)     -> float
    alpha: float = 1.0
    m: int = 3


def default_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--N-interior', type=int, default=3000)
    p.add_argument('--kernel', default='Matern7half',
                   choices=['Matern5half', 'Matern7half', 'Matern9half', 'Gaussian'])
    p.add_argument('--sigma', type=float, default=0.3)
    p.add_argument('--nugget', type=float, default=1e-10)
    p.add_argument('--GN-steps', type=int, default=3)
    p.add_argument('--rho', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--backend', choices=['auto', 'cpu', 'jax'], default='cpu')
    p.add_argument('--platform', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    return p


def _draw_outline(ax, outline: Sequence[np.ndarray]):
    for pl in outline:
        ax.plot(pl[:, 0], pl[:, 1], 'k-', lw=1.8)


def _frame(ax, outline: Sequence[np.ndarray], pad: float = 0.05):
    xs = np.concatenate([pl[:, 0] for pl in outline])
    ys = np.concatenate([pl[:, 1] for pl in outline])
    ax.set_aspect('equal')
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_xlabel('x'); ax.set_ylabel('y')


def plot_three_panel_3d(demo: 'GeometryDemo', sol: np.ndarray, out: pathlib.Path):
    """3D scatter version: three panels (points, solution, error) on
    matplotlib Axes3D. Outlines are lists of (M, 3) polylines.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
    X = demo.X_domain
    Xb = demo.X_boundary
    truth = np.array([demo.u_exact(X[i]) for i in range(X.shape[0])])
    err = np.abs(truth - sol)

    fig = plt.figure(figsize=(16, 5.5), constrained_layout=True)
    ax_kw = dict(projection='3d')
    axes = [fig.add_subplot(1, 3, i + 1, **ax_kw) for i in range(3)]

    all_pts = np.vstack([X, Xb])
    lo_xyz = all_pts.min(axis=0) - 0.03
    hi_xyz = all_pts.max(axis=0) + 0.03

    def _frame(ax):
        ax.set_xlim(lo_xyz[0], hi_xyz[0])
        ax.set_ylim(lo_xyz[1], hi_xyz[1])
        ax.set_zlim(lo_xyz[2], hi_xyz[2])
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(elev=22, azim=40)
        # draw outlines
        for pl in demo.outline:
            ax.plot(pl[:, 0], pl[:, 1], pl[:, 2], 'k-', lw=0.9, alpha=0.45)

    # thin marker for 3D clarity
    axes[0].scatter(X[:, 0], X[:, 1], X[:, 2], s=3, c='tab:blue', alpha=0.35,
                    label=f'interior ({X.shape[0]})')
    axes[0].scatter(Xb[:, 0], Xb[:, 1], Xb[:, 2], s=3, c='tab:red', alpha=0.7,
                    label=f'boundary ({Xb.shape[0]})')
    _frame(axes[0])
    axes[0].set_title('sample points', fontsize=12)
    axes[0].legend(loc='upper left', fontsize=9)

    lo, hi = truth.min(), truth.max()
    sc1 = axes[1].scatter(X[:, 0], X[:, 1], X[:, 2], c=sol, cmap='viridis',
                          s=4, alpha=0.55, vmin=lo, vmax=hi)
    _frame(axes[1])
    axes[1].set_title('numerical solution $u_h$', fontsize=12)
    fig.colorbar(sc1, ax=axes[1], shrink=0.7)

    sc2 = axes[2].scatter(X[:, 0], X[:, 1], X[:, 2], c=err, cmap='inferno',
                          s=4, alpha=0.55)
    _frame(axes[2])
    L2 = float(np.sqrt(np.mean(err ** 2)))
    Linf = float(err.max())
    axes[2].set_title(f'|$u_h$ − $u$|   L²={L2:.1e}, L∞={Linf:.1e}', fontsize=12)
    fig.colorbar(sc2, ax=axes[2], shrink=0.7)

    fig.suptitle(demo.name, fontsize=14, y=1.02)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return L2, Linf


def plot_three_panel(demo: GeometryDemo, sol: np.ndarray, out: pathlib.Path):
    truth = np.array([demo.u_exact(demo.X_domain[i]) for i in range(demo.X_domain.shape[0])])
    err = np.abs(truth - sol)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.scatter(demo.X_domain[:, 0], demo.X_domain[:, 1], s=5,
               c='tab:blue', alpha=0.7, label=f'interior ({demo.X_domain.shape[0]})')
    ax.scatter(demo.X_boundary[:, 0], demo.X_boundary[:, 1], s=12,
               c='tab:red', label=f'boundary ({demo.X_boundary.shape[0]})')
    _draw_outline(ax, demo.outline); _frame(ax, demo.outline)
    ax.set_title('sample points', fontsize=12)
    ax.legend(loc='best', fontsize=10)

    ax = axes[1]
    lo, hi = truth.min(), truth.max()
    sc = ax.scatter(demo.X_domain[:, 0], demo.X_domain[:, 1], c=sol, s=10,
                    cmap='viridis', vmin=lo, vmax=hi)
    _draw_outline(ax, demo.outline); _frame(ax, demo.outline)
    ax.set_title('numerical solution $u_h$', fontsize=12)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    ax = axes[2]
    sc = ax.scatter(demo.X_domain[:, 0], demo.X_domain[:, 1], c=err, s=10,
                    cmap='inferno')
    _draw_outline(ax, demo.outline); _frame(ax, demo.outline)
    L2 = float(np.sqrt(np.mean(err ** 2)))
    Linf = float(err.max())
    ax.set_title(f'|$u_h$ − $u$|     L²={L2:.1e},  L∞={Linf:.1e}', fontsize=12)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    fig.suptitle(demo.name, fontsize=14, y=1.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return L2, Linf


def run_and_plot(demo: GeometryDemo, args) -> dict:
    """Solve the PDE on `demo` and render the three-panel figure."""
    os.environ['JAX_PLATFORMS'] = args.platform

    import kolesky as kl
    from kolesky.pde import NonlinElliptic, solve_nonlin_elliptic

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
        'Gaussian':    kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel]  {args.kernel}, length_scale = {args.sigma}')
    print(f'[points]  interior = {demo.X_domain.shape[0]}, boundary = {demo.X_boundary.shape[0]}')

    # `domain` is only used by the 2D grid sampler; the solver itself is
    # d-agnostic and never touches it. We pass a 2D bbox unconditionally.
    bx = (demo.X_domain[:, 0].min(), demo.X_domain[:, 0].max())
    by = (demo.X_domain[:, 1].min(), demo.X_domain[:, 1].max())
    eqn = NonlinElliptic(
        alpha=demo.alpha, m=demo.m, domain=(bx, by),
        bdy=demo.u_exact, rhs=demo.rhs,
    )
    t0 = time.perf_counter()
    sol = solve_nonlin_elliptic(
        eqn, kernel, demo.X_domain, demo.X_boundary,
        sol_init=np.zeros(demo.X_domain.shape[0]),
        nugget=args.nugget, GN_steps=args.GN_steps,
        rho_big=args.rho, rho_small=args.rho, k_neighbors=args.k_neighbors,
        backend=args.backend, verbose=False,
    )
    t1 = time.perf_counter()
    print(f'[solve]   {t1 - t0:.3f} s')

    out = pathlib.Path(args.out) if args.out else pathlib.Path(f'docs/{demo.name}.png')
    if demo.X_domain.shape[1] == 3:
        L2, Linf = plot_three_panel_3d(demo, sol, out)
    else:
        L2, Linf = plot_three_panel(demo, sol, out)
    print(f'[error]   L² = {L2:.3e}, L∞ = {Linf:.3e}')
    print(f'[figure]  {out}')
    return {'wall': t1 - t0, 'L2': L2, 'Linf': Linf}
