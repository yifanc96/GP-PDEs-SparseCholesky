"""Nonlinear elliptic PDE on an L-shape — a complicated-geometry demo.

Domain:   Ω = [0, 1]² \ [0.5, 1] × [0.5, 1]   (the classic re-entrant L)
Problem:  -Δu + u³ = f                              in  Ω
          u = g                                     on  ∂Ω
Manufactured solution:  u(x, y) = sin(π x) sin(π y)

The point of this example is that the PDE solver is *grid-free*: swap the
rectangular-grid sampler for rejection-sampling inside the L-shape (and
parametric sampling along its six edges), and everything else — maximin
ordering, sparse factorization, Gauss-Newton + pCG — runs unchanged.

Usage:
    python examples/lshape_nonlin_elliptic.py --N-interior 3000 --backend cpu

Outputs:
    docs/lshape.png  (three-panel figure)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import kolesky as kl
from kolesky.pde import NonlinElliptic2d, solve_nonlin_elliptic_2d


# ---------------------------------------------------------------------------
# Geometry: L-shape = [0,1]² \ [0.5, 1] × [0.5, 1]
# ---------------------------------------------------------------------------


def in_lshape(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Boolean mask: True iff (x, y) lies in the L-shape (including boundary)."""
    return (
        (x >= 0.0) & (x <= 1.0)
        & (y >= 0.0) & (y <= 1.0)
        & ~((x > 0.5) & (y > 0.5))
    )


def sample_interior(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly sample the L-shape interior by rejection."""
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform(0, 1, (2 * n, 2))
        keep = in_lshape(batch[:, 0], batch[:, 1])
        # avoid exact boundary
        inner = (
            (batch[:, 0] > 0.005) & (batch[:, 0] < 0.995)
            & (batch[:, 1] > 0.005) & (batch[:, 1] < 0.995)
            & ~((batch[:, 0] > 0.495) & (batch[:, 0] < 0.505)
                & (batch[:, 1] > 0.495))
            & ~((batch[:, 0] > 0.495)
                & (batch[:, 1] > 0.495) & (batch[:, 1] < 0.505))
        )
        pts = np.vstack([pts, batch[keep & inner]])
    return pts[:n]


def sample_boundary(per_unit_length: int) -> np.ndarray:
    """Sample the six L-shape edges at roughly uniform density."""
    def linseg(a, b, n):
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        return (1 - t)[:, None] * np.array(a) + t[:, None] * np.array(b)

    segs = [
        ((0.0, 0.0), (1.0, 0.0),   per_unit_length),              # bottom
        ((1.0, 0.0), (1.0, 0.5),   max(per_unit_length // 2, 2)), # right
        ((1.0, 0.5), (0.5, 0.5),   max(per_unit_length // 2, 2)), # top of notch
        ((0.5, 0.5), (0.5, 1.0),   max(per_unit_length // 2, 2)), # left of notch
        ((0.5, 1.0), (0.0, 1.0),   max(per_unit_length // 2, 2)), # top
        ((0.0, 1.0), (0.0, 0.0),   per_unit_length),              # left
    ]
    return np.concatenate([linseg(a, b, n) for a, b, n in segs], axis=0)


LSHAPE_OUTLINE = np.array([
    [0.0, 0.0], [1.0, 0.0], [1.0, 0.5], [0.5, 0.5],
    [0.5, 1.0], [0.0, 1.0], [0.0, 0.0],
])


# ---------------------------------------------------------------------------
# Manufactured solution + forcing
# ---------------------------------------------------------------------------


def ground_truth(alpha: float = 1.0, m: int = 3):
    def u(x):
        return float(np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    def rhs(x):
        lap_neg = 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
        return float(lap_neg + alpha * u(x) ** m)

    return u, rhs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_solution(X_dom, X_bdy, truth, sol, out_path: pathlib.Path):
    err = np.abs(truth - sol)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    def _frame(ax):
        ax.plot(LSHAPE_OUTLINE[:, 0], LSHAPE_OUTLINE[:, 1], 'k-', lw=2.0)
        ax.set_aspect('equal')
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    # Panel 1: sample points
    ax = axes[0]
    ax.scatter(X_dom[:, 0], X_dom[:, 1], s=6, c='tab:blue', alpha=0.7,
               label=f'interior ({X_dom.shape[0]})')
    ax.scatter(X_bdy[:, 0], X_bdy[:, 1], s=14, c='tab:red',
               label=f'boundary ({X_bdy.shape[0]})')
    _frame(ax)
    ax.set_title('sample points', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    # Panel 2: numerical solution (filled scatter)
    ax = axes[1]
    lo, hi = truth.min(), truth.max()
    sc = ax.scatter(X_dom[:, 0], X_dom[:, 1], c=sol, s=10, cmap='viridis',
                    vmin=lo, vmax=hi)
    _frame(ax)
    ax.set_title('numerical solution $u_h$', fontsize=12)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    # Panel 3: pointwise error
    ax = axes[2]
    sc = ax.scatter(X_dom[:, 0], X_dom[:, 1], c=err, s=10, cmap='inferno')
    _frame(ax)
    L2 = float(np.sqrt(np.mean(err ** 2)))
    Linf = float(err.max())
    ax.set_title(f'|$u_h$ − $u$|     L²={L2:.1e},  L∞={Linf:.1e}', fontsize=12)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--N-interior', type=int, default=3000)
    p.add_argument('--bdy-density', type=int, default=80,
                   help='approx points per unit-length boundary')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--m', type=int, default=3)
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
    p.add_argument('--out', default='docs/lshape.png')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = sample_boundary(args.bdy_density)
    print(f'[points] interior = {X_dom.shape[0]}, boundary = {X_bdy.shape[0]}')

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
        'Gaussian':    kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel]  {args.kernel}, length_scale = {args.sigma}')

    u_exact, rhs_fn = ground_truth(args.alpha, args.m)
    eqn = NonlinElliptic2d(alpha=args.alpha, m=args.m,
                            domain=((0.0, 1.0), (0.0, 1.0)),    # bounding box
                            bdy=u_exact, rhs=rhs_fn)

    t0 = time.perf_counter()
    sol = solve_nonlin_elliptic_2d(
        eqn, kernel, X_dom, X_bdy, sol_init=np.zeros(X_dom.shape[0]),
        nugget=args.nugget, GN_steps=args.GN_steps,
        rho_big=args.rho, rho_small=args.rho, k_neighbors=args.k_neighbors,
        backend=args.backend,
    )
    t1 = time.perf_counter()
    print(f'[solve]   {t1 - t0:.3f} s')

    truth = np.array([u_exact(X_dom[i]) for i in range(X_dom.shape[0])])
    err = truth - sol
    L2 = float(np.sqrt(np.mean(err ** 2)))
    Linf = float(np.max(np.abs(err)))
    print(f'[error]   L² = {L2:.3e}, L∞ = {Linf:.3e}')

    out = pathlib.Path(args.out)
    plot_solution(X_dom, X_bdy, truth, sol, out)
    print(f'[figure]  saved → {out}')


if __name__ == '__main__':
    main()
