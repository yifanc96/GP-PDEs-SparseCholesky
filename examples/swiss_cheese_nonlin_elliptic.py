"""Nonlinear elliptic PDE on a 'Swiss-cheese' domain — fancy-geometry demo.

Domain:   Ω = [0, 1]² \ ⋃ᵢ B(cᵢ, rᵢ)     (a square with 4 circular holes)
Problem:  -Δu + u³ = f                        in Ω
          u = g                               on ∂Ω      (outer square + each hole)
Manufactured solution:  u(x, y) = cos(2π x) · cos(2π y)

This is the kind of domain where classical meshing is *painful* — you'd
need to conform triangles / quads around every curved inner boundary
and propagate refinement into the interior. For this collocation /
Gaussian-process approach, the code change from the rectangular example
is literally the point sampler: rejection outside the holes for the
interior, one parametric circle per hole for the boundary. Everything
else — maximin ordering, sparse factorization, Gauss-Newton + pCG —
is unchanged.

Usage:
    python examples/swiss_cheese_nonlin_elliptic.py

Outputs:
    docs/swiss_cheese.png  (three-panel figure: points / solution / error)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import kolesky as kl
from kolesky.pde import NonlinElliptic2d, solve_nonlin_elliptic_2d


# ---------------------------------------------------------------------------
# Geometry: [0, 1]² with 4 circular holes
# ---------------------------------------------------------------------------


HOLES = [
    ((0.26, 0.28), 0.11),
    ((0.72, 0.22), 0.09),
    ((0.24, 0.74), 0.14),
    ((0.72, 0.72), 0.12),
]


def in_domain(x: np.ndarray, y: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """True iff (x, y) ∈ Ω (optionally excluding a `pad`-thick shell near ∂Ω)."""
    inside = (x >= pad) & (x <= 1.0 - pad) & (y >= pad) & (y <= 1.0 - pad)
    for (cx, cy), r in HOLES:
        inside = inside & ((x - cx) ** 2 + (y - cy) ** 2 >= (r + pad) ** 2)
    return inside


def sample_interior(n: int, rng: np.random.Generator, pad: float = 0.005) -> np.ndarray:
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform(0, 1, (3 * n, 2))
        pts = np.vstack([pts, batch[in_domain(batch[:, 0], batch[:, 1], pad=pad)]])
    return pts[:n]


def sample_outer_boundary(per_side: int) -> np.ndarray:
    t = np.linspace(0, 1, per_side, endpoint=False)
    return np.concatenate([
        np.stack([t,              np.zeros_like(t)],  axis=1),  # bottom
        np.stack([np.ones_like(t), t],                 axis=1),  # right
        np.stack([1.0 - t,         np.ones_like(t)],   axis=1),  # top
        np.stack([np.zeros_like(t), 1.0 - t],           axis=1),  # left
    ], axis=0)


def sample_hole_boundaries(per_hole: int) -> np.ndarray:
    parts = []
    for (cx, cy), r in HOLES:
        theta = np.linspace(0, 2 * np.pi, per_hole, endpoint=False)
        parts.append(np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1))
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Manufactured solution + forcing
# ---------------------------------------------------------------------------


def ground_truth(alpha: float = 1.0, m: int = 3, freq: float = 2.0):
    w = freq * np.pi

    def u(x):
        return float(np.cos(w * x[0]) * np.cos(w * x[1]))

    def rhs(x):
        # -Δu = 2 w² cos(w x) cos(w y) = 2 w² u
        return float(2.0 * w * w * u(x) + alpha * u(x) ** m)

    return u, rhs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _draw_domain_outline(ax):
    # outer square
    ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k-', lw=2.0)
    # holes
    theta = np.linspace(0, 2 * np.pi, 200)
    for (cx, cy), r in HOLES:
        ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), 'k-', lw=1.6)


def plot(X_dom, X_bdy, truth, sol, out: pathlib.Path):
    err = np.abs(truth - sol)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)

    def frame(ax):
        _draw_domain_outline(ax)
        ax.set_aspect('equal')
        ax.set_xlim(-0.04, 1.04); ax.set_ylim(-0.04, 1.04)
        ax.set_xlabel('x'); ax.set_ylabel('y')

    # Panel 1: sample points
    ax = axes[0]
    ax.scatter(X_dom[:, 0], X_dom[:, 1], s=5, c='tab:blue', alpha=0.7,
               label=f'interior ({X_dom.shape[0]})')
    ax.scatter(X_bdy[:, 0], X_bdy[:, 1], s=12, c='tab:red',
               label=f'boundary ({X_bdy.shape[0]})')
    frame(ax)
    ax.set_title('sample points', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)

    # Panel 2: numerical solution
    ax = axes[1]
    lo, hi = truth.min(), truth.max()
    sc = ax.scatter(X_dom[:, 0], X_dom[:, 1], c=sol, s=10, cmap='viridis',
                    vmin=lo, vmax=hi)
    frame(ax)
    ax.set_title('numerical solution $u_h$', fontsize=12)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    # Panel 3: error
    ax = axes[2]
    sc = ax.scatter(X_dom[:, 0], X_dom[:, 1], c=err, s=10, cmap='inferno')
    frame(ax)
    L2 = float(np.sqrt(np.mean(err ** 2)))
    Linf = float(err.max())
    ax.set_title(f'|$u_h$ − $u$|     L²={L2:.1e},  L∞={Linf:.1e}', fontsize=12)
    fig.colorbar(sc, ax=ax, shrink=0.85)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--N-interior', type=int, default=4000)
    p.add_argument('--bdy-outer', type=int, default=80, help='per outer edge')
    p.add_argument('--bdy-hole',  type=int, default=80, help='per inner circle')
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
    p.add_argument('--out', default='docs/swiss_cheese.png')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy_outer = sample_outer_boundary(args.bdy_outer)
    X_bdy_holes = sample_hole_boundaries(args.bdy_hole)
    X_bdy = np.concatenate([X_bdy_outer, X_bdy_holes], axis=0)
    print(f'[points] interior = {X_dom.shape[0]}  boundary = {X_bdy.shape[0]} '
          f'(outer {X_bdy_outer.shape[0]} + holes {X_bdy_holes.shape[0]})')

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
        'Gaussian':    kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel]  {args.kernel}, length_scale = {args.sigma}')

    u_exact, rhs_fn = ground_truth(args.alpha, args.m, freq=2.0)
    eqn = NonlinElliptic2d(alpha=args.alpha, m=args.m,
                            domain=((0.0, 1.0), (0.0, 1.0)),
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

    plot(X_dom, X_bdy, truth, sol, pathlib.Path(args.out))
    print(f'[figure]  saved → {args.out}')


if __name__ == '__main__':
    main()
