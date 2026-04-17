"""Nonlinear elliptic PDE on a flower (gear) domain.

Domain:  Ω enclosed by r(θ) = r₀ + a·cos(k θ), centred at (0.5, 0.5).
         Smooth non-convex boundary with k petals — the outer contour
         oscillates between r₀ − a and r₀ + a.

No corners, no holes — the accuracy story is clean. Demonstrates that
the solver handles smoothly-varying non-convex curves.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


CENTRE = (0.5, 0.5)
R0 = 0.32
AMP = 0.08
K = 6


def r_of_theta(theta):
    return R0 + AMP * np.cos(K * theta)


def in_flower(x, y):
    dx, dy = x - CENTRE[0], y - CENTRE[1]
    r = np.sqrt(dx * dx + dy * dy)
    theta = np.arctan2(dy, dx)
    return r <= r_of_theta(theta)


def sample_interior(n, rng):
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform(CENTRE[0] - R0 - AMP - 0.02,
                            CENTRE[0] + R0 + AMP + 0.02, (3 * n, 2))
        keep = in_flower(batch[:, 0], batch[:, 1])
        # small inward pad
        dx, dy = batch[:, 0] - CENTRE[0], batch[:, 1] - CENTRE[1]
        r = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)
        keep &= (r < r_of_theta(theta) - 0.005)
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def boundary_points(n):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = r_of_theta(theta)
    return np.stack([CENTRE[0] + r * np.cos(theta),
                     CENTRE[1] + r * np.sin(theta)], axis=1)


def u_exact(x):
    return float(np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))


def rhs(x):
    return float(8 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy', type=int, default=400)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = boundary_points(args.bdy)

    theta_dense = np.linspace(0, 2 * np.pi, 800)
    r_dense = r_of_theta(theta_dense)
    outline = [np.stack([CENTRE[0] + r_dense * np.cos(theta_dense),
                         CENTRE[1] + r_dense * np.sin(theta_dense)], axis=1)]

    demo = GeometryDemo(
        name='flower', X_domain=X_dom, X_boundary=X_bdy, outline=outline,
        u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
