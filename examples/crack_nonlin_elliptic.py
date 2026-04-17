"""Nonlinear elliptic PDE on a cracked square — a horizontal slit domain.

Domain:  Ω = [0, 1]² with a horizontal slit from (0.2, 0.5) to (0.7, 0.5).
The slit is zero-thickness; we model it by placing boundary points on
both its top and bottom faces. With a smooth manufactured solution
(not the classical √r corner singularity), this shows that the method
handles geometrically-complex open cuts without special treatment.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


CRACK_X0, CRACK_X1 = 0.2, 0.7
CRACK_Y = 0.5
SLIT_EPS = 0.002   # visual half-thickness for plotting and sampling exclusion


def in_domain(x, y, pad: float = 0.0):
    inside = (x >= pad) & (x <= 1 - pad) & (y >= pad) & (y <= 1 - pad)
    on_crack = (x >= CRACK_X0 - pad) & (x <= CRACK_X1 + pad) \
                & (np.abs(y - CRACK_Y) < SLIT_EPS + pad)
    return inside & ~on_crack


def sample_interior(n, rng):
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform(0, 1, (3 * n, 2))
        pts = np.vstack([pts, batch[in_domain(batch[:, 0], batch[:, 1], pad=0.005)]])
    return pts[:n]


def outer_boundary(per_side):
    t = np.linspace(0, 1, per_side, endpoint=False)
    return np.concatenate([
        np.stack([t, np.zeros_like(t)], axis=1),
        np.stack([np.ones_like(t), t], axis=1),
        np.stack([1 - t, np.ones_like(t)], axis=1),
        np.stack([np.zeros_like(t), 1 - t], axis=1),
    ], axis=0)


def crack_boundary(n):
    s = np.linspace(CRACK_X0, CRACK_X1, n // 2, endpoint=False)
    top = np.stack([s, np.full_like(s, CRACK_Y + SLIT_EPS)], axis=1)
    bot = np.stack([s, np.full_like(s, CRACK_Y - SLIT_EPS)], axis=1)
    return np.concatenate([top, bot], axis=0)


def outline_polylines():
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    crack = np.array([
        [CRACK_X0, CRACK_Y + SLIT_EPS],
        [CRACK_X1, CRACK_Y + SLIT_EPS],
        [CRACK_X1, CRACK_Y - SLIT_EPS],
        [CRACK_X0, CRACK_Y - SLIT_EPS],
        [CRACK_X0, CRACK_Y + SLIT_EPS],
    ])
    return [box, crack]


def u_exact(x):
    # Smooth manufactured solution (avoids the √r singularity at crack tips).
    return float(np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


def rhs(x):
    return float(2 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-outer', type=int, default=80)
    parser.add_argument('--bdy-crack', type=int, default=200)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = np.concatenate([
        outer_boundary(args.bdy_outer),
        crack_boundary(args.bdy_crack),
    ], axis=0)

    demo = GeometryDemo(
        name='crack', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polylines(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
