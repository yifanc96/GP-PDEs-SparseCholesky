"""Nonlinear elliptic PDE on a stadium (pill) domain.

Domain:  Ω = rectangle [0, L] × [−R, R]  union two semicircles of radius R
         capping each end. Mix of flat and curved boundary pieces.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


L = 1.2
R = 0.3


def in_stadium(x, y):
    inside_rect = (x >= 0) & (x <= L) & (np.abs(y) <= R)
    inside_left = (x < 0) & (x * x + y * y <= R * R)
    inside_right = (x > L) & ((x - L) ** 2 + y * y <= R * R)
    return inside_rect | inside_left | inside_right


def sample_interior(n, rng):
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform([-R - 0.02, -R - 0.02], [L + R + 0.02, R + 0.02], (3 * n, 2))
        keep = in_stadium(batch[:, 0], batch[:, 1])
        # inward pad
        dx_r, dy_r = batch[:, 0] - L, batch[:, 1]
        dx_l, dy_l = batch[:, 0], batch[:, 1]
        pad_rect  = (batch[:, 0] > 0.005) & (batch[:, 0] < L - 0.005) & (np.abs(batch[:, 1]) < R - 0.005)
        pad_left  = (batch[:, 0] < 0.005) & (dx_l * dx_l + dy_l * dy_l < (R - 0.005) ** 2)
        pad_right = (batch[:, 0] > L - 0.005) & (dx_r * dx_r + dy_r * dy_r < (R - 0.005) ** 2)
        keep &= (pad_rect | pad_left | pad_right)
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def boundary_points(per_side, per_cap):
    top = np.stack([np.linspace(0, L, per_side, endpoint=False),
                    R * np.ones(per_side)], axis=1)
    bot = np.stack([np.linspace(L, 0, per_side, endpoint=False),
                    -R * np.ones(per_side)], axis=1)
    th_right = np.linspace(-np.pi / 2, np.pi / 2, per_cap, endpoint=False)
    right = np.stack([L + R * np.cos(th_right), R * np.sin(th_right)], axis=1)
    th_left = np.linspace(np.pi / 2, 3 * np.pi / 2, per_cap, endpoint=False)
    left = np.stack([R * np.cos(th_left), R * np.sin(th_left)], axis=1)
    return np.concatenate([top, right, bot, left], axis=0)


def outline_polylines():
    # Walk the boundary counter-clockwise starting at (0, R):
    #   top segment  -> right cap  -> bottom segment  -> left cap  -> back to start.
    top   = np.stack([np.linspace(0, L, 100, endpoint=False), R * np.ones(100)], axis=1)
    th_r  = np.linspace(np.pi / 2, -np.pi / 2, 100, endpoint=False)
    right = np.stack([L + R * np.cos(th_r), R * np.sin(th_r)], axis=1)
    bot   = np.stack([np.linspace(L, 0, 100, endpoint=False), -R * np.ones(100)], axis=1)
    th_l  = np.linspace(-np.pi / 2, -3 * np.pi / 2, 100, endpoint=False)
    left  = np.stack([R * np.cos(th_l), R * np.sin(th_l)], axis=1)
    return [np.concatenate([top, right, bot, left, top[:1]], axis=0)]


def u_exact(x):
    return float(np.sin(np.pi * x[0] / L) * np.cos(np.pi * x[1] / (2 * R)))


def rhs(x):
    d2_x = (np.pi / L) ** 2 * u_exact(x)
    d2_y = (np.pi / (2 * R)) ** 2 * u_exact(x)
    return float(d2_x + d2_y + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-side', type=int, default=120)
    parser.add_argument('--bdy-cap',  type=int, default=100)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = boundary_points(args.bdy_side, args.bdy_cap)

    demo = GeometryDemo(
        name='stadium', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polylines(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
