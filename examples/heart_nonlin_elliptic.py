"""Nonlinear elliptic PDE on a heart-shaped domain.

Parametric heart curve:
    x(t) = 16 sin³ t
    y(t) = 13 cos t − 5 cos 2t − 2 cos 3t − cos 4t
(Then scaled/shifted so it fits comfortably in a bounding box.) Demonstrates
how any parametric / SVG / CAD contour plugs into the collocation solver
without changing anything else.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.path as mpath

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


def _heart_curve(n: int):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    # scale + shift into a friendly box
    x = x / 16.0
    y = (y + 5) / 20.0   # rough centering; bbox ~ [-1, 1] x [-0.5, 0.9]
    return np.stack([x, y], axis=1)


def sample_interior(n, rng, polygon: np.ndarray, inward_pad: float = 0.02):
    path = mpath.Path(polygon)
    pts = np.empty((0, 2))
    xmin, ymin = polygon.min(axis=0) - 0.05
    xmax, ymax = polygon.max(axis=0) + 0.05
    # A kd-tree over boundary vertices to enforce an inward pad
    tree = cKDTree(polygon)
    while pts.shape[0] < n:
        batch = rng.uniform([xmin, ymin], [xmax, ymax], (4 * n, 2))
        inside = path.contains_points(batch)
        d, _ = tree.query(batch)
        keep = inside & (d > inward_pad)
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def u_exact(x):
    return float(np.exp(-(x[0] ** 2 + (x[1] - 0.2) ** 2)) * np.cos(2 * x[0]) * np.cos(2 * x[1]))


def _laplacian_u_numerical(x, eps=1e-4):
    ex = np.array([eps, 0.0]); ey = np.array([0.0, eps])
    return (u_exact(x + ex) + u_exact(x - ex)
            + u_exact(x + ey) + u_exact(x - ey)
            - 4 * u_exact(x)) / (eps * eps)


def rhs(x):
    return float(-_laplacian_u_numerical(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy', type=int, default=500)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_bdy = _heart_curve(args.bdy)
    X_dom = sample_interior(args.N_interior, rng, polygon=X_bdy)

    demo = GeometryDemo(
        name='heart', X_domain=X_dom, X_boundary=X_bdy,
        outline=[np.vstack([X_bdy, X_bdy[:1]])],
        u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
