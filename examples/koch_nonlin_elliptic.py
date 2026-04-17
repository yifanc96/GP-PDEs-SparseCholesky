"""Nonlinear elliptic PDE on a Koch snowflake.

Domain:  Ω enclosed by an n-th-level Koch snowflake (self-similar fractal
         polygon). Visual shock value — illustrates that even nominally
         non-rectifiable boundaries (at the limit) are fine at any finite
         discretization level.
"""

from __future__ import annotations

import numpy as np
import matplotlib.path as mpath
from scipy.spatial import cKDTree

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


def koch_level(level: int, side: float = 1.0, centre=(0.5, 0.4)):
    # Equilateral triangle vertices, ordered counter-clockwise so
    # the right-hand normal points outward.
    cx, cy = centre
    r = side / np.sqrt(3)
    pts = np.array([
        [cx,                 cy + r],      # top
        [cx - 0.5 * side,    cy - 0.5 * r],# bottom-left
        [cx + 0.5 * side,    cy - 0.5 * r],# bottom-right
    ])
    for _ in range(level):
        new_pts = []
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            ab = b - a
            p1 = a + ab / 3.0
            p2 = a + 2 * ab / 3.0
            mid = (p1 + p2) / 2.0
            # Right-hand normal of a CCW polygon is outward.
            perp = np.array([ab[1], -ab[0]])
            perp = perp / np.linalg.norm(perp) * np.linalg.norm(ab) / (2 * np.sqrt(3))
            tip = mid + perp
            new_pts.extend([a, p1, tip, p2])
        pts = np.array(new_pts)
    return pts


def sample_interior(n, rng, polygon, inward_pad: float = 0.01):
    path = mpath.Path(polygon)
    tree = cKDTree(polygon)
    xmin, ymin = polygon.min(axis=0) - 0.02
    xmax, ymax = polygon.max(axis=0) + 0.02
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform([xmin, ymin], [xmax, ymax], (4 * n, 2))
        inside = path.contains_points(batch)
        d, _ = tree.query(batch)
        pts = np.vstack([pts, batch[inside & (d > inward_pad)]])
    return pts[:n]


def u_exact(x):
    return float(np.sin(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]))


def _laplacian(x, eps=1e-4):
    ex = np.array([eps, 0.0]); ey = np.array([0.0, eps])
    return (u_exact(x + ex) + u_exact(x - ex)
            + u_exact(x + ey) + u_exact(x - ey)
            - 4 * u_exact(x)) / (eps * eps)


def rhs(x):
    return float(-_laplacian(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--level', type=int, default=4)
    args = parser.parse_args()

    poly = koch_level(args.level, side=0.9, centre=(0.5, 0.4))
    print(f'[koch]    level={args.level}, boundary vertices = {poly.shape[0]}')

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng, poly)
    X_bdy = poly.copy()

    demo = GeometryDemo(
        name='koch', X_domain=X_dom, X_boundary=X_bdy,
        outline=[np.vstack([poly, poly[:1]])],
        u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
