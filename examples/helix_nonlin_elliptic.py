"""Nonlinear elliptic PDE on a thick helical tube.

Domain:   sweep of a disk of radius r_tube along a helix of major radius
          R and pitch `rise` over `n_turns` turns. Narrow spiral channels
          test the method on curved, elongated geometries.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


R_HELIX = 0.35    # major radius (winding)
RISE    = 0.22    # axial distance per full turn
N_TURNS = 2.5
R_TUBE  = 0.10    # tube radius


def helix_centerline(n):
    t = np.linspace(0, 2 * np.pi * N_TURNS, n)
    return np.stack([R_HELIX * np.cos(t), R_HELIX * np.sin(t), RISE * t / (2 * np.pi)], axis=1)


def build_tree(n_samples=2000):
    centers = helix_centerline(n_samples)
    tree = cKDTree(centers)
    return tree, centers


def sample_interior(n, rng, tree):
    pts = np.empty((0, 3))
    z_lo, z_hi = 0.0, RISE * N_TURNS
    xy_lo, xy_hi = -R_HELIX - R_TUBE - 0.05, R_HELIX + R_TUBE + 0.05
    while pts.shape[0] < n:
        batch = rng.uniform([xy_lo, xy_lo, z_lo - R_TUBE - 0.02],
                            [xy_hi, xy_hi, z_hi + R_TUBE + 0.02], (4 * n, 3))
        d, _ = tree.query(batch)
        pts = np.vstack([pts, batch[d < R_TUBE - 0.005]])
    return pts[:n]


def sample_boundary(n_along, n_around):
    # A Frenet-ish frame: tangent, plus two unit normals orthogonal to it.
    t_vals = np.linspace(0, 2 * np.pi * N_TURNS, n_along, endpoint=False)
    c  = np.stack([R_HELIX * np.cos(t_vals), R_HELIX * np.sin(t_vals), RISE * t_vals / (2 * np.pi)], axis=1)
    dc = np.stack([-R_HELIX * np.sin(t_vals), R_HELIX * np.cos(t_vals),
                    np.full_like(t_vals, RISE / (2 * np.pi))], axis=1)
    dc /= np.linalg.norm(dc, axis=1, keepdims=True)
    # Choose a fixed "up" and project out the tangent component
    up = np.array([0.0, 0.0, 1.0])
    n1 = up - (dc @ up)[:, None] * dc
    n1 /= np.linalg.norm(n1, axis=1, keepdims=True)
    n2 = np.cross(dc, n1)
    phi = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)
    # (n_along, n_around, 3)
    ring = (R_TUBE * (cos_phi[None, :, None] * n1[:, None, :]
                     + sin_phi[None, :, None] * n2[:, None, :]))
    return (c[:, None, :] + ring).reshape(-1, 3)


def outline_polyline(n=400):
    return [helix_centerline(n)]


def u_exact(x):
    return float(np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]))


def rhs(x):
    return float(12 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-along',  type=int, default=120)
    parser.add_argument('--bdy-around', type=int, default=14)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tree, _ = build_tree()
    X_dom = sample_interior(args.N_interior, rng, tree)
    X_bdy = sample_boundary(args.bdy_along, args.bdy_around)

    demo = GeometryDemo(
        name='helix', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polyline(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
