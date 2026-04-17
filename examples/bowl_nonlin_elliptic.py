"""Nonlinear elliptic PDE on a bowl / dimpled sphere.

Domain:  Ω = B₁ \ B₂ where
         B₁ is a ball of radius R_out centred at origin,
         B₂ is a ball of radius R_in  centred at (0, 0, z_shift) with
         R_in < R_out and z_shift > 0 so the inner ball pokes out the top.
Result is a "bowl" shape — a solid ball with a spherical cavity carved
from above.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


R_OUT = 0.50
R_IN  = 0.40
Z_SHIFT = 0.25


def in_bowl(x, y, z, pad: float = 0.0):
    rr2 = x * x + y * y + z * z
    in_outer = rr2 <= (R_OUT - pad) ** 2
    rr2_in = x * x + y * y + (z - Z_SHIFT) ** 2
    in_inner = rr2_in <= (R_IN + pad) ** 2
    return in_outer & ~in_inner


def sample_interior(n, rng):
    pts = np.empty((0, 3))
    while pts.shape[0] < n:
        batch = rng.uniform(-R_OUT - 0.02, R_OUT + 0.02, (6 * n, 3))
        keep = in_bowl(batch[:, 0], batch[:, 1], batch[:, 2], pad=0.01)
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def outer_sphere_boundary(n_theta, n_phi):
    # Sample the part of the outer sphere that is in Ω (i.e. not covered by B₂).
    theta = np.linspace(0, np.pi, n_phi)[1:-1]
    phi   = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = R_OUT * np.sin(TH) * np.cos(PH)
    y = R_OUT * np.sin(TH) * np.sin(PH)
    z = R_OUT * np.cos(TH)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    # discard points that lie inside the cavity
    keep = (pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] - Z_SHIFT) ** 2) >= R_IN ** 2
    return pts[keep]


def inner_cavity_boundary(n_theta, n_phi):
    theta = np.linspace(0, np.pi, n_phi)[1:-1]
    phi   = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = R_IN * np.sin(TH) * np.cos(PH)
    y = R_IN * np.sin(TH) * np.sin(PH)
    z = Z_SHIFT + R_IN * np.cos(TH)
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    # keep only those that lie inside B₁ (otherwise they'd be outside Ω)
    keep = (pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2) <= R_OUT ** 2
    return pts[keep]


def outline_polylines():
    th = np.linspace(0, 2 * np.pi, 200)
    o_xy = np.stack([R_OUT * np.cos(th), R_OUT * np.sin(th), np.zeros_like(th)], axis=1)
    o_xz = np.stack([R_OUT * np.cos(th), np.zeros_like(th), R_OUT * np.sin(th)], axis=1)
    i_xy = np.stack([R_IN  * np.cos(th), R_IN  * np.sin(th), np.full_like(th, Z_SHIFT)], axis=1)
    i_xz = np.stack([R_IN  * np.cos(th), np.zeros_like(th), Z_SHIFT + R_IN * np.sin(th)], axis=1)
    return [o_xy, o_xz, i_xy, i_xz]


def u_exact(x):
    return float(np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]))


def rhs(x):
    return float(12 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--theta', type=int, default=36)
    parser.add_argument('--phi',   type=int, default=18)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = np.concatenate([
        outer_sphere_boundary(args.theta, args.phi),
        inner_cavity_boundary(args.theta, args.phi),
    ], axis=0)

    demo = GeometryDemo(
        name='bowl', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polylines(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
