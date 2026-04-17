"""Nonlinear elliptic PDE on a solid torus (3D).

Domain:   Ω = { (√(x²+y²) − R)² + z² ≤ r² }   with R = 0.6, r = 0.22.
Problem:  -Δu + u³ = f          in Ω,     u = g on ∂Ω.
Manufactured solution:  u(x, y, z) = cos(2πx)·cos(2πy)·cos(2πz).

This is the simplest "canonical" 3D geometry — iconic, multiply-connected,
and nothing special is needed to solve on it beyond sampling points inside
the torus and on its surface.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


R = 0.6        # major radius
r_TUBE = 0.22  # minor radius


def in_torus(x, y, z, pad: float = 0.0):
    rho = np.sqrt(x * x + y * y)
    return (rho - R) ** 2 + z * z <= (r_TUBE - pad) ** 2


def sample_interior(n, rng):
    pts = np.empty((0, 3))
    while pts.shape[0] < n:
        batch = rng.uniform([-R - r_TUBE - 0.05, -R - r_TUBE - 0.05, -r_TUBE - 0.05],
                            [ R + r_TUBE + 0.05,  R + r_TUBE + 0.05,  r_TUBE + 0.05],
                            (5 * n, 3))
        keep = in_torus(batch[:, 0], batch[:, 1], batch[:, 2], pad=0.01)
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def torus_boundary(n_theta, n_phi):
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    phi   = np.linspace(0, 2 * np.pi, n_phi,   endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = (R + r_TUBE * np.cos(PH)) * np.cos(TH)
    y = (R + r_TUBE * np.cos(PH)) * np.sin(TH)
    z = r_TUBE * np.sin(PH)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def torus_outline():
    # Two circles: outer (r = R + r_TUBE at z=0) and inner (r = R − r_TUBE at z=0)
    # plus a meridian circle at θ = 0.
    th = np.linspace(0, 2 * np.pi, 200)
    outer = np.stack([(R + r_TUBE) * np.cos(th), (R + r_TUBE) * np.sin(th), np.zeros_like(th)], axis=1)
    inner = np.stack([(R - r_TUBE) * np.cos(th), (R - r_TUBE) * np.sin(th), np.zeros_like(th)], axis=1)
    meridian = np.stack([R + r_TUBE * np.cos(th), np.zeros_like(th), r_TUBE * np.sin(th)], axis=1)
    return [outer, inner, meridian]


def u_exact(x):
    return float(np.cos(2 * np.pi * x[0]) * np.cos(2 * np.pi * x[1]) * np.cos(2 * np.pi * x[2]))


def rhs(x):
    lap_neg = 12 * np.pi ** 2 * u_exact(x)
    return float(lap_neg + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--theta', type=int, default=40)
    parser.add_argument('--phi',   type=int, default=16)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = torus_boundary(args.theta, args.phi)

    demo = GeometryDemo(
        name='torus', X_domain=X_dom, X_boundary=X_bdy,
        outline=torus_outline(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
