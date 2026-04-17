"""3D Swiss-cheese cube: [0,1]³ minus several spherical holes.

Domain:   Ω = [0, 1]³ \ ⋃ᵢ B(cᵢ, rᵢ)   (6 non-overlapping random balls)
Problem:  -Δu + u³ = f,  u = g on ∂Ω    (both outer faces and hole surfaces).
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


N_HOLES = 6
R_MIN, R_MAX = 0.10, 0.17


def make_holes(seed: int):
    rng = np.random.default_rng(seed)
    centres, radii = [], []
    trials = 0
    while len(centres) < N_HOLES and trials < 20000:
        trials += 1
        r = rng.uniform(R_MIN, R_MAX)
        c = rng.uniform(r + 0.05, 1 - r - 0.05, 3)
        ok = all(np.linalg.norm(c - cj) > r + rj + 0.03 for cj, rj in zip(centres, radii))
        if ok:
            centres.append(c); radii.append(r)
    return np.array(centres), np.array(radii)


def in_domain(x, y, z, centres, radii, pad: float = 0.0):
    inside = (x >= pad) & (x <= 1 - pad) & (y >= pad) & (y <= 1 - pad) & (z >= pad) & (z <= 1 - pad)
    for c, r in zip(centres, radii):
        inside &= (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2 >= (r + pad) ** 2
    return inside


def sample_interior(n, centres, radii, rng):
    pts = np.empty((0, 3))
    while pts.shape[0] < n:
        batch = rng.uniform(0, 1, (5 * n, 3))
        pts = np.vstack([pts, batch[in_domain(batch[:, 0], batch[:, 1], batch[:, 2],
                                              centres, radii, pad=0.005)]])
    return pts[:n]


def cube_boundary(per_face):
    t = np.linspace(0, 1, per_face, endpoint=False)
    g1, g2 = np.meshgrid(t, t, indexing='ij')
    g1, g2 = g1.ravel(), g2.ravel()
    zeros = np.zeros_like(g1); ones = np.ones_like(g1)
    faces = [
        np.stack([g1, g2, zeros], axis=1),   # z = 0
        np.stack([g1, g2, ones ], axis=1),   # z = 1
        np.stack([g1, zeros, g2], axis=1),   # y = 0
        np.stack([g1, ones,  g2], axis=1),   # y = 1
        np.stack([zeros, g1, g2], axis=1),   # x = 0
        np.stack([ones,  g1, g2], axis=1),   # x = 1
    ]
    return np.concatenate(faces, axis=0)


def sphere_boundary(centre, radius, n_theta, n_phi):
    theta = np.linspace(0, np.pi, n_phi)[1:-1]        # latitude
    phi   = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    x = centre[0] + radius * np.sin(TH) * np.cos(PH)
    y = centre[1] + radius * np.sin(TH) * np.sin(PH)
    z = centre[2] + radius * np.cos(TH)
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)


def cube_outline():
    edges = [
        [[0,0,0],[1,0,0]], [[1,0,0],[1,1,0]], [[1,1,0],[0,1,0]], [[0,1,0],[0,0,0]],
        [[0,0,1],[1,0,1]], [[1,0,1],[1,1,1]], [[1,1,1],[0,1,1]], [[0,1,1],[0,0,1]],
        [[0,0,0],[0,0,1]], [[1,0,0],[1,0,1]], [[1,1,0],[1,1,1]], [[0,1,0],[0,1,1]],
    ]
    return [np.array(e) for e in edges]


def sphere_outline(centre, radius):
    th = np.linspace(0, 2 * np.pi, 60)
    c_xy = np.stack([centre[0] + radius * np.cos(th), centre[1] + radius * np.sin(th),
                     np.full_like(th, centre[2])], axis=1)
    c_xz = np.stack([centre[0] + radius * np.cos(th), np.full_like(th, centre[1]),
                     centre[2] + radius * np.sin(th)], axis=1)
    c_yz = np.stack([np.full_like(th, centre[0]), centre[1] + radius * np.cos(th),
                     centre[2] + radius * np.sin(th)], axis=1)
    return [c_xy, c_xz, c_yz]


def u_exact(x):
    return float(np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2]))


def rhs(x):
    return float(3 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-face', type=int, default=16)
    parser.add_argument('--bdy-theta', type=int, default=20)
    parser.add_argument('--bdy-phi',   type=int, default=10)
    args = parser.parse_args()

    centres, radii = make_holes(args.seed)
    print(f'[holes]   {len(centres)} spheres of varying radius')

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, centres, radii, rng)
    X_bdy = np.concatenate([
        cube_boundary(args.bdy_face),
        *[sphere_boundary(c, r, args.bdy_theta, args.bdy_phi) for c, r in zip(centres, radii)],
    ], axis=0)

    outline = cube_outline()
    for c, r in zip(centres, radii):
        outline.extend(sphere_outline(c, r))

    demo = GeometryDemo(
        name='swiss_cheese_cube', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline, u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
