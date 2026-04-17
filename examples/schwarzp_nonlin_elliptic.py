"""Nonlinear elliptic PDE on a Schwarz-P (TPMS) solid.

Triply-periodic minimal surface defined implicitly by
    f(x, y, z) = cos(ω x) + cos(ω y) + cos(ω z)
The surface f = 0 partitions ℝ³ into two interpenetrating regions.
We take one of them (f ≥ 0) inside a cube [−L, L]³ — this gives a
multiply-connected solid with a non-trivial topology that nonetheless
has a simple implicit boundary representation.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


L = 1.0
OMEGA = 2 * np.pi / L        # one full period inside [-L, L]


def schwarz_p(x, y, z):
    return np.cos(OMEGA * x) + np.cos(OMEGA * y) + np.cos(OMEGA * z)


def in_domain(x, y, z, pad: float = 0.0):
    in_cube = (np.abs(x) <= L - pad) & (np.abs(y) <= L - pad) & (np.abs(z) <= L - pad)
    return in_cube & (schwarz_p(x, y, z) >= 0.0 + pad * 3)


def sample_interior(n, rng):
    pts = np.empty((0, 3))
    while pts.shape[0] < n:
        batch = rng.uniform(-L, L, (8 * n, 3))
        pts = np.vstack([pts, batch[in_domain(batch[:, 0], batch[:, 1], batch[:, 2], pad=0.01)]])
    return pts[:n]


def _surface_points(n_total, rng):
    """Sample points on the implicit surface f = 0 by projection: pick
    random points in the cube, then Newton-step along ∇f until |f| is small.
    """
    pts = rng.uniform(-L, L, (4 * n_total, 3))
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    for _ in range(6):
        f  = schwarz_p(x, y, z)
        gx = -OMEGA * np.sin(OMEGA * x)
        gy = -OMEGA * np.sin(OMEGA * y)
        gz = -OMEGA * np.sin(OMEGA * z)
        g2 = gx * gx + gy * gy + gz * gz + 1e-12
        step = f / g2
        x = x - step * gx
        y = y - step * gy
        z = z - step * gz
    keep = (np.abs(schwarz_p(x, y, z)) < 0.03) & \
           (np.abs(x) <= L) & (np.abs(y) <= L) & (np.abs(z) <= L)
    xyz = np.stack([x[keep], y[keep], z[keep]], axis=1)
    return xyz[:n_total]


def cube_face_boundary(per_face):
    t = np.linspace(-L, L, per_face, endpoint=False)
    g1, g2 = np.meshgrid(t, t, indexing='ij')
    g1, g2 = g1.ravel(), g2.ravel()
    faces = []
    for sgn in (-1.0, 1.0):
        faces.append(np.stack([g1, g2, np.full_like(g1, sgn * L)], axis=1))
        faces.append(np.stack([g1, np.full_like(g1, sgn * L), g2], axis=1))
        faces.append(np.stack([np.full_like(g1, sgn * L), g1, g2], axis=1))
    all_faces = np.concatenate(faces, axis=0)
    # keep only face points that are inside Ω (f ≥ 0)
    return all_faces[schwarz_p(all_faces[:, 0], all_faces[:, 1], all_faces[:, 2]) >= 0.0]


def cube_outline():
    edges = [
        [[-L,-L,-L],[ L,-L,-L]], [[ L,-L,-L],[ L, L,-L]],
        [[ L, L,-L],[-L, L,-L]], [[-L, L,-L],[-L,-L,-L]],
        [[-L,-L, L],[ L,-L, L]], [[ L,-L, L],[ L, L, L]],
        [[ L, L, L],[-L, L, L]], [[-L, L, L],[-L,-L, L]],
        [[-L,-L,-L],[-L,-L, L]], [[ L,-L,-L],[ L,-L, L]],
        [[ L, L,-L],[ L, L, L]], [[-L, L,-L],[-L, L, L]],
    ]
    return [np.array(e) for e in edges]


def u_exact(x):
    return float(np.sin(np.pi * x[0] / L) * np.sin(np.pi * x[1] / L) * np.sin(np.pi * x[2] / L))


def rhs(x):
    return float(3 * (np.pi / L) ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-surface', type=int, default=1500)
    parser.add_argument('--bdy-face',    type=int, default=14)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_surface = _surface_points(args.bdy_surface, rng)
    X_faces = cube_face_boundary(args.bdy_face)
    X_bdy = np.concatenate([X_surface, X_faces], axis=0)

    demo = GeometryDemo(
        name='schwarzp', X_domain=X_dom, X_boundary=X_bdy,
        outline=cube_outline(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
