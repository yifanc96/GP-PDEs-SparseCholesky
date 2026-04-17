"""Nonlinear elliptic PDE on a porous medium — many circular inclusions.

Domain:  Ω = [0, 1]² \ ⋃ᵢ B(cᵢ, rᵢ)  with 40 non-overlapping random holes
         of varying radii. The point of this example is that meshing
         forty curved inner boundaries conforming to each hole is
         expensive; meshless collocation just takes more sample points
         and changes nothing else.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


N_HOLES = 40
R_MIN, R_MAX = 0.015, 0.050
PAD = 0.01


def make_holes(seed: int = 0):
    rng = np.random.default_rng(seed)
    centres, radii = [], []
    trials = 0
    while len(centres) < N_HOLES and trials < 20000:
        trials += 1
        r = rng.uniform(R_MIN, R_MAX)
        c = rng.uniform(r + PAD + 0.02, 1 - r - PAD - 0.02, 2)
        ok = True
        for (cj, rj) in zip(centres, radii):
            if np.hypot(c[0] - cj[0], c[1] - cj[1]) < r + rj + 2 * PAD:
                ok = False; break
        if ok:
            centres.append(c); radii.append(r)
    return np.array(centres), np.array(radii)


def in_domain(x, y, centres, radii, pad: float = 0.0):
    inside = (x >= pad) & (x <= 1 - pad) & (y >= pad) & (y <= 1 - pad)
    for (cx, cy), r in zip(centres, radii):
        inside &= (x - cx) ** 2 + (y - cy) ** 2 >= (r + pad) ** 2
    return inside


def sample_interior(n, centres, radii, rng):
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform(0, 1, (5 * n, 2))
        pts = np.vstack([pts, batch[in_domain(batch[:, 0], batch[:, 1], centres, radii, pad=0.005)]])
    return pts[:n]


def outer_boundary(per_side):
    t = np.linspace(0, 1, per_side, endpoint=False)
    return np.concatenate([
        np.stack([t, np.zeros_like(t)], axis=1),
        np.stack([np.ones_like(t), t], axis=1),
        np.stack([1 - t, np.ones_like(t)], axis=1),
        np.stack([np.zeros_like(t), 1 - t], axis=1),
    ], axis=0)


def hole_boundaries(centres, radii, per_hole):
    theta = np.linspace(0, 2 * np.pi, per_hole, endpoint=False)
    return np.concatenate([
        np.stack([cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=1)
        for (cx, cy), r in zip(centres, radii)
    ], axis=0)


def outline_polylines(centres, radii):
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    th = np.linspace(0, 2 * np.pi, 128)
    circles = [
        np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1)
        for (cx, cy), r in zip(centres, radii)
    ]
    return [box, *circles]


def u_exact(x):
    return float(np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))


def rhs(x):
    return float(2 * np.pi ** 2 * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-outer', type=int, default=80)
    parser.add_argument('--bdy-hole',  type=int, default=24)
    args = parser.parse_args()

    centres, radii = make_holes(seed=args.seed)
    print(f'[holes]   {len(centres)} non-overlapping circles')

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, centres, radii, rng)
    X_bdy = np.concatenate([
        outer_boundary(args.bdy_outer),
        hole_boundaries(centres, radii, args.bdy_hole),
    ], axis=0)

    demo = GeometryDemo(
        name='porous', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polylines(centres, radii),
        u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
