"""Nonlinear elliptic PDE on an airfoil-in-a-box domain.

Domain:  Ω = rectangular far-field box \ NACA 0012 airfoil  (chord = 1,
         placed along the x-axis from x=0.3 to x=1.3 in a [0, 1.8] × [−0.5, 0.5] box).
         A practical CFD-adjacent geometry: smooth airfoil contour with a
         sharp trailing-edge cusp.
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


# Far-field box
X0, X1 = 0.0, 1.8
Y0, Y1 = -0.5, 0.5

# Airfoil placement
CHORD = 1.0
X_LE = 0.3           # leading-edge x
T = 0.12             # NACA 0012 thickness ratio


def naca_y(x_local):
    """NACA 0012 upper surface y for x_local ∈ [0, 1]."""
    t = T
    x = np.clip(x_local, 0.0, 1.0)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x * x
                     + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def in_airfoil(x, y):
    x_local = (x - X_LE) / CHORD
    y_thick = naca_y(x_local)
    return (x_local >= 0.0) & (x_local <= 1.0) & (np.abs(y) <= y_thick)


def in_domain(x, y, pad: float = 0.0):
    in_box = (x >= X0 + pad) & (x <= X1 - pad) & (y >= Y0 + pad) & (y <= Y1 - pad)
    return in_box & ~in_airfoil(x, y)


def sample_interior(n, rng):
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform([X0, Y0], [X1, Y1], (4 * n, 2))
        # Avoid a small shell around the airfoil
        x_local = (batch[:, 0] - X_LE) / CHORD
        y_thick = naca_y(x_local)
        near_airfoil = (x_local > -0.01) & (x_local < 1.02) & (np.abs(batch[:, 1]) < y_thick + 0.01)
        keep = in_domain(batch[:, 0], batch[:, 1], pad=0.005) & ~near_airfoil
        pts = np.vstack([pts, batch[keep]])
    return pts[:n]


def outline_polylines():
    # box
    box = np.array([[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1], [X0, Y0]])
    # airfoil (cosine-spaced)
    beta = np.linspace(0, np.pi, 200)
    x = 0.5 * (1 - np.cos(beta))
    yt = naca_y(x)
    upper = np.stack([X_LE + x * CHORD, yt], axis=1)
    lower = np.stack([X_LE + x * CHORD, -yt], axis=1)[::-1]
    airfoil = np.concatenate([upper, lower, upper[:1]], axis=0)
    return [box, airfoil]


def airfoil_boundary(n):
    beta = np.linspace(0, np.pi, n // 2, endpoint=False)
    x = 0.5 * (1 - np.cos(beta))
    yt = naca_y(x)
    upper = np.stack([X_LE + x * CHORD, yt], axis=1)
    lower = np.stack([X_LE + x * CHORD, -yt], axis=1)
    return np.concatenate([upper, lower], axis=0)


def box_boundary(per_long, per_short):
    t_long = np.linspace(0, 1, per_long, endpoint=False)
    t_short = np.linspace(0, 1, per_short, endpoint=False)
    bot   = np.stack([X0 + (X1 - X0) * t_long,  Y0 * np.ones(per_long)], axis=1)
    right = np.stack([X1 * np.ones(per_short),  Y0 + (Y1 - Y0) * t_short], axis=1)
    top   = np.stack([X1 - (X1 - X0) * t_long,  Y1 * np.ones(per_long)], axis=1)
    left  = np.stack([X0 * np.ones(per_short),  Y1 - (Y1 - Y0) * t_short], axis=1)
    return np.concatenate([bot, right, top, left], axis=0)


def u_exact(x):
    # Smooth, non-trivial everywhere
    return float(np.sin(np.pi * (x[0] - X0) / (X1 - X0))
                  * np.cos(np.pi * (x[1] - Y0) / (Y1 - Y0)))


def rhs(x):
    kx = np.pi / (X1 - X0)
    ky = np.pi / (Y1 - Y0)
    return float((kx * kx + ky * ky) * u_exact(x) + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-airfoil', type=int, default=240)
    parser.add_argument('--bdy-box-long',  type=int, default=90)
    parser.add_argument('--bdy-box-short', type=int, default=45)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    X_bdy = np.concatenate([
        box_boundary(args.bdy_box_long, args.bdy_box_short),
        airfoil_boundary(args.bdy_airfoil),
    ], axis=0)

    demo = GeometryDemo(
        name='airfoil', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polylines(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
