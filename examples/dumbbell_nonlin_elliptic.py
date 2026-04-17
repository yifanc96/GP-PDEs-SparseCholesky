"""Nonlinear elliptic PDE on a dumbbell domain.

Domain:  Two disks joined by a narrow rectangular bridge.
Left disk at (−0.8, 0), radius 0.4; right disk at (+0.8, 0), radius 0.4;
bridge [−0.6, 0.6] × [−0.05, 0.05]. The narrow channel tests that the
sparsity-pattern radius is chosen small enough relative to the bridge
width (ρ·ℓ has to fit inside).
"""

from __future__ import annotations

import numpy as np

from _geometry_demo import GeometryDemo, default_argparser, run_and_plot


CL = (-0.8, 0.0); RL = 0.4
CR = ( 0.8, 0.0); RR = 0.4
BRIDGE_X = (-0.6, 0.6)
BRIDGE_Y = (-0.05, 0.05)


def in_domain(x, y, pad: float = 0.0):
    in_left  = (x - CL[0]) ** 2 + (y - CL[1]) ** 2 <= (RL - pad) ** 2
    in_right = (x - CR[0]) ** 2 + (y - CR[1]) ** 2 <= (RR - pad) ** 2
    in_bridge = ((x >= BRIDGE_X[0] + pad) & (x <= BRIDGE_X[1] - pad)
                 & (y >= BRIDGE_Y[0] + pad) & (y <= BRIDGE_Y[1] - pad))
    return in_left | in_right | in_bridge


def sample_interior(n, rng):
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = rng.uniform([-1.3, -0.5], [1.3, 0.5], (4 * n, 2))
        pts = np.vstack([pts, batch[in_domain(batch[:, 0], batch[:, 1], pad=0.005)]])
    return pts[:n]


def disk_boundary(centre, radius, n, theta_range):
    th = np.linspace(theta_range[0], theta_range[1], n, endpoint=False)
    return np.stack([centre[0] + radius * np.cos(th),
                     centre[1] + radius * np.sin(th)], axis=1)


def bridge_boundary(per_side):
    t = np.linspace(BRIDGE_X[0], BRIDGE_X[1], per_side, endpoint=False)
    top = np.stack([t, np.full_like(t, BRIDGE_Y[1])], axis=1)
    bot = np.stack([t, np.full_like(t, BRIDGE_Y[0])], axis=1)
    return np.concatenate([top, bot], axis=0)


def outline_polylines():
    th = np.linspace(0, 2 * np.pi, 200)
    left  = np.stack([CL[0] + RL * np.cos(th), CL[1] + RL * np.sin(th)], axis=1)
    right = np.stack([CR[0] + RR * np.cos(th), CR[1] + RR * np.sin(th)], axis=1)
    bridge = np.array([
        [BRIDGE_X[0], BRIDGE_Y[1]], [BRIDGE_X[1], BRIDGE_Y[1]],
        [BRIDGE_X[1], BRIDGE_Y[0]], [BRIDGE_X[0], BRIDGE_Y[0]],
        [BRIDGE_X[0], BRIDGE_Y[1]],
    ])
    return [left, right, bridge]


def u_exact(x):
    return float(np.cos(np.pi * x[0]) * np.cos(2 * np.pi * x[1]))


def rhs(x):
    lap = (np.pi ** 2 + 4 * np.pi ** 2) * u_exact(x)
    return float(lap + u_exact(x) ** 3)


def main():
    parser = default_argparser()
    parser.add_argument('--bdy-disk',    type=int, default=200)
    parser.add_argument('--bdy-bridge',  type=int, default=120)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(args.N_interior, rng)
    # left and right disk outer arcs (skip region where bridge attaches)
    th_bridge = np.arcsin(BRIDGE_Y[1] / RL)            # angular half-width of bridge mouth
    left_bdy  = disk_boundary(CL, RL, args.bdy_disk,
                               theta_range=(-np.pi + th_bridge, np.pi - th_bridge))
    right_bdy = disk_boundary(CR, RR, args.bdy_disk,
                               theta_range=(th_bridge,          2 * np.pi - th_bridge))
    X_bdy = np.concatenate([left_bdy, right_bdy, bridge_boundary(args.bdy_bridge)], axis=0)

    demo = GeometryDemo(
        name='dumbbell', X_domain=X_dom, X_boundary=X_bdy,
        outline=outline_polylines(), u_exact=u_exact, rhs=rhs,
    )
    run_and_plot(demo, args)


if __name__ == '__main__':
    main()
