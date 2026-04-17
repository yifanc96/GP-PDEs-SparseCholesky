"""2-D scalar advection-diffusion past a cylinder — time-stepped demo.

Domain:  Ω = [0, Lx] × [−Ly/2, Ly/2] \\ disk at (xc, 0) of radius R
Equation:  c_t + u·∇c − D Δc = 0
Velocity:  analytical potential flow past the cylinder (incompressible,
           satisfies no-penetration on the cylinder surface).
BCs:       c = c_in(y, t) at inlet; c = 0 elsewhere.

The transport equation above is exactly the **vorticity transport**
step of an incompressible Navier-Stokes solver in stream-function /
vorticity form. A full NS would add a Poisson solve `−Δψ = ω` (via
`solve_nonlin_elliptic` with α=0) plus a gradient extraction
`u = (ψ_y, −ψ_x)` each step — both one-liners on top of this scaffold.

Numerics:  **fully implicit backward-Euler** for both advection and
diffusion. Each time step solves the linear system
    (1 + dt·u·∇ − dt·D·Δ) c^{n+1} = c^n                on  interior
    c^{n+1} = c_bdy(t_{n+1})                           on  ∂Ω
using a Δ∇δ measurement at each interior point and a δ measurement at
each boundary point. The operator coefficients are time-independent so
the kolesky sparse Cholesky factor is built **once** and reused across
all time steps — per-step cost is one pCG solve.

Output: docs/flow_past_cylinder.gif
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import kolesky as kl
from kolesky.measurements import LaplaceGradDiracPointMeasurement
from kolesky.pde.pcg_ops import BigFactorOperator, LiftedThetaTrainMatVec, SmallPrecond


# ---------------------------------------------------------------------------
# Geometry + velocity + BCs
# ---------------------------------------------------------------------------


@dataclass
class Box:
    Lx: float = 6.0
    Ly: float = 3.0
    x_cyl: float = 1.5
    y_cyl: float = 0.0
    r_cyl: float = 0.30


def in_domain(box: Box, x, y, pad: float = 0.0):
    in_box = ((x >= pad) & (x <= box.Lx - pad)
              & (y >= -box.Ly / 2 + pad) & (y <= box.Ly / 2 - pad))
    dx = x - box.x_cyl; dy = y - box.y_cyl
    return in_box & (dx * dx + dy * dy >= (box.r_cyl + pad) ** 2)


def sample_interior(box: Box, n: int, rng) -> np.ndarray:
    pts = np.empty((0, 2))
    while pts.shape[0] < n:
        batch = np.stack([rng.uniform(0, box.Lx, 4 * n),
                          rng.uniform(-box.Ly / 2, box.Ly / 2, 4 * n)], axis=1)
        pts = np.vstack([pts, batch[in_domain(box, batch[:, 0], batch[:, 1], pad=0.01)]])
    return pts[:n]


def sample_boundary(box: Box, n_long: int, n_short: int, n_cyl: int):
    y = np.linspace(-box.Ly / 2, box.Ly / 2, n_short, endpoint=False)
    inlet  = np.stack([np.zeros_like(y),      y], axis=1)
    outlet = np.stack([np.full_like(y, box.Lx), y], axis=1)
    x = np.linspace(0, box.Lx, n_long, endpoint=False)
    top = np.stack([x, np.full_like(x,  box.Ly / 2)], axis=1)
    bot = np.stack([x, np.full_like(x, -box.Ly / 2)], axis=1)
    th = np.linspace(0, 2 * np.pi, n_cyl, endpoint=False)
    cyl = np.stack([box.x_cyl + box.r_cyl * np.cos(th),
                    box.y_cyl + box.r_cyl * np.sin(th)], axis=1)
    X_bdy = np.concatenate([inlet, outlet, top, bot, cyl], axis=0)
    tag = np.concatenate([np.full(len(inlet),  0, np.int8),
                          np.full(len(outlet), 1, np.int8),
                          np.full(len(top),    2, np.int8),
                          np.full(len(bot),    3, np.int8),
                          np.full(len(cyl),    4, np.int8)])
    return X_bdy, tag


def potential_flow(box: Box, X: np.ndarray, U_inf: float = 1.0) -> np.ndarray:
    """Analytical 2-D potential flow past a circular cylinder."""
    xr = X[:, 0] - box.x_cyl
    yr = X[:, 1] - box.y_cyl
    r2 = xr * xr + yr * yr
    safe = np.where(r2 > 0.0, r2, 1.0)
    R2 = box.r_cyl ** 2
    u =  U_inf * (1.0 - R2 / safe + 2.0 * R2 * yr * yr / (safe * safe))
    v = -U_inf * (2.0 * R2 * xr * yr / (safe * safe))
    return np.stack([u, v], axis=1)


def inlet_pulse(y, t, sigma=0.25, amp_y=0.7, freq=0.6):
    """Plume whose centre y-coordinate oscillates sinusoidally in time
    (∈ [−amp_y, amp_y]) and whose amplitude pulses on/off. Produces a
    visibly moving / wrapping plume in the animation.
    """
    y0 = amp_y * np.sin(2 * np.pi * freq * t)
    envelope = 0.5 * (1.0 + np.cos(2 * np.pi * freq * t * 0.5))  # slow pulse
    return envelope * np.exp(-0.5 * ((y - y0) / sigma) ** 2)


def bc_values(X_bdy, tag, t):
    vals = np.zeros(X_bdy.shape[0])
    vals[tag == 0] = inlet_pulse(X_bdy[tag == 0, 1], t)
    return vals


# ---------------------------------------------------------------------------
# Solver: fully-implicit advection-diffusion, factor-cached across steps
# ---------------------------------------------------------------------------


class AdvDiffSolver:
    """Solves  (1 + dt·u·∇ − dt·D·Δ) c^{n+1} = c^n  on interior with
    Dirichlet data on ∂Ω using Δ∇δ measurements.

    Weights w_L = −dt·D, w_∇ = dt·u(x), w_δ = 1 are time-independent, so
    the kolesky factors are built once and reused every step.
    """

    def __init__(self, box, X_dom, X_bdy, tag, kernel,
                 dt, D, u_dom,
                 rho_big=4.0, rho_small=4.0, k_neighbors=2,
                 nugget=1e-6, backend='cpu'):
        self.box = box
        self.X_dom = X_dom; self.X_bdy = X_bdy; self.tag = tag
        self.dt = dt; self.D = D
        self.kernel = kernel
        self.N_bdy = X_bdy.shape[0]; self.N_dom = X_dom.shape[0]

        # Big factor: (δ_bdy, δ_int, spatial_deriv_int).
        # spatial_deriv = −D·Δ + u·∇  — the non-δ part of the train op.
        m_bdy = LaplaceGradDiracPointMeasurement(
            coordinate=X_bdy, weight_laplace=np.zeros(self.N_bdy),
            weight_grad=np.zeros((self.N_bdy, 2)),
            weight_delta=np.ones(self.N_bdy),
        )
        m_d_int = LaplaceGradDiracPointMeasurement(
            coordinate=X_dom, weight_laplace=np.zeros(self.N_dom),
            weight_grad=np.zeros((self.N_dom, 2)),
            weight_delta=np.ones(self.N_dom),
        )
        m_deriv_int = LaplaceGradDiracPointMeasurement(
            coordinate=X_dom, weight_laplace=np.full(self.N_dom, -D),
            weight_grad=u_dom.astype(np.float64).copy(),
            weight_delta=np.zeros(self.N_dom),
        )
        print('[build] big factor (3-set) …', flush=True)
        t0 = time.perf_counter()
        impl_big = kl.ImplicitKLFactorization.build_diracs_first_then_unif_scale(
            kernel, [m_bdy, m_d_int, m_deriv_int], rho=rho_big, k_neighbors=k_neighbors,
        )
        self.expl_big = kl.ExplicitKLFactorization(impl_big, nugget=nugget, backend=backend)
        self.big_op = BigFactorOperator(self.expl_big.U, self.expl_big.P)
        print(f'[build] big factor {time.perf_counter()-t0:.2f} s, '
              f'U.nnz = {self.expl_big.U.nnz:,}')

        # Small (train) factor: (δ_bdy, train_op_int).
        m_train_int = LaplaceGradDiracPointMeasurement(
            coordinate=X_dom,
            weight_laplace=np.full(self.N_dom, -dt * D),
            weight_grad=(dt * u_dom).astype(np.float64).copy(),
            weight_delta=np.ones(self.N_dom),
        )
        print('[build] small train factor (2-set) …', flush=True)
        t0 = time.perf_counter()
        impl_small = kl.ImplicitKLFactorization.build(
            kernel, [m_bdy, m_train_int], rho=rho_small, k_neighbors=k_neighbors,
        )
        self.expl_small = kl.ExplicitKLFactorization(impl_small, nugget=nugget, backend=backend)
        self.precond = SmallPrecond(self.expl_small.U, self.expl_small.P)
        print(f'[build] small factor {time.perf_counter()-t0:.2f} s, '
              f'U.nnz = {self.expl_small.U.nnz:,}')

        # Lift/extract weights: train op = 1·δ_int + dt·(spatial_deriv).
        self.theta_train_op = LiftedThetaTrainMatVec(
            self.big_op, self.N_bdy, self.N_dom, n_dom_sets=2,
        )
        self.theta_train_op.set_weights([1.0, dt])

    def step(self, c_bdy_next, c_dom_prev):
        rhs = np.concatenate([c_bdy_next, c_dom_prev], axis=0)
        A = self.theta_train_op.as_linear_operator()
        M = self.precond.as_linear_operator()
        x0 = self.precond.matvec(rhs)
        theta_inv_rhs, _info = scipy.sparse.linalg.cg(
            A, rhs, x0=x0, M=M, rtol=1e-6, maxiter=500,
        )
        lifted = self.theta_train_op._lift(theta_inv_rhs)
        t = self.big_op.apply(lifted)
        return t[self.N_bdy:self.N_bdy + self.N_dom]


# ---------------------------------------------------------------------------
# Main + animation
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--N-interior', type=int, default=4000)
    p.add_argument('--bdy-long',   type=int, default=120)
    p.add_argument('--bdy-short',  type=int, default=60)
    p.add_argument('--bdy-cyl',    type=int, default=120)
    p.add_argument('--sigma',  type=float, default=0.15)
    p.add_argument('--D',      type=float, default=0.02)
    p.add_argument('--dt',     type=float, default=0.05)
    p.add_argument('--T',      type=float, default=8.0)
    p.add_argument('--U-inf',  type=float, default=1.0)
    p.add_argument('--nugget', type=float, default=1e-6)
    p.add_argument('--rho-big',   type=float, default=4.0)
    p.add_argument('--rho-small', type=float, default=4.0)
    p.add_argument('--k-neighbors', type=int, default=2)
    p.add_argument('--backend', default='cpu', choices=['cpu', 'jax'])
    p.add_argument('--platform', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='docs/flow_past_cylinder.gif')
    p.add_argument('--frame-every', type=int, default=2)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform

    box = Box()
    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(box, args.N_interior, rng)
    X_bdy, tag = sample_boundary(box, args.bdy_long, args.bdy_short, args.bdy_cyl)
    print(f'[setup] interior = {X_dom.shape[0]}, boundary = {X_bdy.shape[0]}')

    u_dom = potential_flow(box, X_dom, U_inf=args.U_inf)
    kernel = kl.MaternCovariance7_2(args.sigma)

    solver = AdvDiffSolver(
        box, X_dom, X_bdy, tag, kernel,
        dt=args.dt, D=args.D, u_dom=u_dom,
        nugget=args.nugget, backend=args.backend,
        rho_big=args.rho_big, rho_small=args.rho_small,
        k_neighbors=args.k_neighbors,
    )

    c_dom = np.zeros(X_dom.shape[0])
    n_steps = int(round(args.T / args.dt))
    print(f'[simulate] n_steps = {n_steps}, dt = {args.dt}')

    frames = []
    t_loop = time.perf_counter()
    for step in range(n_steps):
        t = (step + 1) * args.dt
        c_bdy_next = bc_values(X_bdy, tag, t)
        t0 = time.perf_counter()
        c_dom = solver.step(c_bdy_next, c_dom)
        if step % args.frame_every == 0:
            frames.append((t, c_dom.copy()))
        if step % 10 == 0:
            print(f'  step {step+1}/{n_steps}  t={t:5.2f}  '
                  f'{time.perf_counter()-t0:.3f} s/step   '
                  f'c range [{c_dom.min():.3f}, {c_dom.max():.3f}]')
    print(f'[simulate] total loop wall: {time.perf_counter()-t_loop:.2f} s')

    print(f'[render] {len(frames)} frames …')
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    sc = ax.scatter(X_dom[:, 0], X_dom[:, 1], c=frames[0][1],
                    s=10, cmap='viridis', vmin=0.0, vmax=1.0)
    cyl = mpatches.Circle((box.x_cyl, box.y_cyl), box.r_cyl,
                          fc='white', ec='black', lw=1.5, zorder=5)
    ax.add_patch(cyl)
    box_outline = np.array([[0, -box.Ly/2], [box.Lx, -box.Ly/2],
                            [box.Lx, box.Ly/2], [0, box.Ly/2],
                            [0, -box.Ly/2]])
    ax.plot(box_outline[:, 0], box_outline[:, 1], 'k-', lw=1.5)
    ax.set_aspect('equal'); ax.set_xlim(-0.1, box.Lx + 0.1)
    ax.set_ylim(-box.Ly / 2 - 0.1, box.Ly / 2 + 0.1)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(sc, ax=ax, shrink=0.85, label='concentration')
    ttl = ax.set_title('')

    from matplotlib.animation import FuncAnimation, PillowWriter

    def update(i):
        t_i, c_i = frames[i]
        sc.set_array(c_i)
        ttl.set_text(f'flow past cylinder — t = {t_i:.2f},  N_int = {X_dom.shape[0]}')
        return sc, ttl

    ani = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_path, writer=PillowWriter(fps=18))
    plt.close(fig)
    print(f'[done] {out_path}')


if __name__ == '__main__':
    main()
