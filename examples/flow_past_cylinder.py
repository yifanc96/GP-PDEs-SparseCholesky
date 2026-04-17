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


def bc_values(X_bdy, tag, t):
    """Zero Dirichlet everywhere: the initial blob carries the signal."""
    return np.zeros(X_bdy.shape[0])


def initial_blob(X: np.ndarray, centre=(0.6, 0.35), width=0.22) -> np.ndarray:
    """Gaussian tracer released upstream of the cylinder. The potential
    flow then advects (and diffuses) it past the obstacle."""
    dx = X[:, 0] - centre[0]
    dy = X[:, 1] - centre[1]
    return np.exp(-0.5 * (dx * dx + dy * dy) / (width * width))


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
    p.add_argument('--render-only', action='store_true',
                   help='skip simulation; load cached .npz next to --out and re-render')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_path.with_suffix('.npz')

    if args.render_only:
        print(f'[render-only] loading {npz_path}')
        d = np.load(npz_path)
        box = Box(Lx=float(d['box_Lx']), Ly=float(d['box_Ly']),
                  x_cyl=float(d['box_x_cyl']), y_cyl=float(d['box_y_cyl']),
                  r_cyl=float(d['box_r_cyl']))
        _render_animation(out_path, d['X_dom'], d['X_bdy'], d['tag'],
                          d['times'], d['c_frames'], box, float(d['U_inf']))
        print(f'[done] {out_path}')
        return

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

    c_dom = initial_blob(X_dom)
    n_steps = int(round(args.T / args.dt))
    print(f'[simulate] n_steps = {n_steps}, dt = {args.dt}')

    frames = [(0.0, c_dom.copy())]
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
    times = np.array([t for t, _ in frames])
    c_frames = np.stack([c for _, c in frames], axis=0)  # (F, N_dom)

    np.savez_compressed(
        npz_path, X_dom=X_dom, X_bdy=X_bdy, tag=tag,
        times=times, c_frames=c_frames,
        box_Lx=box.Lx, box_Ly=box.Ly,
        box_x_cyl=box.x_cyl, box_y_cyl=box.y_cyl, box_r_cyl=box.r_cyl,
        U_inf=args.U_inf,
    )
    print(f'[render] cached frame data at {npz_path}')

    _render_animation(out_path, X_dom, X_bdy, tag, times, c_frames, box, args.U_inf)
    print(f'[done] {out_path}')


def _render_animation(out_path, X_dom, X_bdy, tag, times, c_frames, box, U_inf):
    """Smooth tricontourf field + static streamlines of the potential flow."""
    from matplotlib.tri import Triangulation
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Triangulate interior + boundary so the field reaches every edge.
    X_all = np.vstack([X_dom, X_bdy])
    triang = Triangulation(X_all[:, 0], X_all[:, 1])

    tris = triang.triangles
    centroids = X_all[tris].mean(axis=1)
    inside_cyl = ((centroids[:, 0] - box.x_cyl) ** 2
                  + (centroids[:, 1] - box.y_cyl) ** 2
                  < (box.r_cyl + 0.015) ** 2)
    edge_lens = np.stack([
        np.linalg.norm(X_all[tris[:, 0]] - X_all[tris[:, 1]], axis=1),
        np.linalg.norm(X_all[tris[:, 1]] - X_all[tris[:, 2]], axis=1),
        np.linalg.norm(X_all[tris[:, 2]] - X_all[tris[:, 0]], axis=1),
    ], axis=0)
    too_long = edge_lens.max(axis=0) > 5 * np.median(edge_lens.max(axis=0))
    triang.set_mask(inside_cyl | too_long)

    # Stationary potential-flow streamlines (computed once).
    gx = np.linspace(0.02, box.Lx - 0.02, 100)
    gy = np.linspace(-box.Ly / 2 + 0.02, box.Ly / 2 - 0.02, 50)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
    u_grid = potential_flow(box, grid_pts, U_inf=U_inf)
    inside_grid = ((grid_pts[:, 0] - box.x_cyl) ** 2
                   + (grid_pts[:, 1] - box.y_cyl) ** 2 < box.r_cyl ** 2)
    u_grid[inside_grid] = np.nan
    U = u_grid[:, 0].reshape(GY.shape)
    V = u_grid[:, 1].reshape(GY.shape)

    # Truncate frames after the plume decays below a useful amplitude
    # (so we don't render noise). Then per-frame auto-vmax so the plume
    # stays visible as its absolute amplitude shrinks.
    frame_peak = np.percentile(c_frames, 99.5, axis=1)
    keep = frame_peak >= 0.03
    if keep.sum() < len(keep):
        last = int(np.argmin(keep[: np.argmax(~keep) + 1]))  # noqa
        # Simpler: truncate at the last index where peak >= threshold.
        last_idx = int(np.where(keep)[0].max())
        times = times[: last_idx + 1]
        c_frames = c_frames[: last_idx + 1]
        frame_peak = frame_peak[: last_idx + 1]
    per_frame_vmax = np.maximum(frame_peak, 0.03)
    vmax_display = 1.0
    print(f'[render] keeping {len(times)} frames (peak c ≥ 0.03), '
          f'per-frame vmax range: '
          f'{per_frame_vmax.min():.3f} → {per_frame_vmax.max():.3f}')

    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    # Include boundary values in the field for correct edge rendering.
    c_all0 = np.concatenate([c_frames[0], bc_values(X_bdy, tag, times[0])]) / per_frame_vmax[0]
    tpc = ax.tripcolor(triang, c_all0, shading='gouraud',
                       cmap='magma', vmin=0, vmax=vmax_display)
    ax.streamplot(GX, GY, U, V, color='white', linewidth=0.4,
                  density=0.9, arrowsize=0.7, arrowstyle='-')
    cyl = mpatches.Circle((box.x_cyl, box.y_cyl), box.r_cyl,
                          fc='#222', ec='white', lw=1.2, zorder=5)
    ax.add_patch(cyl)
    box_outline = np.array([[0, -box.Ly / 2], [box.Lx, -box.Ly / 2],
                            [box.Lx, box.Ly / 2], [0, box.Ly / 2],
                            [0, -box.Ly / 2]])
    ax.plot(box_outline[:, 0], box_outline[:, 1], 'k-', lw=1.2)
    ax.set_aspect('equal')
    ax.set_xlim(-0.05, box.Lx + 0.05)
    ax.set_ylim(-box.Ly / 2 - 0.05, box.Ly / 2 + 0.05)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(tpc, ax=ax, shrink=0.85, label='concentration (normalized)')
    ttl = ax.set_title('')

    def update(i):
        c_all_i = np.concatenate([c_frames[i], bc_values(X_bdy, tag, times[i])]) / per_frame_vmax[i]
        tpc.set_array(c_all_i)
        ttl.set_text(
            f'flow past cylinder — t = {times[i]:.2f},  '
            f'peak c ≈ {per_frame_vmax[i]:.3f}'
        )
        return [tpc, ttl]

    ani = FuncAnimation(fig, update, frames=len(times), interval=60, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=18))
    plt.close(fig)


if __name__ == '__main__':
    main()
