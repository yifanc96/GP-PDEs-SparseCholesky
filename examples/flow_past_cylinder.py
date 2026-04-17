"""2-D heat diffusion on the flow-past-cylinder geometry — moving-source demo.

NOTE: EXPERIMENTAL / NOT REFERENCED FROM THE README.
    An earlier version of this script attempted true scalar advection-
    diffusion (c_t + u·∇c − D Δc = 0) and steady -D Δc + u·∇c = S with a
    potential-flow velocity. Neither worked well: isotropic Matérn
    kernels can't represent thin high-Péclet plumes, so the plume never
    propagated downstream. A proper flow-past-cylinder demo needs the
    full Navier-Stokes streamfunction-vorticity system (coupled Poisson
    + transport with a no-slip cylinder BC), which is a separate
    project. This script is kept as a scaffold in the meantime.

Current physics: steady −D Δc = S(x) with a Gaussian source whose
vertical position sweeps over animation frames. Pure diffusion only.

Interior measurements: w_L = −D, w_∇ = 0, w_δ = 0.
Boundary measurements: pure δ (Dirichlet c = 0).
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
    """Zero Dirichlet everywhere: the source drives the plume."""
    return np.zeros(X_bdy.shape[0])


def source_field(X: np.ndarray, centre=(0.6, 0.35), width=0.18,
                 strength: float = 1.0) -> np.ndarray:
    """Localized Gaussian source of tracer — a continuous release at
    ``centre``. The flow then carries it downstream; when the per-step
    decay balances the source, the plume reaches steady state."""
    dx = X[:, 0] - centre[0]
    dy = X[:, 1] - centre[1]
    return strength * np.exp(-0.5 * (dx * dx + dy * dy) / (width * width))


# ---------------------------------------------------------------------------
# Solver: fully-implicit advection-diffusion, factor-cached across steps
# ---------------------------------------------------------------------------


class SteadyAdvDiffSolver:
    """Solves the steady advection-diffusion-reaction equation

        −D Δc + u·∇c = S(x)   on interior    c = 0 on ∂Ω

    as a single GP regression. Big and small sparse Cholesky factors are
    built ONCE for the operator (u, D); per solve is one pCG with a new
    RHS — swapping source positions is essentially free. The demo in
    this file passes ``u_dom = 0`` so the operator reduces to ``−D Δ``
    (pure diffusion), but the machinery is written for the full
    advection-diffusion form.
    """

    def __init__(self, box, X_dom, X_bdy, tag, kernel,
                 D, u_dom,
                 rho_big=4.0, rho_small=4.0, k_neighbors=2,
                 nugget=1e-6, backend='cpu'):
        self.box = box
        self.X_dom = X_dom; self.X_bdy = X_bdy; self.tag = tag
        self.D = D
        self.kernel = kernel
        self.N_bdy = X_bdy.shape[0]; self.N_dom = X_dom.shape[0]

        # Big factor: (δ_bdy, δ_int, spatial_deriv_int).
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

        # Small (train) factor: (δ_bdy, spatial_deriv_int).
        # This is the ACTUAL steady-state operator: no identity part.
        print('[build] small train factor (2-set) …', flush=True)
        t0 = time.perf_counter()
        impl_small = kl.ImplicitKLFactorization.build(
            kernel, [m_bdy, m_deriv_int], rho=rho_small, k_neighbors=k_neighbors,
        )
        self.expl_small = kl.ExplicitKLFactorization(impl_small, nugget=nugget, backend=backend)
        self.precond = SmallPrecond(self.expl_small.U, self.expl_small.P)
        print(f'[build] small factor {time.perf_counter()-t0:.2f} s, '
              f'U.nnz = {self.expl_small.U.nnz:,}')

        # Lift/extract weights: predict δ_int row of Θ_big @ lift(train⁻¹ rhs)
        # where train op = spatial_deriv (weights [0, 1]).
        self.theta_train_op = LiftedThetaTrainMatVec(
            self.big_op, self.N_bdy, self.N_dom, n_dom_sets=2,
        )
        self.theta_train_op.set_weights([0.0, 1.0])

    def solve(self, S_int: np.ndarray, bc_vals: np.ndarray | None = None,
              verbose: bool = False) -> np.ndarray:
        """Solve  −D Δc + u·∇c = S_int  on interior with Dirichlet ``bc_vals``
        (defaults to zero) on the boundary."""
        if bc_vals is None:
            bc_vals = np.zeros(self.N_bdy)
        rhs = np.concatenate([bc_vals, S_int], axis=0)
        A = self.theta_train_op.as_linear_operator()
        M = self.precond.as_linear_operator()
        x0 = self.precond.matvec(rhs)
        nit = [0]
        def cb(_xk): nit[0] += 1
        theta_inv_rhs, info = scipy.sparse.linalg.cg(
            A, rhs, x0=x0, M=M, rtol=1e-7, maxiter=2000, callback=cb,
        )
        if verbose:
            print(f'    pCG: {nit[0]} iters, info={info}')
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
    p.add_argument('--U-inf',  type=float, default=1.0)
    p.add_argument('--n-frames', type=int, default=60,
                   help='number of steady-state solves (one per source position)')
    p.add_argument('--source-width', type=float, default=0.12)
    p.add_argument('--source-amp-y', type=float, default=0.75,
                   help='maximum |y| the source oscillates to')
    p.add_argument('--source-x', type=float, default=0.55,
                   help='x-coordinate of the source (upstream of cylinder)')
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
                          d['times'], d['c_frames'], d['source_positions'],
                          box, float(d['U_inf']))
        print(f'[done] {out_path}')
        return

    box = Box()
    rng = np.random.default_rng(args.seed)
    X_dom = sample_interior(box, args.N_interior, rng)
    X_bdy, tag = sample_boundary(box, args.bdy_long, args.bdy_short, args.bdy_cyl)
    print(f'[setup] interior = {X_dom.shape[0]}, boundary = {X_bdy.shape[0]}')

    # The solver operator is pure diffusion (−D Δ). The `u_dom` we
    # construct from `potential_flow` is ONLY used downstream to draw
    # streamlines — it is NOT fed to the solver.
    kernel = kl.MaternCovariance7_2(args.sigma)

    solver = SteadyAdvDiffSolver(
        box, X_dom, X_bdy, tag, kernel,
        D=args.D, u_dom=np.zeros((X_dom.shape[0], 2)),
        nugget=args.nugget, backend=args.backend,
        rho_big=args.rho_big, rho_small=args.rho_small,
        k_neighbors=args.k_neighbors,
    )

    # Sweep the source vertically: one steady-state solve per frame.
    # (Factors are cached — each solve is just a pCG on a new RHS.)
    phases = np.linspace(0.0, 2 * np.pi, args.n_frames, endpoint=False)
    source_ys = args.source_amp_y * np.sin(phases)

    c_frames_list = []
    source_positions = []
    t_loop = time.perf_counter()
    for i, y_src in enumerate(source_ys):
        S = source_field(
            X_dom, centre=(args.source_x, float(y_src)),
            width=args.source_width, strength=1.0,
        )
        t0 = time.perf_counter()
        c_dom = solver.solve(S, verbose=(i == 0))
        t1 = time.perf_counter()
        c_frames_list.append(c_dom.copy())
        source_positions.append((args.source_x, float(y_src)))
        if i % 5 == 0:
            print(f'  frame {i+1:3d}/{args.n_frames}  y_src={y_src:+.2f}  '
                  f'{t1-t0:.3f} s/solve   '
                  f'c range [{c_dom.min():+.3f}, {c_dom.max():.3f}]')
    print(f'[simulate] total wall: {time.perf_counter()-t_loop:.2f} s  '
          f'({args.n_frames} solves)')

    times = phases  # treat phase as pseudo-time for the animation
    c_frames = np.stack(c_frames_list, axis=0)
    source_positions = np.asarray(source_positions)
    print(f'[render] {len(times)} frames …')

    np.savez_compressed(
        npz_path, X_dom=X_dom, X_bdy=X_bdy, tag=tag,
        times=times, c_frames=c_frames,
        source_positions=source_positions,
        box_Lx=box.Lx, box_Ly=box.Ly,
        box_x_cyl=box.x_cyl, box_y_cyl=box.y_cyl, box_r_cyl=box.r_cyl,
        U_inf=args.U_inf,
    )
    print(f'[render] cached frame data at {npz_path}')

    _render_animation(out_path, X_dom, X_bdy, tag, times, c_frames,
                      source_positions, box, args.U_inf)
    print(f'[done] {out_path}')


def _render_animation(out_path, X_dom, X_bdy, tag, times, c_frames,
                      source_positions, box, U_inf):
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

    # Fixed vmax so the brightness contrast stays consistent across frames.
    # (Each frame is an independent steady-state solve; amplitudes are
    # comparable but not identical.)
    vmax = float(np.percentile(c_frames, 99.5))
    vmax = max(vmax, 0.05)
    print(f'[render] {len(times)} frames, fixed vmax = {vmax:.3f}')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 6.8), facecolor='#0a0a0a')
    ax.set_facecolor('#000000')

    c_all0 = np.concatenate([c_frames[0], bc_values(X_bdy, tag, times[0])])
    tpc = ax.tripcolor(triang, c_all0, shading='gouraud',
                       cmap='inferno', vmin=0, vmax=vmax, zorder=1)

    # Collocation cloud — dim dots so the user sees the scattered sample
    # points under the field. These are the *measurement locations*: the
    # PDE is collocated at each of these.
    ax.scatter(X_dom[:, 0], X_dom[:, 1], s=1.2, c='#70a0ff',
               alpha=0.20, linewidths=0, zorder=2)
    ax.scatter(X_bdy[:, 0], X_bdy[:, 1], s=3.0, c='#ffb060',
               alpha=0.55, linewidths=0, zorder=3)

    # Streamlines — thin pale cyan.
    ax.streamplot(GX, GY, U, V, color='#9fd8ff', linewidth=0.35,
                  density=1.2, arrowsize=0.0, arrowstyle='-')

    cyl = mpatches.Circle((box.x_cyl, box.y_cyl), box.r_cyl,
                          fc='#202428', ec='#e0e0e0', lw=1.2, zorder=5)
    ax.add_patch(cyl)

    # Domain frame.
    box_outline = np.array([[0, -box.Ly / 2], [box.Lx, -box.Ly / 2],
                            [box.Lx, box.Ly / 2], [0, box.Ly / 2],
                            [0, -box.Ly / 2]])
    ax.plot(box_outline[:, 0], box_outline[:, 1], color='#555', lw=0.8, zorder=4)

    # Moving source marker (updated each frame).
    src_marker, = ax.plot([source_positions[0, 0]], [source_positions[0, 1]],
                          marker='o', ms=10, mfc='none',
                          mec='#ff5a5a', mew=1.8, zorder=7)
    src_glow, = ax.plot([source_positions[0, 0]], [source_positions[0, 1]],
                        marker='o', ms=22, mfc='none',
                        mec='#ff5a5a', mew=0.8, alpha=0.35, zorder=6)

    ax.set_aspect('equal')
    ax.set_xlim(-0.05, box.Lx + 0.05)
    ax.set_ylim(-box.Ly / 2 - 0.05, box.Ly / 2 + 0.05)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Header text.
    ax.text(
        0.015, 0.965,
        'heat diffusion on a flow-past-cylinder geometry  —  steady (−D Δc = S),  per-frame solve',
        transform=ax.transAxes, fontsize=12.5,
        color='#f0f0f0', fontweight='bold', va='top',
    )
    ax.text(
        0.015, 0.915,
        f'N_int = {X_dom.shape[0]}    N_bdy = {X_bdy.shape[0]}    '
        f'kernel = Matern 7/2    source sweeps y ∈ [−{abs(source_positions[:,1]).max():.2f}, {abs(source_positions[:,1]).max():.2f}]',
        transform=ax.transAxes, fontsize=9.5,
        color='#a8a8a8', va='top', family='monospace',
    )
    # Legend.
    ax.plot([], [], marker='o', ms=5, ls='', color='#70a0ff',
            alpha=0.7, label=f'interior collocation ({X_dom.shape[0]})')
    ax.plot([], [], marker='o', ms=5, ls='', color='#ffb060',
            alpha=0.9, label=f'boundary collocation ({X_bdy.shape[0]})')
    ax.plot([], [], color='#9fd8ff', lw=1.5, label='streamlines (potential flow)')
    ax.plot([], [], marker='o', ms=8, ls='', mfc='none',
            mec='#ff5a5a', mew=1.5, label='source')
    leg = ax.legend(loc='lower right', fontsize=9, frameon=False,
                    labelcolor='#cccccc')

    src_txt = ax.text(
        0.015, 0.06, '', transform=ax.transAxes,
        fontsize=10, color='#c0c0c0', va='bottom', family='monospace',
    )

    # Progress bar.
    pb_bg = mpatches.Rectangle(
        (0.015, 0.035), 0.55, 0.010,
        transform=ax.transAxes, facecolor='#242424', edgecolor='none', zorder=8,
    )
    pb_fg = mpatches.Rectangle(
        (0.015, 0.035), 0.0, 0.010,
        transform=ax.transAxes, facecolor='#ff9a55', edgecolor='none', zorder=9,
    )
    ax.add_patch(pb_bg); ax.add_patch(pb_fg)

    fig.tight_layout(pad=0.2)

    def update(i):
        c_all_i = np.concatenate([c_frames[i], bc_values(X_bdy, tag, times[i])])
        tpc.set_array(c_all_i)
        src_marker.set_data([source_positions[i, 0]], [source_positions[i, 1]])
        src_glow.set_data([source_positions[i, 0]], [source_positions[i, 1]])
        src_txt.set_text(
            f'frame {i+1:3d}/{len(times)}    '
            f'source @ ({source_positions[i, 0]:.2f}, {source_positions[i, 1]:+.2f})    '
            f'peak c = {c_frames[i].max():.2f}'
        )
        pb_fg.set_width(0.55 * (i + 1) / len(times))
        return [tpc, src_marker, src_glow, src_txt, pb_fg]

    ani = FuncAnimation(fig, update, frames=len(times), interval=60, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=20))
    plt.close(fig)
    plt.style.use('default')


if __name__ == '__main__':
    main()
