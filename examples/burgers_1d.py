"""Python port of main_Burgers1d.jl.

Solves u_t + u u_x - ν u_xx = 0 on [-1, 1] × [0, T] with u(±1, t) = 0 and
u(x, 0) = -sin(π x). The exact solution is obtained from the Cole-Hopf
transform via Gauss-Hermite quadrature (same reference as Julia's code).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def _cole_hopf_solution(x: np.ndarray, t: float, nu: float, n_quad: int = 100):
    """u(x, t) via Cole-Hopf for Burgers with u(x,0) = -sin(π x).

    The exact formula is
        u(x,t) = -∫ sin(π(x - 2√(νt)ξ)) exp(-cos(π(x - 2√(νt)ξ)) / (2πν)) e^{-ξ²} dξ
               / ∫ exp(-cos(π(x - 2√(νt)ξ)) / (2πν)) e^{-ξ²} dξ
    evaluated by Gauss-Hermite.
    """
    xi, w = np.polynomial.hermite.hermgauss(n_quad)
    # broadcast: x is (N,), xi is (n_quad,)
    arg = x[:, None] - np.sqrt(4.0 * nu * t) * xi[None, :]
    cos_arg = np.cos(np.pi * arg)
    expo = np.exp(-cos_arg / (2.0 * np.pi * nu))
    val1 = w[None, :] * np.sin(np.pi * arg) * expo
    val2 = w[None, :] * expo
    return -val1.sum(axis=1) / val2.sum(axis=1)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--nu', type=float, default=0.001)
    p.add_argument('--kernel', choices=['Matern5half', 'Matern7half', 'Matern9half',
                                         'Matern11half', 'Gaussian'],
                   default='Matern7half')
    p.add_argument('--sigma', type=float, default=0.02)
    p.add_argument('--h', type=float, default=0.005)
    p.add_argument('--dt', type=float, default=0.02)
    p.add_argument('--T', type=float, default=1.0)
    p.add_argument('--nugget', type=float, default=1e-10)
    p.add_argument('--GN-steps', type=int, default=2)
    p.add_argument('--rho-big', type=float, default=3.0)
    p.add_argument('--rho-small', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=1)
    p.add_argument('--backend', choices=['auto', 'cpu', 'jax'], default='auto')
    p.add_argument('--platform', default='cpu')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform
    import kolesky as kl
    from kolesky.pde import Burgers1d, solve_burgers_1d, sample_points_grid_1d

    def fun_u0(x):
        return float(-np.sin(np.pi * x))

    def grad_u0(x):
        return float(-np.pi * np.cos(np.pi * x))

    def laplace_u0(x):
        return float(np.pi ** 2 * np.sin(np.pi * x))

    def fun_rhs(x):
        return 0.0

    def fun_bdy(x):
        return 0.0

    eqn = Burgers1d(
        nu=args.nu, bdy=fun_bdy, rhs=fun_rhs,
        init=fun_u0, init_dx=grad_u0, init_dxx=laplace_u0,
    )
    X_domain, X_boundary = sample_points_grid_1d(args.h)
    print(f'[sample points] h = {args.h}, N_domain = {X_domain.shape[0]}, N_boundary = {X_boundary.shape[0]}')
    print(f'[time] dt = {args.dt}, T = {args.T}, Nt = {int(round(args.T/args.dt))}')

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
        'Matern11half': kl.MaternCovariance11_2,
        'Gaussian': kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel] {args.kernel}, length_scale = {args.sigma}, ν = {args.nu}')

    t0 = time.perf_counter()
    sol = solve_burgers_1d(
        eqn, kernel, X_domain, X_boundary,
        dt=args.dt, T=args.T, nugget=args.nugget, GN_steps=args.GN_steps,
        rho_big=args.rho_big, rho_small=args.rho_small,
        k_neighbors=args.k_neighbors, backend=args.backend,
        verbose=False,
    )
    t1 = time.perf_counter()
    print(f'[fast solve] wall time: {t1 - t0:.3f} s')

    truth = _cole_hopf_solution(X_domain, args.T, args.nu)
    err = truth - sol
    print(f'[accuracy] L2 = {np.sqrt(np.mean(err ** 2)):.3e}, '
          f'Linf = {np.max(np.abs(err)):.3e}')


if __name__ == '__main__':
    main()
