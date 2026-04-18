"""Python port of main_MongeAmpere2d.jl.

Solves the Monge-Ampere equation det(∇² u) = f on [0,1]² with u(x) = u_exact(x)
on the boundary. Ground-truth u = exp((x-0.5)²/2 + (y-0.5)²/2), with f
obtained analytically from its Hessian.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def ground_truth_factory():
    """u(x) = exp(|x-0.5|²/2).

    u_x  = (x-0.5) u
    u_y  = (y-0.5) u
    u_xx = u + (x-0.5)² u = (1 + (x-0.5)²) u
    u_yy = (1 + (y-0.5)²) u
    u_xy = (x-0.5)(y-0.5) u
    det H = u_xx u_yy - u_xy² = u² [(1 + (x-0.5)²)(1 + (y-0.5)²) - (x-0.5)²(y-0.5)²]
          = u² [1 + (x-0.5)² + (y-0.5)²]
          = (1 + |x-0.5|²) · u²
    """

    def u_exact(x):
        return float(np.exp(0.5 * ((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)))

    def rhs(x):
        u = u_exact(x)
        return float((1.0 + (x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) * u * u)

    def bdy(x):
        return u_exact(x)

    return u_exact, bdy, rhs


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--kernel', choices=['Matern5half', 'Matern7half', 'Gaussian'],
                   default='Matern5half')
    p.add_argument('--sigma', type=float, default=0.3)
    p.add_argument('--h', type=float, default=0.05)
    p.add_argument('--nugget', type=float, default=1e-10)
    p.add_argument('--GN-steps', type=int, default=3)
    p.add_argument('--rho-big', type=float, default=3.0)
    p.add_argument('--rho-small', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--backend', choices=['auto', 'cpu', 'jax'], default='jax')
    p.add_argument('--platform', default='cpu')
    p.add_argument('--solve-twice', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform
    import kolesky as kl
    from kolesky.pde import sample_points_grid_2d
    from kolesky.pde import MongeAmpere2d, solve_monge_ampere_2d

    u_exact, bdy_fn, rhs_fn = ground_truth_factory()
    eqn = MongeAmpere2d(domain=((0.0, 1.0), (0.0, 1.0)), bdy=bdy_fn, rhs=rhs_fn)
    X_domain, X_boundary = sample_points_grid_2d(eqn.domain, args.h, args.h)
    print(f'[sample points] h = {args.h}, N_domain = {X_domain.shape[0]}, N_boundary = {X_boundary.shape[0]}')

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Gaussian': kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel] {args.kernel}, length_scale = {args.sigma}')

    N_dom = X_domain.shape[0]
    sol_init = np.zeros(N_dom)
    # mild convex initialisation — Julia uses (v_xx, v_xy, v_yy) = (1, 0, 1)
    sol_init_xx = np.ones(N_dom)
    sol_init_xy = np.zeros(N_dom)
    sol_init_yy = np.ones(N_dom)

    truth = np.array([u_exact(X_domain[i]) for i in range(N_dom)])

    def _solve():
        return solve_monge_ampere_2d(
            eqn, kernel, X_domain, X_boundary, sol_init,
            sol_init_xx, sol_init_xy, sol_init_yy,
            nugget=args.nugget, GN_steps=args.GN_steps,
            rho_big=args.rho_big, rho_small=args.rho_small,
            k_neighbors=args.k_neighbors, backend=args.backend,
            verbose=True,
        )

    t0 = time.perf_counter()
    sol = _solve()
    t1 = time.perf_counter()
    print(f'\n[fast solve] wall time: {t1 - t0:.3f} s')

    if args.solve_twice:
        t2 = time.perf_counter()
        sol = _solve()
        t3 = time.perf_counter()
        print(f'[fast solve, 2nd] wall time: {t3 - t2:.3f} s  (steady-state)')

    err = truth - sol
    print(f'[accuracy] L2 = {np.sqrt(np.mean(err ** 2)):.3e}, '
          f'Linf = {np.max(np.abs(err)):.3e}')


if __name__ == '__main__':
    main()
