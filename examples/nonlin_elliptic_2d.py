"""Python port of main_NonLinElliptic2d.jl.

Solves -Δu + α u^m = f on [0,1]^2 via iterative Gauss-Newton + fast sparse
Cholesky factorization + pCG. Ground-truth solution is a truncated sin-sin
series.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def ground_truth_factory(freq: int = 600, s: int = 6, alpha: float = 1.0, m: int = 3):
    """Build (u_exact, rhs) pair for the test problem.

    u(x, y) = Σ_{k=1..freq} sin(π k x) sin(π k y) / k^s
    Then -Δu = Σ 2 k^2 π² sin(π k x) sin(π k y) / k^s.
    Finally f = -Δu + α u^m (so the PDE -Δu + α u^m = f is satisfied).
    """
    ks = np.arange(1, freq + 1)

    def u_exact(x):
        return float(np.sum(np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))

    def rhs(x):
        laplace_part = np.sum(
            2 * ks ** 2 * np.pi ** 2
            * np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1])
            / ks ** s
        )
        return float(laplace_part + alpha * u_exact(x) ** m)

    return u_exact, rhs


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--m', type=int, default=3)
    p.add_argument('--kernel', choices=['Matern5half', 'Matern7half', 'Matern9half',
                                         'Matern11half', 'Gaussian'],
                   default='Matern7half')
    p.add_argument('--sigma', type=float, default=0.3, help='length scale')
    p.add_argument('--h', type=float, default=0.02, help='grid spacing')
    p.add_argument('--nugget', type=float, default=1e-10)
    p.add_argument('--GN-steps', type=int, default=3)
    p.add_argument('--rho-big', type=float, default=3.0)
    p.add_argument('--rho-small', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--backend', choices=['auto', 'cpu', 'jax'], default='auto')
    p.add_argument('--platform', default='cpu',
                   help='jax platform: cpu or cuda')
    p.add_argument('--compare-exact', action='store_true')
    p.add_argument('--solve-twice', action='store_true',
                   help='solve twice to report a warm-cache timing')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform
    import kolesky as kl
    from kolesky.pde import (
        NonlinElliptic2d,
        sample_points_grid_2d,
        solve_nonlin_elliptic_2d,
        iterGPR_exact,
    )

    u_exact, rhs_fn = ground_truth_factory(freq=600, s=6, alpha=args.alpha, m=args.m)
    eqn = NonlinElliptic2d(
        alpha=args.alpha, m=args.m, domain=((0.0, 1.0), (0.0, 1.0)),
        bdy=u_exact, rhs=rhs_fn,
    )
    X_domain, X_boundary = sample_points_grid_2d(eqn.domain, args.h, args.h)
    print(f'[sample points] h = {args.h}')
    print(f'[sample points] N_domain = {X_domain.shape[0]}, N_boundary = {X_boundary.shape[0]}')

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
        'Matern11half': kl.MaternCovariance11_2,
        'Gaussian': kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel] {args.kernel}, length_scale = {args.sigma}')
    print(f'[nugget] {args.nugget}')
    print(f'[GN steps] {args.GN_steps}')
    print(f'[fast Cholesky] rho_big={args.rho_big}, rho_small={args.rho_small}, '
          f'k_neighbors={args.k_neighbors}')

    sol_init = np.zeros(X_domain.shape[0])
    truth = np.array([u_exact(X_domain[i]) for i in range(X_domain.shape[0])])

    def _fast_solve():
        return solve_nonlin_elliptic_2d(
            eqn, kernel, X_domain, X_boundary, sol_init,
            nugget=args.nugget, GN_steps=args.GN_steps,
            rho_big=args.rho_big, rho_small=args.rho_small,
            k_neighbors=args.k_neighbors,
            backend=args.backend,
        )

    t0 = time.perf_counter()
    sol = _fast_solve()
    t1 = time.perf_counter()
    print(f'\n[fast solve] wall time: {t1 - t0:.3f} s')

    if args.solve_twice:
        t2 = time.perf_counter()
        sol = _fast_solve()
        t3 = time.perf_counter()
        print(f'[fast solve, 2nd] wall time: {t3 - t2:.3f} s   (steady-state)')

    err = truth - sol
    L2 = float(np.sqrt(np.sum(err ** 2) / err.size))
    Linf = float(np.max(np.abs(err)))
    print(f'[accuracy] L2 = {L2:.3e}, Linf = {Linf:.3e}')

    if args.compare_exact:
        print('\n[exact reference solve] (dense)')
        t0 = time.perf_counter()
        sol_exact = iterGPR_exact(eqn, kernel, X_domain, X_boundary,
                                   sol_init, nugget=args.nugget, GN_steps=2)
        t1 = time.perf_counter()
        err_e = truth - sol_exact
        L2_e = float(np.sqrt(np.sum(err_e ** 2) / err_e.size))
        Linf_e = float(np.max(np.abs(err_e)))
        print(f'  time {t1 - t0:.3f} s, L2 = {L2_e:.3e}, Linf = {Linf_e:.3e}')


if __name__ == '__main__':
    main()
