"""Python port of main_VarLinElliptic2d.jl.

Solves -∇·(a(x) ∇u) + α u^m = f with a(x) = exp(sin(5πx₁x₂)) and
ground-truth u from a truncated sin-sin series.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def ground_truth_factory(freq: int = 100, s: int = 3, alpha: float = 1.0, m: int = 3, k_var: int = 5):
    ks = np.arange(1, freq + 1)

    def u_exact(x):
        return float(np.sum(np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))

    def grad_u(x):
        du_dx = float(np.sum(np.pi * ks * np.cos(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1]) / ks ** s))
        du_dy = float(np.sum(np.pi * ks * np.sin(np.pi * ks * x[0]) * np.cos(np.pi * ks * x[1]) / ks ** s))
        return np.array([du_dx, du_dy])

    def fun_a(x):
        return float(np.exp(np.sin(k_var * np.pi * x[0] * x[1])))

    def grad_a(x):
        # d/dx_i  exp(sin(k π x_0 x_1)) = cos(...) · k π · x_{1-i} · exp(...)
        arg = k_var * np.pi * x[0] * x[1]
        fa = np.exp(np.sin(arg))
        c = np.cos(arg) * k_var * np.pi * fa
        return np.array([c * x[1], c * x[0]])

    def laplacian_u(x):
        # -Δu part (positive because PDE is -a Δu ...)
        val = float(np.sum(2 * ks ** 2 * np.pi ** 2
                            * np.sin(np.pi * ks * x[0]) * np.sin(np.pi * ks * x[1])
                            / ks ** s))
        return val  # = -Δu

    def rhs(x):
        # f = -∇·(a ∇u) + α u^m
        #   = -∇a · ∇u - a Δu + α u^m
        #   = -∇a · ∇u + a · (-Δu) + α u^m
        return float(-np.dot(grad_a(x), grad_u(x)) + fun_a(x) * laplacian_u(x) + alpha * u_exact(x) ** m)

    def bdy(x):
        return u_exact(x)

    return u_exact, bdy, rhs, fun_a, grad_a


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--m', type=int, default=3)
    p.add_argument('--kernel', choices=['Matern5half', 'Matern7half', 'Matern9half',
                                         'Matern11half', 'Gaussian'],
                   default='Matern7half')
    p.add_argument('--sigma', type=float, default=0.3)
    p.add_argument('--h', type=float, default=0.02)
    p.add_argument('--nugget', type=float, default=1e-10)
    p.add_argument('--GN-steps', type=int, default=3)
    p.add_argument('--rho-big', type=float, default=3.0)
    p.add_argument('--rho-small', type=float, default=3.0)
    p.add_argument('--k-neighbors', type=int, default=3)
    p.add_argument('--backend', choices=['auto', 'cpu', 'jax'], default='auto')
    p.add_argument('--platform', default='cpu')
    p.add_argument('--solve-twice', action='store_true')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ['JAX_PLATFORMS'] = args.platform
    import kolesky as kl
    from kolesky.pde import sample_points_grid_2d
    from kolesky.pde import VarLinElliptic2d, solve_var_lin_elliptic_2d

    u_exact, bdy_fn, rhs_fn, fun_a, grad_a = ground_truth_factory(
        freq=100, s=3, alpha=args.alpha, m=args.m, k_var=5,
    )
    eqn = VarLinElliptic2d(
        alpha=args.alpha, m=args.m, domain=((0.0, 1.0), (0.0, 1.0)),
        a=fun_a, grad_a=grad_a, bdy=bdy_fn, rhs=rhs_fn,
    )
    X_domain, X_boundary = sample_points_grid_2d(eqn.domain, args.h, args.h)
    print(f'[sample points] h = {args.h}, N_domain = {X_domain.shape[0]}, N_boundary = {X_boundary.shape[0]}')

    kernels = {
        'Matern5half': kl.MaternCovariance5_2,
        'Matern7half': kl.MaternCovariance7_2,
        'Matern9half': kl.MaternCovariance9_2,
        'Matern11half': kl.MaternCovariance11_2,
        'Gaussian': kl.GaussianCovariance,
    }
    kernel = kernels[args.kernel](args.sigma)
    print(f'[kernel] {args.kernel}, length_scale = {args.sigma}')

    sol_init = np.zeros(X_domain.shape[0])
    truth = np.array([u_exact(X_domain[i]) for i in range(X_domain.shape[0])])

    def _solve():
        return solve_var_lin_elliptic_2d(
            eqn, kernel, X_domain, X_boundary, sol_init,
            nugget=args.nugget, GN_steps=args.GN_steps,
            rho_big=args.rho_big, rho_small=args.rho_small,
            k_neighbors=args.k_neighbors, backend=args.backend,
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
