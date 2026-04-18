"""KL factorization — the numerical work after the ordering is fixed.

Pipeline per supernode (same math as KLMinimization.jl):

    I  = row_indices,  C = column_indices           (I, C are ascending, C ⊂ I)
    K  = kernel(m[I], m[I])                         (dense, size |I|×|I|)
    L  = cholesky(K + nugget * diag(K))             (lower triangular)
    For each c in C, let p = position of c in I:
        U[I[:p+1], c] = (L^{-T} @ e_p)[:p+1]

The result is a scipy.sparse.csc_matrix upper-triangular factor s.t.
U^T U approximates K^{-1} restricted to the KL-optimal sparsity pattern.

Two implementations are provided:

* `factorize_cpu`: supernode-by-supernode with scipy/lapack. Correct,
  low dispatch overhead per call, excellent for O(10^3-10^4) supernodes.
* `factorize_jax`: size-bucketed batched Cholesky + triangular solve via
  `jax.vmap` + `jit`. Kicks in automatically when the JAX backend is GPU
  (or when the caller forces it). Compiles once per size bucket, so
  keeping the bucket count small (default: power-of-two ceiling) is
  important for throughput.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import scipy.linalg
import scipy.sparse

import jax
import jax.numpy as jnp
from jax import jit

from . import measurements as _m
from .covariance import AbstractCovarianceFunction
from .supernodes import (
    IndexSuperNode,
    IndirectSupernodalAssignment,
    ordering_and_sparsity_pattern,
)
from .ordering import maximin_ordering


def _default_num_threads() -> int:
    env = os.environ.get('KOLESKY_NUM_THREADS')
    if env:
        return max(1, int(env))
    # pick a reasonable default: up to 32 threads, bounded by cores
    return min(32, max(1, os.cpu_count() or 1))


# ---------------------------------------------------------------------------
# Sparsity shell
# ---------------------------------------------------------------------------


def _build_sparsity(supernodes: List[IndexSuperNode], N: int) -> scipy.sparse.csc_matrix:
    """Construct the (empty-data) CSC matrix with the KL sparsity pattern."""
    rows_parts: List[np.ndarray] = []
    cols_parts: List[np.ndarray] = []
    for node in supernodes:
        I = node.row_indices
        C = node.column_indices
        for c in C:
            rows = I[I <= c]
            rows_parts.append(rows)
            cols_parts.append(np.full(rows.size, c, dtype=np.int64))
    if rows_parts:
        rows = np.concatenate(rows_parts)
        cols = np.concatenate(cols_parts)
    else:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
    data = np.zeros(rows.size, dtype=np.float64)
    U = scipy.sparse.csc_matrix((data, (rows, cols)), shape=(N, N))
    U.sum_duplicates()
    U.sort_indices()
    return U


# ---------------------------------------------------------------------------
# CPU path
# ---------------------------------------------------------------------------


def _process_one_supernode(node, kernel, measurements, nugget, U_indptr, U_data):
    """Factorize a single supernode and write into U_data in place.

    U_data is the backing ndarray of a csc_matrix — rows for column c are
    stored contiguously at U_indptr[c]:U_indptr[c+1], so disjoint columns
    can be written from separate threads without locking.
    """
    I = node.row_indices
    C = node.column_indices
    if I.size == 0 or C.size == 0:
        return
    m_I = _m.select(measurements, I)
    # ``np.array(..., copy=True)`` ensures a writeable, owned buffer — some
    # kernels return JAX-backed arrays that are read-only once wrapped.
    K = np.array(kernel(m_I, m_I), dtype=np.float64, copy=True, order='C')
    if nugget != 0.0:
        d = np.arange(K.shape[0])
        K[d, d] *= (1.0 + nugget)
    K = 0.5 * (K + K.T)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-12 * np.trace(K) / K.shape[0] * np.eye(K.shape[0])
        L = np.linalg.cholesky(K)
    positions = np.searchsorted(I, C)
    rhs = np.zeros((I.size, C.size), dtype=np.float64)
    rhs[positions, np.arange(C.size)] = 1.0
    X = scipy.linalg.solve_triangular(L.T, rhs, lower=False, check_finite=False)
    for k in range(C.size):
        c = int(C[k])
        pos = int(positions[k])
        start = U_indptr[c]
        end = U_indptr[c + 1]
        U_data[start:end] = X[: pos + 1, k]


def factorize_cpu(
    kernel: AbstractCovarianceFunction,
    supernodes: List[IndexSuperNode],
    measurements,
    nugget: float = 0.0,
    num_threads: Optional[int] = None,
) -> scipy.sparse.csc_matrix:
    """Factorize per-supernode on CPU. Uses a thread pool over supernodes;
    each worker thread pins BLAS to 1 thread so they don't fight.

    Set KOLESKY_NUM_THREADS env var or pass num_threads to tune.
    """
    N = _m.size(measurements)
    U = _build_sparsity(supernodes, N)

    if num_threads is None:
        num_threads = _default_num_threads()

    # Ensure threading libs don't oversubscribe. These env vars are read by
    # OpenBLAS/MKL at thread-pool init time; setting here also affects the
    # current process's subsequent calls.
    for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ.setdefault(var, '1')
    try:
        # best-effort runtime thread-count toggle
        from threadpoolctl import threadpool_limits  # type: ignore
        limiter = threadpool_limits(limits=1, user_api='blas')
    except Exception:
        limiter = None

    try:
        if num_threads <= 1 or len(supernodes) < 4:
            for node in supernodes:
                _process_one_supernode(
                    node, kernel, measurements, nugget, U.indptr, U.data
                )
        else:
            with ThreadPoolExecutor(max_workers=num_threads) as ex:
                list(ex.map(
                    lambda n: _process_one_supernode(
                        n, kernel, measurements, nugget, U.indptr, U.data
                    ),
                    supernodes,
                ))
    finally:
        if limiter is not None:
            try:
                limiter.__exit__(None, None, None)
            except Exception:
                pass

    return U


# ---------------------------------------------------------------------------
# JAX batched path
# ---------------------------------------------------------------------------


def _bucket_sizes(sizes: np.ndarray, base: float = 1.25) -> np.ndarray:
    """Assign each size to a bucket; bucket bound grows geometrically by `base`.
    Returns an int array of bucket IDs and the list of bucket upper-bounds.
    """
    if sizes.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    smax = int(sizes.max())
    # generate bucket upper bounds: 1, ceil(base), ceil(base^2), ... >= smax
    bounds = []
    b = 1
    while b < smax:
        bounds.append(b)
        b = max(b + 1, int(np.ceil(b * base)))
    bounds.append(smax)
    bounds = np.asarray(bounds, dtype=np.int64)
    # assign: bucket[i] = smallest j such that bounds[j] >= sizes[i]
    bucket = np.searchsorted(bounds, sizes, side='left')
    return bucket, bounds


def _make_bucket_kernel_pp(F_scalar):
    """Bucket kernel for PointMeasurement × PointMeasurement."""

    def _impl(coord, row_idx, row_valid, rhs, a, nugget, jitter):
        xs = coord[row_idx]  # (B, S, d)
        diff = xs[:, :, None, :] - xs[:, None, :, :]
        t2 = jnp.sum(diff * diff, axis=-1)
        t = jnp.sqrt(jnp.maximum(t2, 0.0))
        K = F_scalar(t, a)
        S = row_valid.shape[1]
        idx = jnp.arange(S)
        K = K.at[:, idx, idx].multiply(1.0 + nugget)
        valid_pair = row_valid[:, :, None] & row_valid[:, None, :]
        eye = jnp.eye(S, dtype=K.dtype)[None]
        K = jnp.where(valid_pair, K, eye)
        K = 0.5 * (K + jnp.swapaxes(K, -1, -2))
        # Unconditional tiny regularization. Scaled by the average diagonal
        # entry so it's relative. Mirrors the CPU path's fallback jitter,
        # but we do it up-front because jnp.linalg.cholesky returns NaN
        # (rather than raising) on near-singular input.
        trace_avg = jnp.trace(K, axis1=-2, axis2=-1) / S
        K = K + jitter * trace_avg[:, None, None] * eye
        L = jnp.linalg.cholesky(K)
        X = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(L, -1, -2), rhs, lower=False
        )
        return X

    return jit(_impl)


def _make_bucket_kernel_lgdlgd(F, D2F, D4F, DF, D3F, DDF):
    """Bucket kernel for LaplaceGradDiracPointMeasurement × LaplaceGradDiracPointMeasurement."""

    def _impl(coord, wl, wg, wd, row_idx, row_valid, rhs, a, nugget, jitter):
        xs = coord[row_idx]              # (B, S, d)
        wlx = wl[row_idx]                # (B, S)
        wdx = wd[row_idx]
        wgx = wg[row_idx]                # (B, S, d)
        diff = xs[:, :, None, :] - xs[:, None, :, :]   # (B, S, S, d)
        t2 = jnp.sum(diff * diff, axis=-1)
        t = jnp.sqrt(jnp.maximum(t2, 0.0))
        d = xs.shape[-1]
        f = F(t, a)
        d2 = D2F(t, a, d)
        d4 = D4F(t, a, d)
        df = DF(t, a)
        d3 = D3F(t, a, d)
        ddf = DDF(t, a)

        # symmetric within a supernode: y slots == x slots
        wly = wlx
        wdy = wdx
        wgy = wgx

        diff_wgy = jnp.einsum('bijk,bjk->bij', diff, wgy)
        diff_wgx = jnp.einsum('bijk,bik->bij', diff, wgx)
        wgx_wgy = jnp.einsum('bik,bjk->bij', wgx, wgy)

        K = (
            wlx[:, :, None] * wly[:, None, :] * d4
            + (wdx[:, :, None] * wly[:, None, :] + wlx[:, :, None] * wdy[:, None, :]) * d2
            + wdx[:, :, None] * wdy[:, None, :] * f
            - wlx[:, :, None] * d3 * diff_wgy
            + wly[:, None, :] * d3 * diff_wgx
            - wdx[:, :, None] * df * diff_wgy
            + wdy[:, None, :] * df * diff_wgx
            + (-wgx_wgy * df + diff_wgx * (-diff_wgy) * ddf)
        )

        S = row_valid.shape[1]
        idx = jnp.arange(S)
        K = K.at[:, idx, idx].multiply(1.0 + nugget)
        valid_pair = row_valid[:, :, None] & row_valid[:, None, :]
        eye = jnp.eye(S, dtype=K.dtype)[None]
        K = jnp.where(valid_pair, K, eye)
        K = 0.5 * (K + jnp.swapaxes(K, -1, -2))
        trace_avg = jnp.trace(K, axis1=-2, axis2=-1) / S
        K = K + jitter * trace_avg[:, None, None] * eye
        L = jnp.linalg.cholesky(K)
        X = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(L, -1, -2), rhs, lower=False
        )
        return X

    return jit(_impl)


def _make_bucket_kernel_hdhd(F_xy_scalar):
    """Bucket kernel for HessianDiracPointMeasurement × HessianDiracPointMeasurement (2D).

    Uses JAX autodiff through F_xy(x, y, a) for the Hessian / Hessian-of-Hessian
    derivatives. JIT compile is per supernode output-shape (S, M), same as the
    other bucket kernels.
    """

    def pair_K(x, y, a, wxd, wxh, wyd, wyh):
        F = F_xy_scalar(x, y, a)
        Hx = jax.hessian(lambda xv: F_xy_scalar(xv, y, a))(x)
        HxV = jnp.array([Hx[0, 0], Hx[0, 1], Hx[1, 1]])
        Hy = jax.hessian(lambda yv: F_xy_scalar(x, yv, a))(y)
        HyV = jnp.array([Hy[0, 0], Hy[0, 1], Hy[1, 1]])

        def HxF(xv, yv):
            h = jax.hessian(lambda zv: F_xy_scalar(zv, yv, a))(xv)
            return jnp.array([h[0, 0], h[0, 1], h[1, 1]])

        T = jax.hessian(lambda yv: HxF(x, yv))(y)  # (3, 2, 2)
        T33 = jnp.stack([T[:, 0, 0], T[:, 0, 1], T[:, 1, 1]], axis=1)
        return (
            wxd * wyd * F
            + wxd * jnp.dot(HyV, wyh)
            + wyd * jnp.dot(HxV, wxh)
            + wxh @ T33 @ wyh
        )

    over_j = jax.vmap(pair_K, in_axes=(None, 0, None, None, None, 0, 0))
    over_ij = jax.vmap(over_j, in_axes=(0, None, None, 0, 0, None, None))
    over_bij = jax.vmap(over_ij, in_axes=(0, 0, None, 0, 0, 0, 0))

    def _impl(coord, w_hess, w_delta, row_idx, row_valid, rhs, a, nugget, jitter):
        xs = coord[row_idx]             # (B, S, 2)
        wh = w_hess[row_idx]            # (B, S, 3)
        wd = w_delta[row_idx]           # (B, S)
        K = over_bij(xs, xs, a, wd, wh, wd, wh)   # (B, S, S)

        S = row_valid.shape[1]
        idx = jnp.arange(S)
        K = K.at[:, idx, idx].multiply(1.0 + nugget)
        valid_pair = row_valid[:, :, None] & row_valid[:, None, :]
        eye = jnp.eye(S, dtype=K.dtype)[None]
        K = jnp.where(valid_pair, K, eye)
        K = 0.5 * (K + jnp.swapaxes(K, -1, -2))
        trace_avg = jnp.trace(K, axis1=-2, axis2=-1) / S
        K = K + jitter * trace_avg[:, None, None] * eye
        L = jnp.linalg.cholesky(K)
        X = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(L, -1, -2), rhs, lower=False
        )
        return X

    return jit(_impl)


def _make_bucket_kernel_ldld(F_scalar, D2F_scalar, D4F_scalar):
    """Bucket kernel for LaplaceDiracPointMeasurement × LaplaceDiracPointMeasurement.
    Uses the closed-form Δδ×Δδ covariance in terms of F, D2F, D4F.
    """

    def _impl(coord, wl, wd, row_idx, row_valid, rhs, a, nugget, jitter):
        xs = coord[row_idx]                       # (B, S, d)
        wlx = wl[row_idx]                         # (B, S)
        wdx = wd[row_idx]
        diff = xs[:, :, None, :] - xs[:, None, :, :]
        t2 = jnp.sum(diff * diff, axis=-1)
        t = jnp.sqrt(jnp.maximum(t2, 0.0))
        d = xs.shape[-1]
        f = F_scalar(t, a)
        d2 = D2F_scalar(t, a, d)
        d4 = D4F_scalar(t, a, d)
        # K[b, i, j] = wlx_i wly_j D4 + (wdx_i wly_j + wlx_i wdy_j) D2 + wdx_i wdy_j F
        # where both sides come from the same batch slice.
        wly = wlx
        wdy = wdx
        K = (
            wlx[:, :, None] * wly[:, None, :] * d4
            + (wdx[:, :, None] * wly[:, None, :] + wlx[:, :, None] * wdy[:, None, :]) * d2
            + wdx[:, :, None] * wdy[:, None, :] * f
        )
        S = row_valid.shape[1]
        idx = jnp.arange(S)
        K = K.at[:, idx, idx].multiply(1.0 + nugget)
        valid_pair = row_valid[:, :, None] & row_valid[:, None, :]
        eye = jnp.eye(S, dtype=K.dtype)[None]
        K = jnp.where(valid_pair, K, eye)
        K = 0.5 * (K + jnp.swapaxes(K, -1, -2))
        trace_avg = jnp.trace(K, axis1=-2, axis2=-1) / S
        K = K + jitter * trace_avg[:, None, None] * eye
        L = jnp.linalg.cholesky(K)
        X = jax.scipy.linalg.solve_triangular(
            jnp.swapaxes(L, -1, -2), rhs, lower=False
        )
        return X

    return jit(_impl)


# Cache compiled bucket-kernel per (kernel-class, measurement-type).
_bucket_kernel_cache = {}


def _get_bucket_kernel(kernel: AbstractCovarianceFunction, meas_type: type):
    from .covariance import (
        _jax_F_of,
        MaternCovariance5_2, MaternCovariance7_2, MaternCovariance9_2, GaussianCovariance,
        _SQRT5, _SQRT7,
    )
    from .measurements import (
        PointMeasurement, DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement,
        LaplaceGradDiracPointMeasurement, HessianDiracPointMeasurement,
    )
    from .covariance import (
        _matern52_F_xy, _matern72_F_xy, _gauss_F_xy,
    )

    key = (type(kernel), meas_type)
    fn = _bucket_kernel_cache.get(key)
    if fn is not None:
        return fn

    F_scalar = _jax_F_of(kernel)

    def _derivs_for(kcls):
        """Return jnp (D2F, D4F, DF, D3F, DDF) for a given kernel class."""
        if kcls is MaternCovariance5_2:
            def D2F(t, a, d):
                return -5.0 * (d * a * a + _SQRT5 * d * a * t - 5.0 * t * t) / (3.0 * a ** 4) * jnp.exp(-_SQRT5 * t / a)

            def D4F(t, a, d):
                return 25.0 * (d * (d + 2) * a * a - (3 + 2 * d) * _SQRT5 * a * t + 5.0 * t * t) / (3.0 * a ** 6) * jnp.exp(-_SQRT5 * t / a)

            def DF(t, a):
                return -5.0 * (a + _SQRT5 * t) * jnp.exp(-_SQRT5 * t / a) / (3.0 * a ** 3)

            def D3F(t, a, d):
                return 25.0 * jnp.exp(-_SQRT5 * t / a) * (a * (2 + d) - _SQRT5 * t) / (3.0 * a ** 5)

            def DDF(t, a):
                return 25.0 * jnp.exp(-_SQRT5 * t / a) / (3.0 * a ** 4)
        elif kcls is MaternCovariance7_2:
            def D2F(t, a, d):
                return -7.0 * (3.0 * d * a ** 3 + 3.0 * _SQRT7 * a * a * d * t + 7.0 * a * (d - 1) * t * t - 7.0 * _SQRT7 * t ** 3) / (15.0 * a ** 5) * jnp.exp(-_SQRT7 * t / a)

            def D4F(t, a, d):
                return 49.0 * (d * (d + 2) * a ** 3 + d * (d + 2) * _SQRT7 * a * a * t - 14.0 * a * (2 + d) * t * t + 7.0 * _SQRT7 * t ** 3) / (15.0 * a ** 7) * jnp.exp(-_SQRT7 * t / a)

            def DF(t, a):
                return -7.0 * (3.0 * a * a + 3.0 * _SQRT7 * a * t + 7.0 * t * t) * jnp.exp(-_SQRT7 * t / a) / (15.0 * a ** 4)

            def D3F(t, a, d):
                return 49.0 * jnp.exp(-_SQRT7 * t / a) * (a * a * (2 + d) + _SQRT7 * a * (2 + d) * t - 7.0 * t * t) / (15.0 * a ** 6)

            def DDF(t, a):
                return 49.0 * jnp.exp(-_SQRT7 * t / a) * (a + _SQRT7 * t) / (15.0 * a ** 5)
        elif kcls is MaternCovariance9_2:
            def D2F(t, a, d):
                return -9.0 * (5.0 * d * a ** 4 + 15.0 * d * a ** 3 * t + 9.0 * a * a * (2 * d - 1) * t * t + 9.0 * a * (d - 3) * t ** 3 - 27.0 * t ** 4) / (35.0 * a ** 6) * jnp.exp(-3.0 * t / a)

            def D4F(t, a, d):
                return 81.0 * (d * (d + 2) * a ** 4 + 3.0 * a ** 3 * d * (d + 2) * t + 3.0 * a * a * (d * d - 4) * t * t - 18.0 * a * (d + 2) * t ** 3 + 27.0 * t ** 4) / (35.0 * a ** 8) * jnp.exp(-3.0 * t / a)

            # DF, D3F, DDF for Matern 9/2 not supplied by Julia; not supported here either.
            DF = D3F = DDF = None
        elif kcls is GaussianCovariance:
            def D2F(t, a, d):
                return (t * t - a * a * d) / (a ** 4) * jnp.exp(-t * t / (2.0 * a * a))

            def D4F(t, a, d):
                return (a ** 4 * d * (2 + d) - 2.0 * a * a * (2 + d) * t * t + t ** 4) * jnp.exp(-t * t / (2.0 * a * a)) / (a ** 8)

            def DF(t, a):
                return -jnp.exp(-t * t / (2.0 * a * a)) / (a * a)

            def D3F(t, a, d):
                return jnp.exp(-t * t / (2.0 * a * a)) * (a * a * (2 + d) - t * t) / (a ** 6)

            def DDF(t, a):
                return jnp.exp(-t * t / (2.0 * a * a)) / (a ** 4)
        else:
            return None, None, None, None, None
        return D2F, D4F, DF, D3F, DDF

    if meas_type is PointMeasurement:
        fn = _make_bucket_kernel_pp(F_scalar)
    elif meas_type is LaplaceGradDiracPointMeasurement:
        kcls = type(kernel)
        D2F, D4F, DF, D3F, DDF = _derivs_for(kcls)
        if D2F is None or DF is None:
            raise NotImplementedError(
                f'JAX-batched Δ∇δ factorization not implemented for {kcls.__name__}'
            )
        fn = _make_bucket_kernel_lgdlgd(F_scalar, D2F, D4F, DF, D3F, DDF)
    elif meas_type in (DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement):
        kcls = type(kernel)
        D2F, D4F, _DF, _D3F, _DDF = _derivs_for(kcls)
        if D2F is None:
            raise NotImplementedError(
                f'JAX-batched Δδ factorization not implemented for {kcls.__name__}'
            )
        fn = _make_bucket_kernel_ldld(F_scalar, D2F, D4F)
    elif meas_type is HessianDiracPointMeasurement:
        kcls = type(kernel)
        if kcls is MaternCovariance5_2:
            F_xy = _matern52_F_xy
        elif kcls is MaternCovariance7_2:
            F_xy = _matern72_F_xy
        elif kcls is GaussianCovariance:
            F_xy = _gauss_F_xy
        else:
            raise NotImplementedError(
                f'JAX-batched HessianDirac factorization not implemented for {kcls.__name__}'
            )
        fn = _make_bucket_kernel_hdhd(F_xy)
    else:
        raise NotImplementedError(
            f'JAX-batched factorization not implemented for measurement type {meas_type.__name__}'
        )

    _bucket_kernel_cache[key] = fn
    return fn


def factorize_jax(
    kernel: AbstractCovarianceFunction,
    supernodes: List[IndexSuperNode],
    measurements,
    nugget: float = 0.0,
    bucket_growth: float = 1.5,
    jitter: float = 1e-10,
) -> scipy.sparse.csc_matrix:
    """GPU-resident variant: gathers coordinate slices, computes kernel,
    Cholesky, and triangular solve entirely on device. Host↔device transfer
    is per-bucket: upload (row_idx, row_valid, rhs), download X.

    Only the PointMeasurement case is supported (this is what main.jl uses).
    """
    from .measurements import (
        PointMeasurement, DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement,
        LaplaceGradDiracPointMeasurement, HessianDiracPointMeasurement,
    )

    N = _m.size(measurements)
    meas_type = type(measurements)
    supported = (
        PointMeasurement, DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement,
        LaplaceGradDiracPointMeasurement, HessianDiracPointMeasurement,
    )
    if meas_type not in supported:
        raise NotImplementedError(
            f'factorize_jax supports PointMeasurement / Δδ / Δ∇δ / ∂∂+δ; '
            f'got {meas_type.__name__}. Use backend="cpu" for other measurement types.'
        )

    U = _build_sparsity(supernodes, N)

    coord_device = jnp.asarray(measurements.coordinate)
    wl_device = None
    wd_device = None
    wg_device = None
    wh_device = None
    if meas_type in (DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement,
                     LaplaceGradDiracPointMeasurement):
        wl_device = jnp.asarray(measurements.weight_laplace)
        wd_device = jnp.asarray(measurements.weight_delta)
    if meas_type is LaplaceGradDiracPointMeasurement:
        wg_device = jnp.asarray(measurements.weight_grad)
    if meas_type is HessianDiracPointMeasurement:
        wh_device = jnp.stack([
            jnp.asarray(measurements.weight_11),
            jnp.asarray(measurements.weight_12),
            jnp.asarray(measurements.weight_22),
        ], axis=1)
        wd_device = jnp.asarray(measurements.weight_delta)

    a = kernel.length_scale
    bucket_fn = _get_bucket_kernel(kernel, meas_type)

    row_sizes = np.asarray([n.row_indices.size for n in supernodes], dtype=np.int64)
    col_sizes = np.asarray([n.column_indices.size for n in supernodes], dtype=np.int64)
    buckets, _bounds = _bucket_sizes(row_sizes, base=bucket_growth)

    for b in np.unique(buckets):
        idx = np.flatnonzero(buckets == b)
        S = int(row_sizes[idx].max())
        M = int(col_sizes[idx].max())
        B = idx.size

        row_idx = np.zeros((B, S), dtype=np.int32)
        row_valid = np.zeros((B, S), dtype=np.bool_)
        rhs_buf = np.zeros((B, S, M), dtype=np.float64)
        meta = []
        for bi, si in enumerate(idx):
            node = supernodes[int(si)]
            I = node.row_indices
            C = node.column_indices
            n = I.size
            row_idx[bi, :n] = I
            row_valid[bi, :n] = True
            positions = np.searchsorted(I, C)
            if not np.all(I[positions] == C):
                raise RuntimeError('column_indices is not a subset of row_indices')
            rhs_buf[bi, positions, np.arange(C.size)] = 1.0
            meta.append((int(si), positions, C))

        if meas_type is PointMeasurement:
            X = bucket_fn(
                coord_device,
                jnp.asarray(row_idx),
                jnp.asarray(row_valid),
                jnp.asarray(rhs_buf),
                a, nugget, jitter,
            )
        elif meas_type is LaplaceGradDiracPointMeasurement:
            X = bucket_fn(
                coord_device, wl_device, wg_device, wd_device,
                jnp.asarray(row_idx),
                jnp.asarray(row_valid),
                jnp.asarray(rhs_buf),
                a, nugget, jitter,
            )
        elif meas_type is HessianDiracPointMeasurement:
            X = bucket_fn(
                coord_device, wh_device, wd_device,
                jnp.asarray(row_idx),
                jnp.asarray(row_valid),
                jnp.asarray(rhs_buf),
                a, nugget, jitter,
            )
        else:
            X = bucket_fn(
                coord_device, wl_device, wd_device,
                jnp.asarray(row_idx),
                jnp.asarray(row_valid),
                jnp.asarray(rhs_buf),
                a, nugget, jitter,
            )
        # single host↔device transfer per bucket
        X = np.asarray(X)

        for bi, (si, positions, C) in enumerate(meta):
            for k in range(C.size):
                c = int(C[k])
                pos = int(positions[k])
                start = U.indptr[c]
                end = U.indptr[c + 1]
                U.data[start:end] = X[bi, : pos + 1, k]

    return U


# ---------------------------------------------------------------------------
# Factor classes
# ---------------------------------------------------------------------------


def _measurements_from_list(measurements):
    """Permit callers to pass either a single batched measurement object or a
    Python list/tuple of them (each possibly batched).
    """
    if isinstance(measurements, (list, tuple)):
        # concatenate. If it's a list of lists (multi-set case), flatten first.
        if len(measurements) > 0 and isinstance(measurements[0], (list, tuple)):
            flat = []
            for sub in measurements:
                flat.extend(sub)
            return _m.stack_measurements(flat)
        return _m.stack_measurements(list(measurements))
    return measurements


def _coords_list_from_measurements_list(measurements_list):
    """Return a list of (N_k, d) arrays from a list (or list of lists) of
    measurements, matching the Julia multi-set layout.
    """
    out = []
    for bucket in measurements_list:
        if isinstance(bucket, (list, tuple)):
            coords = np.stack([np.atleast_1d(m.coordinate) for m in bucket], axis=0)
            out.append(coords)
        else:
            out.append(np.atleast_2d(bucket.coordinate))
    return out


@dataclass
class ImplicitKLFactorization:
    """Stores the ordering, sparsity pattern, and permuted measurements —
    but no numerical factor. Call `ExplicitKLFactorization(implicit)` to
    materialize the sparse Cholesky factor.
    """

    P: np.ndarray
    supernodes: IndirectSupernodalAssignment
    kernel: AbstractCovarianceFunction

    @classmethod
    def build_diracs_first_then_unif_scale(
        cls,
        kernel: AbstractCovarianceFunction,
        measurements,
        rho: float,
        k_neighbors: Optional[int] = None,
        lambda_: float = 1.5,
        alpha: float = 1.0,
    ) -> 'ImplicitKLFactorization':
        """DiracsFirstThenUnifScale ordering (matches Julia's
        ImplicitKLFactorization_DiracsFirstThenUnifScale).

        Layout same as build_follow_diracs — first 2 sets used for maximin
        ordering, all domain-located sets re-use the same relative order —
        BUT the remaining sets are concatenated *after* the δ_int block
        (instead of interleaved per-point). Length-scales for the repeated
        blocks use the smallest ℓ seen in the 2-set ordering.
        """
        import warnings

        if not isinstance(measurements, (list, tuple)) or len(measurements) < 3:
            raise ValueError('build_diracs_first_then_unif_scale expects a list of 3+ measurement groups')

        batched_groups = [
            _m.stack_measurements(list(b)) if isinstance(b, (list, tuple)) else b
            for b in measurements
        ]
        if len(set(type(g) for g in batched_groups)) > 1:
            raise TypeError(
                'build_diracs_first_then_unif_scale: all measurement groups must be the same concrete type.'
            )
        lm = len(batched_groups)
        N_bdy = batched_groups[0].coordinate.shape[0]
        N_dom = batched_groups[1].coordinate.shape[0]
        for k in range(2, lm):
            if batched_groups[k].coordinate.shape[0] != N_dom:
                raise ValueError('domain groups must all have the same length')

        # maximin on (boundary, domain_δ)
        coord_groups_2 = [batched_groups[0].coordinate, batched_groups[1].coordinate]
        P, ell, _sns = ordering_and_sparsity_pattern(
            coord_groups_2, rho, k_neighbors=k_neighbors, lambda_=lambda_, alpha=alpha,
        )
        # discard the sns; we rebuild from the expanded P_all below

        n_dom_sets = lm - 1
        n_total = N_bdy + n_dom_sets * N_dom
        P_all = np.empty(n_total, dtype=np.int64)
        P_all[:N_bdy] = P[:N_bdy]
        P_dom = P[N_bdy:]  # (N_dom,)  values in [N_bdy, N_bdy+N_dom)
        for s in range(n_dom_sets):
            P_all[N_bdy + s * N_dom : N_bdy + (s + 1) * N_dom] = P_dom + s * N_dom

        ell_all = np.empty(n_total, dtype=np.float64)
        ell_all[:N_bdy + N_dom] = ell
        ell_all[N_bdy + N_dom:] = ell[-1]

        x_cat = np.concatenate([g.coordinate for g in batched_groups], axis=0)
        from .supernodes import supernodal_reverse_maximin_sparsity_pattern
        sns = supernodal_reverse_maximin_sparsity_pattern(
            x_cat, P_all, ell_all, rho, lambda_=lambda_, alpha=alpha,
        )

        merged = _m.stack_measurements(batched_groups)
        assert _m.size(merged) == n_total
        permuted = _m.select(merged, P_all)
        assignment = IndirectSupernodalAssignment(supernodes=sns, measurements=permuted)
        return cls(P=P_all, supernodes=assignment, kernel=kernel)

    @classmethod
    def build_follow_diracs(
        cls,
        kernel: AbstractCovarianceFunction,
        measurements,
        rho: float,
        k_neighbors: Optional[int] = None,
        lambda_: float = 1.5,
        alpha: float = 1.0,
    ) -> 'ImplicitKLFactorization':
        """FollowDiracs ordering (matches Julia's
        ImplicitKLFactorization_FollowDiracs).

        `measurements` must be a list of `lm >= 3` groups, all of the same
        measurement type:
            [boundary_points,
             domain_points (δ variant),
             domain_points (derivative variant #1),
             ...]
        All domain groups must live at the same coordinates (in the same
        order). The maximin ordering is computed only on (boundary, domain_δ).
        For each domain point, the remaining (lm - 2) derivative measurements
        are inserted immediately after its δ index in the ordering — so
        matching δ / derivative pairs end up in the same supernode, which
        is what the paper relies on for accuracy.
        """
        if not isinstance(measurements, (list, tuple)) or len(measurements) < 3:
            raise ValueError('build_follow_diracs expects a list of 3+ measurement groups')

        batched_groups = [
            _m.stack_measurements(list(b)) if isinstance(b, (list, tuple)) else b
            for b in measurements
        ]
        if len(set(type(g) for g in batched_groups)) > 1:
            raise TypeError(
                'build_follow_diracs: all measurement groups must be the same concrete type. '
                'Promote e.g. with LaplaceDiracPointMeasurement(..., weight_laplace=0, weight_delta=1) first.'
            )
        lm = len(batched_groups)
        N_bdy = batched_groups[0].coordinate.shape[0]
        N_dom = batched_groups[1].coordinate.shape[0]
        for k in range(2, lm):
            if batched_groups[k].coordinate.shape[0] != N_dom:
                raise ValueError('domain groups must all have the same length')

        # maximin ordering on [boundary, domain_δ] only
        coord_groups = [batched_groups[0].coordinate, batched_groups[1].coordinate]
        P, _ell, sns = ordering_and_sparsity_pattern(
            coord_groups, rho, k_neighbors=k_neighbors, lambda_=lambda_, alpha=alpha,
        )
        assert P.shape[0] == N_bdy + N_dom

        n_dom_sets = lm - 1  # number of groups that live at domain points
        P_all = np.empty(N_bdy + n_dom_sets * N_dom, dtype=np.int64)
        P_all[:N_bdy] = P[:N_bdy]
        # For each interior pick P[N_bdy + k], in the full ordering it gets
        # n_dom_sets adjacent positions carrying the δ and each derivative.
        # Global flat indices in the stacked measurement are:
        #   boundary[i]   → i
        #   domain_δ[j]   → N_bdy + j
        #   domain_ds[j]  → N_bdy + s*N_dom + j   (s = 1..lm-2)
        offsets = np.arange(n_dom_sets, dtype=np.int64) * N_dom
        # P[N_bdy:] holds the domain ordering (values in [N_bdy, N_bdy+N_dom))
        P_dom = P[N_bdy:]
        P_all_dom = P_dom[:, None] + offsets[None, :]  # (N_dom, n_dom_sets)
        P_all[N_bdy:] = P_all_dom.reshape(-1)

        # Expand each supernode: any index k ≥ N_bdy becomes n_dom_sets
        # consecutive new indices.
        n_total = N_bdy + n_dom_sets * N_dom

        def _expand(indices: np.ndarray) -> np.ndarray:
            if indices.size == 0:
                return indices.astype(np.int64)
            bdy_part = indices[indices < N_bdy]
            int_part = indices[indices >= N_bdy]
            # an interior index k (with k >= N_bdy) corresponds to domain position
            # (k - N_bdy). In the new ordering, its block is at
            # [N_bdy + n_dom_sets*(k-N_bdy), N_bdy + n_dom_sets*(k-N_bdy) + n_dom_sets)
            dom_pos = int_part - N_bdy
            starts = N_bdy + n_dom_sets * dom_pos  # (len(int_part),)
            span = np.arange(n_dom_sets, dtype=np.int64)
            expanded = (starts[:, None] + span[None, :]).reshape(-1)
            out = np.concatenate([bdy_part.astype(np.int64), expanded])
            out.sort()
            return out

        new_sns = []
        for node in sns:
            new_sns.append(IndexSuperNode(
                row_indices=_expand(node.row_indices),
                column_indices=_expand(node.column_indices),
            ))

        merged = _m.stack_measurements(batched_groups)
        assert _m.size(merged) == n_total
        permuted = _m.select(merged, P_all)
        assignment = IndirectSupernodalAssignment(supernodes=new_sns, measurements=permuted)
        return cls(P=P_all, supernodes=assignment, kernel=kernel)

    @classmethod
    def build(
        cls,
        kernel: AbstractCovarianceFunction,
        measurements,
        rho: float,
        k_neighbors: Optional[int] = None,
        lambda_: float = 1.5,
        alpha: float = 1.0,
    ) -> 'ImplicitKLFactorization':
        """Build ordering + sparsity pattern.

        `measurements` accepts:
          * a single batched measurement (e.g. from `point_measurements`),
          * a list of batched measurements — treated as multi-set ordering
            (each group retains its relative position in the final ordering;
            mirrors Julia's `Vector{<:AbstractVector{<:AbstractPointMeasurement}}`).
        """
        if isinstance(measurements, (list, tuple)):
            batched_groups = [
                _m.stack_measurements(list(b)) if isinstance(b, (list, tuple)) else b
                for b in measurements
            ]
            coord_groups = [g.coordinate for g in batched_groups]
            P, _ell, sns = ordering_and_sparsity_pattern(
                coord_groups, rho,
                k_neighbors=k_neighbors, lambda_=lambda_, alpha=alpha,
            )
            permuted = _m.select(_m.stack_measurements(batched_groups), P)
        else:
            P, _ell, sns = ordering_and_sparsity_pattern(
                measurements.coordinate, rho,
                k_neighbors=k_neighbors, lambda_=lambda_, alpha=alpha,
            )
            permuted = _m.select(measurements, P)

        assignment = IndirectSupernodalAssignment(supernodes=sns, measurements=permuted)
        return cls(P=P.astype(np.int64), supernodes=assignment, kernel=kernel)


@dataclass
class ExplicitKLFactorization:
    P: np.ndarray
    measurements: object  # batched, P-ordered
    kernel: AbstractCovarianceFunction
    U: scipy.sparse.csc_matrix

    @classmethod
    def from_implicit(
        cls,
        implicit: ImplicitKLFactorization,
        nugget: float = 0.0,
        backend: str = 'auto',
    ) -> 'ExplicitKLFactorization':
        """Materialize the sparse Cholesky factor.

        backend:
            'cpu'  — scipy/lapack, one supernode at a time.
            'jax'  — JAX vmapped, size-bucketed. Use this on GPU.
            'auto' — 'jax' if jax.default_backend() != 'cpu', else 'cpu'.
        """
        if backend == 'auto':
            backend = 'jax' if jax.default_backend() != 'cpu' else 'cpu'

        sns = implicit.supernodes.supernodes
        meas = implicit.supernodes.measurements
        if backend == 'jax':
            U = factorize_jax(implicit.kernel, sns, meas, nugget=nugget)
        elif backend == 'cpu':
            U = factorize_cpu(implicit.kernel, sns, meas, nugget=nugget)
        else:
            raise ValueError(f'unknown backend: {backend}')

        return cls(P=implicit.P, measurements=meas, kernel=implicit.kernel, U=U)

    def __init__(self, *args, nugget: float = 0.0, backend: str = 'auto', **kwargs):
        """Convenience: call ExplicitKLFactorization(implicit) directly to
        build from an ImplicitKLFactorization, mirroring the Julia constructor.
        """
        if len(args) == 1 and isinstance(args[0], ImplicitKLFactorization):
            tmp = self.from_implicit(args[0], nugget=nugget, backend=backend)
            self.P = tmp.P
            self.measurements = tmp.measurements
            self.kernel = tmp.kernel
            self.U = tmp.U
            return
        if len(args) == 4:
            self.P, self.measurements, self.kernel, self.U = args
            return
        # dataclass-style
        self.P = kwargs['P']
        self.measurements = kwargs['measurements']
        self.kernel = kwargs['kernel']
        self.U = kwargs['U']


# ---------------------------------------------------------------------------
# Covariance assembly
# ---------------------------------------------------------------------------


def assemble_covariance(factor: ExplicitKLFactorization) -> np.ndarray:
    """Dense version of the approximate covariance K ≈ (U^T U)^{-1}, in the
    *original* (un-permuted) ordering. For debugging / small-N validation.
    """
    U_dense = factor.U.toarray()
    invU = np.linalg.inv(U_dense)
    inv_P = np.empty_like(factor.P)
    inv_P[factor.P] = np.arange(factor.P.shape[0])
    M = invU.T @ invU
    return M[np.ix_(inv_P, inv_P)]
