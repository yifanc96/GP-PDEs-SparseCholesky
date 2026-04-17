"""Covariance kernels mirroring CovarianceFunctions.jl.

Two execution backends live in this module:

1. **NumPy "eager" path** — a direct `kernel(x, y)` call returns a numpy
   array computed via numpy broadcasting. This is what the CPU
   supernode-at-a-time factorization uses; each call is cheap and there
   is no per-shape JIT compile tax (which would dominate runtime when
   there are thousands of differently-shaped supernodes).

2. **JAX "batched" path** — for the GPU factorization backend we expose
   `kernel.pair_fn_jax(tuple_of_raw_arrays)` JIT-able closures over the
   batched Cholesky solve. These are compiled once per size bucket.

Kernels implemented:

* Matern 1/2, 3/2, 5/2, 7/2, 9/2, 11/2 on PointMeasurement × PointMeasurement
* Matern 5/2 on Δδ × Δδ, Δ∇δ × Δ∇δ, ∂∂ × ∂∂ (and p↔∂∂ cross-pairs in d=2)
* Matern 7/2 similarly (where Julia supplied it)
* Matern 9/2 on Δδ × Δδ and Δ∇δ × Δ∇δ
* Gaussian on PointMeasurement and on Δδ / Δ∇δ / ∂∂
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

# We keep JAX imports lazy-ish; they only matter for the batched path. But
# we DO want jax available as a module for factorization_jax, so import at
# top. We default to CPU JAX if not configured, to avoid a busy-GPU crash
# on import.
os.environ.setdefault('JAX_PLATFORMS', os.environ.get('JAX_PLATFORMS', 'cpu'))

import jax
import jax.numpy as jnp
from jax import jit

jax.config.update('jax_enable_x64', True)

from .measurements import (
    PointMeasurement,
    DeltaDiracPointMeasurement,
    LaplaceDiracPointMeasurement,
    LaplaceGradDiracPointMeasurement,
    PartialPartialPointMeasurement,
    HessianDiracPointMeasurement,
)


_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)
_SQRT7 = math.sqrt(7.0)
_SQRT11 = math.sqrt(11.0)


# ---------------------------------------------------------------------------
# NumPy pairwise distance utilities
# ---------------------------------------------------------------------------


def _np_pairwise_sq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance, rows of x (N, d) vs rows of y (M, d)."""
    x2 = np.einsum('ij,ij->i', x, x)[:, None]
    y2 = np.einsum('ij,ij->i', y, y)[None, :]
    cross = x @ y.T
    sq = x2 + y2 - 2.0 * cross
    np.maximum(sq, 0.0, out=sq)
    return sq


def _np_pairwise(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sqrt(_np_pairwise_sq(x, y))


def _np_pairwise_diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return (N, M, d) difference tensor."""
    return x[:, None, :] - y[None, :, :]


# ---------------------------------------------------------------------------
# NumPy scalar kernels F and derivatives (broadcast-friendly)
# ---------------------------------------------------------------------------


def _np_matern12_F(t, a):
    return np.exp(-t / a)


def _np_matern32_F(t, a):
    r = _SQRT3 * t / a
    return (1.0 + r) * np.exp(-r)


def _np_matern52_F(t, a):
    return (1.0 + _SQRT5 * t / a + 5.0 * t * t / (3.0 * a * a)) * np.exp(-_SQRT5 * t / a)


def _np_matern52_D2F(t, a, d):
    a2 = a * a
    return -5.0 * (d * a2 + _SQRT5 * d * a * t - 5.0 * t * t) / (3.0 * a2 * a2) * np.exp(-_SQRT5 * t / a)


def _np_matern52_D4F(t, a, d):
    a2 = a * a
    return 25.0 * (d * (d + 2) * a2 - (3 + 2 * d) * _SQRT5 * a * t + 5.0 * t * t) / (3.0 * a2 ** 3) * np.exp(-_SQRT5 * t / a)


def _np_matern52_DF(t, a):
    return -5.0 * (a + _SQRT5 * t) * np.exp(-_SQRT5 * t / a) / (3.0 * a ** 3)


def _np_matern52_D3F(t, a, d):
    return 25.0 * np.exp(-_SQRT5 * t / a) * (a * (2 + d) - _SQRT5 * t) / (3.0 * a ** 5)


def _np_matern52_DDF(t, a):
    return 25.0 * np.exp(-_SQRT5 * t / a) / (3.0 * a ** 4)


def _np_matern72_F(t, a):
    a3 = a ** 3
    return (15.0 * a3 + 15.0 * _SQRT7 * a * a * t + 42.0 * a * t * t + 7.0 * _SQRT7 * t ** 3) / (15.0 * a3) * np.exp(-_SQRT7 * t / a)


def _np_matern72_D2F(t, a, d):
    a5 = a ** 5
    return -7.0 * (3.0 * d * a ** 3 + 3.0 * _SQRT7 * a * a * d * t + 7.0 * a * (d - 1) * t * t - 7.0 * _SQRT7 * t ** 3) / (15.0 * a5) * np.exp(-_SQRT7 * t / a)


def _np_matern72_D4F(t, a, d):
    a7 = a ** 7
    return 49.0 * (d * (d + 2) * a ** 3 + d * (d + 2) * _SQRT7 * a * a * t - 14.0 * a * (2 + d) * t * t + 7.0 * _SQRT7 * t ** 3) / (15.0 * a7) * np.exp(-_SQRT7 * t / a)


def _np_matern72_DF(t, a):
    return -7.0 * (3.0 * a * a + 3.0 * _SQRT7 * a * t + 7.0 * t * t) * np.exp(-_SQRT7 * t / a) / (15.0 * a ** 4)


def _np_matern72_D3F(t, a, d):
    return 49.0 * np.exp(-_SQRT7 * t / a) * (a * a * (2 + d) + _SQRT7 * a * (2 + d) * t - 7.0 * t * t) / (15.0 * a ** 6)


def _np_matern72_DDF(t, a):
    return 49.0 * np.exp(-_SQRT7 * t / a) * (a + _SQRT7 * t) / (15.0 * a ** 5)


def _np_matern92_F(t, a):
    a4 = a ** 4
    return (35.0 * a4 + 105.0 * a ** 3 * t + 135.0 * a * a * t * t + 90.0 * a * t ** 3 + 27.0 * t ** 4) / (35.0 * a4) * np.exp(-3.0 * t / a)


def _np_matern92_D2F(t, a, d):
    a6 = a ** 6
    return -9.0 * (5.0 * d * a ** 4 + 15.0 * d * a ** 3 * t + 9.0 * a * a * (2 * d - 1) * t * t + 9.0 * a * (d - 3) * t ** 3 - 27.0 * t ** 4) / (35.0 * a6) * np.exp(-3.0 * t / a)


def _np_matern92_D4F(t, a, d):
    a8 = a ** 8
    return 81.0 * (d * (d + 2) * a ** 4 + 3.0 * a ** 3 * d * (d + 2) * t + 3.0 * a * a * (d * d - 4) * t * t - 18.0 * a * (d + 2) * t ** 3 + 27.0 * t ** 4) / (35.0 * a8) * np.exp(-3.0 * t / a)


def _np_matern112_F(t, a):
    a5 = a ** 5
    return (
        945.0 * a5
        + 945.0 * _SQRT11 * a ** 4 * t
        + 4620.0 * a ** 3 * t * t
        + 1155.0 * _SQRT11 * a * a * t ** 3
        + 1815.0 * a * t ** 4
        + 121.0 * _SQRT11 * t ** 5
    ) / (945.0 * a5) * np.exp(-_SQRT11 * t / a)


def _np_gauss_F(t, a):
    return np.exp(-t * t / (2.0 * a * a))


def _np_gauss_D2F(t, a, d):
    return (t * t - a * a * d) / (a ** 4) * np.exp(-t * t / (2.0 * a * a))


def _np_gauss_D4F(t, a, d):
    a8 = a ** 8
    return (a ** 4 * d * (2 + d) - 2.0 * a * a * (2 + d) * t * t + t ** 4) * np.exp(-t * t / (2.0 * a * a)) / a8


def _np_gauss_DF(t, a):
    return -np.exp(-t * t / (2.0 * a * a)) / (a * a)


def _np_gauss_D3F(t, a, d):
    return np.exp(-t * t / (2.0 * a * a)) * (a * a * (2 + d) - t * t) / (a ** 6)


def _np_gauss_DDF(t, a):
    return np.exp(-t * t / (2.0 * a * a)) / (a ** 4)


# ---------------------------------------------------------------------------
# Point × Point evaluators (numpy)
# ---------------------------------------------------------------------------


def _np_pp(F):
    def _impl(xc, yc, a):
        t = _np_pairwise(xc, yc)
        return F(t, a)

    return _impl


# ---------------------------------------------------------------------------
# Δδ × Δδ evaluator (numpy)
# ---------------------------------------------------------------------------


def _np_ldld(F, D2F, D4F):
    def _impl(xc, yc, wlx, wdx, wly, wdy, a):
        t = _np_pairwise(xc, yc)
        d = xc.shape[1]
        f = F(t, a)
        d2 = D2F(t, a, d)
        d4 = D4F(t, a, d)
        return (
            wlx[:, None] * wly[None, :] * d4
            + (wdx[:, None] * wly[None, :] + wlx[:, None] * wdy[None, :]) * d2
            + wdx[:, None] * wdy[None, :] * f
        )

    return _impl


# ---------------------------------------------------------------------------
# Δ∇δ × Δ∇δ evaluator (numpy)
# ---------------------------------------------------------------------------


def _np_lgdlgd(F, D2F, D4F, DF, D3F, DDF):
    def _impl(xc, yc, wlx, wgx, wdx, wly, wgy, wdy, a):
        diff = _np_pairwise_diff(xc, yc)  # (N, M, d)
        t2 = np.einsum('ijk,ijk->ij', diff, diff)
        t = np.sqrt(t2)
        d = xc.shape[1]
        f = F(t, a)
        d2 = D2F(t, a, d)
        d4 = D4F(t, a, d)
        df = DF(t, a)
        d3 = D3F(t, a, d)
        ddf = DDF(t, a)
        diff_wgy = np.einsum('ijk,jk->ij', diff, wgy)
        diff_wgx = np.einsum('ijk,ik->ij', diff, wgx)
        wgx_wgy = wgx @ wgy.T
        return (
            wlx[:, None] * wly[None, :] * d4
            + (wdx[:, None] * wly[None, :] + wlx[:, None] * wdy[None, :]) * d2
            + wdx[:, None] * wdy[None, :] * f
            - wlx[:, None] * d3 * diff_wgy
            + wly[None, :] * d3 * diff_wgx
            - wdx[:, None] * df * diff_wgy
            + wdy[None, :] * df * diff_wgx
            + (-wgx_wgy * df + diff_wgx * (-diff_wgy) * ddf)
        )

    return _impl


# ---------------------------------------------------------------------------
# ∂∂ × ∂∂ (2D only): analytical Hessians
# ---------------------------------------------------------------------------
#
# For these we compute F(t^2) where t^2 = (x1-y1)^2 + (x2-y2)^2 (plus eps
# for Matern to keep sqrt smooth away from zero). Then ∂^2 F / ∂x_i ∂x_j
# and further derivatives w.r.t. y are expressed analytically in terms of
# g(s) := F(sqrt(s + eps), a) and its derivatives with respect to s.
# This avoids any autodiff cost on the hot path.


def _matern52_g_and_derivs(s, a):
    """Return g(s), g'(s), g''(s), g'''(s), g''''(s) for
    g(s) = F(sqrt(s+eps), a), F being the Matern 5/2 covariance.
    """
    eps = 1e-8
    s_e = s + eps
    t = np.sqrt(s_e)
    F = (1.0 + _SQRT5 * t / a + 5.0 * s_e / (3.0 * a * a)) * np.exp(-_SQRT5 * t / a)
    # Chain-rule via derivatives of F with respect to t. g(s) = F(t(s)).
    # dt/ds = 1/(2 t)
    # F'(t) = derivative of F w.r.t. t.
    # Using closed form: F(t) = (1 + √5 t/a + 5 t²/(3a²)) exp(-√5 t/a)
    # Simpler approach: compute numerically via multiple base derivatives,
    # but for a clean implementation I'll just differentiate symbolically.
    #
    # Let p(t) = 1 + √5 t/a + 5 t²/(3a²), e(t) = exp(-√5 t/a)
    # F'(t) = p'(t) e(t) + p(t) e'(t)
    #       = (√5/a + 10t/(3a²)) e - (√5/a) p e
    #       = e * (√5/a + 10 t / (3 a²) - √5 p / a)
    # The expansion simplifies to  F'(t) = -5 t / (3 a²) (1 + √5 t / a) e.
    # (Matches Julia's DF/D2F structure.)
    # Then g(s) = F(t), with chain rule g' = F' / (2 t), g'' needs F''.
    # Rather than re-derive all derivatives, we use the formulas directly
    # derived from Julia's DF/D2F etc, but those are t-space derivatives.
    # We only need up to the 4-th order for ∂∂ × ∂∂.
    raise NotImplementedError


# Because the closed-form Hessian-of-Hessian for Matern is messy, we keep
# a JAX-AD path for the ∂∂ evaluator — the kernels are called rarely (only
# for the specific ∂∂ PDE constraints) so JIT compile overhead is tolerable.


def _jax_make_pp2(F_scalar):
    def Hx(x, y, a):
        h = jax.hessian(lambda xv: F_scalar(xv, y, a))(x)
        return jnp.array([h[0, 0], h[0, 1], h[1, 1]])

    def HxHy(x, y, a):
        return jax.hessian(lambda yv: Hx(x, yv, a))(y)

    def triple(x, y, a):
        t = HxHy(x, y, a)  # (3, 2, 2)
        return jnp.stack([t[:, 0, 0], t[:, 0, 1], t[:, 1, 1]], axis=1)  # (3, 3)

    batched = jax.vmap(
        jax.vmap(triple, in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )

    @jit
    def _impl(xc, yc, wx, wy, a):
        T = batched(xc, yc, a)
        return jnp.einsum('nmij,ni,mj->nm', T, wx, wy)

    return _impl


def _jax_make_hdhd(F_scalar):
    """(∂∂ + δ) × (∂∂ + δ) kernel for 2D. Weights per point: (w11, w12, w22, wd)."""

    def pair_K(x, y, a, wxd, wxh, wyd, wyh):
        F = F_scalar(x, y, a)
        Hy = jax.hessian(lambda yv: F_scalar(x, yv, a))(y)
        HyV = jnp.array([Hy[0, 0], Hy[0, 1], Hy[1, 1]])
        Hx = jax.hessian(lambda xv: F_scalar(xv, y, a))(x)
        HxV = jnp.array([Hx[0, 0], Hx[0, 1], Hx[1, 1]])

        def HxF(xv, yv):
            h = jax.hessian(lambda zv: F_scalar(zv, yv, a))(xv)
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

    @jit
    def _impl(xc, yc, wxd, wxh, wyd, wyh, a):
        return over_ij(xc, yc, a, wxd, wxh, wyd, wyh)

    return _impl


def _jax_make_p_pp2(F_scalar):
    def Hy(x, y, a):
        h = jax.hessian(lambda yv: F_scalar(x, yv, a))(y)
        return jnp.array([h[0, 0], h[0, 1], h[1, 1]])

    batched = jax.vmap(
        jax.vmap(Hy, in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )

    @jit
    def _impl(xc, yc, wy, a):
        V = batched(xc, yc, a)
        return jnp.einsum('nmk,mk->nm', V, wy)

    return _impl


def _jax_make_pp2_p(F_scalar):
    def Hx(x, y, a):
        h = jax.hessian(lambda xv: F_scalar(xv, y, a))(x)
        return jnp.array([h[0, 0], h[0, 1], h[1, 1]])

    batched = jax.vmap(
        jax.vmap(Hx, in_axes=(None, 0, None)),
        in_axes=(0, None, None),
    )

    @jit
    def _impl(xc, yc, wx, a):
        V = batched(xc, yc, a)
        return jnp.einsum('nmk,nk->nm', V, wx)

    return _impl


def _matern52_F_xy(x, y, a):
    eps = 1e-8
    t = jnp.sqrt(jnp.sum((x - y) ** 2) + eps)
    return (1.0 + _SQRT5 * t / a + 5.0 * t * t / (3.0 * a * a)) * jnp.exp(-_SQRT5 * t / a)


def _matern72_F_xy(x, y, a):
    eps = 1e-8
    t = jnp.sqrt(jnp.sum((x - y) ** 2) + eps)
    a3 = a ** 3
    return (15.0 * a3 + 15.0 * _SQRT7 * a * a * t + 42.0 * a * t * t + 7.0 * _SQRT7 * t ** 3) / (15.0 * a3) * jnp.exp(-_SQRT7 * t / a)


def _gauss_F_xy(x, y, a):
    t = jnp.sum((x - y) ** 2)
    return jnp.exp(-t / (2.0 * a * a))


_pp2_matern52_jit = _jax_make_pp2(_matern52_F_xy)
_pp2_matern72_jit = _jax_make_pp2(_matern72_F_xy)
_pp2_gauss_jit = _jax_make_pp2(_gauss_F_xy)
_p_pp2_matern52_jit = _jax_make_p_pp2(_matern52_F_xy)
_p_pp2_matern72_jit = _jax_make_p_pp2(_matern72_F_xy)
_p_pp2_gauss_jit = _jax_make_p_pp2(_gauss_F_xy)
_pp2_p_matern52_jit = _jax_make_pp2_p(_matern52_F_xy)
_pp2_p_matern72_jit = _jax_make_pp2_p(_matern72_F_xy)
_pp2_p_gauss_jit = _jax_make_pp2_p(_gauss_F_xy)

_hdhd_matern52_jit = _jax_make_hdhd(_matern52_F_xy)
_hdhd_matern72_jit = _jax_make_hdhd(_matern72_F_xy)
_hdhd_gauss_jit = _jax_make_hdhd(_gauss_F_xy)


# ---------------------------------------------------------------------------
# Base class with type-pair dispatch
# ---------------------------------------------------------------------------


class AbstractCovarianceFunction:
    """Each subclass implements some subset of the type-pair evaluators.

    Call `kernel(x, y)` or `kernel(x)` (symmetric) to get a numpy (N, M)
    covariance matrix.
    """

    def __init__(self, length_scale: float):
        self.length_scale = float(length_scale)

    # overrides ---------------------------------------------------------
    def _pp(self, x, y):
        raise NotImplementedError

    def _ldld(self, x, y):
        raise NotImplementedError

    def _lgdlgd(self, x, y):
        raise NotImplementedError

    def _pp2(self, x, y):
        raise NotImplementedError

    def _p_pp2(self, x, y):
        raise NotImplementedError

    def _pp2_p(self, x, y):
        raise NotImplementedError

    def _hdhd(self, x, y):
        raise NotImplementedError

    # promotion ---------------------------------------------------------
    @staticmethod
    def _promote_to_ldld(m):
        if isinstance(m, PointMeasurement):
            N = m.coordinate.shape[0]
            return LaplaceDiracPointMeasurement(
                coordinate=m.coordinate,
                weight_laplace=np.zeros(N, dtype=np.float64),
                weight_delta=np.ones(N, dtype=np.float64),
            )
        return m

    @staticmethod
    def _promote_to_lgdlgd(m):
        if isinstance(m, PointMeasurement):
            N, d = m.coordinate.shape
            return LaplaceGradDiracPointMeasurement(
                coordinate=m.coordinate,
                weight_laplace=np.zeros(N, dtype=np.float64),
                weight_grad=np.zeros((N, d), dtype=np.float64),
                weight_delta=np.ones(N, dtype=np.float64),
            )
        return m

    def __call__(self, x, y=None):
        if y is None:
            y = x
        tx, ty = type(x), type(y)
        ldld = (DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement)

        if tx is PointMeasurement and ty is PointMeasurement:
            return np.asarray(self._pp(x, y))
        if tx in ldld and ty in ldld:
            return np.asarray(self._ldld(x, y))
        if tx in ldld and ty is PointMeasurement:
            return np.asarray(self._ldld(x, self._promote_to_ldld(y)))
        if tx is PointMeasurement and ty in ldld:
            return np.asarray(self._ldld(self._promote_to_ldld(x), y))
        if tx is LaplaceGradDiracPointMeasurement and ty is LaplaceGradDiracPointMeasurement:
            return np.asarray(self._lgdlgd(x, y))
        if tx is LaplaceGradDiracPointMeasurement and ty is PointMeasurement:
            return np.asarray(self._lgdlgd(x, self._promote_to_lgdlgd(y)))
        if tx is PointMeasurement and ty is LaplaceGradDiracPointMeasurement:
            return np.asarray(self._lgdlgd(self._promote_to_lgdlgd(x), y))
        if tx is PartialPartialPointMeasurement and ty is PartialPartialPointMeasurement:
            return np.asarray(self._pp2(x, y))
        if tx is PointMeasurement and ty is PartialPartialPointMeasurement:
            return np.asarray(self._p_pp2(x, y))
        if tx is PartialPartialPointMeasurement and ty is PointMeasurement:
            return np.asarray(self._pp2_p(x, y))
        if tx is HessianDiracPointMeasurement and ty is HessianDiracPointMeasurement:
            return np.asarray(self._hdhd(x, y))
        raise NotImplementedError(
            f'{type(self).__name__} not implemented for ({tx.__name__}, {ty.__name__})'
        )


# ---------------------------------------------------------------------------
# Concrete kernels
# ---------------------------------------------------------------------------


class MaternCovariance1_2(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_matern12_F(t, self.length_scale)


class MaternCovariance3_2(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_matern32_F(t, self.length_scale)


class MaternCovariance5_2(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_matern52_F(t, self.length_scale)

    def _ldld(self, x, y):
        return _np_ldld(_np_matern52_F, _np_matern52_D2F, _np_matern52_D4F)(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_delta,
            y.weight_laplace, y.weight_delta,
            self.length_scale,
        )

    def _lgdlgd(self, x, y):
        return _np_lgdlgd(
            _np_matern52_F, _np_matern52_D2F, _np_matern52_D4F,
            _np_matern52_DF, _np_matern52_D3F, _np_matern52_DDF,
        )(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_grad, x.weight_delta,
            y.weight_laplace, y.weight_grad, y.weight_delta,
            self.length_scale,
        )

    def _pp2(self, x, y):
        wx = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        wy = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_pp2_matern52_jit(x.coordinate, y.coordinate, wx, wy, self.length_scale))

    def _p_pp2(self, x, y):
        wy = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_p_pp2_matern52_jit(x.coordinate, y.coordinate, wy, self.length_scale))

    def _pp2_p(self, x, y):
        wx = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        return np.asarray(_pp2_p_matern52_jit(x.coordinate, y.coordinate, wx, self.length_scale))

    def _hdhd(self, x, y):
        wxh = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        wyh = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_hdhd_matern52_jit(
            x.coordinate, y.coordinate, x.weight_delta, wxh, y.weight_delta, wyh, self.length_scale,
        ))


class MaternCovariance7_2(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_matern72_F(t, self.length_scale)

    def _ldld(self, x, y):
        return _np_ldld(_np_matern72_F, _np_matern72_D2F, _np_matern72_D4F)(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_delta,
            y.weight_laplace, y.weight_delta,
            self.length_scale,
        )

    def _lgdlgd(self, x, y):
        return _np_lgdlgd(
            _np_matern72_F, _np_matern72_D2F, _np_matern72_D4F,
            _np_matern72_DF, _np_matern72_D3F, _np_matern72_DDF,
        )(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_grad, x.weight_delta,
            y.weight_laplace, y.weight_grad, y.weight_delta,
            self.length_scale,
        )

    def _pp2(self, x, y):
        wx = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        wy = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_pp2_matern72_jit(x.coordinate, y.coordinate, wx, wy, self.length_scale))

    def _p_pp2(self, x, y):
        wy = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_p_pp2_matern72_jit(x.coordinate, y.coordinate, wy, self.length_scale))

    def _pp2_p(self, x, y):
        wx = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        return np.asarray(_pp2_p_matern72_jit(x.coordinate, y.coordinate, wx, self.length_scale))

    def _hdhd(self, x, y):
        wxh = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        wyh = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_hdhd_matern72_jit(
            x.coordinate, y.coordinate, x.weight_delta, wxh, y.weight_delta, wyh, self.length_scale,
        ))


class MaternCovariance9_2(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_matern92_F(t, self.length_scale)

    def _ldld(self, x, y):
        return _np_ldld(_np_matern92_F, _np_matern92_D2F, _np_matern92_D4F)(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_delta,
            y.weight_laplace, y.weight_delta,
            self.length_scale,
        )


class MaternCovariance11_2(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_matern112_F(t, self.length_scale)


class GaussianCovariance(AbstractCovarianceFunction):
    def _pp(self, x, y):
        t = _np_pairwise(x.coordinate, y.coordinate)
        return _np_gauss_F(t, self.length_scale)

    def _ldld(self, x, y):
        return _np_ldld(_np_gauss_F, _np_gauss_D2F, _np_gauss_D4F)(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_delta,
            y.weight_laplace, y.weight_delta,
            self.length_scale,
        )

    def _lgdlgd(self, x, y):
        return _np_lgdlgd(
            _np_gauss_F, _np_gauss_D2F, _np_gauss_D4F,
            _np_gauss_DF, _np_gauss_D3F, _np_gauss_DDF,
        )(
            x.coordinate, y.coordinate,
            x.weight_laplace, x.weight_grad, x.weight_delta,
            y.weight_laplace, y.weight_grad, y.weight_delta,
            self.length_scale,
        )

    def _pp2(self, x, y):
        wx = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        wy = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_pp2_gauss_jit(x.coordinate, y.coordinate, wx, wy, self.length_scale))

    def _p_pp2(self, x, y):
        wy = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_p_pp2_gauss_jit(x.coordinate, y.coordinate, wy, self.length_scale))

    def _pp2_p(self, x, y):
        wx = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        return np.asarray(_pp2_p_gauss_jit(x.coordinate, y.coordinate, wx, self.length_scale))

    def _hdhd(self, x, y):
        wxh = np.stack([x.weight_11, x.weight_12, x.weight_22], axis=1)
        wyh = np.stack([y.weight_11, y.weight_12, y.weight_22], axis=1)
        return np.asarray(_hdhd_gauss_jit(
            x.coordinate, y.coordinate, x.weight_delta, wxh, y.weight_delta, wyh, self.length_scale,
        ))


# ---------------------------------------------------------------------------
# Scalar-F registry — used by the GPU batched factorization. Each kernel
# below, applied element-wise to a distance array, gives the value of the
# kernel between two PointMeasurements. This is the only piece of the GPU
# path that needs JAX, and we expose it as a JAX-traceable function.
# ---------------------------------------------------------------------------


def _jax_F_of(kernel: AbstractCovarianceFunction):
    """Return a JAX-traceable scalar function F(t, a) for the PointMeasurement
    case of `kernel`. Used to JIT a batched Cholesky in factorization_jax.
    """
    cls = type(kernel)
    if cls is MaternCovariance1_2:
        return lambda t, a: jnp.exp(-t / a)
    if cls is MaternCovariance3_2:
        def f(t, a):
            r = _SQRT3 * t / a
            return (1.0 + r) * jnp.exp(-r)
        return f
    if cls is MaternCovariance5_2:
        def f(t, a):
            return (1.0 + _SQRT5 * t / a + 5.0 * t * t / (3.0 * a * a)) * jnp.exp(-_SQRT5 * t / a)
        return f
    if cls is MaternCovariance7_2:
        def f(t, a):
            a3 = a ** 3
            return (15.0 * a3 + 15.0 * _SQRT7 * a * a * t + 42.0 * a * t * t + 7.0 * _SQRT7 * t ** 3) / (15.0 * a3) * jnp.exp(-_SQRT7 * t / a)
        return f
    if cls is MaternCovariance9_2:
        def f(t, a):
            a4 = a ** 4
            return (35.0 * a4 + 105.0 * a ** 3 * t + 135.0 * a * a * t * t + 90.0 * a * t ** 3 + 27.0 * t ** 4) / (35.0 * a4) * jnp.exp(-3.0 * t / a)
        return f
    if cls is MaternCovariance11_2:
        def f(t, a):
            a5 = a ** 5
            return (
                945.0 * a5
                + 945.0 * _SQRT11 * a ** 4 * t
                + 4620.0 * a ** 3 * t * t
                + 1155.0 * _SQRT11 * a * a * t ** 3
                + 1815.0 * a * t ** 4
                + 121.0 * _SQRT11 * t ** 5
            ) / (945.0 * a5) * jnp.exp(-_SQRT11 * t / a)
        return f
    if cls is GaussianCovariance:
        return lambda t, a: jnp.exp(-t * t / (2.0 * a * a))
    raise NotImplementedError(f'no JAX-F for {cls.__name__}')
