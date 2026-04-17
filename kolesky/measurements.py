"""Measurement types mirroring Measurements.jl.

Each concrete measurement is represented as a dataclass whose fields are
numpy arrays. A single measurement has scalar weight fields and a 1-D
coordinate of shape (d,); a *batched* measurement has a leading batch
dimension N on every field. Kernel functions downstream always operate on
batched measurements (built by `stack_measurements`), which is how we
achieve Julia-level throughput from Python: one kernel call processes
O(N x M) pairs, not one pair at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np


ArrayLike = Union[np.ndarray, float, int]


@dataclass
class PointMeasurement:
    """Point (Dirac delta) measurement at `coordinate`.

    Single: coordinate is shape (d,).
    Batched: coordinate is shape (N, d).
    """

    coordinate: np.ndarray

    @property
    def d(self) -> int:
        return self.coordinate.shape[-1]

    def is_batched(self) -> bool:
        return self.coordinate.ndim == 2


@dataclass
class DeltaDiracPointMeasurement:
    """w_Δ * Δu(x) + w_δ * u(x) — Laplacian-plus-Dirac functional.

    Single: coordinate (d,), weight_laplace (), weight_delta ().
    Batched: coordinate (N, d), weight_laplace (N,), weight_delta (N,).
    """

    coordinate: np.ndarray
    weight_laplace: np.ndarray
    weight_delta: np.ndarray

    @property
    def d(self) -> int:
        return self.coordinate.shape[-1]

    def is_batched(self) -> bool:
        return self.coordinate.ndim == 2


@dataclass
class LaplaceDiracPointMeasurement:
    """Alias dataclass for ΔδPointMeasurement (same as DeltaDiracPointMeasurement).

    Kept for naming consistency with the Julia type hierarchy where
    ΔδPointMeasurement and Δ∇δPointMeasurement are distinct types.
    This is the plain "Δ + δ" version; see Δ∇δ below for the gradient variant.
    """

    coordinate: np.ndarray
    weight_laplace: np.ndarray
    weight_delta: np.ndarray

    @property
    def d(self) -> int:
        return self.coordinate.shape[-1]

    def is_batched(self) -> bool:
        return self.coordinate.ndim == 2


@dataclass
class LaplaceGradDiracPointMeasurement:
    """w_Δ * Δu + <w_∇, ∇u> + w_δ * u — Δ∇δ measurement.

    Single: coordinate (d,), weight_laplace (), weight_grad (d,), weight_delta ().
    Batched: coordinate (N, d), weight_laplace (N,), weight_grad (N, d), weight_delta (N,).
    """

    coordinate: np.ndarray
    weight_laplace: np.ndarray
    weight_grad: np.ndarray
    weight_delta: np.ndarray

    @property
    def d(self) -> int:
        return self.coordinate.shape[-1]

    def is_batched(self) -> bool:
        return self.coordinate.ndim == 2


@dataclass
class PartialPartialPointMeasurement:
    """Second-derivative measurement in 2D: w11*∂11 u + w12*∂12 u + w22*∂22 u.

    Only d=2 is supported in the Julia code.
    """

    coordinate: np.ndarray
    weight_11: np.ndarray
    weight_12: np.ndarray
    weight_22: np.ndarray

    @property
    def d(self) -> int:
        return self.coordinate.shape[-1]

    def is_batched(self) -> bool:
        return self.coordinate.ndim == 2


@dataclass
class HessianDiracPointMeasurement:
    """Combined ∂∂ + δ measurement in 2D:

        L(u)(x) = w11 ∂11 u(x) + w12 ∂12 u(x) + w22 ∂22 u(x) + w_δ u(x)

    Single type used by the Monge-Ampere solver, where the big factor
    combines pure-Dirac measurements (w_* = 0, w_δ = 1) with pure-Hessian
    measurements (w_δ = 0) in one ordering.
    """

    coordinate: np.ndarray
    weight_11: np.ndarray
    weight_12: np.ndarray
    weight_22: np.ndarray
    weight_delta: np.ndarray

    @property
    def d(self) -> int:
        return self.coordinate.shape[-1]

    def is_batched(self) -> bool:
        return self.coordinate.ndim == 2


# ---------------------------------------------------------------------------
# Constructors and helpers
# ---------------------------------------------------------------------------


def point_measurements(x: np.ndarray, dims: int = 1) -> PointMeasurement:
    """Mirror `point_measurements(x; dims=1)` from Julia.

    The Julia version returns a Vector{PointMeasurement}. Here we return a
    single *batched* PointMeasurement because every kernel call wants the
    batched form — constructing a Python list of N dataclass instances then
    re-stacking for each kernel call would be a major perf regression.

    `dims=1` means each column of `x` is a coordinate (Julia default, x is
    (d, N)); `dims=2` means each row is a coordinate (x is (N, d)).
    """
    x = np.asarray(x, dtype=np.float64)
    if dims == 1:
        coords = np.ascontiguousarray(x.T)  # (N, d)
    elif dims == 2:
        coords = np.ascontiguousarray(x)
    else:
        raise ValueError('dims must be 1 or 2')
    return PointMeasurement(coordinate=coords)


def stack_measurements(
    measurements: Sequence[
        Union[
            PointMeasurement,
            DeltaDiracPointMeasurement,
            LaplaceDiracPointMeasurement,
            LaplaceGradDiracPointMeasurement,
            PartialPartialPointMeasurement,
        ]
    ],
):
    """Concatenate a list of (possibly batched) measurements of the *same*
    concrete type along the batch axis.
    """
    if len(measurements) == 0:
        raise ValueError('empty measurement list')
    cls = type(measurements[0])
    if not all(type(m) is cls for m in measurements):
        raise TypeError('stack_measurements: all inputs must be same type')

    def _cat(field: str, expand_scalar_to: int):
        parts = []
        for m in measurements:
            v = getattr(m, field)
            if v.ndim == 0 or (v.ndim == 1 and expand_scalar_to == 1):
                # scalar field, single measurement -> (1,)
                v = np.atleast_1d(v)
            elif v.ndim == 1 and expand_scalar_to > 1:
                # could be coord of single (d,) or batched scalar (N,)
                v = v[None, :] if v.shape[0] == expand_scalar_to else v
            parts.append(v)
        return np.concatenate(parts, axis=0)

    if cls is PointMeasurement:
        coords = [np.atleast_2d(m.coordinate) for m in measurements]
        return PointMeasurement(coordinate=np.concatenate(coords, axis=0))

    if cls in (DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement):
        coords = [np.atleast_2d(m.coordinate) for m in measurements]
        wl = [np.atleast_1d(m.weight_laplace) for m in measurements]
        wd = [np.atleast_1d(m.weight_delta) for m in measurements]
        return cls(
            coordinate=np.concatenate(coords, axis=0),
            weight_laplace=np.concatenate(wl, axis=0),
            weight_delta=np.concatenate(wd, axis=0),
        )

    if cls is LaplaceGradDiracPointMeasurement:
        coords = [np.atleast_2d(m.coordinate) for m in measurements]
        wl = [np.atleast_1d(m.weight_laplace) for m in measurements]
        wg = [np.atleast_2d(m.weight_grad) for m in measurements]
        wd = [np.atleast_1d(m.weight_delta) for m in measurements]
        return cls(
            coordinate=np.concatenate(coords, axis=0),
            weight_laplace=np.concatenate(wl, axis=0),
            weight_grad=np.concatenate(wg, axis=0),
            weight_delta=np.concatenate(wd, axis=0),
        )

    if cls is PartialPartialPointMeasurement:
        coords = [np.atleast_2d(m.coordinate) for m in measurements]
        w11 = [np.atleast_1d(m.weight_11) for m in measurements]
        w12 = [np.atleast_1d(m.weight_12) for m in measurements]
        w22 = [np.atleast_1d(m.weight_22) for m in measurements]
        return cls(
            coordinate=np.concatenate(coords, axis=0),
            weight_11=np.concatenate(w11, axis=0),
            weight_12=np.concatenate(w12, axis=0),
            weight_22=np.concatenate(w22, axis=0),
        )

    if cls is HessianDiracPointMeasurement:
        coords = [np.atleast_2d(m.coordinate) for m in measurements]
        w11 = [np.atleast_1d(m.weight_11) for m in measurements]
        w12 = [np.atleast_1d(m.weight_12) for m in measurements]
        w22 = [np.atleast_1d(m.weight_22) for m in measurements]
        wd = [np.atleast_1d(m.weight_delta) for m in measurements]
        return cls(
            coordinate=np.concatenate(coords, axis=0),
            weight_11=np.concatenate(w11, axis=0),
            weight_12=np.concatenate(w12, axis=0),
            weight_22=np.concatenate(w22, axis=0),
            weight_delta=np.concatenate(wd, axis=0),
        )

    raise TypeError(f'unsupported measurement type: {cls}')


def get_coordinates(measurements) -> np.ndarray:
    """Return the coordinates of a (possibly batched) measurement as (N, d)."""
    if hasattr(measurements, 'coordinate'):
        c = measurements.coordinate
        return np.atleast_2d(c)
    # list of measurements
    return np.concatenate([np.atleast_2d(m.coordinate) for m in measurements], axis=0)


def select(measurements, idx: np.ndarray):
    """Return a batched measurement consisting of rows `idx` of `measurements`."""
    idx = np.asarray(idx, dtype=np.int64)
    cls = type(measurements)
    coord = measurements.coordinate[idx]
    if cls is PointMeasurement:
        return PointMeasurement(coordinate=coord)
    if cls in (DeltaDiracPointMeasurement, LaplaceDiracPointMeasurement):
        return cls(
            coordinate=coord,
            weight_laplace=measurements.weight_laplace[idx],
            weight_delta=measurements.weight_delta[idx],
        )
    if cls is LaplaceGradDiracPointMeasurement:
        return cls(
            coordinate=coord,
            weight_laplace=measurements.weight_laplace[idx],
            weight_grad=measurements.weight_grad[idx],
            weight_delta=measurements.weight_delta[idx],
        )
    if cls is PartialPartialPointMeasurement:
        return cls(
            coordinate=coord,
            weight_11=measurements.weight_11[idx],
            weight_12=measurements.weight_12[idx],
            weight_22=measurements.weight_22[idx],
        )
    if cls is HessianDiracPointMeasurement:
        return cls(
            coordinate=coord,
            weight_11=measurements.weight_11[idx],
            weight_12=measurements.weight_12[idx],
            weight_22=measurements.weight_22[idx],
            weight_delta=measurements.weight_delta[idx],
        )
    raise TypeError(f'unsupported measurement type: {cls}')


def size(measurements) -> int:
    """Number of measurements in a batched container."""
    return measurements.coordinate.shape[0]
