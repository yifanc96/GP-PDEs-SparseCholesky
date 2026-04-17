"""Generate interior / boundary sample points on a 2D rectangle."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def sample_points_grid_2d(
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    h_in: float,
    h_bd: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform grid in the interior + uniform grid on the boundary.

    Returns (X_domain, X_boundary) each of shape (N, 2).

    Matches the construction in main_NonLinElliptic2d.jl's sample_points_grid,
    except we return each array in (N, 2) layout instead of Julia's (2, N).
    """
    (x1l, x1r), (x2l, x2r) = domain
    xs = np.arange(x1l + h_in, x1r - h_in + 1e-12, h_in)
    ys = np.arange(x2l + h_in, x2r - h_in + 1e-12, h_in)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    X_domain = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)

    # boundary: 4 sides concatenated, matching the Julia version's layout
    # bottom edge: x1l..x1r-h_bd, y=x2l
    t_bot = np.arange(x1l, x1r - h_bd + 1e-12, h_bd)
    bottom = np.stack([t_bot, np.full_like(t_bot, x2l)], axis=1)
    # right edge: x=x1r, y=x2l..x2r-h_bd
    t_right = np.arange(x2l, x2r - h_bd + 1e-12, h_bd)
    right = np.stack([np.full_like(t_right, x1r), t_right], axis=1)
    # top edge: x=x1r..x1l+h_bd (descending), y=x2r
    t_top = np.arange(x1r, x1l + h_bd - 1e-12, -h_bd)
    top = np.stack([t_top, np.full_like(t_top, x2r)], axis=1)
    # left edge: x=x1l, y=x2r..x1l+h_bd (descending)
    t_left = np.arange(x2r, x1l + h_bd - 1e-12, -h_bd)
    left = np.stack([np.full_like(t_left, x1l), t_left], axis=1)

    X_boundary = np.concatenate([bottom, right, top, left], axis=0)
    return X_domain, X_boundary


def sample_points_rdm_2d(
    domain: Tuple[Tuple[float, float], Tuple[float, float]],
    N_domain: int,
    N_boundary: int,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random interior + uniformly-random boundary (equal on each side)."""
    if rng is None:
        rng = np.random.default_rng()
    (x1l, x1r), (x2l, x2r) = domain
    X_domain = np.stack([
        rng.uniform(x1l, x1r, N_domain),
        rng.uniform(x2l, x2r, N_domain),
    ], axis=1)

    per_side, rem = divmod(N_boundary, 4)
    if rem != 0:
        N_boundary = 4 * per_side
    bot = np.stack([rng.uniform(x1l, x1r, per_side), np.full(per_side, x2l)], axis=1)
    right = np.stack([np.full(per_side, x1r), rng.uniform(x2l, x2r, per_side)], axis=1)
    top = np.stack([rng.uniform(x1l, x1r, per_side), np.full(per_side, x2r)], axis=1)
    left = np.stack([np.full(per_side, x1l), rng.uniform(x2l, x2r, per_side)], axis=1)
    X_boundary = np.concatenate([bot, right, top, left], axis=0)
    return X_domain, X_boundary
