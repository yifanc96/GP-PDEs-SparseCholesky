"""Supernodal data structures and reverse-maximin sparsity pattern.

Direct translation of SuperNodes.jl + the sparsity-pattern half of
MaximinNN.jl. All indices are 0-based and refer to positions in the
*maximin-permuted* point cloud x[P].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .ordering import maximin_ordering


@dataclass
class IndexSuperNode:
    """A supernode: a group of columns that share a common row support.

    row_indices and column_indices are ascending-sorted, unique, int64,
    and expressed in the reverse-maximin ordering (0-based).
    """

    column_indices: np.ndarray
    row_indices: np.ndarray

    def shape(self):
        return (self.row_indices.shape[0], self.column_indices.shape[0])

    def __len__(self):
        return self.row_indices.shape[0]


@dataclass
class IndirectSupernodalAssignment:
    """Pairs a list of IndexSuperNodes with the (P-permuted) batched
    measurement container they index into.
    """

    supernodes: List[IndexSuperNode]
    measurements: object  # a batched measurement dataclass


# ---------------------------------------------------------------------------
# Supernodal sparsity pattern
# ---------------------------------------------------------------------------


def _gather_assignments(assignments: np.ndarray, first_parent: int) -> List[np.ndarray]:
    """Group parent indices by their nearest-aggregation-center label.

    Returns a list of arrays; each array contains parent global indices
    (in the P-permuted ordering) that share the same aggregation center.
    Output groups are ordered by the *first* occurrence of the label in
    the sorted sequence.
    """
    if assignments.size == 0:
        return []
    perm = np.argsort(assignments, kind='stable')
    sorted_ass = assignments[perm]
    # boundary positions: first occurrence of each distinct value in sorted_ass
    changes = np.concatenate(([True], sorted_ass[1:] != sorted_ass[:-1]))
    first_idx = np.flatnonzero(changes)
    boundaries = np.concatenate([first_idx, [assignments.size]])
    return [
        perm[boundaries[k]:boundaries[k + 1]] + first_parent
        for k in range(boundaries.size - 1)
    ]


def supernodal_reverse_maximin_sparsity_pattern(
    x: np.ndarray,
    P: np.ndarray,
    ell: np.ndarray,
    rho: float,
    lambda_: float = 1.5,
    alpha: float = 1.0,
    reconstruct_ordering: bool = True,
) -> List[IndexSuperNode]:
    """Compute the supernodal reverse-maximin sparsity pattern.

    Parameters
    ----------
    x : (N, d) array
        Original (unpermuted) point coordinates.
    P : (N,) int64 array
        Reverse-maximin ordering (0-based).
    ell : (N,) float64 array
        Length-scale at each step, in P-ordering.
    rho : float
        Sparsity radius multiplier.
    lambda_ : float > 1
        Scale reduction factor between levels.
    alpha : float in [0, 1]
        Fraction of rho used to decide which points are "coarse enough" to
        serve as aggregation centers at a given level.
    reconstruct_ordering : bool
        If True, a fresh 1-maximin ordering of x[P] is used for the
        aggregation centers. If False, reuse P.
    """
    assert lambda_ > 1.0
    assert 0.0 <= alpha <= 1.0
    assert alpha * rho > 1.0

    x = np.asarray(x, dtype=np.float64)
    P = np.asarray(P, dtype=np.int64)
    ell = np.asarray(ell, dtype=np.float64)
    N = x.shape[0]
    assert P.shape == (N,)
    assert ell.shape == (N,)

    # permute
    x_p = np.ascontiguousarray(x[P])

    if reconstruct_ordering:
        P_temp, ell_temp = maximin_ordering(x_p)
    else:
        P_temp = np.arange(N, dtype=np.int64)
        ell_temp = ell.copy()

    supernodes: List[IndexSuperNode] = []
    children_tree = cKDTree(x_p)

    finite_mask = np.isfinite(ell)
    if not finite_mask.any():
        # all-Inf: nothing to factorize non-trivially
        return supernodes
    min_ell = float(ell[np.argmax(finite_mask)])

    last_aggregation_point = 0  # number of agg-centers already committed
    last_parent = -1

    while last_parent < N - 1:
        # --- find last_aggregation_point ---
        # first index l >= last_aggregation_point with ell_temp[l] < alpha*rho*min_ell,
        # or N if none.
        tail = ell_temp[last_aggregation_point:]
        thresh = alpha * rho * min_ell
        below = tail < thresh
        if below.any():
            last_aggregation_point = last_aggregation_point + int(np.argmax(below))
        else:
            last_aggregation_point = N

        if last_aggregation_point == 0:
            # can't happen for finite min_ell because ell_temp[P_temp[0]]=inf
            min_ell /= lambda_
            continue

        agg_global = P_temp[:last_aggregation_point]
        aggregation_tree = cKDTree(x_p[agg_global])

        # --- find last_parent ---
        first_parent = last_parent + 1
        if first_parent >= N:
            break
        # find the first l in [first_parent, N-1] with (l == N-1) or ell[l+1] < min_ell
        if first_parent + 1 >= N:
            last_parent = N - 1
        else:
            ell_next = ell[first_parent + 1:]  # (N - first_parent - 1,)
            below_min = ell_next < min_ell
            if below_min.any():
                last_parent = first_parent + int(np.argmax(below_min))
            else:
                last_parent = N - 1

        # --- NN assignment of parents to aggregation centers ---
        parent_coords = x_p[first_parent:last_parent + 1]
        _, assigned = aggregation_tree.query(parent_coords, k=1)
        assigned = np.atleast_1d(assigned).astype(np.int64, copy=False)

        groups = _gather_assignments(assigned, first_parent)

        for column_indices in groups:
            column_indices = np.sort(column_indices)
            rows_parts = []
            for c in column_indices:
                c = int(c)
                nb = children_tree.query_ball_point(
                    x_p[c], r=rho * ell[c], return_sorted=False
                )
                if not nb:
                    continue
                nb = np.asarray(nb, dtype=np.int64)
                # upper-triangular in P-ordering: row <= column
                nb = nb[nb <= c]
                if nb.size == 0:
                    continue
                d = np.linalg.norm(x_p[nb] - x_p[c], axis=1)
                keep = d <= rho * ell[nb]
                nb = nb[keep]
                if nb.size:
                    rows_parts.append(nb)
            if rows_parts:
                row_indices = np.unique(np.concatenate(rows_parts))
            else:
                row_indices = np.empty(0, dtype=np.int64)
            supernodes.append(
                IndexSuperNode(
                    column_indices=column_indices.astype(np.int64),
                    row_indices=row_indices.astype(np.int64),
                )
            )

        min_ell /= lambda_

    return supernodes


# ---------------------------------------------------------------------------
# Driver matching Julia's ordering_and_sparsity_pattern
# ---------------------------------------------------------------------------


def ordering_and_sparsity_pattern(
    x,
    rho: float,
    k_neighbors: Optional[int] = None,
    lambda_: float = 1.5,
    alpha: float = 1.0,
    init_distances=None,
):
    """Compute (P, ell, supernodes) in one call. Accepts either a single
    (N, d) array or a list of (N_k, d) arrays (multi-set ordering).
    """
    if isinstance(x, (list, tuple)):
        x_cat = np.concatenate([np.asarray(xk, dtype=np.float64) for xk in x], axis=0)
        P, ell = maximin_ordering(x, k_neighbors=k_neighbors, init_distances=init_distances)
    else:
        x_cat = np.asarray(x, dtype=np.float64)
        P, ell = maximin_ordering(x_cat, k_neighbors=k_neighbors, init_distances=init_distances)

    supernodes = supernodal_reverse_maximin_sparsity_pattern(
        x_cat, P, ell, rho, lambda_=lambda_, alpha=alpha
    )
    return P, ell, supernodes
