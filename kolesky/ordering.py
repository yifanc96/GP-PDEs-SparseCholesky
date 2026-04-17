"""Maximin reverse ordering, mirroring MaximinNN.jl.

The irregular, sequential heap + range-query loop is left on CPU
(scipy.spatial.cKDTree + a lazy-deletion heapq). Moving this to a GPU
would require redesigning the algorithm; staying on CPU costs nothing
because the inner work (tree queries) is already C-backed and every
subsequent step is bandwidth-bound.

We always use 0-based indexing on the Python side.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# 1-maximin: nearest_distances is a vector of scalars
# ---------------------------------------------------------------------------


def _maximin_1(
    x: np.ndarray,
    init_distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """1-maximin ordering: each point's key is its current nearest-neighbor
    distance to already-picked points.

    Returns (P, ell) with P[k] = index chosen at step k (0-based) and
    ell[k] = the distance key at that step (== the 'length scale').
    """
    N = x.shape[0]
    if init_distances is None:
        nearest = np.full(N, np.inf, dtype=np.float64)
    else:
        nearest = np.asarray(init_distances, dtype=np.float64).copy()
        assert nearest.shape == (N,)

    tree = cKDTree(x)

    # Max-heap via negation. Entries: (neg_key, id).
    # Lazy deletion: an entry is stale if -neg_key != nearest[id] or id was
    # already processed.
    heap: List[Tuple[float, int]] = [(-nearest[i], i) for i in range(N)]
    heapq.heapify(heap)

    processed = np.zeros(N, dtype=bool)
    P = np.empty(N, dtype=np.int64)
    ell = np.empty(N, dtype=np.float64)

    step = 0
    while step < N:
        while heap:
            neg_d, i = heapq.heappop(heap)
            if processed[i]:
                continue
            if -neg_d != nearest[i]:
                continue  # stale
            break
        else:
            break  # empty heap (should not happen for N>0)

        processed[i] = True
        P[step] = i
        ell[step] = nearest[i]
        r = nearest[i]
        step += 1

        # Candidate neighbors whose nearest-distance might be updated.
        # For r = inf (typical at step 0 when `nearest` starts all-inf),
        # scipy's query_ball_point would be asked for every point anyway —
        # cheaper to enumerate directly and skip the KD-tree call.
        if not np.isfinite(r):
            idx = np.arange(N, dtype=np.int64)
        else:
            if r <= 0.0:
                continue
            idx_list = tree.query_ball_point(x[i], r=r, return_sorted=False)
            if not idx_list:
                continue
            idx = np.asarray(idx_list, dtype=np.int64)

        if idx.size == 0:
            continue
        d = np.linalg.norm(x[idx] - x[i], axis=1)
        improved = (d < nearest[idx]) & (idx != i)
        if not improved.any():
            continue
        imp_idx = idx[improved]
        imp_d = d[improved]
        nearest[imp_idx] = imp_d
        for j, dj in zip(imp_idx.tolist(), imp_d.tolist()):
            heapq.heappush(heap, (-dj, int(j)))

    return P, ell


# ---------------------------------------------------------------------------
# k-maximin: nearest_distances is a (k, N) matrix, column sorted descending.
# Row 0 is the heap key = k-th nearest distance so far.
# ---------------------------------------------------------------------------


def _update_k_column(nd_col: np.ndarray, new_d: float) -> float:
    """Insert `new_d` into a size-k descending column, keeping the k smallest.

    Returns the new top (row 0) = k-th smallest seen so far.
    """
    k = nd_col.shape[0]
    if new_d >= nd_col[0]:
        return float(nd_col[0])
    # shift down and insert
    for i in range(1, k):
        if nd_col[i] <= new_d:
            nd_col[i - 1] = new_d
            return float(nd_col[0])
        nd_col[i - 1] = nd_col[i]
    nd_col[k - 1] = new_d
    return float(nd_col[0])


def _maximin_k(
    x: np.ndarray,
    k_neighbors: int,
    init_distances: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """k-maximin ordering."""
    N = x.shape[0]
    if init_distances is None:
        nd = np.full((k_neighbors, N), np.inf, dtype=np.float64)
    else:
        nd = np.asarray(init_distances, dtype=np.float64).copy()
        assert nd.shape == (k_neighbors, N)
        # each column must be sorted descending for the insertion logic
        for col in range(N):
            nd[:, col].sort()
            nd[:, col] = nd[::-1, col]

    tree = cKDTree(x)

    heap: List[Tuple[float, int]] = [(-nd[0, i], i) for i in range(N)]
    heapq.heapify(heap)

    processed = np.zeros(N, dtype=bool)
    P = np.empty(N, dtype=np.int64)
    ell = np.empty(N, dtype=np.float64)

    step = 0
    while step < N:
        while heap:
            neg_d, i = heapq.heappop(heap)
            if processed[i]:
                continue
            if -neg_d != nd[0, i]:
                continue
            break
        else:
            break

        processed[i] = True
        P[step] = i
        ell[step] = nd[0, i]
        r = nd[0, i]
        step += 1

        if not np.isfinite(r):
            idx_list = list(range(N))
        else:
            if r <= 0.0:
                continue
            idx_list = tree.query_ball_point(x[i], r=r, return_sorted=False)
        if not idx_list:
            continue
        idx = np.asarray(idx_list, dtype=np.int64)
        # distances
        d = np.linalg.norm(x[idx] - x[i], axis=1)
        # Update each column's sorted list and push updated heap entry
        for jj in range(len(idx)):
            j = int(idx[jj])
            if j == i:
                continue
            new_top = _update_k_column(nd[:, j], float(d[jj]))
            heapq.heappush(heap, (-new_top, j))

    return P, ell


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def maximin_ordering(
    x,
    k_neighbors: Optional[int] = None,
    init_distances=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Maximin reverse ordering.

    Parameters
    ----------
    x : np.ndarray of shape (N, d), OR a list of such arrays
        Point cloud(s). When `x` is a list, the ordering is constrained so
        that points of x[0] come before x[1], etc. (matches the Julia
        multi-level variant).
    k_neighbors : int or None
        None for 1-maximin (single-nearest key), int for k-maximin.
    init_distances : None, np.ndarray, or list of np.ndarrays
        Optional precomputed "so far" distances. Shape matches the variant:
            1-maximin, single set: (N,)
            k-maximin, single set: (k, N)
            1-maximin, multi set:   list of (N_k,)
            k-maximin, multi set:   list of (k, N_k)

    Returns
    -------
    P : np.ndarray of shape (sum N_k,), int64
        P[step] = original index of the point chosen at this step.
    ell : np.ndarray of shape (sum N_k,), float64
        ell[step] = the length-scale associated with this step.

    Multi-set mode concatenates outputs with index offsets so that the
    returned P indexes into `np.concatenate(x, axis=0)`.
    """
    if isinstance(x, (list, tuple)):
        return _maximin_multi(x, k_neighbors=k_neighbors, init_distances=init_distances)

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('x must be a 2D array (N, d)')

    if k_neighbors is None or k_neighbors == 1 and (init_distances is None or init_distances.ndim == 1):
        return _maximin_1(x, init_distances=init_distances)
    return _maximin_k(x, k_neighbors=k_neighbors, init_distances=init_distances)


def _init_distance_from_tree(
    tree: cKDTree, xq: np.ndarray, k: Optional[int] = None
) -> np.ndarray:
    """Compute, for each query point in xq, either the nearest or k-nearest
    distance(s) to points already in the tree.

    For k=None returns shape (Nq,). For k set returns (k, Nq) descending.
    """
    if k is None:
        d, _ = tree.query(xq, k=1)
        return d
    k_eff = min(k, tree.n)
    d, _ = tree.query(xq, k=k_eff)
    if d.ndim == 1:
        d = d[:, None]
    # d is (Nq, k_eff), nearest first. We want (k, Nq) descending.
    d = d[:, ::-1].T  # (k_eff, Nq) ascending? No: reversing along k axis then transposing.
    # tree.query returns nearest-first along axis 1, so d[:,0]=smallest, d[:,k-1]=largest.
    # Desired format: row 0 = largest among the k-nearest. So:
    d_desc = np.sort(d, axis=0)[::-1]  # (k_eff, Nq), desc along axis 0
    if k_eff < k:
        # pad leading rows with inf
        pad = np.full((k - k_eff, d_desc.shape[1]), np.inf, dtype=np.float64)
        d_desc = np.concatenate([pad, d_desc], axis=0)
    return d_desc


def _maximin_multi(
    x_list: Sequence[np.ndarray],
    k_neighbors: Optional[int] = None,
    init_distances=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-set maximin: forces each set to be ordered in sequence."""
    x_list = [np.asarray(x, dtype=np.float64) for x in x_list]
    Ns = [x.shape[0] for x in x_list]

    if init_distances is None:
        if k_neighbors is None:
            init_distances = [np.full(N, np.inf, dtype=np.float64) for N in Ns]
        else:
            init_distances = [
                np.full((k_neighbors, N), np.inf, dtype=np.float64) for N in Ns
            ]
    else:
        # defensive copy
        init_distances = [np.array(d, dtype=np.float64, copy=True) for d in init_distances]

    # Before ordering set k, adjust init_distances[k+1..] to reflect the
    # points of set k (they'll be picked earlier and so lower the scale for later sets).
    for k, xk in enumerate(x_list):
        if k + 1 >= len(x_list):
            continue
        tree_k = cKDTree(xk)
        for ell_idx in range(k + 1, len(x_list)):
            dist_to_k = _init_distance_from_tree(tree_k, x_list[ell_idx], k=k_neighbors)
            if k_neighbors is None:
                init_distances[ell_idx] = np.minimum(init_distances[ell_idx], dist_to_k)
            else:
                # elementwise min on each (k, N_l) grid
                init_distances[ell_idx] = np.minimum(init_distances[ell_idx], dist_to_k)

    P_parts: List[np.ndarray] = []
    ell_parts: List[np.ndarray] = []
    offset = 0
    for k, xk in enumerate(x_list):
        Pk, ell_k = maximin_ordering(
            xk,
            k_neighbors=k_neighbors,
            init_distances=init_distances[k],
        )
        P_parts.append(Pk + offset)
        ell_parts.append(ell_k)
        offset += xk.shape[0]

    return np.concatenate(P_parts, axis=0), np.concatenate(ell_parts, axis=0)
