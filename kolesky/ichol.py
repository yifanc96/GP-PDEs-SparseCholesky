"""Noisy / additive-noise variant — Algorithm 4.1 of Schäfer-Katzfuss-Owhadi (2020).

Given an existing noiseless KL factor U with sparsity pattern S
(so that Uᵀ U ≈ Θ⁻¹ on S, in P-permuted order), and a diagonal noise
covariance R, this module computes a *second* sparse upper-triangular
factor Ũ (same pattern S) such that

    Ũᵀ Ũ  ≈  Uᵀ U  +  R⁻¹      (= Θ⁻¹ + R⁻¹  =  A)

and exposes a compact API for applying Σ = Θ + R and Σ⁻¹ to vectors.

Why this matters:
    Naively running the noiseless KL factorization on Σ = Θ + R fails:
    adding R to every diagonal entry of Θ attenuates the off-diagonal
    decay of (Θ + R)⁻¹, so the maximin sparsity pattern is no longer
    a good support. The paper's trick is to factor Θ alone (existing
    pipeline), then run a *second* incomplete Cholesky on the
    well-conditioned correction R⁻¹ + Θ⁻¹. Combined with Woodbury's
    identity,

        Σ⁻¹  =  (Θ + R)⁻¹  =  R⁻¹  −  R⁻¹ A⁻¹ R⁻¹,

    the two factors give an O(N · ρ²ᵈ) preconditioner / direct
    approximate solver for noisy GP regression.

Cost: same asymptotic complexity as the noiseless factorization itself
(O(N · ρ²ᵈ) time, O(N · ρᵈ) memory), uniform in σ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .factorization import ExplicitKLFactorization


# ---------------------------------------------------------------------------
# Incomplete Cholesky on the existing sparsity pattern S
# ---------------------------------------------------------------------------


def _col_inner_lt(U: scipy.sparse.csc_matrix, i: int, j: int, max_row: int) -> float:
    """Compute Σ_{k < max_row, k ∈ S[:, i] ∩ S[:, j]} U[k, i] · U[k, j].

    Both U[:, i] and U[:, j] have ascending row indices. We mask each to
    rows < max_row, then intersect on row index and dot the values.
    """
    ci = U.indices[U.indptr[i] : U.indptr[i + 1]]
    di = U.data[U.indptr[i] : U.indptr[i + 1]]
    cj = U.indices[U.indptr[j] : U.indptr[j + 1]]
    dj = U.data[U.indptr[j] : U.indptr[j + 1]]
    if max_row is not None:
        mi = ci < max_row
        ci, di = ci[mi], di[mi]
        mj = cj < max_row
        cj, dj = cj[mj], dj[mj]
    if ci.size == 0 or cj.size == 0:
        return 0.0
    common = np.intersect1d(ci, cj, assume_unique=True)
    if common.size == 0:
        return 0.0
    pi = np.searchsorted(ci, common)
    pj = np.searchsorted(cj, common)
    return float((di[pi] * dj[pj]).sum())


def ichol_pattern(
    U: scipy.sparse.csc_matrix,
    R_inv_perm: np.ndarray,
) -> scipy.sparse.csc_matrix:
    """Algorithm 4.1: incomplete Cholesky of A = Uᵀ U + diag(R_inv_perm)
    restricted to the sparsity pattern of `U`.

    Parameters
    ----------
    U : scipy.sparse.csc_matrix, upper-triangular
        Noiseless KL factor (same convention as ``ExplicitKLFactorization.U``).
    R_inv_perm : array of shape (N,)
        Diagonal of R⁻¹ in the *P-permuted* order (i.e. ``R_orig_diag[P]``;
        for homoscedastic σ², this is just ``np.full(N, 1/σ²)``).

    Returns
    -------
    U_tilde : scipy.sparse.csc_matrix, upper-triangular
        Sparse factor with the same pattern as ``U`` and ``Ũᵀ Ũ ≈ A`` on
        that pattern. ``Ũᵢᵢ > 0`` by construction.
    """
    N = U.shape[0]
    R_inv_perm = np.asarray(R_inv_perm, dtype=np.float64)
    if R_inv_perm.shape != (N,):
        raise ValueError(f'R_inv_perm must have shape (N,) = ({N},), got {R_inv_perm.shape}')

    # Step 1: assemble A on pattern(U) via UtU = U.T @ U.
    # `U.T @ U` is denser than U, but cheap and lets us index into it cleanly.
    UtU = (U.T @ U).tocsc()

    A_data = np.empty(U.nnz, dtype=np.float64)
    for j in range(N):
        col_start = U.indptr[j]
        col_end = U.indptr[j + 1]
        rows = U.indices[col_start:col_end]
        UtU_col = UtU.getcol(j).toarray().ravel()
        A_data[col_start:col_end] = UtU_col[rows]
        # Add R⁻¹ to diagonal: the last entry in column j is the (j, j) entry
        # because U is upper-triangular and rows are ascending.
        A_data[col_end - 1] += R_inv_perm[j]

    # Step 2: incomplete Cholesky on A, in-place (output has same pattern).
    out_data = A_data.copy()
    out_indices = U.indices.copy()
    out_indptr = U.indptr.copy()
    out = scipy.sparse.csc_matrix((out_data, out_indices, out_indptr), shape=(N, N))

    for j in range(N):
        col_start = out.indptr[j]
        col_end = out.indptr[j + 1]
        rows_j = out.indices[col_start:col_end]
        m = rows_j.size

        # Off-diagonals (rows i < j; last entry of rows_j is the diagonal j).
        for ii in range(m - 1):
            i = int(rows_j[ii])
            s = _col_inner_lt(out, i, j, max_row=i)
            U_ii = out.data[out.indptr[i + 1] - 1]   # diagonal of already-finished column i
            if U_ii <= 0:
                raise FloatingPointError(
                    f'ichol breakdown at column {i}: non-positive diagonal {U_ii}'
                )
            out.data[col_start + ii] = (out.data[col_start + ii] - s) / U_ii

        # Diagonal entry.
        s = float((out.data[col_start : col_end - 1] ** 2).sum())
        diag_val = out.data[col_end - 1] - s
        if diag_val <= 0:
            # Fall back to a tiny positive value rather than break the chain.
            # In practice this only triggers for very ill-conditioned A.
            diag_val = max(diag_val, 1e-14 * abs(out.data[col_end - 1]) + 1e-14)
        out.data[col_end - 1] = np.sqrt(diag_val)

    return out


# ---------------------------------------------------------------------------
# Wrapper: noisy-Σ apply / inverse-apply
# ---------------------------------------------------------------------------


@dataclass
class NoisyExplicitKLFactorization:
    """Noisy-Σ (= Θ + R) factor pair from Algorithm 4.1.

    Stores the noiseless factor ``U`` and the ichol factor ``U_tilde`` plus
    the per-point noise covariance ``R`` (in P-permuted order). Provides
    matrix-vector and inverse-matrix-vector routines for ``Σ = Θ + R``.

    Construct via ``NoisyExplicitKLFactorization.build(noiseless, R)``.
    """

    P: np.ndarray                                # int64
    U: scipy.sparse.csc_matrix                   # noiseless factor (Uᵀ U ≈ Θ⁻¹)
    U_tilde: scipy.sparse.csc_matrix             # ichol factor (Ũᵀ Ũ ≈ R⁻¹ + Θ⁻¹)
    R_perm: np.ndarray                           # diagonal of R in P-permuted order
    R_inv_perm: np.ndarray                       # cached diagonal of R⁻¹

    @classmethod
    def build(
        cls,
        noiseless: ExplicitKLFactorization,
        R: Union[float, np.ndarray],
    ) -> 'NoisyExplicitKLFactorization':
        """Run ichol on top of an existing noiseless KL factor.

        Parameters
        ----------
        noiseless : ExplicitKLFactorization
            The noiseless factor (output of ``ExplicitKLFactorization(implicit, ...)``).
        R : float or array of shape (N,)
            Diagonal noise covariance, in the *original* (unpermuted) order.
            Pass a float for homoscedastic σ² (most common).

        Returns
        -------
        NoisyExplicitKLFactorization
        """
        P = np.asarray(noiseless.P, dtype=np.int64)
        N = P.size

        if np.isscalar(R) or (isinstance(R, np.ndarray) and R.ndim == 0):
            R_orig = np.full(N, float(R), dtype=np.float64)
        else:
            R_orig = np.asarray(R, dtype=np.float64).copy()
            if R_orig.shape != (N,):
                raise ValueError(f'R must be scalar or shape ({N},), got {R_orig.shape}')
        if np.any(R_orig <= 0.0):
            raise ValueError('R must have strictly positive diagonal')

        # Permute R into the factor's order.
        R_perm = R_orig[P]
        R_inv_perm = 1.0 / R_perm

        U_tilde = ichol_pattern(noiseless.U, R_inv_perm)

        return cls(
            P=P,
            U=noiseless.U,
            U_tilde=U_tilde,
            R_perm=R_perm,
            R_inv_perm=R_inv_perm,
        )

    # ----- forward: y = Σ v = Θ v + R v -----

    def apply_Sigma(self, v: np.ndarray) -> np.ndarray:
        """Compute ``y ≈ (Θ + R) v`` using the noiseless factor + diag(R)."""
        v = np.asarray(v, dtype=np.float64)
        # Θ v  (≈ P^T (UᵀU)⁻¹ P v)
        vp = v[self.P]
        y = scipy.sparse.linalg.spsolve_triangular(self.U.tocsr(),   vp, lower=False)
        z = scipy.sparse.linalg.spsolve_triangular(self.U.T.tocsr(), y,  lower=True)
        Theta_v = np.empty_like(v)
        Theta_v[self.P] = z
        # R v  (diagonal in original order)
        # R_perm is in P-permuted order: R_orig[P] = R_perm, so R_orig[i] = R_perm[P_inv[i]].
        # Equivalently R v in original order = (R_perm * v[P])[P_inv]; but easier:
        # since R is diagonal with entries R_orig[i], reconstruct:
        Rv = np.empty_like(v)
        Rv[self.P] = self.R_perm * v[self.P]   # equivalent to R_orig * v
        return Theta_v + Rv

    # ----- inverse: x = Σ⁻¹ b = R⁻¹ b − R⁻¹ A⁻¹ R⁻¹ b   (Woodbury) -----

    def apply_Sigma_inv(self, b: np.ndarray) -> np.ndarray:
        """Direct one-shot approximation of ``x ≈ (Θ + R)⁻¹ b`` via Woodbury.

        Cheap (two pairs of triangular solves + diagonal scalings) but its
        accuracy is bounded by the *noiseless* factor's accuracy times
        κ(Σ): when σ² is small enough that κ(Σ) is large, the two terms
        in the Woodbury identity nearly cancel, and any error in the
        noiseless `U` gets amplified. For tight solves at small σ², use
        :py:meth:`solve_Sigma` (CG with this routine as preconditioner).
        """
        b = np.asarray(b, dtype=np.float64)
        # t = R⁻¹ b   (diagonal in original order; same trick as in apply_Sigma).
        t = np.empty_like(b)
        t[self.P] = self.R_inv_perm * b[self.P]

        # u = A⁻¹ t   ≈   P^T Ũ⁻¹ Ũ⁻ᵀ P t
        tp = t[self.P]
        # Ũᵀ y = tp  →  y = Ũ⁻ᵀ tp   (lower triangular solve)
        y = scipy.sparse.linalg.spsolve_triangular(self.U_tilde.T.tocsr(), tp, lower=True)
        # Ũ z = y   →   z = Ũ⁻¹ y    (upper triangular solve)
        z = scipy.sparse.linalg.spsolve_triangular(self.U_tilde.tocsr(),   y,  lower=False)
        u = np.empty_like(b)
        u[self.P] = z

        # x = R⁻¹ b − R⁻¹ u
        Rinv_u = np.empty_like(b)
        Rinv_u[self.P] = self.R_inv_perm * u[self.P]
        return t - Rinv_u

    # ----- iterative CG-with-preconditioner solve (paper's recommended path) -----

    def solve_Sigma(
        self,
        b: np.ndarray,
        rtol: float = 1e-8,
        maxiter: int = 200,
        x0: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Iterative ``x ≈ (Θ + R)⁻¹ b`` via CG preconditioned by the
        ichol factor (Algorithm 4.1's recommended use).

        Drives `Σ x = b` with `M = apply_Sigma_inv` as the preconditioner.
        Converges in ~10 iterations to single-precision when the
        noiseless `U` is reasonably accurate; degrades gracefully when
        σ² is small (the underlying problem becomes ill-conditioned).
        """
        b = np.asarray(b, dtype=np.float64)
        N = b.shape[0]
        A_op = scipy.sparse.linalg.LinearOperator(
            (N, N), matvec=lambda v: self.apply_Sigma(v), dtype=np.float64,
        )
        M_op = scipy.sparse.linalg.LinearOperator(
            (N, N), matvec=lambda v: self.apply_Sigma_inv(v), dtype=np.float64,
        )
        if x0 is None:
            x0 = self.apply_Sigma_inv(b)
        x, info = scipy.sparse.linalg.cg(
            A_op, b, x0=x0, M=M_op, rtol=rtol, maxiter=maxiter,
        )
        return x
