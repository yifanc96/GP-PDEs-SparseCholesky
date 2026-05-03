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
    well-conditioned correction R⁻¹ + Θ⁻¹. Combining the two factors
    via the simple algebraic identity (paper §4.1 — *not*
    Sherman-Morrison-Woodbury)

        Σ  =  Θ̂ + R  =  Θ̂ (R⁻¹ + Θ̂⁻¹) R                              (1)

    and using Uᵀ U ≈ Θ̂⁻¹, Ũᵀ Ũ ≈ R⁻¹ + Θ̂⁻¹, the apply / inverse-apply
    chains are

        Σ    ≈  (Uᵀ U)⁻¹  Ũᵀ Ũ  R                                     (apply Σ)
        Σ⁻¹  ≈  R⁻¹  Ũ⁻¹ Ũ⁻ᵀ  Uᵀ U                                    (apply Σ⁻¹)

    each of which is four cheap sparse pieces. The paper recommends
    using ``Ũ`` as a *preconditioner* for an inner CG that solves
    ``(R⁻¹ + Θ̂⁻¹) α = c`` for tighter accuracy at small σ²;
    :py:meth:`solve_Sigma` runs that pattern (single-precision in ~10
    iterations).

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
    use_squared_pattern: bool = True,
) -> scipy.sparse.csc_matrix:
    """Algorithm 4.1: incomplete Cholesky of  A = Uᵀ U + diag(R_inv_perm).

    Parameters
    ----------
    U : scipy.sparse.csc_matrix, upper-triangular
        Noiseless KL factor (same convention as ``ExplicitKLFactorization.U``).
    R_inv_perm : array of shape (N,)
        Diagonal of R⁻¹ in the *P-permuted* order (i.e. ``R_orig_diag[P]``;
        for homoscedastic σ² this is just ``np.full(N, 1/σ²)``).
    use_squared_pattern : bool, default True
        If True (paper §4.1, line 579–589), compute the ichol on the
        sparsity pattern of ``UᵀU`` (~2× denser than ``U``'s pattern but
        "as accurate as the exact Cholesky of A over a wide range of ρ"
        per the paper's experiments). If False, use the pattern of ``U``
        (cheaper but lower accuracy at small ρ).

    Returns
    -------
    U_tilde : scipy.sparse.csc_matrix, upper-triangular
        Sparse factor satisfying ``Ũᵀ Ũ ≈ A`` on the chosen pattern.
        ``Ũᵢᵢ > 0`` by construction.
    """
    N = U.shape[0]
    R_inv_perm = np.asarray(R_inv_perm, dtype=np.float64)
    if R_inv_perm.shape != (N,):
        raise ValueError(f'R_inv_perm must have shape (N,) = ({N},), got {R_inv_perm.shape}')

    # Step 1: assemble A on the chosen pattern.
    UtU = (U.T @ U).tocsc()                      # exact UᵀU, denser than U
    if use_squared_pattern:
        # Take the upper triangle of UtU (same triangular convention as U).
        # UᵀU's nonzero structure IS the "LL⊤ pattern" the paper recommends.
        UtU_upper = scipy.sparse.triu(UtU, k=0).tocsc()
        UtU_upper.eliminate_zeros()
        UtU_upper.sort_indices()
        out_indices = UtU_upper.indices.copy()
        out_indptr  = UtU_upper.indptr.copy()
        A_data = UtU_upper.data.copy()
    else:
        out_indices = U.indices.copy()
        out_indptr  = U.indptr.copy()
        A_data = np.empty(U.nnz, dtype=np.float64)
        for j in range(N):
            col_start = U.indptr[j]
            col_end = U.indptr[j + 1]
            rows = U.indices[col_start:col_end]
            UtU_col = UtU.getcol(j).toarray().ravel()
            A_data[col_start:col_end] = UtU_col[rows]

    # Add R⁻¹ to the diagonal: in CSC upper-triangular layout the diagonal
    # of column j is the LAST entry of column j (rows ascending; max row = j).
    for j in range(N):
        col_end = out_indptr[j + 1]
        # Defensive: confirm the last row index in this column is j.
        # (For correctly-built U / UtU this always holds.)
        if col_end > out_indptr[j] and out_indices[col_end - 1] == j:
            A_data[col_end - 1] += R_inv_perm[j]

    # Step 2: incomplete Cholesky on A, in-place (output keeps the pattern).
    out_data = A_data.copy()
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

    # ----- inverse: x = Σ⁻¹ b ≈ R⁻¹ Ũ⁻¹ Ũ⁻ᵀ Uᵀ U b   (paper §4.1) -----

    def apply_Sigma_inv(self, b: np.ndarray) -> np.ndarray:
        """One-shot approximation of ``x ≈ (Θ + R)⁻¹ b`` via the paper's
        factor-product identity  ``Σ⁻¹ ≈ R⁻¹ Ũ⁻¹ Ũ⁻ᵀ Uᵀ U`` (paper §4.1).

        Four sparse pieces: two matvecs with U, Uᵀ on the noiseless
        factor (gives Θ̂⁻¹ b), then two triangular solves with Ũᵀ, Ũ
        on the ichol factor (gives A⁻¹ on the result), then a diagonal
        scaling by R⁻¹.

        Cheap — but accuracy is bounded by the noiseless factor's
        accuracy as a Θ̂⁻¹ approximation. The bare ``Uᵀ U`` matvec form
        is less accurate than the forward triangular-solve form, so at
        small ρ this one-shot apply can have substantial residual. For
        tighter accuracy use :py:meth:`solve_Sigma` (CG on the inner
        ``A x = c`` with ``Ũ`` as preconditioner — the path the paper
        recommends).
        """
        b = np.asarray(b, dtype=np.float64)

        # Step 1 — Θ̂⁻¹ b   ≈   Pᵀ (Uᵀ U) P b     (two sparse matvecs).
        bp = b[self.P]
        s = self.U.T @ (self.U @ bp)             # in P-perm order

        # Step 2 — A⁻¹ s   ≈   Pᵀ Ũ⁻¹ Ũ⁻ᵀ P s     (two triangular solves).
        y = scipy.sparse.linalg.spsolve_triangular(self.U_tilde.T.tocsr(), s, lower=True)
        z = scipy.sparse.linalg.spsolve_triangular(self.U_tilde.tocsr(),   y, lower=False)
        u = np.empty_like(b); u[self.P] = z

        # Step 3 — R⁻¹ u   (diagonal in original order).
        out = np.empty_like(b)
        out[self.P] = self.R_inv_perm * u[self.P]
        return out

    # ----- iterative CG-with-preconditioner solve (paper's recommended path) -----

    def solve_Sigma(
        self,
        b: np.ndarray,
        rtol: float = 1e-8,
        maxiter: int = 100,
    ) -> np.ndarray:
        """Iterative ``x ≈ (Θ + R)⁻¹ b`` following paper §4.1's
        recommendation that "the accuracy for solving systems of
        equations in Σ can easily be increased by adding a few
        iterations of conjugate gradient" (paper line 770).

        Outer CG on the symmetric system ``Σ x = b``:

          * **Matvec** ``Σ v  =  Θ v + R v`` — Θ via the noiseless KL
            factor's *forward* apply (two triangular solves on U;
            highly accurate), plus a diagonal R-scaling. This is what
            `apply_Sigma` computes.
          * **Preconditioner** ``M ≈ Σ⁻¹`` — the symmetric (Sherman-
            Morrison) rearrangement of the paper's chain identity:

                 M  =  R⁻¹  −  R⁻¹ A⁻¹ R⁻¹    (with A = R⁻¹ + Θ̂⁻¹)

            which is mathematically the same matrix as the paper's
            asymmetric chain ``R⁻¹ A⁻¹ Θ̂⁻¹`` (exact-arithmetic
            equivalence) but in the symmetric form CG requires.
            ``A⁻¹`` is applied via ``Ũ`` triangular solves.

        Converges in ~10 iterations to single precision (paper's
        empirical claim).
        """
        b = np.asarray(b, dtype=np.float64)
        N = b.shape[0]

        Ut_csr   = self.U_tilde.T.tocsr()
        Utop_csr = self.U_tilde.tocsr()

        # Symmetric SMW-form preconditioner — same matrix as the paper's
        # chain via Sherman-Morrison.
        def Minv_matvec(v):
            t = np.empty_like(v); t[self.P] = self.R_inv_perm * v[self.P]
            tp = t[self.P]
            y = scipy.sparse.linalg.spsolve_triangular(Ut_csr,   tp, lower=True)
            z = scipy.sparse.linalg.spsolve_triangular(Utop_csr, y,  lower=False)
            u = np.empty_like(v); u[self.P] = z
            Rinv_u = np.empty_like(v); Rinv_u[self.P] = self.R_inv_perm * u[self.P]
            return t - Rinv_u

        A_op = scipy.sparse.linalg.LinearOperator(
            (N, N), matvec=lambda v: self.apply_Sigma(v), dtype=np.float64,
        )
        M_op = scipy.sparse.linalg.LinearOperator(
            (N, N), matvec=Minv_matvec, dtype=np.float64,
        )
        x0 = Minv_matvec(b)
        x, _info = scipy.sparse.linalg.cg(
            A_op, b, x0=x0, M=M_op, rtol=rtol, maxiter=maxiter,
        )
        return x
