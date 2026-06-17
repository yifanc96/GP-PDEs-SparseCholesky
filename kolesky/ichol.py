"""Noisy / additive-noise variant — Algorithm 4.1 of Schäfer-Katzfuss-Owhadi (2020).

Convention: kolesky's noiseless factor U has ``U Uᵀ ≈ Θ⁻¹`` (paper
Eq 1.2, page 8 footnote — forward-maximin ordering, upper-triangular).
Algorithm 4.1 outputs Ũ satisfying ``Ũ Ũᵀ ≈ U Uᵀ + R⁻¹ = Θ⁻¹ + R⁻¹``.
The factorization is computed by a **reverse-order column Cholesky**
(processes j = N−1 down to 0) — this is the standard upper-Cholesky
recurrence for the ``M = U Uᵀ`` form (each column j depends on entries
in LATER columns k > j). Triangular solves for ``A⁻¹ v`` are
``Ũ⁻ᵀ Ũ⁻¹ v`` (upper solve on Ũ, then lower solve on Ũᵀ).

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


def _row_inner_gt(U_csr, i: int, j: int, min_col: int) -> float:
    """Compute Σ_{k > min_col, k ∈ S[i, :] ∩ S[j, :]} U[i, k] · U[j, k].

    Used by the reverse-order ichol that factors ``U Uᵀ + R⁻¹``: each
    column j (processed from N−1 down to 0) needs the row-inner-product
    Σ_{k > j} U[i, k] U[j, k] across LATER columns. Row access via CSR.
    """
    ci = U_csr.indices[U_csr.indptr[i] : U_csr.indptr[i + 1]]
    di = U_csr.data[U_csr.indptr[i] : U_csr.indptr[i + 1]]
    cj = U_csr.indices[U_csr.indptr[j] : U_csr.indptr[j + 1]]
    dj = U_csr.data[U_csr.indptr[j] : U_csr.indptr[j + 1]]
    mi = ci > min_col
    ci, di = ci[mi], di[mi]
    mj = cj > min_col
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
    """Algorithm 4.1: incomplete Cholesky of  A = U Uᵀ + diag(R_inv_perm).

    Per paper Eq 1.2 / kolesky convention (forward maximin → upper-tri),
    ``K_perm⁻¹ = U Uᵀ``, so the matrix to factor is ``U Uᵀ + R⁻¹`` and
    the result Ũ satisfies ``Ũ Ũᵀ ≈ K_perm⁻¹ + R⁻¹``.

    Parameters
    ----------
    U : scipy.sparse.csc_matrix, upper-triangular
        Noiseless KL factor (same convention as ``ExplicitKLFactorization.U``).
    R_inv_perm : array of shape (N,)
        Diagonal of R⁻¹ in the *P-permuted* order (``R_orig_diag[P]``).
    use_squared_pattern : bool, default True
        If True, factor on the sparsity pattern of ``U Uᵀ`` (the paper's
        "LLᵀ pattern", ~2× denser than U's). If False, use U's pattern.

    Returns
    -------
    U_tilde : scipy.sparse.csc_matrix, upper-triangular
        Sparse factor satisfying ``Ũ Ũᵀ ≈ A`` on the chosen pattern.
    """
    N = U.shape[0]
    R_inv_perm = np.asarray(R_inv_perm, dtype=np.float64)
    if R_inv_perm.shape != (N,):
        raise ValueError(f'R_inv_perm must have shape (N,) = ({N},), got {R_inv_perm.shape}')

    # Step 1: assemble A = U Uᵀ + diag(R⁻¹) on the chosen sparsity pattern.
    UUT = (U @ U.T).tocsc()
    if use_squared_pattern:
        UUT_upper = scipy.sparse.triu(UUT, k=0).tocsc()
        UUT_upper.eliminate_zeros()
        UUT_upper.sort_indices()
        out_indices = UUT_upper.indices.copy()
        out_indptr  = UUT_upper.indptr.copy()
        A_data = UUT_upper.data.copy()
    else:
        out_indices = U.indices.copy()
        out_indptr  = U.indptr.copy()
        A_data = np.empty(U.nnz, dtype=np.float64)
        for j in range(N):
            col_start = U.indptr[j]
            col_end = U.indptr[j + 1]
            rows = U.indices[col_start:col_end]
            UUT_col = UUT.getcol(j).toarray().ravel()
            A_data[col_start:col_end] = UUT_col[rows]

    # Add R⁻¹ to the diagonal (last entry of each upper-tri CSC column).
    for j in range(N):
        col_end = out_indptr[j + 1]
        if col_end > out_indptr[j] and out_indices[col_end - 1] == j:
            A_data[col_end - 1] += R_inv_perm[j]

    # Step 2: REVERSE-order incomplete Cholesky for ``Ũ Ũᵀ = A``.
    # Process columns j = N-1 down to 0. Each column needs entries at
    # LATER columns (k > j), accessed by ROW; we keep a CSR view in
    # lock-step with the CSC ``out`` for `_row_inner_gt`.
    out_data = A_data.copy()
    out = scipy.sparse.csc_matrix((out_data, out_indices, out_indptr), shape=(N, N))
    out_csr = out.tocsr().copy()

    def _sync_csr(i, j, value):
        rstart = out_csr.indptr[i]; rend = out_csr.indptr[i + 1]
        cols = out_csr.indices[rstart:rend]
        idx = np.searchsorted(cols, j)
        if idx < cols.size and cols[idx] == j:
            out_csr.data[rstart + idx] = value

    for j in range(N - 1, -1, -1):
        col_start = out.indptr[j]
        col_end = out.indptr[j + 1]
        rows_j = out.indices[col_start:col_end]
        m = rows_j.size
        if m == 0:
            continue

        # Diagonal: Ũ[j, j]² = A[j, j] − Σ_{k > j} Ũ[j, k]²
        s_diag = _row_inner_gt(out_csr, j, j, min_col=j)
        diag_val = out.data[col_end - 1] - s_diag
        if diag_val <= 0:
            diag_val = max(diag_val, 1e-14 * abs(out.data[col_end - 1]) + 1e-14)
        ujj = float(np.sqrt(diag_val))
        out.data[col_end - 1] = ujj
        _sync_csr(j, j, ujj)

        # Off-diagonals: Ũ[i, j] = (A[i, j] − Σ_{k > j} Ũ[i, k] Ũ[j, k]) / Ũ[j, j]
        for ii in range(m - 1):
            i = int(rows_j[ii])
            s = _row_inner_gt(out_csr, i, j, min_col=j)
            uij = (out.data[col_start + ii] - s) / ujj
            out.data[col_start + ii] = uij
            _sync_csr(i, j, uij)

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

        # Step 1 — Θ̂⁻¹ b   ≈   Pᵀ (U Uᵀ) P b     (two sparse matvecs).
        # K_perm⁻¹ = U Uᵀ per paper Eq 1.2 / kolesky convention.
        bp = b[self.P]
        s = self.U @ (self.U.T @ bp)             # in P-perm order

        # Step 2 — A⁻¹ s   ≈   Pᵀ Ũ⁻ᵀ Ũ⁻¹ P s     (two triangular solves).
        # With Ũ Ũᵀ ≈ A, A⁻¹ = Ũ⁻ᵀ Ũ⁻¹.
        y = scipy.sparse.linalg.spsolve_triangular(self.U_tilde.tocsr(),   s, lower=False)
        z = scipy.sparse.linalg.spsolve_triangular(self.U_tilde.T.tocsr(), y, lower=True)
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

        Ut_csr   = self.U_tilde.T.tocsr()    # lower-tri (Ũᵀ)
        Utop_csr = self.U_tilde.tocsr()       # upper-tri (Ũ)

        # Symmetric SMW-form preconditioner — same matrix as the paper's
        # chain via Sherman-Morrison. With ``Ũ Ũᵀ ≈ A`` (paper convention),
        # ``A⁻¹ = Ũ⁻ᵀ Ũ⁻¹`` — first upper-tri solve on Ũ, then lower-tri
        # solve on Ũᵀ.
        def Minv_matvec(v):
            t = np.empty_like(v); t[self.P] = self.R_inv_perm * v[self.P]
            tp = t[self.P]
            y = scipy.sparse.linalg.spsolve_triangular(Utop_csr, tp, lower=False)
            z = scipy.sparse.linalg.spsolve_triangular(Ut_csr,   y,  lower=True)
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
