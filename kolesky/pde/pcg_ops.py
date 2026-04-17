"""Linear-operator wrappers for the approximate Θ_train and its preconditioner.

The big factor covers `n_sets` measurement groups:
    [δ_boundary ; measurement_1_at_domain ; ... ; measurement_{n_sets-1}_at_domain]
and has size `N_bdy + n_dom_sets * N_dom`, where `n_dom_sets = n_sets - 1`.

Every PDE in this package has a Θ_train that can be written as
    Θ_train @ b = extract( Θ_big @ lift(b) )
with per-PDE weights `w_k` that change each GN step / time step.

When CuPy (+ cuSPARSE) is available, the sparse triangular solves in
`BigFactorOperator.apply` and the sparse matvecs in `SmallPrecond.matvec`
run on GPU automatically. This is the critical path inside the pCG loop
and is where Julia's SuiteSparse backend would otherwise out-perform
scipy's Python-scheduled `spsolve_triangular`.

To force CPU sparse ops (e.g. for debugging), pass `use_gpu=False` to the
operators or set `KOLESKY_DISABLE_GPU_SPARSE=1` in the environment.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
from typing import List, Sequence, Union

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator


# ---------------------------------------------------------------------------
# Optional CuPy import + cuSPARSE library preload.
# ---------------------------------------------------------------------------


def _preload_nvidia_libs():
    """PyPI-installed JAX puts its bundled CUDA libs under `nvidia/<lib>/lib/`
    but doesn't add them to LD_LIBRARY_PATH, so CuPy can't find them at
    import time. We dlopen each shared object with RTLD_GLOBAL, which places
    them in the process namespace and lets CuPy's own dlopen succeed.
    """
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    root = pathlib.Path(nvidia.__file__).parent
    for libdir in root.iterdir():
        if not libdir.is_dir():
            continue
        lib_dir = libdir / 'lib'
        if not lib_dir.exists():
            continue
        for so in lib_dir.glob('lib*.so*'):
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_cp = None
_cp_sp = None
_cp_spla = None


def _try_cupy():
    global _cp, _cp_sp, _cp_spla
    if _cp is not None:
        return True
    if os.environ.get('KOLESKY_DISABLE_GPU_SPARSE'):
        return False
    _preload_nvidia_libs()
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cp_sp
        import cupyx.scipy.sparse.linalg as cp_spla
        # touch device to confirm it works
        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return False
    _cp, _cp_sp, _cp_spla = cp, cp_sp, cp_spla
    return True


def _gpu_sparse_available() -> bool:
    """Whether CuPy+cuSPARSE is importable. Does NOT imply 'use it by default'."""
    return _try_cupy()


def _gpu_sparse_default() -> bool:
    """Default behavior for use_gpu=None. CPU by default because CuPy's
    `spsolve_triangular` re-runs cuSPARSE's analysis phase every call, so
    the per-call overhead wipes out the compute speedup for the N≲10⁴
    problems typical of this solver. Opt in via `use_gpu=True` or env
    var `KOLESKY_ENABLE_GPU_SPARSE=1` if you have N ≫ 10⁴ or measure a win.
    """
    return bool(os.environ.get('KOLESKY_ENABLE_GPU_SPARSE'))


# ---------------------------------------------------------------------------
# Big-factor sparse triangular solve (Θ_big ≈ U^{-T} U^{-1}).
# ---------------------------------------------------------------------------


class BigFactorOperator:
    """Apply Θ_big ≈ U^{-T} U^{-1} to a dense vector.

    GPU path: upload U_csr and L_csr to cuSPARSE once; each `apply` is then
    two cuSPARSE triangular solves + one scatter for the P permutation.
    CPU path: scipy's `spsolve_triangular`.
    """

    def __init__(
        self,
        U: scipy.sparse.csc_matrix,
        P: np.ndarray,
        use_gpu: bool | None = None,
    ):
        self.U = U.tocsc()
        self.U_csr = self.U.tocsr()
        self.L_csr = self.U.T.tocsr()
        self.P = np.asarray(P, dtype=np.int64)

        if use_gpu is None:
            use_gpu = _gpu_sparse_default() and _gpu_sparse_available()
        self.use_gpu = bool(use_gpu)

        if self.use_gpu:
            self.U_gpu = _cp_sp.csr_matrix(self.U_csr)
            self.L_gpu = _cp_sp.csr_matrix(self.L_csr)
            self.P_gpu = _cp.asarray(self.P)

    # CPU helper
    def _apply_cpu(self, x: np.ndarray) -> np.ndarray:
        permuted = x[self.P]
        y = scipy.sparse.linalg.spsolve_triangular(self.U_csr, permuted, lower=False)
        z = scipy.sparse.linalg.spsolve_triangular(self.L_csr, y, lower=True)
        out = np.empty_like(x)
        out[self.P] = z
        return out

    # GPU helper
    def _apply_gpu(self, x: np.ndarray) -> np.ndarray:
        x_gpu = _cp.asarray(x)
        permuted = x_gpu[self.P_gpu]
        y = _cp_spla.spsolve_triangular(self.U_gpu, permuted, lower=False)
        z = _cp_spla.spsolve_triangular(self.L_gpu, y, lower=True)
        out_gpu = _cp.empty_like(x_gpu)
        out_gpu[self.P_gpu] = z
        return _cp.asnumpy(out_gpu)

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            return self._apply_gpu(x)
        return self._apply_cpu(x)


WeightT = Union[np.ndarray, float]


def _as_scaled(w: WeightT, b_int: np.ndarray) -> np.ndarray:
    if np.isscalar(w):
        if float(w) == 0.0:
            return np.zeros_like(b_int)
        return float(w) * b_int
    return np.asarray(w) * b_int


# ---------------------------------------------------------------------------
# Θ_train ≈ extract ∘ Θ_big ∘ lift — the forward operator for pCG.
# ---------------------------------------------------------------------------


class LiftedThetaTrainMatVec:
    def __init__(self, big: BigFactorOperator, N_bdy: int, N_dom: int, n_dom_sets: int):
        self.big = big
        self.N_bdy = N_bdy
        self.N_dom = N_dom
        self.n_dom_sets = n_dom_sets
        self.weights: List[WeightT] = [0.0] * n_dom_sets

    def set_weights(self, weights: Sequence[WeightT]) -> None:
        assert len(weights) == self.n_dom_sets
        self.weights = list(weights)

    def _lift(self, b: np.ndarray) -> np.ndarray:
        N_bdy, N_dom = self.N_bdy, self.N_dom
        b_int = b[N_bdy:]
        out = np.empty(N_bdy + self.n_dom_sets * N_dom, dtype=np.float64)
        out[:N_bdy] = b[:N_bdy]
        for k, w in enumerate(self.weights):
            out[N_bdy + k * N_dom : N_bdy + (k + 1) * N_dom] = _as_scaled(w, b_int)
        return out

    def _extract(self, t: np.ndarray) -> np.ndarray:
        N_bdy, N_dom = self.N_bdy, self.N_dom
        out = np.zeros(N_bdy + N_dom, dtype=np.float64)
        out[:N_bdy] = t[:N_bdy]
        accum = np.zeros(N_dom, dtype=np.float64)
        for k, w in enumerate(self.weights):
            block = t[N_bdy + k * N_dom : N_bdy + (k + 1) * N_dom]
            accum += _as_scaled(w, block)
        out[N_bdy:] = accum
        return out

    def matvec(self, b: np.ndarray) -> np.ndarray:
        return self._extract(self.big.apply(self._lift(b)))

    def predict_blocks(self, theta_inv_rhs: np.ndarray) -> np.ndarray:
        return self.big.apply(self._lift(theta_inv_rhs))

    def as_linear_operator(self) -> LinearOperator:
        n = self.N_bdy + self.N_dom
        return LinearOperator(shape=(n, n), matvec=self.matvec, dtype=np.float64)


# ---------------------------------------------------------------------------
# Small-factor preconditioner: Θ_train^{-1} ≈ U_small U_small^T
# ---------------------------------------------------------------------------


class SmallPrecond:
    """Apply x ≈ Θ_train^{-1} @ b  via  x[P] = U_s * (U_s^T * b[P]).

    GPU path: upload U_csr and L_csr to cuSPARSE once; each `matvec` runs
    two cuSPARSE matvecs.
    """

    def __init__(
        self,
        U: scipy.sparse.csc_matrix,
        P: np.ndarray,
        use_gpu: bool | None = None,
    ):
        self.U = U.tocsr()
        self.L = U.T.tocsr()
        self.P = np.asarray(P, dtype=np.int64)

        if use_gpu is None:
            use_gpu = _gpu_sparse_default() and _gpu_sparse_available()
        self.use_gpu = bool(use_gpu)

        if self.use_gpu:
            self.U_gpu = _cp_sp.csr_matrix(self.U)
            self.L_gpu = _cp_sp.csr_matrix(self.L)
            self.P_gpu = _cp.asarray(self.P)

    def _matvec_cpu(self, b: np.ndarray) -> np.ndarray:
        permuted = b[self.P]
        y = self.L @ permuted
        z = self.U @ y
        out = np.empty_like(b)
        out[self.P] = z
        return out

    def _matvec_gpu(self, b: np.ndarray) -> np.ndarray:
        b_gpu = _cp.asarray(b)
        permuted = b_gpu[self.P_gpu]
        y = self.L_gpu @ permuted
        z = self.U_gpu @ y
        out_gpu = _cp.empty_like(b_gpu)
        out_gpu[self.P_gpu] = z
        return _cp.asnumpy(out_gpu)

    def matvec(self, b: np.ndarray) -> np.ndarray:
        if self.use_gpu:
            return self._matvec_gpu(b)
        return self._matvec_cpu(b)

    def as_linear_operator(self) -> LinearOperator:
        n = self.U.shape[0]
        return LinearOperator(shape=(n, n), matvec=self.matvec, dtype=np.float64)


# ---------------------------------------------------------------------------
# Backwards-compat shim for the older NonLinElliptic pcg_ops interface.
# ---------------------------------------------------------------------------


class ThetaTrainMatVec(LiftedThetaTrainMatVec):
    def __init__(self, big, N_bdy, N_dom):
        super().__init__(big, N_bdy, N_dom, n_dom_sets=2)

    def set_delta_coefs(self, delta_coefs: np.ndarray) -> None:
        self.set_weights([np.asarray(delta_coefs, dtype=np.float64), 1.0])


def predict_via_big(big: BigFactorOperator, theta_inv_rhs, delta_coefs, N_bdy, N_dom):
    op = ThetaTrainMatVec(big, N_bdy, N_dom)
    op.set_delta_coefs(delta_coefs)
    t = op.predict_blocks(theta_inv_rhs)
    return t[N_bdy:N_bdy + N_dom]
