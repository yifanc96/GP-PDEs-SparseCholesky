# GP-PDEs-SparseCholesky

**Sparse-Cholesky-accelerated Gaussian-process PDE solver — in Python.**

Meshless PDE solver on arbitrary point clouds, backed by an approximate
sparse Cholesky factorization of the kernel matrix that runs in
`O(N · ρᵈ)` time and memory. Handles 2D and 3D domains with complicated
geometry (curved boundaries, holes, cracks, CAD-style shapes); optional
JAX/CUDA acceleration. 

Two pieces, same repo:

1. **`kolesky`** — sparse approximate Cholesky of kernel matrices on
   point clouds in any dimension, using the Kullback–Leibler
   minimization and maximin ordering in
   [Schäfer, Katzfuss, Owhadi 2020](https://arxiv.org/abs/2004.14455).
2. **`kolesky.pde`** — a GP-regression PDE solver that obtains the factor in the case of PDE measurements at point clouds
   as a fast matvec *and* a preconditioner, following
   [Chen, Owhadi, Schäfer 2025](https://arxiv.org/abs/2304.01294).

Both are Python ports of the original Julia code:

* [KoLesky.jl](https://github.com/f-t-s/KoLesky.jl) — [arXiv:2004.14455](https://arxiv.org/abs/2004.14455) Thanks Claude Code's help to make it Pythonic.
* [PDEs-GP-KoleskySolver](https://github.com/yifanc96/GP-PDEs-SparseCholesky/tree/initial-julia-code) — [arXiv:2304.01294](https://arxiv.org/abs/2304.01294)

The original Julia source is preserved on the
[`initial-julia-code`](../../tree/initial-julia-code) branch.

```bibtex
@article{chen2025sparse,
  title={Sparse Cholesky factorization for solving nonlinear PDEs via Gaussian processes},
  author={Chen, Yifan and Owhadi, Houman and Sch{\"a}fer, Florian},
  journal={Mathematics of Computation},
  volume={94}, number={353}, pages={1235--1280}, year={2025}
}
```

## Contents

- [Install](#install)
- [Part 1 — `kolesky`: sparse Cholesky of kernel matrices](#part-1--kolesky-sparse-cholesky-of-kernel-matrices)
- [Part 2 — `kolesky.pde`: Gauss-Newton + pCG PDE solver](#part-2--koleskypde-gauss-newton--pcg-pde-solver)
- [Geometry gallery (2D + 3D)](#geometry-gallery-2d--3d)
- [Package layout](#package-layout)
- [Backends and timings](#backends-and-timings)

## Install

```bash
pip install -e .           # CPU only (NumPy + SciPy)
pip install -e '.[gpu]'    # + JAX/CUDA + optional CuPy
```

Python ≥ 3.10. GPU path requires a JAX build with CUDA.

---

## Part 1 — `kolesky`: sparse Cholesky of kernel matrices

Given `N` points `x₁ … x_N ∈ Rᵈ` and a kernel `K`, the `N×N` kernel
matrix is too big to store densely. `kolesky` returns a sparse
upper-triangular factor `U` such that

    Θ ≈ P (Uᵀ U)⁻¹ Pᵀ        equivalently    Θ⁻¹ ≈ P Uᵀ U Pᵀ

where `P` is the reverse-maximin permutation (coarse to fine). Storage
is `O(N · ρᵈ)`; `Θ v` and `Θ⁻¹ b` each cost `O(N · ρᵈ)`. `ρ` is the
user's accuracy–speed knob.

![U sparsity](docs/U_sparsity.png)

*40×40 grid → N=1600. At ρ=3 the factor has 2.3% of dense nnz; the
ratio shrinks as N grows. Figure produced by
[`docs/make_figures.py`](docs/make_figures.py).*

| operation        | cost       | how                                |
| ---------------- | ---------- | ---------------------------------- |
| `Θ v`            | `O(N·ρᵈ)`  | two triangular solves on U         |
| `Θ⁻¹ b`          | `O(N·ρᵈ)`  | two matvecs with U, Uᵀ             |
| sample N(0, Θ)   | `O(N·ρᵈ)`  | solve `U ξ = z` for `z ∼ N(0, I)`  |
| log-det          | `O(N)`     | `−2 Σ log Uᵢᵢ`                     |

### Quickstart

```python
import numpy as np, scipy.sparse.linalg as spla
import kolesky as kl

# 40×40 grid on [0,1]²
xs  = np.linspace(0.02, 0.98, 40)
pts = np.stack(np.meshgrid(xs, xs, indexing='ij'), axis=-1).reshape(-1, 2)

kernel = kl.MaternCovariance5_2(length_scale=0.15)
meas   = kl.point_measurements(pts, dims=2)

implicit = kl.ImplicitKLFactorization.build(kernel, meas, rho=3.0, k_neighbors=1)
explicit = kl.ExplicitKLFactorization(implicit, nugget=1e-8, backend='auto')

U, P = explicit.U, explicit.P   # U: scipy.sparse.csc_matrix;  P: np.ndarray[int64]
print(f'N = {U.shape[0]},  nnz = {U.nnz}')

# Verify: Θ v  ≈  (dense) K v
v  = np.random.default_rng(0).standard_normal(U.shape[0])
vp = v[P]
y  = spla.spsolve_triangular(U.tocsr(),   vp, lower=False)
z  = spla.spsolve_triangular(U.T.tocsr(), y,  lower=True)
Theta_v = np.empty_like(v); Theta_v[P] = z
print('rel err:', np.linalg.norm(Theta_v - kernel(meas) @ v) / np.linalg.norm(kernel(meas) @ v))
```

Output at N = 1600, ρ = 3:

```
N = 1600,  nnz = 59104
rel err: 2.15e-02
```

The factor is stored in the P-permuted order; all built-in
matvec/solve routines permute automatically. When using `U` by hand,
remember `Θ v ≈ P Uᵀ⁻¹ U⁻¹ Pᵀ v` (exactly what the verification block
above computes).

### What do those knobs mean?

The quickstart had four parameters that quietly do all the work:

- **`ImplicitKLFactorization.build(...)`** — *stage 1 of the
  factorization.* Computes the **reverse-maximin ordering** of the
  points (coarse scales first, fine last) and the **sparsity pattern**
  of the factor (which entries of `U` will be nonzero). Graph-theoretic
  work only, CPU, `O(N log N)` ish. Doesn't touch the kernel values
  yet, so it's cheap and can be reused across different kernels or
  nuggets.

- **`ExplicitKLFactorization(implicit, nugget=..., backend=...)`** —
  *stage 2.* Fills in the actual numbers: for each column group
  (a "supernode"), evaluate the local kernel matrix, add the nugget to
  the diagonal, Cholesky-factorize it, and write the resulting
  sub-columns into `U`. This is where all the arithmetic happens; it's
  what runs on GPU when `backend='jax'`.

- **`rho`** (aka ρ) — *accuracy ↔ cost knob.* At each column, only
  points within `ρ × ℓ_i` get a nonzero entry (where `ℓ_i` is that
  point's maximin length scale). Bigger ρ → denser factor, more
  accurate; cost scales like `O(N · ρᵈ)`. **ρ = 3 is the sweet spot**
  for the PDE examples in this README; empirically the relative
  forward-matvec error `‖Θv − Kv‖ / ‖Kv‖` decays roughly
  exponentially with ρ (≈ 2 × 10⁻² at ρ = 3, ≈ 5 × 10⁻³ at ρ = 4 on
  Matern 5/2). Every additional unit of ρ trades a factor of `~ρᵈ`
  more nnz for roughly an order of magnitude in accuracy.

- **`k_neighbors`** — *variant of the maximin ordering.* In standard
  1-maximin (`k_neighbors = 1`, the default when unspecified), the
  next point picked is the one whose **nearest** already-processed
  point is farthest. With `k_neighbors = k > 1` ("k-maximin"), the
  next point is instead the one whose **k-th nearest** already-
  processed point is farthest. The resulting length scales `ell[i]`
  are larger (k-th nearest ≥ 1st nearest), which enlarges every
  column's sparsity neighborhood (`within ρ · ell[i]`) — so more nnz
  and better accuracy, at extra cost. `k = 3` is the usual default
  for PDE problems; `k = 1` is fine for a plain point cloud.

- **`nugget`** — *diagonal regularization.* A small `nugget · I` is
  added to each local kernel block before the Cholesky. Covariance
  matrices are typically numerically semi-definite at machine
  precision (especially for small kernel length scales or densely
  packed points); the nugget keeps Cholesky happy. Typical values:
  `1e-10` when you have plenty of conditioning headroom, up to `1e-6`
  or `1e-4` when you don't. Bigger nugget ⇒ more stable, less
  accurate.

### A note on kernel smoothness

The sparsity of `U` depends on how fast the kernel's screened inverse
decays — smoother kernels (e.g. Gaussian, Matérn with large ν) leave
*long-range* entries in `U` that the maximin truncation has to keep if
you want accuracy. Concretely, smoother ⇒ needs larger `ρ` at a given
target error, and storage / cost scale like `O(N · ρᵈ)`.

**Rule of thumb.**

- **Matérn 5/2, 7/2** work very well here — they're the defaults in
  every example.
- **Matérn 9/2, 11/2, Gaussian** — usable but expect to push `ρ` up
  before you're happy with the error; in high `d` this can be
  expensive.
- If you genuinely need a very smooth kernel: the kernel matrix in
  that regime is often numerically *low-rank*, so a low-rank
  approximation (Nyström, pivoted Cholesky) can be more efficient than
  a sparse KL factor. For high-dimensional data we recommend
  [eepperly/Randomly-Pivoted-Cholesky](https://github.com/eepperly/Randomly-Pivoted-Cholesky),
  which is robust and parameter-light.

### Derivative measurements (beyond point values)

Everything works for **any** linear functional of the GP, not just
point evaluations `u(xᵢ)` — critical for PDEs, where you need things
like `Δu(xᵢ)` or `∂₁₁u(xᵢ)` at each collocation point. A linear
functional `L` of a GP is itself a GP; its covariance kernel is
`K(Lₓ, Lᵧ)` (apply `L` twice). The rest of the pipeline is unchanged.

| class                              | measurement                                           |
| ---------------------------------- | ----------------------------------------------------- |
| `PointMeasurement`                 | `u(x)`                                                |
| `LaplaceDiracPointMeasurement`     | `w_Δ Δu(x) + w_δ u(x)`                                |
| `LaplaceGradDiracPointMeasurement` | `w_Δ Δu + ⟨w_∇, ∇u⟩ + w_δ u`                          |
| `HessianDiracPointMeasurement`     | `w₁₁ ∂₁₁u + w₁₂ ∂₁₂u + w₂₂ ∂₂₂u + w_δ u`  (d = 2)      |

These are precisely the linearizations the PDE solvers below feed in:

- `−Δu + α uᵐ = f`: Δδ measurement (+ a δ term from linearizing `uᵐ`).
- `−∇·(a∇u) + α uᵐ = f`: Δ∇δ measurement (picks up the `∇a` term).
- `uₜ + u uₓ − ν uₓₓ = 0` (Crank-Nicolson): Δ∇δ in 1D.
- `det(∇²u) = f`: ∂∂ measurement after linearization.

#### Worked example: factorize `K` over boundary δ + interior Δ

```python
import numpy as np, scipy.sparse.linalg as spla
import kolesky as kl
from kolesky import LaplaceDiracPointMeasurement

# 19×19 interior grid (Laplacian measurements) + grid boundary (Dirac).
h  = 0.05
xs = np.arange(h, 1 - h + 1e-12, h)
XX, YY   = np.meshgrid(xs, xs, indexing='ij')
interior = np.stack([XX.ravel(), YY.ravel()], axis=1)                      # (N_int, 2)
bt       = np.arange(0, 1 + 1e-12, h)
z, o     = np.zeros_like(bt), np.ones_like(bt)
boundary = np.unique(np.vstack([np.c_[bt, z], np.c_[bt, o],
                                np.c_[z, bt], np.c_[o, bt]]), axis=0)      # (N_bdy, 2)
N_int, N_bdy = len(interior), len(boundary)

# Two measurement groups:
m_bdy = LaplaceDiracPointMeasurement(                   # u(x_i)   (Dirichlet)
    coordinate=boundary,
    weight_laplace=np.zeros(N_bdy), weight_delta=np.ones(N_bdy),
)
m_lap = LaplaceDiracPointMeasurement(                   # −Δu(x_j)
    coordinate=interior,
    weight_laplace=-np.ones(N_int), weight_delta=np.zeros(N_int),
)

kernel   = kl.MaternCovariance7_2(length_scale=0.2)

# Multi-set build: pass a *list* of measurement groups. Plain `build` works
# because the two groups live at different locations (boundary vs interior —
# no co-location). For groups with co-located δ/Δδ pairs, use
# `.build_follow_diracs(...)` or `.build_diracs_first_then_unif_scale(...)`.
implicit = kl.ImplicitKLFactorization.build(kernel, [m_bdy, m_lap],
                                            rho=3.0, k_neighbors=3)
explicit = kl.ExplicitKLFactorization(implicit, nugget=1e-10, backend='cpu')

U, P = explicit.U, explicit.P
print(f'N = {N_bdy + N_int},  U.nnz = {U.nnz}')

# Verify: Θ v  ≈  (dense) K v
all_meas = kl.stack_measurements([m_bdy, m_lap])      # (N_bdy + N_int, 2)
K = kernel(all_meas)                                  # dense (same size)

v  = np.random.default_rng(0).standard_normal(K.shape[0])
vp = v[P]
y  = spla.spsolve_triangular(U.tocsr(),   vp, lower=False)
z  = spla.spsolve_triangular(U.T.tocsr(), y,  lower=True)
Theta_v = np.empty_like(v); Theta_v[P] = z
print('rel err:', np.linalg.norm(Theta_v - K @ v) / np.linalg.norm(K @ v))
```

Output at N = 441:

```
N = 441,  U.nnz = 19558
rel err: 6.09e-02
```

Take-aways:

- **Pass a list of measurement groups**, one per kind, to `.build`. Each
  group is itself batched (shape `(N_k, d)` coordinates + weights);
  `kolesky` figures out where each group sits in the ordering.
- `stack_measurements` merges those groups into one batched measurement,
  which is also what you apply a kernel to (`kernel(all_meas)` gives
  the full `(N × N)` covariance in the same row order the factor uses).
- For co-located groups (δ and Δδ at the *same* interior points, as in
  PDE solves), use `.build_follow_diracs(...)` or
  `.build_diracs_first_then_unif_scale(...)` instead — plain maximin
  would see distance-zero ties. See Part 2 for details.

---

## Part 2 — `kolesky.pde`: Gauss-Newton + pCG PDE solver

Each PDE is reduced to a sequence of linear GP regressions — one per
Gauss-Newton (GN) step. You supply: domain, right-hand side, boundary
data, initial iterate. With `backend='auto'` the heavy factorization
runs on GPU whenever JAX resolves to a CUDA device.

### Where `Θ_train` and `Θ_big` come from — walked through on `−Δu + α uᵐ = f`

Take the nonlinear elliptic PDE `−Δu + α uᵐ = f` on Ω with Dirichlet
BCs `u = g on ∂Ω`. Sample `N_bdy` boundary points `{xᵢᵇ}` and `N_int`
interior points `{xⱼⁱ}`.

**Step 1 — Gauss-Newton linearization.** Define `F(u) = −Δu + α uᵐ −
f`. Around a current iterate `v`,

    F(u) ≈ F(v) + F'(v) (u − v),

so setting `F(u) = 0` gives a *linear* problem for `u`:

    −Δu(x) + c(x)·u(x) = f(x) + α(m−1)·v(x)ᵐ    on interior,     u = g   on ∂Ω,

with the spatially varying coefficient `c(x) = α·m·v(x)ᵐ⁻¹`. Each GN
step is this linear PDE — only `c(x)` and the right-hand side change
from step to step.

**Step 2 — GP regression as the solver.** Place a GP prior
`u ~ N(0, K)` and impose the linearized equations as linear
functionals of `u` at the collocation points:

| at point               | linear functional `Lu`          | value    | measurement type         |
| ---------------------- | ------------------------------- | -------- | ------------------------ |
| `xᵢᵇ` (boundary)       | `u(xᵢᵇ)`                        | `g(xᵢᵇ)` | pure δ                   |
| `xⱼⁱ` (interior)       | `−Δu(xⱼⁱ) + c(xⱼⁱ)·u(xⱼⁱ)`      | rhsⱼ     | **Δδ** with weights `w_Δ = −1`, `w_δ = c(xⱼⁱ)` |

Stack them into one training vector
`train = [δ_bdy, (−Δ + c·δ)_int]` of length `N_bdy + N_int`. The
posterior mean of `u` at interior points, which is the next GN iterate,
is the standard GP answer

    u(xⁱ) = K(δ_{xⁱ}, train) · Θ_train⁻¹ · rhs,                       (∗)

where

> **`Θ_train` = K(train, train)** — the `(N_bdy + N_int)²` kernel
> matrix over the *linearized* training measurements at the current
> iterate `v`.

This is what pCG inverts, once per GN step.

**Step 3 — why we don't build `Θ_train` directly.** Across GN steps
the weights `c(x)` change, so every entry of `Θ_train` changes. We'd
be building a fresh `(N_bdy + N_int)²` sparse Cholesky at every step.

But notice: each training measurement is a *linear combination of a
few fixed ones*. Concretely

    (−Δ + c·δ) u  =  (−1) · (Δu)  +  c · (δu),

so on the interior row-block,

    K(train, train)_int,int
      = K(−Δ + c·δ, −Δ + c·δ)_int
      = K(Δ, Δ) − c · K(δ, Δ) − c · K(Δ, δ) + c · cᵀ · K(δ, δ),

and the bdy-int cross block is similarly a linear combination of
`K(δ, δ)` and `K(δ, Δ)`. All of these sub-blocks live inside **one
bigger kernel matrix** whose measurements do *not* depend on `c(x)`:

> **`Θ_big` = K over the fixed 3-set measurement list
> `{ δ_bdy, δ_int, Δ_int }`** — size `(N_bdy + 2·N_int)²`.

**Step 4 — the lift / apply / extract trick.** Because `Θ_train · v`
is a linear combination of blocks of `Θ_big`, we compute it without
ever materializing `Θ_train`:

1. **lift** `v ∈ ℝ^{N_bdy + N_int}` to
   `v_lift ∈ ℝ^{N_bdy + 2·N_int}` by copying `v_bdy` and, at the
   interior, placing `c(xⱼⁱ) · v_int(j)` in the δ-block and
   `−1 · v_int(j)` in the Δ-block;
2. **apply** `Θ_big` to `v_lift` via the sparse factor of `Θ_big`
   (two triangular solves);
3. **extract** the `train`-row entries (boundary δ output + the
   same weighted linear combination in the interior) back to
   `ℝ^{N_bdy + N_int}`.

That's `LiftedThetaTrainMatVec` — the forward operator fed to
`scipy.sparse.linalg.cg`. The **expensive `Θ_big` sparse factor is
built only once**, outside the GN loop; what changes from step to step
is the cheap lift / extract weights.

**The small preconditioner.** pCG also wants a cheap approximation to
`Θ_train⁻¹`. We get one by building a smaller, 2-set sparse Cholesky
factor over just `{ δ_bdy, (−Δ + c·δ)_int }` — same size as
`Θ_train` itself — at the current `c(x)`. This does get rebuilt each
GN step, but it's much cheaper (one small factor, one-shot) and gives
a strong preconditioner; pCG converges in ~10 iterations.

**Summary.**

| matrix     | size                      | what it is                                                                                 | what it costs                                                                           |
| ---------- | ------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `Θ_train`  | `(N_bdy + N_int)²`        | kernel of the *linearized* PDE training measurements at the current GN iterate `v`.         | 2-set sparse factor, **rebuilt cheaply each GN step** as the pCG preconditioner.        |
| `Θ_big`    | `(N_bdy + 2·N_int)²`      | kernel over the *fixed union* of measurement types: boundary δ, interior δ, interior Δ.     | 3-set sparse factor, **built ONCE outside the GN loop**, reused as fast `Θ_train` matvec. |

The same pattern generalizes: `VarLinElliptic2d` has a `Δ∇δ`
measurement so its `Θ_big` has blocks `{δ_bdy, δ_int, (−a∆ − ∇a·∇)_int}`;
`MongeAmpere2d` needs `∂∂` blocks; etc. Every `kolesky.pde` solver
follows this lift → apply → extract pattern.

### Ordering multi-set measurements

Plain reverse-maximin is ill-defined when PDE problems have
**co-located measurement groups** — e.g. both `u(xᵢ)` and `Δu(xᵢ)` at
every interior point, distance zero. Two canonical variants:

- **FollowDiracs** — maximin on `(boundary δ, interior δ)`, then insert
  each derivative measurement immediately after its δ. Keeps co-located
  `(δ, Δδ)` pairs in the same supernode. We found this performs better
  and leads to a sparser factor in our experiments. *Used by
  `NonlinElliptic2d` and `Burgers1d`.*
- **DiracsFirstThenUnifScale** — same maximin step, then append each
  derivative block at the finest length scale. This is the variant
  theoretically analyzed in our paper. *Used by `VarLinElliptic2d` and
  `MongeAmpere2d`.*

The measurement set for `Θ_big` is baked into each solver:

| PDE                 | ordering                      | `Θ_big` measurement sets                    |
| ------------------- | ----------------------------- | ------------------------------------------- |
| `NonlinElliptic2d`  | FollowDiracs (3 sets)         | δ_bdy, δ_int, −Δ_int                         |
| `VarLinElliptic2d`  | DiracsFirstThenUnifScale (3)  | δ_bdy, δ_int, `−a∆ − ∇a·∇` on int            |
| `Burgers1d`         | FollowDiracs (4 sets)         | δ_bdy, δ_int, ∇_int, Δ_int                    |
| `MongeAmpere2d`     | DiracsFirstThenUnifScale (5)  | δ_bdy, δ_int, ∂₁₁, ∂₂₂, ∂₁₂                   |

### Quickstart: `−Δu + α uᵐ = f` on `[0,1]²`

```python
import numpy as np, kolesky as kl
from kolesky.pde import (
    NonlinElliptic2d, solve_nonlin_elliptic_2d, sample_points_grid_2d,
)

def u_exact(x): return float(np.sin(np.pi*x[0]) * np.sin(np.pi*x[1]))
def rhs(x):     return 2*np.pi**2 * u_exact(x) + u_exact(x)**3

eqn       = NonlinElliptic2d(alpha=1.0, m=3, domain=((0,1),(0,1)),
                             bdy=u_exact, rhs=rhs)
X_d, X_b  = sample_points_grid_2d(eqn.domain, 0.02, 0.02)
kernel    = kl.MaternCovariance7_2(length_scale=0.3)

sol = solve_nonlin_elliptic_2d(
    eqn, kernel, X_d, X_b, sol_init=np.zeros(X_d.shape[0]),
    GN_steps=3, rho_big=3, rho_small=3, k_neighbors=3, backend='auto',
)
```

Ground truth vs numerical on a 50×50 grid, Matern 7/2, 3 GN steps, ρ=3:

![NonlinElliptic comparison](docs/nonlin_elliptic_compare.png)

### Other PDEs

```python
from kolesky.pde import (
    VarLinElliptic2d, solve_var_lin_elliptic_2d,   # −∇·(a∇u) + α uᵐ = f
    Burgers1d,        solve_burgers_1d,             # uₜ + u uₓ − ν uₓₓ = 0
    MongeAmpere2d,    solve_monge_ampere_2d,        # det(∇²u) = f
)
```

See [`examples/`](examples/) for runnable scripts that mirror the Julia
reference code.

### Bring your own PDE

The four built-in solvers (`NonlinElliptic`, `VarLinElliptic`,
`Burgers1d`, `MongeAmpere2d`) cover a lot, but if your PDE isn't one of
them you build a new solver out of three low-level pieces in
`kolesky.pde.pcg_ops`:

| piece                    | role                                                                 |
| ------------------------ | -------------------------------------------------------------------- |
| `BigFactorOperator`      | applies `Θ_big` to a dense vector (two sparse triangular solves)     |
| `LiftedThetaTrainMatVec` | assembles `Θ_train` from `Θ_big` without materializing it            |
| `SmallPrecond`           | applies `Θ_train⁻¹` via the small sparse factor (two sparse matvecs) |

The **four-step recipe** every built-in solver follows:

1. **Pick measurements.** A measurement is a linear functional of `u`
   applied at a point. Match your operator:

   | operator terms you need  | measurement                                              |
   | ------------------------ | -------------------------------------------------------- |
   | `u` only                 | `PointMeasurement`                                       |
   | `Δu`, `u`                | `LaplaceDiracPointMeasurement` (weights `w_Δ`, `w_δ`)    |
   | `Δu`, `∇u`, `u`          | `LaplaceGradDiracPointMeasurement` (+ `w_∇` is a `d`-vec)|
   | `∂ᵢⱼu` in 2D             | `HessianDiracPointMeasurement` (2D only)                 |
   | anything else            | write a new dataclass + kernel pair evaluator (see below)|

2. **Build the big factor** with the multi-set measurement list
   `(δ_bdy, δ_int, L_int …)` — call `ImplicitKLFactorization.build_follow_diracs`
   or `.build_diracs_first_then_unif_scale`, then `ExplicitKLFactorization`.
   This is the expensive step; do it once.

3. **Build a small preconditioner factor** on the 2-set list `(δ_bdy,
   L_int)` where `L` is the full linear(ized) operator.

4. **pCG solve.** Wrap the big factor in `LiftedThetaTrainMatVec` and
   the small factor in `SmallPrecond`, then `scipy.sparse.linalg.cg`
   drives `Θ_train · α = rhs` in ~10–50 iters. Prediction at interior
   points is one final `Θ_big · lift(α)`.

For **nonlinear PDEs**, wrap steps 3–4 in a Gauss-Newton loop that
updates the operator's weights on each iterate — see
[`kolesky/pde/nonlin_elliptic.py`](kolesky/pde/nonlin_elliptic.py).

A ~80-line runnable template for a **linear reaction-diffusion**
`−Δu + c(x)·u = f(x)` lives at
[`examples/custom_pde_minimal.py`](examples/custom_pde_minimal.py).
Start from there, change `c(x)`, `f(x)`, the boundary data, and (if
needed) the measurement weights.

### When you need a new measurement type

If your operator involves a linear functional `L` the built-in
dataclasses don't cover (biharmonic `Δ²u`, mixed third derivatives, 3-D
Hessian, curl …), you need to:

1. **Add a dataclass** in `kolesky/measurements.py` with the weight
   fields `L` uses. Supply a `.d` property and an `.is_batched()`
   method so the rest of the pipeline (ordering, supernodes,
   factorization) treats it uniformly with the built-ins.
2. **Extend two helpers** in the same file — `stack_measurements`
   concatenates a list of batched measurements of one type along the
   batch axis (used whenever a multi-set group is merged into one
   training set); `select(m, idx)` returns the rows `idx` of a batched
   measurement (used by the maximin / supernode logic to grab a
   sub-point-cloud without copying every weight field). Both are plain
   `if cls is …:` dispatch tables; add a branch for your new dataclass
   so it round-trips through the factorization.
3. **Implement the kernel pair evaluator** `K(Lₓ, Lᵧ)` in
   `kolesky/covariance.py`. Two routes:
   - *Analytic* — differentiate the Matérn radial twice by hand (once
     per side of `L`) and plug into a broadcast NumPy evaluator. The
     existing `_np_ldld` (Δδ × Δδ) and `_np_lgdlgd` (Δ∇δ × Δ∇δ) paths
     are templates. Fast at runtime, no per-pair JIT.
   - *Autodiff* — let JAX compute `L₁ L₂ K(x, y)` via `jax.grad`,
     `jax.hessian`, `jax.jvp`, etc.; `MongeAmpere2d`'s Hessian kernel
     already takes this route (see the HessianDirac evaluator in
     `covariance.py`). Much less code, but slower: each supernode-size
     bucket JIT-compiles its own autodiff graph, and nested
     `jax.hessian` is expensive. Generally fine if your operator is
     rarely exercised or for a first prototype.

The Julia reference [KoLesky.jl](https://github.com/f-t-s/KoLesky.jl)
has more measurement types (higher-order derivatives, etc.) if you
need a starting point for the analytic route.

---

## Geometry gallery (2D + 3D)

The solver takes raw point arrays — no mesh, no element assembly, no
boundary re-derivation. Change the geometry by changing the sampler;
everything downstream (maximin ordering, factorization, Gauss-Newton,
pCG) runs unchanged. Every script is ~80 lines of Python describing
the geometry + one solver call.

### 2D

All 3000 interior + a few hundred boundary points, Matern 7/2 at σ=0.3,
ρ=3, 3 GN steps, `backend='cpu'` (any of them also runs on
`backend='jax'`).

| script                              | geometry                                     | L²     |
| ----------------------------------- | -------------------------------------------- | -----: |
| `lshape_nonlin_elliptic.py`         | L-shape with re-entrant corner               | 4e-6   |
| `swiss_cheese_nonlin_elliptic.py`   | square with 4 circular holes                 | 6e-6   |
| `flower_nonlin_elliptic.py`         | smooth non-convex, oscillating radius        | 7e-6   |
| `stadium_nonlin_elliptic.py`        | curved + straight boundary segments          | 5e-5   |
| `airfoil_nonlin_elliptic.py`        | NACA 0012 + box far-field                    | 5e-5   |
| `porous_nonlin_elliptic.py`         | 40 random circular inclusions                | 2e-6   |
| `heart_nonlin_elliptic.py`          | parametric heart curve                       | 3e-5   |
| `crack_nonlin_elliptic.py`          | zero-thickness horizontal slit               | 8e-6   |
| `koch_nonlin_elliptic.py`           | level-4 Koch snowflake                       | 1e-4   |
| `dumbbell_nonlin_elliptic.py`       | two disks joined by a narrow bridge          | 3e-5   |

|     |     |
| --- | --- |
| ![lshape](docs/lshape.png)   | ![swiss cheese](docs/swiss_cheese.png) |
| ![flower](docs/flower.png)   | ![stadium](docs/stadium.png)   |
| ![airfoil](docs/airfoil.png) | ![porous](docs/porous.png)     |
| ![heart](docs/heart.png)     | ![crack](docs/crack.png)       |
| ![koch](docs/koch.png)       | ![dumbbell](docs/dumbbell.png) |

### 3D

The nonlinear elliptic solver is **dimension-agnostic** (Δδ kernel and
maximin ordering both work in any `d`). A d-general alias
`solve_nonlin_elliptic` is exported alongside `_2d`. Same function,
same ρ, 5k interior points except torus at 10k:

| script                                  | geometry                             | L²    |
| --------------------------------------- | ------------------------------------ | ----: |
| `torus_nonlin_elliptic.py`              | solid torus                          | 1e-4\*|
| `swiss_cheese_cube_nonlin_elliptic.py`  | cube with 6 spherical holes          | 2e-4  |
| `bowl_nonlin_elliptic.py`               | ball minus an off-centre ball        | 1e-4  |
| `schwarzp_nonlin_elliptic.py`           | Schwarz-P triply-periodic surface    | 1e-3  |
| `helix_nonlin_elliptic.py`              | thick helical tube                   | 3e-4  |
| `bunny_nonlin_elliptic.py`              | Stanford bunny (OBJ mesh)            | 6e-4  |
| `bracket_nonlin_elliptic.py`            | L-bracket + bolt holes via CSG       | 5e-6  |

\*10k points; rest 5k. Reaching 2D-level accuracy in 3D needs
roughly `N^{3/2}` points (~10⁵); 5k is enough to show the method
working, not the asymptotic regime.

|     |     |
| --- | --- |
| ![torus](docs/torus.png)       | ![swiss cheese cube](docs/swiss_cheese_cube.png) |
| ![bowl](docs/bowl.png)         | ![Schwarz-P](docs/schwarzp.png) |
| ![helix](docs/helix.png)       | ![bunny](docs/bunny.png)        |
| ![bracket](docs/bracket.png)   |                                 |

**Samplers used.** Implicit inequality (torus, bowl, Schwarz-P),
numerical projection onto `f(x)=0` (Schwarz-P surface), distance-to-
curve via `cKDTree` (helix), mesh `contains` + surface sampling
(bunny, via `trimesh`), and CSG boolean via `manifold3d` (L-bracket —
no STEP reader needed).

*Optional deps:* `trimesh + rtree` (bunny), `manifold3d` (bracket).

---

## Package layout

```
kolesky/
├── measurements.py      # PointMeasurement / Δδ / Δ∇δ / ∂∂ / ∂∂+δ dataclasses
├── covariance.py        # Matern 1/2 … 11/2 + Gaussian
├── ordering.py          # Reverse-maximin ordering via mutable max-heap
├── supernodes.py        # Supernodal reverse-maximin sparsity pattern
├── factorization.py     # Implicit / Explicit KLFactorization
└── pde/
    ├── pdes.py              # PDE dataclasses
    ├── sampling.py          # 1D / 2D grid + random sample-point helpers
    ├── pcg_ops.py           # BigFactorOperator, LiftedThetaTrainMatVec, SmallPrecond
    ├── nonlin_elliptic.py
    ├── varlin_elliptic.py
    ├── burgers.py
    └── monge_ampere.py

examples/    # one script per PDE and per geometry
tests/       # pytest smoke tests
docs/        # figure generation for this README
```

---

## Backends and timings

Every factorization / solver accepts `backend={'cpu', 'jax', 'auto'}`:

- `'cpu'` — NumPy + SciPy per supernode, thread-pooled over supernodes.
- `'jax'` — JAX with size-bucketed batched Cholesky; GPU if CUDA present.
- `'auto'` — `'jax'` if `jax.default_backend() != 'cpu'`, else `'cpu'`.

Environment knobs:

- `KOLESKY_NUM_THREADS` (default 32) — CPU thread-pool size per
  supernode.
- `KOLESKY_ENABLE_GPU_SPARSE=1` — opt into CuPy cuSPARSE triangular
  solves inside pCG. Only helpful for N ≫ 10⁴ (below that, CuPy re-runs
  cuSPARSE analysis every call and loses to SciPy).

**Warm timings** (JIT cache hot):

| example             | h      | N       | CPU¹   | GPU²   | L² error |
| ------------------- | :----- | ------: | -----: | -----: | -------: |
| `NonlinElliptic2d`  | 0.02   |  2 600  | 3.6 s  | 1.5 s  | ~2e-5    |
| `NonlinElliptic2d`  | 0.01   | 10 200  | 16.6 s | 6.0 s  | ~1e-5    |
| `VarLinElliptic2d`  | 0.05   |    520  | 1.0 s  | 0.3 s  | 3.7e-2   |
| `Burgers1d`, T=0.1  | 0.01   |    200  | 0.30 s | 0.3 s  | 5e-3     |
| `MongeAmpere2d`     | 0.1    |    120  | 0.47 s | 0.11 s | 1.5e-2   |

¹ `backend='cpu'`, 32-thread pool with OpenBLAS pinned to 1 thread per
worker (`pip install threadpoolctl`). AMD EPYC dual-socket.
² `backend='jax'` on a single NVIDIA H200 GPU. *Cold* first-call times
are larger due to per-supernode-size JIT compilation; `MongeAmpere2d`
cold is ~60 s (each pair evaluator fires `jax.hessian` calls). For
`Burgers1d` at N=200 there's no GPU win — per-call dispatch overhead
matches CPU SciPy.

GPU advantage grows with N: ~2.4× at N≈2 600, ~2.8× at N≈10 200 in the
`NonlinElliptic2d` column.

### Comparison vs the Julia reference

Head-to-head against the original
[PDEs-GP-KoleskySolver](https://github.com/yifanc96/GP-PDEs-SparseCholesky/tree/initial-julia-code)
on the same machine (AMD EPYC 9554 / single NVIDIA H200) at matched
parameters (Matern 7/2, σ=0.3, ρ_big = ρ_small = 3, k_neighbors = 3,
3 Gauss-Newton steps):

| NonlinElliptic, h | N      | Julia CPU³ | Python CPU | Python GPU | L² error |
| ----------------: | -----: | ---------: | ---------: | ---------: | -------: |
| 0.02              |  2 400 | **1.3 s**  | 3.7 s      | 1.7 s      | ~2e-5    |
| 0.01              |  9 800 | **6.5 s**  | 16.7 s     | 7.8 s      | ~6e-6    |

³ Julia 1.11, IntelVectorMath/MKL. Matches `iterGPR_fast_pcg` in
`main_NonLinElliptic2d.jl`; `@elapsed` warm call, compilation excluded.
At matched ρ = 3 (the value used in every example in this README),
Python on GPU gets within ~1.2× of Julia on CPU; on CPU the Python port
is ~3× slower, mostly the BLAS difference (MKL vs OpenBLAS) and the
lack of JIT'd inner loops. Accuracy is the same order across all three
paths; tiny differences come from the nondeterministic maximin seed.

---

## License

MIT — see [LICENSE](LICENSE).
