# Protected-coordinate Sparse JL research variant

The original experiment runners and manuscript are unchanged. Experimental
features are opt-in and use `zk/bin/commitjl-experimental-v3.exe`; the existing
default prover binary is retained. Source changes can trigger a rebuild through
the normal bridge, but unset feature flags retain the existing projection and
distance path. A verification key from another circuit cannot be reused.

## Native representation

Given fixed public centroids c, take leading left singular vectors U of the
matrix of differences `c_j-c_0`. U is fixed before the fresh map challenge.
For points shifted by c_0, output `[U^T x; A(I-UU^T)x]`, with A the canonical
fresh Sparse JL matrix normalized by sqrt(q). The implementation uses
`Ax-(AU)(U^T x)` and avoids forming the dense residual matrix.

When U spans all centroid differences (rank at most k-1), every centroid's
residual is zero in exact real arithmetic. The residual point norm is common
to all center distances, so original nearest-center labels are preserved.
Protecting fewer directions has no exact-assignment guarantee. Degenerate
centers reduce the required rank. Float ties and fixed-point rounding require
separate evaluation.

The protected-only ablation also preserves fixed-center labels at full rank.
Thus perfect ARI alone does not establish a contribution from JL. The random
residual's possible contribution is preservation of geometry/objective and
conditional guarantees. These require updated theory; the original oblivious
map theorem must not be copied unchanged onto the adaptive hybrid. Conditional
on a fixed orthogonal U, a valid residual JL event yields a full-space event by
adding the exact protected squared norm. This observation is not a new
end-to-end security proof or a novelty claim.

## Exact integer proof representation

For B=round(T U^T), the circuit outputs

`[T B x; T^2 A x - (A B^T)(B x)]`.

The first r distance coordinates have weight q; the residual coordinates have
weight one. Dividing total squared distance by q*T^4 recovers the intended
normalization, up to basis rounding. No field division or irrational sqrt(q)
is used. X and the original centroids remain authenticated and range-checked;
the sparse records remain witnessed, distribution-checked and authenticated.
The fixed integer basis is part of the compiled relation/verification key.
The experimental CLI does not enforce that it is the prescribed centroid SVD:
a production verifier must validate/pin the basis, centroids and parameters
before the challenge and validate the challenge-derived A. No production
setup ceremony or deployed randomness service was evaluated.

## Circuit alternatives

- `score_distances`: replace repeated coordinate differences by center norms
  and dot products, with a signed score shifted into a constrained positive
  interval. Ties still select the lowest index; the exact original objective
  for the chosen representation is retained. Full and PCA get the same option.
- `lookup_sparse`: generic sparse accumulation paths. Each operation adds
  sign*x to a same-bucket predecessor verified by two indexed lookup tables.
  Predecessors have earlier indices. A backend-committed-challenge permutation
  argument checks that predecessors plus terminals consume each node once.
  This prevents forks, cycles, dropped updates and wrong-bucket terminals.
  Cost scales with d*q+m per vector (including index checks and lookup
  overhead), rather than m*d coefficient products. Tiny m can make it worse.
  It adds probabilistic argument soundness assumptions and needs independent
  cryptographic review; tests do not establish universal security.
- `specialize_sparse`: direct routing for a compiled public row pattern;
  witnessed rows are constrained to it and CA authentication remains. A
  changed pattern requires a new circuit/setup/key. Setup must be included
  when comparing fresh maps; this option is a diagnostic, not a free generic
  optimization.

All features are opt-in. `solve_only` compiles and runs a constraint solver
with a test-only commitment-challenge substitution. It generates zero proofs
and performs no cryptographic verification. Real proving uses the backend's
actual commitment hints; the diagnostic override is never passed to Prove.
PLONK runs use the existing unsafe test KZG SRS and are correctness controls.

## Reproduction

From `code/`, choose a new output folder for each run:

```powershell
python -m experiments.protected_jl_study --out results/protected_native_new --seeds 5 --starts 5
python -m experiments.protected_jl_precision --out results/protected_precision_new
python -m experiments.protected_jl_proofs --out results/protected_compile_new --phase compile
python -m experiments.protected_jl_proofs --out results/protected_prove_new --phase prove --shapes 4x16x8x3x2
python -m tools.rebuild_protected_jl --study results/protected_jl_20261010_native_v3 --row 0 --out results/recovered_row0.npz
```

Native runs recover the exact eight saved server subsets and all five original
initializations without downloading/retraining. The screen covers k=5/10,
five fresh maps, m=16/32/64/128 and q=1/2, r=0/1/2/full rank, PCA and
protected-only controls. Crossed bootstrap intervals are pointwise and
conditional on fixed datasets. This is exploratory selection without an
independent confirmation split. The initial precision screen uses only the
first min(128,n) points, k=5/10, two starts and three maps. The subsequent
`protected_jl_20261010_precision_full_final` check uses all rows of every
current paper subset, k=5, two starts, three maps, and T=1024/4096. It compares
the four-protected-plus-sixteen-residual hybrid with PCA20 at matched total
dimension, using exact integer arithmetic. It does not test full-subset k=10
integer fidelity or full-subset ZK proving. The builder adjusts range widths
to the conservative fixed-point coordinate bound and falls back to ordinary
Sparse JL when the centroid-difference rank is zero.

Raw CSV rows, bases, lossless gzip assignment streams, archive input hashes,
source snapshots, proof/solver records, exact instance JSON and figures permit
later plots and replay. Native projection timings overlapped correctness
checks/profiling and are not publication-grade speed measurements. The final
small repeated-proof benchmark ran after the other computational experiments
finished, with four Go threads. Its tiny synthetic shape is an engineering
control and does not establish real-corpus superiority over PCA. See the
[consolidated evaluation](../results/protected_jl_20261010/report.md).
