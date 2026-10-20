# Reusing projected data across different workloads

`commitjl.reusable_native` prepares a representation once and keeps it available
for subsequent native distance queries. It supports Full, PCA, and the protected
centroid-span plus residual Sparse-JL method. Original experiment scripts and
results are unchanged. This module does not change the integer or ZK protocol.

The protected model validates its basis once and precomputes `A @ U` once.
Projection is performed in row blocks; it never creates a dense combined JL
matrix. All methods cache point norms and use the same BLAS distance formula.
Full can reuse an already immutable original array without another copy.

Projected data, norms, and the actual projection parameters can be saved and
loaded in a later Python session. Cache loading checks the source/model binding,
payload hashes, dimensions, and norms. These checks detect stale or corrupted
local files; they are not a substitute for a cryptographic proof.

```python
import numpy as np
from commitjl.protected_jl import centroid_basis
from commitjl.sparse_gen import sparse_matrix_gen_csr
from commitjl.reusable_native import NativeProjection, PreparedDataset, array_hash

# X and C are the validated data and already fixed centroids.
# seed is the same pinned map seed for the whole workload.
u = centroid_basis(C)
q, total_m = 1, 256
_, a = sparse_matrix_gen_csr(seed, total_m-u.shape[1], X.shape[1], q)
a = a.astype(float)
a.data /= np.sqrt(q)
model = NativeProjection('hybrid', X.shape[1], basis=u, sparse_map=a, origin=C[0])
state = model.prepare(X)
state.save('results/my_reusable_cache')  # requires a new folder

# Each query may choose different data points and landmarks.
distances = state.squared_distances(point_indices, landmark_indices)

# In another session, validate the current input and the intended model again.
state = PreparedDataset.load(
    'results/my_reusable_cache',
    expected_source_sha256=array_hash(X),
    expected_model_sha256=model.fingerprint,
)
distances = state.squared_distances(new_point_indices, new_landmark_indices)
```

A changed dataset, basis, origin, PCA components, or sparse map requires a new
matching cache. New incoming points must still be transformed by the pinned
model. A fresh proof challenge can define a different sparse map; projected
values from another challenge cannot be substituted into that proof instance.

If there is enough RAM, `state.precompute_distances(max_bytes=...)` caches the
entire distance matrix. It requires `n*n*8` bytes in every method; projected
dimension does not reduce the size of that matrix. Future queries then perform
table lookups. Full receives exactly the same option. Repeating an identical
input/task also allows every method to cache the final answer itself.

## Bounded laptop experiment

Run from CMD in a new output folder:

```bat
cd /d C:\PyProjects\zero-knowledge\code
python -u -m experiments.laptop_reuse_workload --out results\laptop_reuse_next
```

Default configuration:

- Preserved Gisette `3000 x 5000`, 20NG TF-IDF `3000 x 2000`, and MNIST
  `5000 x 784`; no new data download and no extra point subsampling before
  representation preparation.
- Same existing models: `k=5`, saved Full start 3, total dimension 256 for
  PCA and protected JL; protected rank 4, residual dimension 252, q=1,
  archived fresh-map seed 100. Full retains its original dimension.
- 32 different deterministic distance queries. Each selects 1,024 points and
  128 other landmarks without overlap within the query, computing 131,072
  squared distances. Point sets may overlap between different queries.
- Four BLAS threads, three timing passes, shuffled method order.
- Two identical cache policies for every method: representation plus norms;
  representation/norms plus complete distance matrix. Default distance budget
  200 MiB per model (Gisette/text matrix 68.7 MiB, MNIST matrix 190.7 MiB).
- Query outputs and indices, every individual timing, cumulative timing
  checkpoints, persistent compressed representations/models, cache replay
  checks, source snapshots, and a measured report/plot are saved.

The one-time timing includes model validation/AU preparation, whole-subset
projection, input hashing, and point norms. Existing PCA fitting and centroid
training, archive I/O, and cache disk writing/loading are excluded from these
compute comparisons. Cache disk times are reported separately in validation.
This is a native amortization screen for already learned models, not
training-inclusive timing, held-out quality confirmation, or a ZK proof test.

The relevant comparison is cumulative cost:

`T(Q) = T_prepare + sum(T_query)`

Reusing projected data can amortize compression, but cannot guarantee a speed
win for every workload. Once complete distance matrices are cached, query costs
are similar across methods. Original-space accuracy and compressed geometry
must be reported alongside runtime. Three timing passes are exploratory;
substantial speed claims need confirmation under controlled laptop load.
