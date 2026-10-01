# Implementation review

## Verified components

- Sparse-generator checks passed for boundary and ordinary shapes: `qd` records,
  exactly `q` distinct nonzeros per column, and coefficients in `{-1,0,+1}`.
- MiMC and commitment golden-vector tests passed.
- Encoding, commitment, protocol, quantization, projection, metric, and
  certificate unit tests passed.
- The Go circuit package passed `go test ./circuit`.
- The 2,000-trial unbiasedness result differed from theory by 1.47 standard
  errors, consistent with the expectation identity.

## Protocol-to-proof gap

The protocol and bridge now share field-native `C_theta` and `C_A` commitments.
The RQ2 path creates a protocol transcript first, and the bridge regenerates the
matrix from `H(CommitJL-v1 || C_X || C_theta || rho)` before checking the Merkle
opening, matrix commitment, assignments, and projected cost. An integration test
now validates the complete sequence:

```text
commit -> challenge -> derive matrix -> authenticate matrix -> prove -> verify
```

The circuit shape binds `m` and `q`. Arbitrary metadata `mu` is not part of the
current circuit commitment, so the protocol rejects non-empty `mu` rather than
silently claiming it is authenticated.

## Circuit arithmetic

The circuit checks projected point coordinates but does not fully enforce the
manuscript bounds for input coordinates, projected centroids, coordinate
differences, squared distances, comparison values, and `PsiHat`.

The sparse-row constraint range-checks `m-row-1`. Field arithmetic makes this
different from a direct canonical unsigned proof that `0 <= row < m`. It needs
an explicit adversarial test and canonical range construction.

## Native numeric behavior

`nearest_centroid_assign` and `psi_hat` now use int64 only after a worst-case
bound proves subtraction, squaring, and summation safe. Larger integer inputs
fall back to exact Python integers. The previous overflow counterexample now
selects the mathematically nearest center.

`psi_hat` now retains float64 precision for native inputs and returns an exact
Python integer for quantized inputs. The `0.9` versus `0.1` regression case now
returns `0.64`; its scale-10 integer equivalent returns `64`.

## Platform support

The committed proof binary is Mach-O and cannot execute on Windows.
`binary_available()` checks only for file existence, so ZK tests fail instead of
skipping or rebuilding it. The complete CLI also fails to compile on Windows
because it directly uses Unix-specific `syscall.Getrusage` APIs. The circuit
package passed from source, but full proving and verification were not executed
on this host.

## Proof coverage

The only stored proof row is `n=12,d=8,m=8,k=2,q=1` with one repetition. Since
`m=d`, it performs no dimensionality reduction. The preregistered evaluation
requires ten repetitions, a fidelity-selected operating point, adjacent
dimensions, and a measured full-dimensional control. That evidence is absent.

