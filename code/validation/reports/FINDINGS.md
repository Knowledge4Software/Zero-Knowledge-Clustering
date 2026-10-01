# CommitJL audit findings

These findings apply to commit
`2d8e74cd6f2884181a0c87d466eaa7eee4050a9a`.

## Critical findings

1. **Resolved in the working tree:** the native protocol and ZK bridge now use
   the same field-native `C_theta` and `C_A`. The bridge regenerates the matrix
   from the transcript-derived seed and validates every witness value before
   invoking the Go prover. Non-empty `mu` is rejected until the circuit binds it.

2. **The circuit does not enforce all required arithmetic bounds.** Inputs,
   projected centroids, differences, distances, comparisons, and aggregate cost
   are not all constrained under the manuscript's `M`. Integer-field equivalence
   is not established for the artifact.

3. **Sparse row-index validation is incomplete.** The `m-row-1` check needs a
   canonical `[0,m)` construction and adversarial tests for invalid field values.

## Major findings

4. **Resolved in the working tree:** `psi_hat` previously truncated floating
   inputs to `int64`. It now preserves float64 values and returns `0.64` for
   the regression case.

5. **Resolved in the working tree:** squared-distance calculations now prove
   an int64 safety bound and otherwise fall back to exact Python integers.

6. **Resolved in the working tree:** RQ4 now treats the `A/sqrt(q)` cost as
   already normalized and no longer divides by `q` twice.

7. The symbolic cost model does not represent the current selector-based circuit
   construction. Symbolic reductions are not measured constraint reductions.

8. Stored proof evidence consists of one `n=12,d=m=8` run, with no dimension
   reduction and no measured full-dimensional baseline.

## Data and reproducibility findings

9. MNIST was independently fetched and its 2,000-row hash matched. Other planned
   real datasets have no current data artifacts or result rows.

10. Loaders do not verify upstream checksums or freeze complete source IDs,
    licenses, splits, and preprocessing versions. The processed-array manifest
    is useful but is not an independent download-integrity mechanism.

11. Stored results cover only a subset of the declared grid. The preregistered
    proof grid and ten proof repetitions have not been completed.

12. Twenty-five certificate rows failed because `gamma` was passed to the DBSCAN
    generator. They must remain in the attempted-run denominator.

13. The stored DBSCAN synthetic hash does not match regeneration from the
    current generator and recorded configuration. Those rows are not currently
    reproducible.

## Overall assessment

The fixed-center theory, sparse generator, encoding/hash tests, and many support
functions have a sound base. The artifact does not yet provide enough end-to-end
evidence for the primary claim: reduced proof cost at a fidelity-eligible
operating point under the complete secure protocol.

