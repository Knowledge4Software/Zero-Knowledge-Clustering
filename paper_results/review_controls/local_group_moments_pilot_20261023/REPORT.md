# Group-moment correction: exploratory local pilot

This native numerical experiment completed in 97.74 seconds with two threads.
It reuses the two regression-bearing fits selected for the prior failed
block/refinement pilot: MNIST k5 seed11 and Fashion-MNIST k3 seed11, N1024.
All four actual updates at early and mature baselines are retained: eight
improvements and eight regressions, evaluated with head and PCA rank63.
It does not cover all 24 natural regressions or independent confirmation fits.

## Result

| Method | Tested regressions with exact change recovered | Median exact-row fraction | Range |
| --- | ---: | ---: | ---: |
| Head | 8/8 | 2.00% | 0.78%-7.23% |
| PCA | 8/8 | 4.79% | 0.98%-12.01% |

These are fractions requiring exact original-space nearest-center calculations,
not prover-time savings. The earlier 49-block, width-priority refinement needed
median 81.74%/81.45% exact rows on these same tasks. Early updates require more
correction here: median 11.43% head / 13.92% PCA, maximum 48.73% / 52.15%.
A different mechanism can therefore complement the original early-update
certificate rather than uniformly replace it.

## Why the calculation is exact

Fix a partition of rows before future centers, using the deployed approximate
labels g_i. Correctness does not assume those labels are nearest centers.
For each group g, retain coordinate sums S_g and count n_g. The sum of distance
changes under those fixed labels is exactly

    A = sum_g [2 S_g dot (old_g - new_g)
               + n_g (||new_g||^2 - ||old_g||^2)].

The assignment certificates identify rows whose old and new true nearest
labels both equal g_i. A certified stable row contributes exactly its fixed
label change. For every other row i, compute both exact nearest-center costs
and subtract its fixed-label contribution from A, replacing it with the
actual contribution. The result is the exact cost change, not an interval
obtained by adding per-record norm errors. Full refinement remains correct
if no assignments are certified.

The groups were chosen once from deployed approximate labels before looking
at each update. They do not use future centers or held-out regression signs.
This is compatible with unknown future center values, but usefulness depends
on stable group assignments. The experiment uses fixed k and center index
correspondence. It covers the full fixed audit catalog; arbitrary subset
queries need corresponding authenticated moments or a new aggregation proof.

## Evidence limits

All 32 method/request total changes match independently evaluated exact Full
and the frozen true changes. Every row treated as stable was independently
checked against original-space old/new argmins. Data hashes match the frozen
campaign. PCA bases match; all locally refitted head bases differ from the
historical ones and are labelled fresh local ablations. Actual basis arrays,
fixed groups, centers, counts and moments are persisted in per-fit NPZ files.
Group moments are public-benchmark diagnostics here; production group moments
must not be exposed without an explicit privacy contract.

Nineteen targeted Python tests pass, covering this reference, prior block
bounds, refinement and existing paired arithmetic. This is not a security
audit, a proof execution or a measured cryptographic speedup.

## Required next steps

1. Evaluate all existing 24 regressions and retain all companion improvements;
   then freeze the design and check independent fits. No general sensitivity
   claim follows from eight development cases.
2. Design authenticated moment binding. Prove each original row belongs to
   exactly one fixed group, authenticate both membership and all group sums
   and counts against the original catalog, and prevent omitted/duplicate rows.
   Unchecked prover-supplied sums make the argument unsound.
3. Keep moments and memberships private/committed as required. Prove stable
   assignment flags or exact corrections inside the circuit; a private flag
   alone is not evidence. Range-check signed integer moments and aggregates.
4. Measure the new Stage1 aggregation, query certification, uncertain-row
   exact work, key/setup, memory and public verification costs against equally
   reusable Full. Extra aggregation may erase the advantage suggested by row
   counts. No proving speedup is claimed until measured.

No new Mobin jobs or manuscript scientific claims were changed by this pilot.
