# Local certificate refinement pilot: 23 October 2026

Completed in 142.23 seconds with two native compute threads.

Exploratory selection: MNIST k=5/seed11 and Fashion-MNIST k=3/seed11, N=1024 each; all four actual updates at early and mature baselines. Sixteen unique requests: eight early improvements and eight mature regressions. Both head and PCA evaluated at rank63 with 1/4/16/49 fixed contiguous residual blocks. No new cryptographic circuit or proof-speed measurement.

## Main result

| Method | Residual blocks | Mature regressions certified without exact rows | Median interval-width reduction | Median exact-row fraction needed (width priority) |
| --- | ---: | ---: | ---: | ---: |
| learned | 1 | 0/8 | 0.00% | 84.57% |
| learned | 4 | 0/8 | 3.10% | 83.11% |
| learned | 16 | 0/8 | 6.61% | 82.52% |
| learned | 49 | 0/8 | 11.20% | 81.74% |
| pca | 1 | 0/8 | 0.00% | 84.38% |
| pca | 4 | 0/8 | 3.18% | 82.91% |
| pca | 16 | 0/8 | 6.76% | 82.23% |
| pca | 49 | 0/8 | 11.38% | 81.45% |

A 50% exact-row budget resolves zero mature regressions for either method at every tested block count. A 75% budget resolves two of eight; complete exact refinement resolves eight of eight. With 49 blocks, width-priority refinement needs 68.75%-94.43% of rows for head and 69.34%-94.53% for PCA. Public row-order refinement has median requirements of 88.77%/88.67%, respectively.

## Provenance and validation

Training and audit array hashes match the frozen campaign for both datasets. Every exact update cost change matches its frozen value. PCA basis hashes match in all four dataset/age fits, and the one-block endpoints match the frozen intervals exactly. All four locally refitted head basis hashes differ; head results are a fresh local representation ablation on the same rows/centers, not a byte-identical replay of the historical head. No cause for that difference is assumed.

All 131,072 per-row method/block intervals contain the exact integer change. New intervals are contained in their original global-bound counterparts. Every fully refined interval collapses to the exact total; zero false sign decisions. Sixteen targeted Python tests pass, including nonorthogonal rounded bases, integer truth, original-bound equivalence and refinement monotonicity.

The reported minimum exact-row fractions are retrospective numerical paths, not an implemented private stopping protocol. Width-priority ordering may expose data-dependent information if not protected in a future circuit. Native runtime includes exact reference computation; it is not a proving speed estimate.

## Decision

Do not start a large proof campaign for this block design. The improvement is too small to resolve these regressions directly, and refinement needs most rows. More informative directional or properly authenticated aggregate residual statistics are the next research possibility; they are not implemented or shown effective here. Direct Full remains the reliable comparator for mature-regression tasks. The other sixteen observed natural regressions were not evaluated by this small pilot.
