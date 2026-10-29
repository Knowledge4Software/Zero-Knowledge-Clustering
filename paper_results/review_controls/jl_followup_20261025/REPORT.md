# Further JL experiments, 25 October 2026

All fixed variants/seeds and negative results are retained. The ensemble compares every160 stored head update; this reuses development fits. The covariance/sketch and refinement screens use all6 previously difficult natural regressions, a post-hoc stress set. Neither is new independent validation. No production predictor or proof binary changed.

## Assignment estimates

| Variant | Mean agreement (%) | Worst update (%) | Extra errors vs original | Extra sketch scalars/row |
|---|---:|---:|---:|---:|
| median3x16 | 99.828491 | 98.339844 | 149 | 48 |
| median3x16_gate | 99.917603 | 99.316406 | 3 | 48 |
| median3x16_shrink | 99.921265 | 99.316406 | -3 | 48 |
| median3x32 | 99.887695 | 99.023438 | 52 | 96 |
| median3x32_gate | 99.918213 | 99.316406 | 2 | 96 |
| median3x64 | 99.904785 | 99.023438 | 24 | 192 |
| single64 | 99.884033 | 99.218750 | 58 | 64 |

The original predictor mean is99.919434%. The three-sketch shrunken estimate removes only3 errors among163840 record/update comparisons; its worst update is lower than the original worst. This tiny development gain does not establish a useful improvement or justify adding its48 sketch scalars and authentication cost. Margin gating does not produce a net accuracy improvement. All variants retain the separate no-JL rounded-Gram control in the raw exports.

## Certificate subspaces

| Variant | Difficult regressions certified |
|---|---:|
| base63 | 0/6 |
| jl_cov_p0_s20261101 | 0/6 |
| jl_cov_p0_s20261102 | 0/6 |
| jl_cov_p0_s20261103 | 0/6 |
| jl_cov_p1_s20261101 | 4/6 |
| jl_cov_p1_s20261102 | 4/6 |
| jl_cov_p1_s20261103 | 4/6 |
| jl_cov_p2_s20261101 | 4/6 |
| jl_cov_p2_s20261102 | 4/6 |
| jl_cov_p2_s20261103 | 4/6 |
| trained79 | 4/6 |

Covariance sketches are sparse random range finders with zero/one/two training-covariance multiplications. One/two powers use training data only and reuse the unchanged deterministic rounded-basis bounds. This is randomized subspace iteration, not a new JL distortion theorem. All variants have the same79 certificate rank; power methods use8 oversampling columns before selecting16 directions. Fitting times are retained per fit. A matching decision count does not imply tighter intervals or a faster prover.

## Exact-correction scheduling estimates

| Certificate/hint | Decisions | Total exact rows | Maximum per request |
|---|---:|---:|---:|
| r63_median3x16 | 6/6 | 33 | 22 |
| r63_median3x32 | 6/6 | 33 | 22 |
| r63_median3x64 | 6/6 | 33 | 22 |
| r63_single64 | 6/6 | 33 | 22 |
| r63_width | 6/6 | 34 | 23 |
| r79_median3x16 | 6/6 | 17 | 16 |
| r79_median3x32 | 6/6 | 17 | 16 |
| r79_median3x64 | 6/6 | 17 | 16 |
| r79_single64 | 6/6 | 17 | 16 |
| r79_width | 6/6 | 18 | 17 |

Exact rows here are numerical oracle refinements selected only after an untrusted sketch/width ordering. Certificate acceptance still checks deterministic intervals. An authenticated selective exact-replacement proof circuit is not implemented; no faster-proof claim follows. Compare equal sketch budgets: median3x16 stores48 values, single64 stores64, median3x64 stores192.

All126 final interval rows were independently rechecked against recorded integer Full truth. Zero incorrect conclusive decisions; zero new SNARKs. Total numerical run:7.26minutes. Source/file hashes are in checked_receipt.json and the run protocol. Artifacts remain private for supervisor review.

## Lower-rank follow-up

This post-hoc extension keeps all6 difficult cases and all three seeds. It compares training-aware sparse sketches against the trained residual-SVD control at each equal rank.

| Rank | Trained SVD | JL, one covariance multiplication (three seeds) | JL, two covariance multiplications (three seeds) |
|---:|---:|---|---|
| 71 | 4/6 | 2/6, 3/6, 2/6 | 4/6, 4/6, 4/6 |
| 75 | 4/6 | 4/6, 4/6, 4/6 | 4/6, 4/6, 4/6 |
| 79 | 4/6 | 4/6, 4/6, 4/6 | 4/6, 4/6, 4/6 |

Rank71 with two covariance multiplications matches4/6 from trained79, but trained71 already matches4/6 too. Thus this screen supports testing lower auxiliary rank, not attributing a unique gain to JL. A full stored-suite/independent-fit lower-rank sensitivity and matched proving-cost comparison is not established by these6 post-hoc cases. All126 additional intervals contain Full truth; zero new proofs.
