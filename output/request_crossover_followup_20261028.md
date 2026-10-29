# Matched laptop comparison, 28 October 2026

Serial outer invocation wall time; actual cold setup, native work, proofs, public-only replay and paid fallback. ZKML diagnostic MockProver excluded and reported separately; local test KZG parameter creation included.

| Case | Method | Seconds | Full/method | Agreement % | Decisions before Full fallback | Fallback | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| text_n32 | Exact Full / Groth16 | 624.418 | 1.000 | 100.000 | 16/16 | 0 | 1.192 |
| text_n32 | ZKML primitive adapter / Halo2-KZG | 446.937 | 1.397 | 100.000 | 16/16 | 0 | 0.382 |
| text_n32 | Centroid certificate | 1651.455 | 0.378 | 95.703 | 4/16 | 12 | 1.247 |
| text_n32 | PCA certificate | 1648.700 | 0.379 | 95.508 | 4/16 | 12 | 1.248 |

## Interpretation boundaries

- One exploratory execution; previous8-request times and new16-request times are different executions, not paired statistical replications. Conditional cost models are not observed response times at untested counts.
- ZKML uses the primitive adapter, not the unmodified compiler. Different backend/commitment and exact-cost versus sign outputs remain explicit.

The paper is unchanged. Exact controls resolve all costs; approximate prediction agreement is separate from interval soundness.
