# Matched laptop comparison, 28 October 2026

Serial outer invocation wall time; actual cold setup, native work, proofs, public-only replay and paid fallback. ZKML diagnostic MockProver excluded and reported separately; local test KZG parameter creation included.

| Case | Method | Seconds | Full/method | Agreement % | Decisions before Full fallback | Fallback | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| fashion_n128 | Exact Full / Groth16 | 1370.278 | 1.000 | 100.000 | 4/4 | 0 | 1.756 |
| fashion_n128 | ZKML primitive adapter / Halo2-KZG | 931.473 | 1.471 | 100.000 | 4/4 | 0 | 2.393 |
| fashion_n128 | Centroid certificate | 747.992 | 1.832 | 99.805 | 4/4 | 0 | 1.269 |

## Interpretation boundaries

- Exploratory Fashion scale test chosen after native screen; first4 fixed proposals, N128, one timing execution.
- Same frozen fitted models for all controls; basis-fitting excluded equally from this service-scale comparison.
- All attempted phase work and failures preserved; incomplete comparator phases are not speed evidence.
- Halo2 primitive adapter uses a different commitment/backend and exact-cost output.

The paper is unchanged. Exact controls resolve all costs; approximate prediction agreement is separate from interval soundness.
