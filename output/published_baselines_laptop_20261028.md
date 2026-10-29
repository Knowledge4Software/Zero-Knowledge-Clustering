# Matched laptop comparison, 28 October 2026

Serial outer invocation wall time; actual cold setup, native work, proofs, public-only replay and paid fallback. ZKML diagnostic MockProver excluded and reported separately; local test KZG parameter creation included.

| Case | Method | Seconds | Full/method | Agreement % | Decisions before Full fallback | Fallback | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| fashion_n32 | Exact Full / Groth16 | 422.416 | 1.000 | 100.000 | 3/3 | 0 | 1.725 |
| fashion_n32 | ZKML primitive adapter / Halo2-KZG | 197.684 | 2.137 | 100.000 | 3/3 | 0 | 0.747 |
| fashion_n32 | Centroid certificate | 387.324 | 1.091 | 100.000 | 3/3 | 0 | 1.249 |
| fashion_n32 | PCA certificate | 392.274 | 1.077 | 100.000 | 3/3 | 0 | 1.248 |
| text_n32 | Exact Full / Groth16 | 292.366 | 1.000 | 100.000 | 6/6 | 0 | 1.182 |
| text_n32 | ZKML primitive adapter / Halo2-KZG | 177.655 | 1.646 | 100.000 | 6/6 | 0 | 0.371 |
| text_n32 | Centroid certificate | 554.768 | 0.527 | 93.750 | 5/6 | 1 | 1.236 |
| text_n32 | PCA certificate | 551.186 | 0.530 | 89.583 | 5/6 | 1 | 1.244 |

## Interpretation boundaries

- ZKML uses published exact dot-product, adder, Poseidon chips and Halo2-KZG with our integer audit adapter; this does not benchmark the unmodified optimizing compiler.
- Different owner-approved commitment constructions bind the same original integers; this is not the same Poseidon2 Merkle root.
- Two previously observed fitted workloads, fixed first32 audit records and one timing execution each; no population or large-catalog claim.
- Assignment agreement is agreement with exact quantized labels, not ground-truth classification accuracy.

The paper is unchanged. Exact controls resolve all costs; approximate prediction agreement is separate from interval soundness.
