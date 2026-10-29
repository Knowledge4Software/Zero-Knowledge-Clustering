# Matched laptop comparison, 28 October 2026

Serial outer invocation wall time; actual cold setup, native work, proofs, public-only replay and paid fallback. ZKML diagnostic MockProver excluded and reported separately; local test KZG parameter creation included.

| Case | Method | Seconds | Full/method | Agreement % | Decisions before Full fallback | Fallback | Peak RSS GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| fashion_n32 | Exact Full / Groth16 | 784.572 | 1.000 | 100.000 | 8/8 | 0 | 1.775 |
| fashion_n32 | ZKML primitive adapter / Halo2-KZG | 444.781 | 1.764 | 100.000 | 8/8 | 0 | 0.749 |
| fashion_n32 | Centroid certificate | 590.846 | 1.328 | 100.000 | 8/8 | 0 | 1.252 |
| fashion_n32 | PCA certificate | 598.409 | 1.311 | 100.000 | 8/8 | 0 | 1.253 |
| text_n32 | Exact Full / Groth16 | 357.501 | 1.000 | 100.000 | 8/8 | 0 | 1.191 |
| text_n32 | ZKML primitive adapter / Halo2-KZG | 224.725 | 1.591 | 100.000 | 8/8 | 0 | 0.370 |
| text_n32 | Centroid certificate | 965.053 | 0.370 | 95.703 | 2/8 | 6 | 1.236 |
| text_n32 | PCA certificate | 970.804 | 0.368 | 95.703 | 2/8 | 6 | 1.244 |

## Interpretation boundaries

- ZKML uses published exact dot-product, adder, Poseidon chips and Halo2-KZG with our integer audit adapter; this does not benchmark the unmodified optimizing compiler.
- Different owner-approved commitment constructions bind the same original integers; this is not the same Poseidon2 Merkle root.
- Two previously observed fitted workloads, fixed first32 audit records and one timing execution each; no population or large-catalog claim.
- Assignment agreement is agreement with exact quantized labels, not ground-truth classification accuracy.

The paper is unchanged. Exact controls resolve all costs; approximate prediction agreement is separate from interval soundness.
