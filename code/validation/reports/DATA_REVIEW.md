# Dataset and result-file review

## Dataset status

| Dataset | Status | Evidence |
|---|---|---|
| Gaussian mixture | Verified | Three unique configuration/seed combinations were regenerated and their SHA256 hashes matched. |
| DBSCAN margin synthetic | Failed reproducibility | Regeneration from the recorded configuration and seed produced a different hash. |
| MNIST, 2,000-row sample | Verified | Fetched again through the declared OpenML loader. The hash matched; shape was `2000 x 784`; values were finite and in `[0,255]`; all ten labels were present. |
| Fashion-MNIST | No stored result | Planned, but no current Parquet row or downloaded artifact was found. |
| Gisette | No stored result | Planned, but no current Parquet row or downloaded artifact was found. |
| Arcene | No stored result | Planned, but no current Parquet row or downloaded artifact was found. |
| 20 Newsgroups TF-IDF | No stored result | Planned, but no current Parquet row or downloaded artifact was found. |
| Embedding corpus | Not defined | The manuscript retains a placeholder and the primary configuration does not run it. |

The loaders hash the final contiguous NumPy array. This confirms whether two
runs consumed the same processed array, but does not independently authenticate
the upstream download. The manifest does not verify upstream checksums or freeze
the exact source ID, dataset version, license, split, and preprocessing version.

MNIST was independently fetched during this audit and matched the stored hash.
The same process remains necessary for every other real dataset once its results
exist.

## Stored result files

| File | Rows | Status counts |
|---|---:|---|
| `rq1_fidelity.parquet` | 769 | 769 ok |
| `rq2_proof_cost.parquet` | 1 | 1 ok |
| `rq3_overhead.parquet` | 3 | 3 ok |
| `rq3_grinding.parquet` | 5 | 5 ok |
| `rq4_certificates.parquet` | 412 | 387 ok, 25 failed |
| `rq5_scaling.parquet` | 769 | 769 ok |
| `ablation.parquet` | 14 | 2 ok, 12 symbolic-only |

No duplicate `_hash` values were found. ARI, NMI, and assignment agreement were
within valid ranges; costs were nonnegative; and every stored `r_phi` matched
`phi_proj/phi_full`. These establish internal consistency, not full correctness.

The 25 failed RQ4 rows resulted from passing `gamma` to
`dbscan_margin_dataset`, which does not accept that parameter. They must remain
in the attempted-run denominator.

