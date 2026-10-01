# CommitJL formula review

Mathematical validity and implementation enforcement are assessed separately.
A correct manuscript formula does not imply that the current circuit constrains
every required condition.

| Topic | Assessment | Evidence and limitation |
|---|---|---|
| Sparse JL structure | Correct | Every tested column contains exactly `q` distinct signed entries; the mathematical map is `P=A/sqrt(q)`. |
| Unbiased squared norm | Correct | Independent signs remove cross terms, giving `E[||Av||^2/q]=||v||^2`. A 2,000-trial check was consistent with the identity. |
| Fixed-center cost lemma | Correct, conditional | If the simultaneous JL event holds for every `x_i-c_j`, taking a minimum and summing preserves the bounds. |
| Nearest-centroid margin | Correct, sufficient | `(d2^2/d1^2) > (1+eps)/(1-eps)` is sufficient for a unique nearest center, but is not necessary. |
| DBSCAN ambiguity band | Correct, conditional | Exact output also requires identical graph, vertex order, queue order, and border-point tie rules. |
| Quantization error | Correct | The bounds follow from coordinate rounding and Cauchy-Schwarz. Independent numerical checks remained inside them. |
| No-wraparound bound | Formula correct; circuit incomplete | `2dB`, `D_max=4md^2B^2`, and `Psi_max=nD_max` are conservative. The circuit does not enforce every required bound. |
| Symbolic circuit cost | Does not match current circuit | The model counts about `nqd` projection contributions. The circuit constructs selectors over `m*d*q` combinations and also projects centroids. |
| RQ4 cost certificate | Incorrect for `q>1` | The experiment uses `A/sqrt(q)` and then divides the squared cost by `q` again. |

The soundness-fidelity composition theorem has a valid logical structure but is
conditional on commitment binding, collision resistance, knowledge soundness,
identical canonical encodings, signed-range enforcement, and the geometric
embedding event. The protocol and ZK bridge now share field-native encodings for
`C_theta` and `C_A`, and the transcript-to-proof chain has an integration test.
Circuit range enforcement remains incomplete, so the theorem still cannot be
applied fully to unrestricted inputs.

The unbiasedness proposition is only a first-moment result and supplies no JL
tail bound. The manuscript states this limitation correctly.

The task-level `Q` union bound is valid if an authenticated deployment actually
enforces the eligibility log and attempt budget. The current Python `TaskLog` is
a local reference model, not a publicly auditable production log.

