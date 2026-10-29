# Response to the two 27 October reviews

The revision strengthens the evidence and presentation. It does not establish
that Neurocomputing will request only minor revision. The new numerical study
also exposes a difficult neural-text regime; that result is retained.

## Completed evidence

The fixed profile remains prediction rank 63, certificate rank 79, precision
14 bits and at most 16 authenticated exact-row corrections before paid Full.
There is no change to the mathematical guarantee or relaxed correctness test.

New public-benchmark experiments use official train/test partitions of UCI
smartphone activity recognition (HAR) and 20 Newsgroups. The latter uses a
frozen MiniLM transformer. Each corpus has two row-disjoint fitted
configurations. Only the past 500 training rows determine scaling, reference
centers and bases. Two new 250-row training minibatches propose updates at
fixed rates 0.25, 0.5 and 1.0. All six candidates per fit are retained.
The numerical study completed eight representation fits and 240 records
across catalog prefixes and diagnostic proposal families. These are not 240
independent learning trials.

Results below pool only the twelve minibatch requests per corpus at N=1024.
"Decisions" and "regressions" count certificates before complete-catalog
fallback. Agreement compares approximate assignments with exact quantized
Full; it is not downstream classification accuracy. Label coverage is the
percentage of assignments individually guaranteed by the certificate.

| Corpus / basis | Decisions | Regressions certified | Mean agreement | Worst agreement | Mean label coverage |
|---|---:|---:|---:|---:|---:|
| HAR / centroid | 11/12 | 3/4 | 99.691% | 99.023% | 92.871% |
| HAR / PCA | 11/12 | 3/4 | 99.642% | 98.926% | 91.829% |
| Neural text / centroid | 8/12 | 0/4 | 96.370% | 92.188% | 28.743% |
| Neural text / PCA | 8/12 | 0/4 | 95.793% | 91.992% | 22.176% |

All new numerical intervals contained the independently computed true change;
no false conclusive decisions occurred. This is numerical validation, not
proof evidence for those full 1024-row workloads. Their bounded proof controls
use smaller genuine catalog prefixes and are reported only when completed.

The earlier completed Fashion 1024 proof comparison remains the headline:
centroid 42.534 minutes, PCA 44.156, reusable exact Full 101.575, and
authenticated moments-only including newly generated fallback 134.413.
The Full/centroid ratio is 2.388; all three natural switching regressions
are certified. No universal 5x or universal 99% claim is made.

## Referee response matrix

| Concern | Action and current limit |
|---|---|
| Only related images and Lloyd updates | Added subject-separated sensor data and pretrained neural-text features with new-training-minibatch proposals. Neural text is an explicit failure regime. Public HAR is a surrogate for a sensitive sensor application, not a private deployment study. |
| No neural learning example | Frozen transformer features are measured. A further trained neural activity prototype-head diagnostic is queued with all seeds and outcomes retained; it uses the same HAR corpus, so it is not another independent dataset. Encoder execution and classification accuracy are outside the certificate. |
| Weak exact comparator | Added real PLONK exact Full, not only compilation. A real 16-proof pilot and a 448-proof workload passed fresh public replay. Runtime alone selected the N=64 workload. Clean repetitions, matched Groth16 batch-1 controls and larger feasible PLONK batches are queued. A Windows sleep contaminated repetition 1 timing; raw data remain, and a clean replacement is predeclared. No performance ranking is claimed from that interrupted timing. |
| Full optimization unclear | Main evaluation states reusable public-center setup, eight-row batching, matched workers and guards, cached keys and buffered binary I/O. Separate batch-1 controls will distinguish backend from batching. No claim against unimplemented GKR or matrix-product provers. |
| Fixed-reference motivation | Introduction specifies a release batch: several candidate rates/minibatches are audited against one deployed reference, then one is installed. Main evaluation includes cumulative cost and the measured break-even plot. Replacing the reference requires new group preprocessing and changes the economics. |
| Privacy / moment reconstruction | Abstract, introduction and limitations now target authorized aggregate disclosure. The concrete recovery attack remains explicit, including singleton recovery. No adaptive record non-reconstruction claim. Minimum group size or query spacing alone is not presented as a privacy proof. |
| Too many overlapping tables | One consolidated evaluation now presents numerical quality, completed timing pairs, exact controls, proof/verifier/memory resources, timing variation and rank/cap sensitivity. Historical numerical tables remain preserved. |
| Thin statistics | Main text adds sample SD and conditional log-ratio t intervals for three serial timing repetitions. These describe one fitted workload and host, not independent learning trials. |
| Missing proof size, verifier time and memory | Main resource table counts actual proof/key files, fresh public replay and peak prover RSS. It explicitly reports that compact proofs can require more verifier work than Full. |
| Rank/cap tuning | First declared HAR/text fits have rank 39/63/79 and cap 0/4/8/16/32 controls. HAR improves from 5/6 to 6/6 with four corrections; text stays 4/6 even with 32. Higher rank does not improve resolution in these cases. Neither 79 nor 16 is claimed optimal. |
| Incremental novelty | One precise paragraph distinguishes established clustering summaries from the verified full-catalog exact-change interval, authenticated old memberships and row correction mechanism. No superior learned-representation claim. |
| Fingerprint / references | Concrete BN254 session and repeated-session bounds added. Recent related works use pinned archived arXiv versions and identifiers. |
| Stale affine / PLONK / confirmation wording | Appendix online-rebuild statement and exact-baseline caption corrected; earlier two-fit controls distinguished from newer eight-split confirmation. |
| Author and public artifact readiness | Author declarations remain with the supervisor as requested. Everything remains private as explicitly requested; public accessibility is still an author decision and is not marked solved. |

## Remaining execution and reproduction

The current serial queue is `code/validation/outputs/referee_two_reviews_20261027/`.
Its `queue.json`, `followups/queue.json`, resource logs and completed-only export
receipt are authoritative. `output/two_review_campaign_status_20261027.json`
is a timestamped snapshot, not a promise that a job is currently live.
No Mobin access, 4096/16384 proof job, push or public release is scheduled.
Proof work has a shared 24-hour campaign budget and memory/disk guards.

From the repository root, set `PYTHONPATH` to its `code` directory, then run:

```text
python -m analysis.two_review_campaign_status
python -m analysis.update_optimized_evidence
python -m analysis.export_two_review_evidence
python -m pytest code/tests/test_two_review_evidence.py code/tests/test_referee_learning_workflows.py code/tests/test_optimized_evidence_export.py code/tests/test_online_affine_control.py -q
```

The four test files passed 14 tests during this revision. Export verifies
manifest hashes, independently recomputes each exact backend batch and checks
complete nonduplicated catalog coverage. The PLONK control uses explicitly
unsafe benchmark KZG parameters; its replay validates benchmark execution,
not production cryptographic soundness under those parameters.

The automatic finalizer exports completed evidence when the queue ends and
builds a separate `*_completed_candidate.pdf`. It preserves the manually
reviewed paper and refuses to update changed authored text. Every automatic
candidate still requires visual review before delivery.
