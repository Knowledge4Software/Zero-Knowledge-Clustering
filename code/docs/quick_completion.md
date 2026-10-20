# Affordable paper completion campaign - 7 October 2026

Launcher: `campaigns/quick_completion.py`. Service: `commitjl-quick-20261007.service`.
Output root: `results/quick_completion_20261007`. Original campaigns remain intact.

- Add dense Rademacher, coordinate and public PCA proof baselines on the exact
  prior Gaussian-easy **32x256** row subset, k=5, m=16. All features retained.
  Each baseline must pass a new 10-seed x5-start integer-fidelity check before
  proving. Five measured proofs and one warmup. This is feasibility-subset
  evidence, not a new full-corpus proof or a substitute for every real baseline.
- Matched backend pilot: synthetic **8x16**, k=5, sparse m=8,q=2 and full.
  Exact input/centroids/matrix/assignments are reused for Groth16 and PLONK.
  Five proofs plus one warmup per arm. PLONK uses an unsafe test-only KZG SRS.
- An instrumented copy of the original prover exports proof, verifying key and
  public witness after its final proof. A fresh process deserializes and verifies
  those artifacts (one warmup, ten verifications). Linux ru_maxrss is converted
  from KiB to bytes; because it may include pre-exec launcher memory, the primary
  metric polls the new verifier address space's `/proc/PID/status` VmHWM after
  confirming the executable, at 1ms intervals. The largest observed high-water
  value includes verifier runtime/deserialization, excludes
  compile/setup/proving. The original circuit sources and prover binary remain
  unchanged; the instrumented binary has its own SHA-256 identity.
- Reanalyze original 30 paired ordering tasks into separate commitment, parameter
  digest, log, challenge, seed, expansion and matrix-hash timings with task-cluster
  bootstrap intervals. Perform 30 expected local policy rejections (budget,
  deadline, nonce replay). Audit JSON sizes are not network-wire measurements;
  per-stage circuit counts and production network/deployment costs remain absent.

Expected duration: **45-90 minutes**, conditional on eligible baselines and
backend timings. Queue deadlines sum to **2h35m**; service deadline **3h**.
Proof runs are serial, up to 16 Go threads; no full-size proof or new circuit
generator is launched. No automatic retry loops. Each task and phase has time,
RAM and disk guards, and checkpoint/log outputs. Exclusions are retained.

Manuscript scope is revised to the completed native 10x5 design, actual corpus
subsets and feasible proof domain. Missing empirical Pisot/in-circuit-generator
rows and unspecified embedding corpus are removed. Per-gadget cost is explicitly
symbolic, and unmeasured cross-hardware/full-size claims are omitted. Remaining
numeric placeholders require exporting the resulting tables; no planned
measurement should be described as completed before its result is available.
