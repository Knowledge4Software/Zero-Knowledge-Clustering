# Final cost-aware optimization and tests

The final implementation chooses the proof path before generating query proofs.
It caches fixed private geometry, simulates at most16 exact row corrections,
and selects compact proofs only for a conclusive, cheaper plan. A batch check
also charges loading the certificate/refinement proving keys and requires a10%
predicted saving. Otherwise the entire batch uses Full. Every request remains
covered by a valid public proof; native computations are only untrusted planning.

| Case | Verified policy | Measured warm response/proof total | Full reference |
|---|---|---:|---:|
| Fashion32,8 requests |8 certificate requests,4 correction proofs; final proof plan unchanged |3.942 min |9.061 min, same first paired execution |
| Text32,16 requests | Final policy sends all16 directly to Full, zero certificate/correction proofs |8.427 min, new final execution |8.049 min, earlier Full execution; not a new timing pair |

Fashion's measured paired speed ratio is 2.299x.
The image route and exact correction list were checked to be identical under the
final policy; the image proofs were not regenerated after that policy-only fix.
No speed advantage over Full is claimed on text: the final policy deliberately
uses the same Full relation, with 1.925 seconds of native
planning. Differences between separate text executions are not a speedup claim.

The first mixed text policy used3 compact and13 Full requests. It measured
8.221 minutes against
8.049 for Full, a
2.15% slowdown.
This negative result is preserved in `output/cost_aware_20261029.md`. The final
fix adds batch key-loading overhead and a batch saving margin. Its cost numbers
still come from the earlier Q8 pilot; its design was informed by the first-round
result, so this follow-up is engineering validation on inspected data.

Validation:10 native/coverage tests passed;15 real-proof substitutions were
rejected. All24 first-round responses and16 repeated final text responses had
correct publicly verified decisions. Every interval contained the exact change.
All64 final text query proofs were generated anew, all17 absolute costs were
independently recomputed, and reused keys/old evidence were SHA256 checked.
No private input/proving-key files are in the public exports. Previously written
verified output is ignored and replaced by fresh verification.

Times are warm phase-wall sums including query witness/proof generation, key
loading, native planning and public replay. Existing authenticated rows, setup
and applicable old-cost evidence are reused. Copy/export and independent oracle
checks are excluded. These are single exploratory executions, not replicated
confidence intervals. There is no newly measured cold time or break-even count.

The prediction representation and assignment agreement were not changed.
The speed fix does not improve classification accuracy. The routing rule can
also be applied to PCA; neither PCA nor the Halo2 ZKML primitive adapter was
retimed here. This result establishes no universal advantage over all baselines.
Path choices and refinement indices are observable outputs. No new privacy
guarantee is claimed. Paper files and circuit/binary definitions are unchanged.

Implementation: `code/commitjl/cost_aware.py`,
`code/validation/verify_cost_aware.py`, and
`code/campaigns/run_cost_aware_control.py`. Tests:
`code/tests/test_cost_aware.py` and
`code/validation/check_cost_aware_proofs.py`.
Evidence: `code/validation/outputs/cost_aware_20261029` and
`code/validation/outputs/cost_aware_v2_20261029`.
