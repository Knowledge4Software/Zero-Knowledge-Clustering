"""Verify the final batch-overhead fix without hiding the first timing loss."""
from pathlib import Path
import argparse
import json

from campaigns.published_baselines_laptop import REPO, read, save, sha
from campaigns.cost_aware_pilot import ROOT as FIRST, BASIC, calibrate
from commitjl.cost_aware import Planner, exact_distances

FINAL=REPO/'code/validation/outputs/cost_aware_v2_20261029'


def report(root=FINAL):
    assert read(root/'checks.json')['complete']
    for p,digest in read(root/'source_pin.json').items():
        assert sha(REPO/p)==digest, p
    assert all(sha(REPO/p)==digest for p,digest in read(root/'preservation.json').items())
    out=root/'text_n32';public=out/'routed/public'
    final=read(out/'checked.json');plan=read(out/'planner.json')
    assert plan['certificate']==[] and plan['full']==list(range(16))
    assert not (public/'certificate').exists() and not (public/'refinement').exists()
    assert not final['certificate_requests'] and not final['selected_correction_proofs']
    task=read(out/'approved.json');oldtask=read(FIRST/'text_n32/approved.json')
    assert task==oldtask
    assert read(out/'owner_pin.json')==read(FIRST/'text_n32/owner_pin.json')
    private=out/'routed/routed_full/private'
    full=read(private/'full_reuse_8.json')
    source=BASIC/'matched/text_n32/ordered_head/full/bundle'
    # All64 changing-request proofs are fresh; the four old-reference proofs
    # and cached setup are exactly the declared reused evidence.
    fresh=0;previous=FIRST/'text_n32/baseline_full/private'
    for p in (private/'full_reuse').glob('*.proof'):
        if p.name.startswith('s000_'):
            assert sha(p)==sha(source/'full_reuse'/p.name)
        else:
            fresh+=1;assert sha(p)!=sha(previous/'full_reuse'/p.name)
    assert fresh==64
    assert all(sha(p)==sha(source/'keys'/p.name) for p in (private/'keys').glob('*.vk'))
    for manifest in public.rglob('public_manifest.json'):
        assert all(sha(manifest.parent/p)==digest for p,digest in read(manifest)['sha256'].items())
    costs=[0]*17
    for sample in full['samples']:costs[sample.get('centre_set',0)]+=int(sample['psi'])
    rows=read(BASIC/'matched/text_n32/ordered_head/full/input.json')['x']
    expected=[int(exact_distances(rows,c).min(1).sum()) for c in [task['comparison_base']]+task['reuse_centers']]
    assert costs==expected
    verified=read(public/'verified.json')
    assert len(verified['decisions'])==16
    for i,decision in enumerate(verified['decisions']):
        assert decision['path']=='full' and int(decision['lower'])==int(decision['upper'])==expected[i+1]-expected[0]
    # Check the final policy preserves every image route/correction already
    # actually timed. This is a route-regression check, not a second image run.
    fashion_source=BASIC/'matched/fashion_n32'
    fin=read(fashion_source/'ordered_head_base/input.json')
    ft=read(FIRST/'fashion_n32/approved.json')
    fc, calibration=calibrate(fashion_source)
    fplan=Planner(fin['model'],fin['x'],fc).batch(ft['reuse_centers'])
    original=read(FIRST/'fashion_n32/planner.json')
    assert fplan['certificate']==original['certificate'] and fplan['full']==original['full']
    assert [r['refinement_indices'] for r in fplan['records']]==[r['refinement_indices'] for r in original['records']]
    save(root/'fashion_route_regression.json',dict(complete=True,identical_proof_plan=True,
         newly_timed_image_run=False,calibration_sources_sha256=calibration,plan=fplan))
    fashion=read(FIRST/'fashion_n32/checked.json');text1=read(FIRST/'text_n32/checked.json')
    result=dict(complete=True,unit_tests=10,real_proof_tamper_checks=15,
         final_text_warm_seconds=final['measured_warm_routed_seconds'],
         earlier_paired_text_full_seconds=text1['measured_warm_full_seconds'],
         final_text_planner_seconds=final['planner_seconds'],final_text_certificate_requests=0,
         final_text_full_requests=16,final_text_correction_proofs=0,fresh_text_request_proofs=64,
         final_text_all_costs_checked=True,final_text_all_decisions_correct=True,
         final_fashion_proof_plan_unchanged=True,earlier_measured_fashion_seconds=fashion['measured_warm_routed_seconds'],
         earlier_paired_fashion_full_seconds=fashion['measured_warm_full_seconds'],
         earlier_fashion_full_over_routed=fashion['full_over_routed'],
         first_text_mixed_route_seconds=text1['measured_warm_routed_seconds'],
         first_text_mixed_route_slower_percent=100*(text1['measured_warm_routed_seconds']/text1['measured_warm_full_seconds']-1),
         reused_evidence_and_fresh_proofs_checked=True,paper_unchanged=True,
         no_new_cold_measurement=True,no_new_pca_or_halo2_timing=True,
         text_final_vs_earlier_full_not_a_new_paired_comparison=True)
    save(root/'final_evidence_checks.json',result);save(REPO/'output/cost_aware_final_20261029.json',result)
    text=f"""# Final cost-aware optimization and tests

The final implementation chooses the proof path before generating query proofs.
It caches fixed private geometry, simulates at most16 exact row corrections,
and selects compact proofs only for a conclusive, cheaper plan. A batch check
also charges loading the certificate/refinement proving keys and requires a10%
predicted saving. Otherwise the entire batch uses Full. Every request remains
covered by a valid public proof; native computations are only untrusted planning.

| Case | Verified policy | Measured warm response/proof total | Full reference |
|---|---|---:|---:|
| Fashion32,8 requests |8 certificate requests,4 correction proofs; final proof plan unchanged |{fashion['measured_warm_routed_seconds']/60:.3f} min |{fashion['measured_warm_full_seconds']/60:.3f} min, same first paired execution |
| Text32,16 requests | Final policy sends all16 directly to Full, zero certificate/correction proofs |{final['measured_warm_routed_seconds']/60:.3f} min, new final execution |{text1['measured_warm_full_seconds']/60:.3f} min, earlier Full execution; not a new timing pair |

Fashion's measured paired speed ratio is {fashion['full_over_routed']:.3f}x.
The image route and exact correction list were checked to be identical under the
final policy; the image proofs were not regenerated after that policy-only fix.
No speed advantage over Full is claimed on text: the final policy deliberately
uses the same Full relation, with {final['planner_seconds']:.3f} seconds of native
planning. Differences between separate text executions are not a speedup claim.

The first mixed text policy used3 compact and13 Full requests. It measured
{text1['measured_warm_routed_seconds']/60:.3f} minutes against
{text1['measured_warm_full_seconds']/60:.3f} for Full, a
{100*(text1['measured_warm_routed_seconds']/text1['measured_warm_full_seconds']-1):.2f}% slowdown.
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
"""
    (REPO/'output/cost_aware_final_20261029.md').write_text(text,encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=FINAL)
    report(p.parse_args().root.resolve())
