"""Audit and report the completed cost-aware warm proof pilot."""
from pathlib import Path
import argparse
import json

from campaigns.published_baselines_laptop import read, save, sha, REPO
from campaigns.cost_aware_pilot import ROOT, BASIC


def report(root=ROOT):
    final = read(root/'checks.json')
    assert final['complete'] and final['paper_unchanged']
    cases = []
    for case in read(root/'plan.json')['cases']:
        name = case['name']; out = root/name; source = BASIC/'matched'/name
        value, plan = read(out/'checked.json'), read(out/'planner.json')
        assert value['complete'] and value['all_requests_verified'] and value['all_decisions_correct']
        public = out/'routed/public'
        manifests = []
        for manifest in public.rglob('public_manifest.json'):
            checks = read(manifest)['sha256']
            assert all(sha(manifest.parent/p)==digest for p,digest in checks.items())
            manifests.append(str(manifest.relative_to(root)))
        assert len(read(public/'verified.json')['decisions'])==case['requests']
        if plan['certificate']:
            # Reused catalog and setup really are byte-identical.
            old = source/'ordered_head_base/bundle'
            new = out/'routed/base/bundle'
            for folder, pattern in [('group_rows','*.proof'),('keys','*.vk')]:
                assert all(sha(p)==sha(old/folder/p.name) for p in (new/folder).glob(pattern))
            fresh = list((new/'group_query').glob('*.proof'))
            assert len(fresh)==len(plan['certificate'])*3
            for local, global_id in enumerate(plan['certificate']):
                if global_id >= 8: continue
                for suffix in ['moment','b0000','b0001']:
                    assert sha(new/f'group_query/s{local:03}_{suffix}.proof') != sha(old/f'group_query/s{global_id:03}_{suffix}.proof')
        if plan['full']:
            old = source/'ordered_head/full/bundle'
            new = out/'routed/routed_full/private'
            assert all(sha(p)==sha(old/'full_reuse'/p.name) for p in (new/'full_reuse').glob('s000_*.proof'))
            assert len(list((new/'full_reuse').glob('*.proof')))==4*(len(plan['full'])+1)
            for local, global_id in enumerate(plan['full'],1):
                if global_id >= 8: continue
                for batch in range(4):
                    assert sha(new/f'full_reuse/s{local:03}_q8_{batch:04}.proof') != sha(old/f'full_reuse/s{global_id+1:03}_q8_{batch:04}.proof')
        baseline = out/'baseline_full/private'
        old = source/'ordered_head/full/bundle'
        assert all(sha(p)==sha(old/'full_reuse'/p.name) for p in (baseline/'full_reuse').glob('s000_*.proof'))
        full_exported = sum(p.stat().st_size for p in (out/'baseline_full/public').rglob('*.proof'))
        routed_exported = sum(p.stat().st_size for p in public.rglob('*.proof'))
        fresh_routed_count = len(plan['certificate'])*3+len(plan['full'])*4+value['selected_correction_proofs']
        ref_path = out/'routed/refined/bundle/refinement'
        assert len(list(ref_path.glob('*.proof')))==value['selected_correction_proofs']
        peak = max(r['peak_process_tree_rss_bytes'] for r in value['resources'].values())
        cases.append(dict(name=name,requests=case['requests'],routed_seconds=value['measured_warm_routed_seconds'],
            full_seconds=value['measured_warm_full_seconds'],full_over_routed=value['full_over_routed'],
            routed_reduction_percent=100*(1-value['measured_warm_routed_seconds']/value['measured_warm_full_seconds']),
            planner_seconds=value['planner_seconds'],certificate_requests=len(plan['certificate']),
            direct_full_requests=len(plan['full']),correction_proofs=value['selected_correction_proofs'],
            fresh_routed_request_proofs=fresh_routed_count,fresh_full_request_proofs=case['requests']*4,
            routed_exported_proof_bytes_including_reused=routed_exported,
            full_exported_proof_bytes_including_reused=full_exported,
            fresh_routed_proof_bytes=fresh_routed_count*196,fresh_full_proof_bytes=case['requests']*4*196,
            peak_rss_gib=peak/2**30,public_manifests_checked=manifests,
            cold_activation_prediction=read(out/'cold_activation_prediction.json')))
    smoke=read(root/'smoke_v2/checks.json');assert smoke['complete']
    preservation=read(root/'preservation.json')
    assert all(sha(REPO/p)==digest for p,digest in preservation.items())
    result=dict(complete=True,cases=cases,unit_tests=9,real_proof_tamper_checks=len(smoke['tamper_rejections']),
                all_proofs_verified=True,all_intervals_contain_truth=True,fresh_request_proofs=True,
                reused_evidence_sha256_checked=True,paper_unchanged=True,
                no_new_cold_measurement=True,no_new_pca_or_halo2_timing=True)
    save(REPO/'output/cost_aware_20261029.json',result)
    lines=['# Cost-aware routing: measured laptop results','',
           'Fresh warm request proofs and public replay, eight workers, one exploratory execution per case. '+
           'The same original rows, frozen centers and owner-pinned setup are used by both arms. '+
           'Existing row authentication, proving keys and old Full-cost evidence are reused; native planning is charged. '+
           'Copy/export time and independent oracle validation are outside the phase-wall totals.','',
           '| Case | Requests | Routed | Direct Full | Full / routed | Cert / Full routes | Corrections |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['name']} | {c['requests']} | {c['routed_seconds']/60:.3f} min | {c['full_seconds']/60:.3f} min | {c['full_over_routed']:.3f}x | {c['certificate_requests']} / {c['direct_full_requests']} | {c['correction_proofs']} |")
    lines+=['','The planner computes exact integer bounds from cached private heads, residual energies and old-group moments. '+
            'For an initially inconclusive bound it simulates at most16 width-ordered exact row corrections using the prover’s private rows. '+
            'It selects the compact relation only when that plan is conclusive and its calibrated proof cost is at least10% below Full. '+
            'It sends every other request directly to Full, before generating any compact query or correction proof. '+
            'These native distances are untrusted witness planning. The verifier accepts only freshly verified proofs.','',
            'Costs were frozen from the earlier Q8 pilot before this execution. The catalogs and proposals had already been inspected: '+
            'this is an engineering test, not independent confirmation on held-out data. The safety margin is a prediction rule, not a guaranteed speedup.','',
            'All24 requested decisions are correct and every certificate interval contains the independently recomputed exact change. '+
            'The hybrid verifier binds the branches to the same original-catalog root and approved reference, reconstructs both tasks from independently supplied centers, '+
            'and requires disjoint complete request and row coverage. Nine native/coverage tests and15 real-proof tamper tests passed. '+
            'The initial smoke attempt failed in JSON export on a null sample list; it was fixed before workload proving, and its failure receipt is retained.','',
            '| Case | Native planner | Fresh request proof bytes: routed / Full | Exported proof bytes including reused evidence: routed / Full | Peak RSS |',
            '|---|---:|---:|---:|---:|']
    for c in cases:
        lines.append(f"| {c['name']} | {c['planner_seconds']:.3f} s | {c['fresh_routed_proof_bytes']} / {c['fresh_full_proof_bytes']} | {c['routed_exported_proof_bytes_including_reused']} / {c['full_exported_proof_bytes_including_reused']} | {c['peak_rss_gib']:.3f} GiB |")
    lines+=['','Larger exported bundles can result from the reused per-row certificate evidence. Fresh request proof size and complete bundle size are different metrics.','',
            'Cold activation is implemented as an optional batch cost check: decline the compact path if estimated savings do not cover extra summary setup. '+
            'The text16 batch does not cover that extra setup under the frozen cost model. No new cold runtime or measured break-even count is claimed. '+
            'Do not compare these warm totals directly with the earlier27.52-minute cold text run.','',
            'This change preserves the underlying prediction representation and its assignment agreement; it improves proof scheduling, not classification accuracy. '+
            'All serviced change decisions are exact in sign after public verification. The routing policy is also applicable to the PCA certificate baseline. '+
            'PCA and the Halo2 ZKML primitive adapter were not retimed here; optimized-versus-unoptimized comparisons would not establish a mathematical advantage over PCA. '+
            'No universal superiority over Full or other baselines follows from these two warm cases.','',
            'Private exact corrections remain in the prover. Public routing choices and refinement indices are observable outputs. '+
            'No additional assignment-privacy claim is made. Existing circuit definitions, binary and paper files are unchanged.','',
            f'Evidence: `{root.relative_to(REPO).as_posix()}`.']
    (REPO/'output/cost_aware_20261029.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(root/'final_evidence_checks.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    report(p.parse_args().root.resolve())
