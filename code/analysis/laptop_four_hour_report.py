"""Report measured tradeoffs and failures from the bounded local RQ campaign."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from campaigns.laptop_four_hour_research import ROOT,BASIC
from analysis.published_baseline_report import report,read,sha
from analysis.laptop_advantage_pilot_report import main as basic_report,proof_footprint


def main(root):
    snapshot=read(BASIC/'authorization_and_scope.json')
    assert all(sha(REPO/name)==value for name,value in snapshot['paper_sha256'].items())
    assert sha(root/'plan.json')==read(root/'plan_pin.json')['plan_sha256']
    measurements=[]
    if read(BASIC/'matched_queue.json').get('complete'):
        basic_report(BASIC)
        base=read(REPO/'output/laptop_advantage_pilot_20261028.json')
        for item in base['records']:
            measurements.append(dict(item,experiment='basic_cold',seconds=item['cold_seconds_including_fitting']))
    scaled=root/'scaled'
    if (scaled/'matched').exists():
        audited=report(scaled,output_stem='laptop_four_hour_scaled_20261028',emit=False,expected_case_count=1,
            limitations=['Exploratory Fashion scale test chosen after native screen; first4 fixed proposals, N128, one timing execution.',
                         'Same frozen fitted models for all controls; basis-fitting excluded equally from this service-scale comparison.',
                         'All attempted phase work and failures preserved; incomplete comparator phases are not speed evidence.',
                         'Halo2 primitive adapter uses a different commitment/backend and exact-cost output.'])
        for item in audited['records']:
            target=scaled/'matched'/item['case']
            if item['method']=='Exact Full / Groth16':directories=[target/'ordered_head/full/public_bundle']
            elif item['method']=='ZKML primitive adapter / Halo2-KZG':directories=[target/'zkml_public']
            else:
                method='ordered_head' if item['method']=='Centroid certificate' else 'pca'
                directories=[target/(method+'_base/public'),target/(method+'_refined/public'),
                             target/(method+'_refined/fallback/full/public_bundle')]
            measurements.append(dict(item,experiment='scaled_cold',**proof_footprint(directories)))
    warm_path=root/'warm64/checked_comparison.json'
    warm=read(warm_path) if warm_path.exists() else None
    comparisons=[]
    for experiment,case in sorted({(r['experiment'],r['case']) for r in measurements}):
        group=[r for r in measurements if r['experiment']==experiment and r['case']==case]
        ours=next((r for r in group if r['method']=='Centroid certificate'),None)
        if ours is None:continue
        for baseline in group:
            if baseline is ours:continue
            value=dict(experiment=experiment,case=case,baseline=baseline['method'],
                baseline_over_our_time=baseline['seconds']/ours['seconds'],
                baseline_over_our_proof_bytes=baseline['public_proof_bytes']/ours['public_proof_bytes'],
                baseline_over_our_peak_memory=baseline['peak_rss_bytes']/ours['peak_rss_bytes'],
                our_minus_baseline_assignment_agreement_pp=ours['agreement_percent']-baseline['agreement_percent'],
                our_minus_baseline_conclusive=ours['coverage']-baseline['coverage'],
                timing_repetitions=1,statistical_superiority_established=False)
            if 'Halo2' in baseline['method']:
                value['attribution']='Different proof backend and commitment. Proof-size advantage is a system tradeoff and cannot establish superiority of the certificate mathematics.'
            elif baseline['method']=='PCA certificate':
                value['attribution']='Same certificate backend, prediction/certificate ranks and precision; isolates representation choice on this fit.'
            else:value['attribution']='Same Groth16 backend; exact-cost output is stronger than the certificate sign decision.'
            comparisons.append(value)
    verdict=[]
    for name in ['Exact Full / Groth16','PCA certificate','ZKML primitive adapter / Halo2-KZG']:
        rows=[r for r in comparisons if r['baseline']==name]
        advantages=[];losses=[]
        for r in rows:
            label=f"{r['experiment']}:{r['case']}"
            if r['baseline_over_our_time']>=1.05:advantages.append(dict(case=label,metric='observed total time',ratio=r['baseline_over_our_time']))
            elif r['baseline_over_our_time']<=1/1.05:losses.append(dict(case=label,metric='observed total time',ratio=r['baseline_over_our_time']))
            if r['baseline_over_our_proof_bytes']>1:advantages.append(dict(case=label,metric='exported proof bytes',ratio=r['baseline_over_our_proof_bytes']))
            elif r['baseline_over_our_proof_bytes']<1:losses.append(dict(case=label,metric='exported proof bytes',ratio=r['baseline_over_our_proof_bytes']))
            if r['baseline_over_our_peak_memory']>=1.1:advantages.append(dict(case=label,metric='observed process-tree peak RAM',ratio=r['baseline_over_our_peak_memory']))
            if r['our_minus_baseline_assignment_agreement_pp']>0:advantages.append(dict(case=label,metric='assignment agreement pp',difference=r['our_minus_baseline_assignment_agreement_pp']))
            elif r['our_minus_baseline_assignment_agreement_pp']<0:losses.append(dict(case=label,metric='assignment agreement pp',difference=r['our_minus_baseline_assignment_agreement_pp']))
        if warm and 'Halo2' in name:
            entry=dict(case='warm64:fashion_n32',metric='measured warm time',ratio=warm['exact_over_certificate'])
            if warm['exact_over_certificate']>=1.05:advantages.append(entry)
            elif warm['exact_over_certificate']<=1/1.05:losses.append(entry)
        verdict.append(dict(baseline=name,observed_advantages=advantages,observed_losses=losses,
                            universal_superiority=False,
                            interpretation='One exploratory run per workload; repeat before making a statistical runtime claim. Report all measured tradeoffs.'))
    ablation_path=root/'native_ablations.json'
    ablations=read(ablation_path) if ablation_path.exists() else None
    grouped=[]
    if ablations:
        records=ablations['records']
        keys=sorted({(r['dataset'],r['method'],r['n'],r['certificate_rank'],r['cap']) for r in records})
        for dataset,method,n,rank,cap in keys:
            selected=[r for r in records if (r['dataset'],r['method'],r['n'],r['certificate_rank'],r['cap'])==(dataset,method,n,rank,cap)]
            grouped.append(dict(dataset=dataset,method=method,n=n,rank=rank,cap=cap,requests=len(selected),
                conclusive=sum(r['decision']!='inconclusive' for r in selected),exact_rows=sum(r['exact_rows'] for r in selected)))
    native_pca_comparisons=[]
    pca_verdict=next(v for v in verdict if v['baseline']=='PCA certificate')
    for ours in grouped:
        if ours['method']!='ordered_head':continue
        baseline=next(v for v in grouped if (v['dataset'],v['method'],v['n'],v['rank'],v['cap'])==
                      (ours['dataset'],'pca',ours['n'],ours['rank'],ours['cap']))
        native_pca_comparisons.append(dict(dataset=ours['dataset'],n=ours['n'],rank=ours['rank'],cap=ours['cap'],
            requests=ours['requests'],our_conclusive=ours['conclusive'],pca_conclusive=baseline['conclusive'],
            our_correction_rows=ours['exact_rows'],pca_correction_rows=baseline['exact_rows'],native_only=True))
        difference=100*(ours['conclusive']-baseline['conclusive'])/ours['requests']
        if difference:
            field='observed_advantages' if difference>0 else 'observed_losses'
            pca_verdict[field].append(dict(case=f"native_ablation:{ours['dataset']}_n{ours['n']}_r{ours['rank']}_cap{ours['cap']}",
                metric='native-only conclusive coverage pp',difference=difference,native_only=True,
                qualification='Exact integer bounds checked; no low-rank cryptographic proof timing measured. Prediction remains rank63.'))
    synthetic_path=REPO/'code/validation/outputs/advantage_design_20261028/native_control.json'
    synthetic=read(synthetic_path)['synthetic_summary'] if synthetic_path.exists() else []
    native_screen=read(BASIC/'native_checks.json')['summary']
    manifest={}
    sources=[Path(__file__),REPO/'code/campaigns/laptop_advantage_pilot.py',
             REPO/'code/campaigns/laptop_four_hour_research.py',REPO/'code/campaigns/laptop_warm64_control.py',
             REPO/'code/campaigns/run_group_refinement_control.py',REPO/'code/analysis/published_baseline_report.py',
             REPO/'code/analysis/laptop_advantage_pilot_report.py',root/'plan.json',
             root/'queue_order_amendment.json',REPO/'code/tools/laptop_deadline_awake.py',
             REPO/'code/tools/laptop_host_memory_guard.py']
    for path in sources:manifest[path.relative_to(REPO).as_posix()]=sha(path)
    result=dict(report_complete=True,paper_unchanged=True,plan_sha256=sha(root/'plan.json'),
                queue=read(root/'queue.json'),measurements=measurements,comparisons=comparisons,
                baseline_verdicts=verdict,warm64=warm,native_ablation_summary=grouped,
                native_screen_summary=native_screen,
                native_pca_comparisons=native_pca_comparisons,
                queue_order_amendment=read(root/'queue_order_amendment.json'),
                host_memory_guard=(read(root/'host_memory_guard.json') if (root/'host_memory_guard.json').exists() else None),
                scaled_status=(read(scaled/'matched_queue.json') if (scaled/'matched_queue.json').exists() else None),
                hardware=json.loads((root/'host_hardware.json').read_text(encoding='utf-8-sig')),
                earlier_synthetic_control=synthetic,source_sha256=manifest,
                all_implemented_baselines_beaten_on_every_metric=False,
                not_implemented=['ezDPS same-task adaptation','Zhang2021 same-task adaptation'],
                claim_boundary='Exploratory evidence about specified implementations and workloads. Different outputs/backend choices, setup reuse, synthetic cases and numerical-only studies remain explicit. No proof of universal or statistical superiority.')
    out=REPO/'output/laptop_four_hour_research_20261028.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Four-hour local RQ study — 28 October 2026','',result['claim_boundary'],'',
           '| Experiment | Case | Method | Minutes | Proof KiB | Peak RAM GiB | Assignment agreement % | Before fallback |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in measurements:
        lines.append(f"| {r['experiment']} | {r['case']} | {r['method']} | {r['seconds']/60:.3f} | {r['public_proof_bytes']/1024:.3f} | {r['peak_rss_bytes']/2**30:.3f} | {r['agreement_percent']:.3f} | {r['coverage']}/{r['requests']} |")
    lines+=['','## Warm64','']
    if warm:
        lines += [f"Actual warm certificate: {warm['measured_warm_certificate_seconds']/60:.3f}min; exact Halo2: {warm['measured_warm_exact_seconds']/60:.3f}min; exact/certificate: {warm['exact_over_certificate']:.3f}x.",
                  f"All64 request proofs were freshly generated for both systems; data authentication and setup reused for both. Certificate fallback updates: {warm['fallback_updates']}."]
    else:lines+=['No completed matched warm64 result. Partial or unadmitted work is not speedup evidence.']
    if (root/'host_memory_guard.json').exists() and read(root/'host_memory_guard.json').get('stopped_for_pressure'):
        lines+=['','The host Windows OS issued a low-memory notification during the N128 PCA run. The guard stopped the run; PCA N128 is incomplete and is not counted as a win or a complete timing. The completed N128 centroid, Full and Halo2 runs remain independently checked.']
    lines+=['','## Per-baseline findings','']
    for v in verdict:
        lines += ['### '+v['baseline'],'']
        for kind in ['observed_advantages','observed_losses']:
            lines.append(kind.replace('_',' ').capitalize()+':')
            if not v[kind]:lines.append('- None above the reporting thresholds.')
            for x in v[kind]:
                number=f"{x['ratio']:.3f}x" if 'ratio' in x else f"{x['difference']:+.3f}pp"
                lines.append(f"- {x['case']}: {x['metric']} ({number}).")
            lines.append('')
    lines+=['Proof bytes exclude keys, parameters and public task/witness files. A smaller Groth16 proof footprint versus Halo2 is a backend-level system tradeoff, not isolated evidence for the mathematical method. Timings are single executions.','',
            '## Real-data native screen (not proof timings)','',
            '| Dataset | Basis | N | Requests | Conclusive after bounded correction | Assignment agreement % |',
            '|---|---|---:|---:|---:|---:|']
    for r in native_screen:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['n']} | {r['requests']} | {r['conclusive']}/{r['requests']} | {r['agreement_percent']:.3f} |")
    lines+=['',
            '## Certificate-rank ablation (native only)','',
            '| Dataset | Basis | N | Certificate rank | Correction cap | Conclusive | Exact rows |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in grouped:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['n']} | {r['rank']} | {r['cap']} | {r['conclusive']}/{r['requests']} | {r['exact_rows']} |")
    lines+=['','The prediction model remains rank63. These are checked numerical bounds, not proof timing results.','',
            'At certificate rank4 and cap16, Fashion N128 has6/8 conclusive decisions for the centroid basis versus1/8 for PCA. This is a numerical representation/coverage result, not a measured low-rank ZK runtime. The default high-rank profile remains close to PCA.','',
            '## Boundaries','',
            '- Native/synthetic controls are separated from real proof measurements.',
            '- The Fashion N128 choice followed an exploratory native screen; it is not independent confirmation.',
            '- ezDPS and Zhang2021 were not implemented as matched systems and have no claimed runtime.',
            '- The paper is unchanged. No cloud server was purchased or accessed.',
            '- All unfinished or resource-stopped cells remain in the queue/receipts and are not counted as wins.']
    (REPO/'output/laptop_four_hour_research_20261028.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(out),measurements=len(measurements),baseline_verdicts=verdict,warm64_complete=warm is not None),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();main(args.root.resolve())
