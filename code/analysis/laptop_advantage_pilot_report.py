"""Independent final audit and conservative online-cost gate for the pilot."""
from pathlib import Path
import argparse
import json
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'code'))
from analysis.published_baseline_report import report,read,sha,resource
from campaigns.laptop_advantage_pilot import DEFAULT


def proof_footprint(directories):
    files=[p for directory in directories if directory.exists() for p in directory.rglob('*')
           if p.is_file() and (p.suffix=='.proof' or p.name=='proof')]
    return dict(public_proof_bytes=sum(p.stat().st_size for p in files),public_proof_files=len(files),
                proof_size_scope='Actual exported proof bytes; parameters, keys, public witness/task files reported separately from proof size.')


def main(root):
    pin=read(root/'plan_pin.json')
    assert sha(root/'plan.json')==pin['plan_sha256']
    assert read(root/'matched_queue.json')['complete']
    stem='laptop_advantage_pilot_20261028'
    result=report(root,output_stem=stem+'_audit',emit=False)
    native=read(root/'native_checks.json')
    assert native['complete'] and native['all_intervals_contain_truth'] and native['no_wrong_decisions']
    online=[]
    for row in result['records']:
        case=root/'matched'/row['case']
        common=read(case/'source_pin.json')['common_center_proposal_fitting_seconds']
        row['common_input_fitting_seconds']=common
        extra=0.
        if row['method'] in ['Centroid certificate','PCA certificate']:
            method='ordered_head' if row['method']=='Centroid certificate' else 'pca'
            protocol=read(case/method/'protocol.json')
            extra=protocol['prediction_fitting_seconds']+protocol['auxiliary_fitting_seconds']
            row.update(proof_footprint([case/(method+'_base/public'),case/(method+'_refined/public'),
                                       case/(method+'_refined/fallback/full/public_bundle')]))
        elif row['method']=='Exact Full / Groth16':
            row.update(proof_footprint([case/'ordered_head/full/public_bundle']))
        else:row.update(proof_footprint([case/'zkml_public']))
        row['method_specific_basis_fitting_seconds']=extra
        row['cold_seconds_including_fitting']=row['seconds']+common+extra
    for name in ['fashion_n32','text_n32']:
        case=root/'matched'/name
        count=read(case/'zkml_checked.json')['proofs']-1
        # Charge shared query loading to the online stage. Remove only the actual
        # old-bank work; fresh replay of each new proof is included explicitly.
        rust_queries=resource(root,name+'_zkml_queries')['wall_seconds']
        old=read(case/'zkml_private/bank_000/result.json')['total_seconds']
        verifies=sum(resource(root,name+f'_zkml_verify{i:03}')['wall_seconds'] for i in range(1,count+1))
        exact_online=rust_queries-old+verifies
        for method in ['ordered_head','pca']:
            base=read(case/(method+'_base/status.json'))['resources']
            checked=read(case/(method+'_refined/checked_comparison.json'))
            certificate_online=(base['group_query']['wall_seconds']+base['group_verify']['wall_seconds']
                                +checked['refinement_active_seconds']+checked['fallback_active_seconds'])
            online.append(dict(case=name,method=method,requests=count,
                measured_certificate_online_component_seconds=certificate_online,
                measured_exact_online_component_seconds=exact_online,
                exact_over_certificate_online=exact_online/certificate_online,
                fallback_updates=checked['fallback_updates'],
                qualification='Observed phase-component accounting for this8-request workload, not a new warm trial. Certificate charges all base replay and lazy correction/fallback setup; exact charges shared loading and all new-bank public replay. No asymptotic forecast.'))
    result.update(plan_sha256=pin['plan_sha256'],native_summary=native['summary'],online_diagnostic=online,
                  limitations=[
        'New seed521, existing public data sources; exploratory, one timing execution, N32, eight chronological requests per dataset.',
        'All first8 requests retained; only identical public centers cached equally. The64-request native screen did not select proof candidates.',
        'Agreement measures exact quantized assignments, not classification accuracy. All interval endpoints independently checked.',
        'ZKML primitive adapter, not original compiler; different commitment and proof backend. Exact-cost output is stronger than sign-only output.',
        'No server purchase or remote access, and no manuscript edits.',
    ])
    out=REPO/'output'/f'{stem}.json';out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    lines=['# Basic laptop gate before server rental — 28 October 2026','',
        'New fit seed521; same32 original integer rows and first8 proposed updates per dataset; actual proofs, fresh setup, standalone replay, bounded refinement and new Full fallback. Frozen published adapter binaries reused, historical workload times not reused. One exploratory timing execution.','',
        '| Dataset | Method | Cold minutes incl. fitting | Assignment agreement % | Conclusive before Full | Fallback | Proof KiB | Peak RSS GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['records']:
        lines.append(f"| {r['case']} | {r['method']} | {r['cold_seconds_including_fitting']/60:.3f} | {r['agreement_percent']:.3f} | {r['coverage']}/{r['requests']} | {r['fallback']} | {r['public_proof_bytes']/1024:.3f} | {r['peak_rss_bytes']/2**30:.3f} |")
    lines+=['','## Reuse diagnostic','',
            '| Dataset | Certificate basis | Observed online component seconds | Exact online component seconds | Exact / certificate |',
            '|---|---|---:|---:|---:|']
    for r in online:
        lines.append(f"| {r['case']} | {r['method']} | {r['measured_certificate_online_component_seconds']:.3f} | {r['measured_exact_online_component_seconds']:.3f} | {r['exact_over_certificate_online']:.3f} |")
    lines+=['','These are phase-component diagnostics, not independent warm measurements or predictions. Certificate online costs include all base verification, lazy correction setup and paid fallback. Exact online costs include shared loading and new-bank public replay.','',
            '## Native screen (not proof timings)','',
            '| Dataset | Basis | N | Requests | Conclusive after <=16 corrections | Correction rows | Agreement % |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in native['summary']:
        lines.append(f"| {r['dataset']} | {r['method']} | {r['n']} | {r['requests']} | {r['conclusive']} | {r['correction_rows']} | {r['agreement_percent']:.3f} |")
    lines+=['','## Boundaries','']+['- '+x for x in result['limitations']]
    (REPO/'output'/f'{stem}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=result['complete'],records=result['records'],online_diagnostic=online,paper_unchanged=result['paper_unchanged']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT)
    args=parser.parse_args();main(args.root.resolve())
