"""Check and report only completed matched published-baseline adaptations."""
from pathlib import Path
import argparse,hashlib,json,statistics
import numpy as np
from commitjl.relative_certificates import relative_certificate

REPO=Path(__file__).resolve().parents[2]
DEFAULT=REPO/'code/validation/outputs/published_baselines_20261027'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def resource(root,label):
    paths=sorted((root/'resources').glob(label+'.attempt_*.json'))
    values=[read(p) for p in paths]
    assert values and values[-1]['complete'],label+' is not complete'
    assert all(v['stop'] is None for v in values),'Guarded/interrupted timing is not a clean comparison'
    return dict(wall_seconds=sum(v['wall_seconds'] for v in values),
        peak_rss_bytes=max(v['peak_process_tree_rss_bytes'] for v in values),
        receipts=[p.relative_to(REPO).as_posix() for p in paths])


def report(root,output_stem='published_baselines_laptop_20261028',emit=True,expected_case_count=2,limitations=None):
    records=[];manifest={}
    for target in sorted((root/'matched').glob('*_n*')):
        if not (target/'zkml_checked.json').exists():continue
        available=[b for b in ['ordered_head','pca'] if (target/(b+'_refined/checked_comparison.json')).exists()]
        if not available or not (target/'ordered_head/full/bundle/full_reuse_8.json').exists():continue
        name=target.name;source=target/'ordered_head/full/input.json';inp=read(source);n=len(inp['x'])
        input_pin=read(target/'input_pin.json')
        assert sha(source)==input_pin['full_input_sha256']
        for basis,key in [('ordered_head','centroid_input_sha256'),('pca','pca_input_sha256')]:
            path=target/basis/'head_r63/input.json';assert sha(path)==input_pin[key]
            original=read(path);assert original['x']==inp['x'] and original['reuse_centers']==inp['reuse_centers']
        zk=read(target/'zkml_checked.json');assert zk['complete'] and zk['all_costs_match_integer_oracle']
        public=target/'zkml_public'
        for file,value in zk['public_manifest'].items():assert sha(public/file)==value
        pin=read(target/'zkml_owner_pin.json')
        assert sha(public/'vk')==pin['vk_sha256'] and sha(public/'params')==pin['params_sha256']
        reference=inp['model']['centers'];proposals=[c for c in inp['reuse_centers'] if c!=reference]
        exact=lambda centers:sum(min(sum((int(x)-int(c))**2 for x,c in zip(row,center)) for center in centers) for row in inp['x'])
        costs=[exact(c) for c in inp['reuse_centers']];assert costs==zk['costs']
        full_result=read(target/'ordered_head/full/bundle/full_reuse_8.json');assert full_result['complete']
        full_costs=[0]*len(costs);seen=[[] for _ in costs]
        for sample in full_result['samples']:
            bank=sample.get('centre_set',0);indices=sample['indices'];seen[bank].extend(indices)
            batch_truth=sum(min(sum((int(x)-int(c))**2 for x,c in zip(inp['x'][i],center)) for center in inp['reuse_centers'][bank]) for i in indices)
            assert int(sample['psi'])==batch_truth;full_costs[bank]+=int(sample['psi'])
        assert full_costs==costs and all(sorted(indices)==list(range(n)) for indices in seen)
        old=exact(reference);changes=[exact(c)-old for c in proposals]
        prep=resource(root,name+'_zkml_prepare');queries=resource(root,name+'_zkml_queries')
        verifies=[resource(root,name+f'_zkml_verify{i:03}') for i in range(len(costs))]
        setup=read(target/'zkml_private/setup.json')
        # Charge all real cold work, including local KZG parameter creation.
        # The diagnostic MockProver is reported separately from the service.
        zk_seconds=prep['wall_seconds']-setup['mock_check_seconds']+queries['wall_seconds']+sum(v['wall_seconds'] for v in verifies)
        base=dict(case=name,n=n,d=inp['model']['d'],k=inp['model']['k'],requests=len(proposals),regressions=sum(v>0 for v in changes))
        full=resource(root,name+'_groth_full')
        records.append(dict(**base,method='Exact Full / Groth16',seconds=full['wall_seconds'],
            agreement_percent=100.,coverage=len(proposals),fallback=0,
            peak_rss_bytes=full['peak_rss_bytes'],exact=True))
        records.append(dict(**base,method='ZKML primitive adapter / Halo2-KZG',seconds=zk_seconds,
            including_diagnostic_mock_seconds=zk_seconds+setup['mock_check_seconds'],
            cached_universal_parameters_seconds=zk_seconds-setup['parameter_generation_seconds'],
            agreement_percent=100.,coverage=len(proposals),fallback=0,
            peak_rss_bytes=max([prep['peak_rss_bytes'],queries['peak_rss_bytes']]+[v['peak_rss_bytes'] for v in verifies]),
            proof_bytes=zk['proof_bytes'],proofs=zk['proofs'],exact=True,
            parameter_generation_seconds=setup['parameter_generation_seconds'],mock_check_seconds=setup['mock_check_seconds']))
        for basis,label in [('ordered_head','Centroid certificate'),('pca','PCA certificate')]:
            if basis not in available:continue
            checked=read(target/(basis+'_refined/checked_comparison.json'))
            assert checked['complete'] and checked['no_wrong_decisions'] and checked['all_intervals_contain_full']
            decisions=checked['decisions'];assert len(decisions)==len(proposals)
            for decision,change in zip(decisions,changes):
                assert int(decision['lower'])<=change*int(decision['denominator'])<=int(decision['upper'])
                if decision['decision']!='inconclusive':
                    assert decision['decision']==('improvement' if change<0 else 'worsening' if change>0 else 'equal')
            model=read(target/basis/'head_r63/input.json')['model'];x=np.asarray(inp['x'],np.int64);agreements=[];coverages=[]
            for centers in proposals:
                q=relative_certificate(model,x,centers)
                truths=((x[:,None,:]-np.asarray(centers,np.int64)[None,:,:])**2).sum(2).argmin(1)
                predictions=np.asarray(q['labels']);flags=np.asarray(q['certified_flags'],bool)
                assert not np.any(flags&(predictions!=truths))
                agreements.append(100*float(np.mean(predictions==truths)));coverages.append(100*float(flags.mean()))
            r1=resource(root,name+'_'+basis+'_base');r2=resource(root,name+'_'+basis+'_refined')
            records.append(dict(**base,method=label,seconds=r1['wall_seconds']+r2['wall_seconds'],
                component_active_seconds=checked['active_phase_seconds_including_fitting_and_fallback'],
                agreement_percent=statistics.mean(agreements),worst_agreement_percent=min(agreements),
                label_certificate_percent=statistics.mean(coverages),coverage=checked['conclusive_without_fallback'],
                fallback=checked['fallback_updates'],peak_rss_bytes=max(r1['peak_rss_bytes'],r2['peak_rss_bytes']),exact=False,
                all_intervals_contain_truth=True,certified_regressions=sum(change>0 and decision['decision']=='worsening' for change,decision in zip(changes,decisions))))
        for p in [source,target/'input_pin.json',target/'zkml_checked.json',target/'zkml_owner_pin.json']+[target/(b+'_refined/checked_comparison.json') for b in available]:
            manifest[p.relative_to(REPO).as_posix()]=sha(p)
    snapshot=read(root/'authorization_and_scope.json')
    assert all(sha(REPO/name)==value for name,value in snapshot['paper_sha256'].items()),'Paper changed during a no-edit task'
    for case in {r['case'] for r in records}:
        rows=[r for r in records if r['case']==case];full=next(r['seconds'] for r in rows if r['method']=='Exact Full / Groth16')
        for row in rows:row['full_over_method']=full/row['seconds']
    cases={r['case'] for r in records};sizes=sorted({r['n'] for r in records})
    result=dict(complete=len(cases)==expected_case_count and all(sum(r['case']==c for r in records)==4 for c in cases),completed_case_count=len(cases),records=records,
        paper_unchanged=True,source_sha256=manifest,os='Linux/WSL',workers=8,catalog_records=(sizes[0] if len(sizes)==1 else sizes),executions_per_fit=1,
        timing='Serial outer invocation wall time; actual cold setup, native work, proofs, public-only replay and paid fallback. ZKML diagnostic MockProver excluded and reported separately; local test KZG parameter creation included.',
        limitations=['ZKML uses published exact dot-product, adder, Poseidon chips and Halo2-KZG with our integer audit adapter; this does not benchmark the unmodified optimizing compiler.',
            'Different owner-approved commitment constructions bind the same original integers; this is not the same Poseidon2 Merkle root.',
            'Two previously observed fitted workloads, fixed first32 audit records and one timing execution each; no population or large-catalog claim.',
            'Assignment agreement is agreement with exact quantized labels, not ground-truth classification accuracy.'])
    if limitations is not None:result['limitations']=limitations
    out=REPO/'output'/f'{output_stem}.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Matched laptop comparison, 28 October 2026','',result['timing'],'',
        '| Case | Method | Seconds | Full/method | Agreement % | Decisions before Full fallback | Fallback | Peak RSS GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['case']} | {r['method']} | {r['seconds']:.3f} | {r['full_over_method']:.3f} | {r['agreement_percent']:.3f} | {r['coverage']}/{r['requests']} | {r['fallback']} | {r['peak_rss_bytes']/2**30:.3f} |")
    lines+=['','## Interpretation boundaries','']+['- '+s for s in result['limitations']]
    lines+=['','The paper is unchanged. Exact controls resolve all costs; approximate prediction agreement is separate from interval soundness.']
    (REPO/'output'/f'{output_stem}.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if emit:print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT);a=p.parse_args();report(a.root.resolve())
