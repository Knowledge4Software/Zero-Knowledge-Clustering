"""Cross-check complete proof outputs and account for matched case wall times."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from campaigns.local_fast_tradeoff import save,sha
from commitjl.relative_certificates import relative_certificate
from commitjl.reuse_certificates import certificate


def cold_wall(root,case):
    state=json.loads((root/'status.json').read_text())['cases'][case]
    assert state['complete'] and not state['resumed'] and not state['interrupted_checkpoint']
    attempts=[r['attempts'][0] for r in state['phase_resources']]
    assert all(r['complete'] and not r['running'] for r in attempts)
    fitting=0 if case=='full' else json.loads((root/'protocol.json').read_text())['fitting_seconds']
    # Encloses public export and Python work between the first/last invocation.
    span=max(a['started_unix']+a['wall_seconds'] for a in attempts)-min(a['started_unix'] for a in attempts)+fitting
    return max(span,state.get('enclosing_case_wall_seconds',0),state['wall_seconds'])


def audit_case(root,case):
    inp=json.loads((root/case/'input.json').read_text());model=inp['model'];x=np.asarray(inp['x'],dtype=np.int64)
    queries=inp['queries'];assert sorted(i for q in queries for i in q)==list(range(len(x)))
    is_full=case=='full';is_exact=is_full or case=='family_affine';phase=f'full_reuse_{len(queries[0])}.json' if is_full else ('family_affine_16.json' if case=='family_affine' else 'relative_reuse_16.json')
    data=json.loads((root/case/'bundle'/phase).read_text());assert data['complete']
    expected=len(queries)*len(inp['reuse_centers']);assert len(data['samples'])==data['verified']==expected
    replay=json.loads((root/case/'public_bundle/verify.json').read_text())
    assert replay['complete'] and replay['verified']==expected+(0 if is_full else len(x))
    records=[]
    for s,centers in enumerate(inp['reuse_centers']):
        c=np.asarray(centers,dtype=np.int64);distances=((x[:,None,:]-c[None,:,:])**2).sum(2)
        labels=distances.argmin(1);true=int(distances.min(1).sum())
        samples=data['samples'][s*len(queries):(s+1)*len(queries)]
        for j,sample in enumerate(samples):
            assert sample.get('centre_set',0)==s and sample['id']==j and sample['indices']==queries[j]
        if is_exact:
            for sample in samples:
                assert int(sample['psi'])==int(distances[sample['indices']].min(1).sum())
                if not is_full:assert sample['assign']==labels[sample['indices']].tolist()
            records.append(dict(case=case,set=s,agreement_percent=100.,certified_coverage_percent=100.,cost_error_percent=0.,cost_interval_width_percent=0.))
            continue
        native=relative_certificate(model,x,centers);absolute=certificate(model,x,centers)
        assert all(not flag or lab==truth for flag,lab,truth in zip(native['certified_flags'],native['labels'],labels))
        for sample in samples:
            indices=sample['indices']
            assert sample['assign']==[native['labels'][i] for i in indices]
            assert sample.get('relative_certified_count',0)==sum(native['certified_flags'][i] for i in indices)
            assert sample.get('certified_count',0)==sum(absolute['certified_flags'][i] for i in indices)
        for key,reference in [('psi',absolute['psi']),('margin_sum',absolute['margin_sum']),('cost_lower_scaled',absolute['cost_lower_scaled']),('cost_upper_scaled',absolute['cost_upper_scaled'])]:
            assert sum(int(sample[key]) for sample in samples)==reference,(case,s,key)
        assert absolute['cost_lower_scaled']<=true*absolute['cost_denominator']<=absolute['cost_upper_scaled']
        records.append(dict(case=case,set=s,agreement_percent=100*np.mean(labels==native['labels']),
            certified_coverage_percent=100*native['certified_count']/len(x),
            cost_error_percent=100*abs(absolute['psi']/model['basis_scale']**4/true-1),
            cost_interval_width_percent=100*(absolute['cost_upper_scaled']-absolute['cost_lower_scaled'])/absolute['cost_denominator']/true))
    measured=json.loads((root/'status.json').read_text())['cases'][case]
    interrupted=bool(measured.get('resumed') or measured.get('interrupted_checkpoint'))
    timing=dict(wall_seconds=measured['wall_seconds'] if interrupted else cold_wall(root,case),
        active_phase_wall_seconds=measured['wall_seconds'],uninterrupted_cold_wall_available=not interrupted,
        timing_kind='summed active phase wall including restart/replay overhead; offline pause excluded' if interrupted else 'enclosing uninterrupted case wall')
    return dict(case=case,**timing,public_proofs=replay['verified'],
        native_output_checks_passed=True,input_sha256=sha(root/case/'input.json'),
        proof_phase_sha256=sha(root/case/'bundle'/phase)),records,inp


def run(baseline,optimized,out):
    out.mkdir(parents=True,exist_ok=True)
    full,records,full_input=audit_case(baseline,'full')
    summaries=[full]
    for root,case,label in [(baseline,'head_r39','head_r39_original'),(baseline,'head_r63','head_r63_original')]+([(optimized,'head_r63','head_r63_cached')] if optimized else []):
        item,quality,inp=audit_case(root,case)
        assert inp['x']==full_input['x'] and inp['reuse_centers']==full_input['reuse_centers']
        item['label']=label
        item['full_over_method_ratio']=full['wall_seconds']/item['wall_seconds'] if item['uninterrupted_cold_wall_available'] else None
        item['full_over_active_phase_ratio']=full['active_phase_wall_seconds']/item['active_phase_wall_seconds']
        item['minimum_agreement_percent']=float(min(q['agreement_percent'] for q in quality))
        item['passes_user_99_floor_on_every_set']=bool(item['minimum_agreement_percent']>=99)
        summaries.append(item);records.extend(dict(q,label=label) for q in quality)
    pd.DataFrame(records).to_csv(out/'quality_checked_against_proofs.csv',index=False)
    save(out/'summary.json',dict(complete=True,cases=summaries,baseline_root=str(baseline),optimized_root=str(optimized) if optimized else None,
        timing='Uninterrupted cases use enclosing cold case wall, including compilation/setup, native prepare/answers, proof generation/checks, public export, fresh public replay, and head fitting. User-paused cases report summed active phase wall including restart/replay overhead; offline pause excluded, between-phase export/orchestration excluded and no uninterrupted cold speed ratio claimed. One fit and one repetition; exploratory. Code compilation and numerical research screening excluded.',
        limitations=['No independent-fit timing replication.','Assignment certificates cover only the reported fraction.','Absolute cost intervals do not guarantee conclusive cost-update decisions.','No known-family exact affine proof baseline measured.']+(['The optimized run was interrupted by a user-requested laptop shutdown; its active phase ratio is not an uninterrupted cold benchmark.'] if any(not c['uninterrupted_cold_wall_available'] for c in summaries) else [])))
    print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--optimized',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run(a.baseline.resolve(),a.optimized.resolve() if a.optimized else None,a.out.resolve())
