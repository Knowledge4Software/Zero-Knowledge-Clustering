"""Evidence checks for the selected bounded exact-row refinement profile."""
from pathlib import Path
import json
import numpy as np
from campaigns.local_fast_tradeoff import sha


def read(path): return json.loads(Path(path).read_text())


def checked_refinement(root):
    r=read(root/'checked_comparison.json');p=read(root/'protocol.json');base=Path(p['base'])
    assert r['complete'] and r['all_intervals_contain_full'] and r['no_wrong_decisions']
    for key,name in [('base_protocol_sha256','protocol.json'),('base_owner_pin_sha256','owner_pin.json'),('base_task_sha256','approved.json'),('base_public_manifest_sha256','public/public_manifest.json')]:
        assert sha(base/name)==p[key],name
    for folder in [base/'public']+([root/'public'] if r['new_refinement_proofs'] else []):
        assert all(sha(folder/name)==digest for name,digest in read(folder/'public_manifest.json')['sha256'].items())
    replay=read(base/'public/group_verify.json');assert replay['complete'] and replay['verified']==r['base_public_proofs']
    if r['new_refinement_proofs']:
        replay=read(root/'public/refinement_verify.json');assert replay['complete']
        assert replay['verified']==r['base_public_proofs']+r['new_refinement_proofs']
        assert read(root/'public/refinement_verified_decisions.json')==r['decisions']
    assert sum(v['refined_rows'] for v in r['decisions'])==r['new_refinement_proofs']
    assert all(0<=v['refined_rows']<=16 for v in r['decisions'])
    assert len(r['decisions'])==r['proposal_updates']
    native=read(base/'native_checks.json')['decisions'];assert len(native)==len(r['decisions'])
    for index,(truth,decision) in enumerate(zip(native,r['decisions'])):
        assert decision['set']==index and int(decision['denominator'])==int(truth['denominator'])
        scaled=truth['truth']*int(decision['denominator'])
        assert int(decision['lower'])<=scaled<=int(decision['upper'])
        expected='improvement' if scaled<0 else ('worsening' if scaled>0 else 'equal')
        assert decision['decision'] in ['inconclusive',expected]
    base_seconds=sum(v['wall_seconds'] for v in read(base/'status.json')['resources'].values())+read(base/'protocol.json')['fitting_seconds']
    extra_seconds=sum(v['wall_seconds'] for v in read(root/'status.json')['resources'].values())+read(root/'selection.json')['wall_seconds']
    assert abs(base_seconds-r['base_active_seconds'])<1e-6 and abs(extra_seconds-r['refinement_active_seconds'])<1e-6
    assert abs(base_seconds+extra_seconds+r['fallback_active_seconds']-r['active_phase_seconds_including_fitting_and_fallback'])<1e-6
    if r['fresh_fallback_proofs']:
        from analysis.audit_local_fast_tradeoff import audit_case
        f,_,_=audit_case(root/'fallback','full');assert f['public_proofs']==r['fresh_fallback_proofs']
    return r


def refined_quality(base,root,quality):
    result=quality(base);r=checked_refinement(root);inp=read(base/'input.json');task=read(base/'approved.json')
    x=np.asarray(inp['x'],dtype=np.int64);c=np.asarray(task['comparison_base'],dtype=np.int64)
    cost=int(((x[:,None,:]-c[None,:,:])**2).sum(2).min(1).sum())
    result['mean_change_width']=float(np.mean([100*(int(v['upper'])-int(v['lower']))/int(v['denominator'])/cost for v in r['decisions']]))
    truth=read(base/'native_checks.json')['decisions']
    result['regressions_certified_without_fallback']=sum(t['truth']>0 and v['decision']=='worsening' for t,v in zip(truth,r['decisions']))
    result['certified_switching_regressions']=sum(s['truth']>0 and s['strict_switch_rows']>0 and v['decision']=='worsening' for s,v in zip(result['switching'],r['decisions']))
    return result


def export_extra(repo,out,campaign,evidence,table,quality):
    native=evidence(repo/'paper_results/review_controls/rank79_independent_replication_20261025/distinct_summary.json')
    devroot=repo/'code/validation/outputs/selective_switch_refinement_20261025_v2'
    summary=evidence(devroot/'summary.json');assert summary['complete'] and summary['false_conclusive']==0
    for name,digest in summary['sources'].items():assert sha(repo/name)==digest
    records=evidence(devroot/'records.json');rows=[]
    for item in summary['results']:
        if item['cap']!=16:continue
        n=item['n'];method=item['representation'];q=next(r for r in native['results'] if r['n']==n and r['rank']==79 and r['representation']==method)
        old=next(r for r in summary['results'] if r['n']==n and r['representation']==method and r['cap']==0)
        name='Head' if method=='ordered_head' else 'PCA'
        rows.append(f"{n} & {name} & {q['mean_agreement_percent']:.3f} & {q['minimum_agreement_percent']:.3f} & {q['mean_assignment_certified_percent']:.3f} & {old['conclusive']}/{item['updates']} & {item['conclusive']}/{item['updates']} & {item['detected_regressions']}/{item['regressions']} & {item['total_exact_rows']}")
    (out/'native_quality.tex').write_text(table('Native development grid: fixed prediction rank63 and certificate rank79, before/after bounded refinement. Agreement and label certificates are percentages. Refined regression coverage and total exact rows count distinct updates within each deployed fit. Cap16 was selected on this previously observed grid; these are not confirmatory proof timings.','tab:native-quality','rlrrrrrrr',r'$N$ & Basis & Agr. & Worst & Label cert. & Base & Refined & Regr. & Rows',rows,True),encoding='utf-8')
    confirmroot=repo/'code/validation/outputs/refinement_confirmation_20261025'
    confirm=evidence(confirmroot/'summary.json');assert confirm['complete'] and confirm['false_conclusive']==0 and confirm['new_proofs']==0
    evidence(confirmroot/'protocol.json');evidence(confirmroot/'records.json')
    for fit in evidence(confirmroot/'fit_records.json'):
        assert sha(confirmroot/'fits'/f"{fit['fit']}_{fit['representation']}.npz")==fit['archive_sha256']
    rows=[]
    for r in confirm['results']:
        name='Head' if r['representation']=='ordered_head' else 'PCA'
        rows.append(f"{r['n']} & {name} & {r['mean_agreement_percent']:.3f} & {r['minimum_agreement_percent']:.3f} & {r['mean_assignment_certified_percent']:.3f} & {r['conclusive']}/{r['updates']} & {r['certified_regressions']}/{r['regressions']} & {r['mean_width_percent_old_cost']:.4f} & {r['total_exact_rows']}")
    (out/'confirmation_quality.tex').write_text(table('Native confirmation after freezing ranks, precision, width ordering and cap16: MNIST ($k=3$,seed137) and Fashion-MNIST ($k=5$,seed149), each at ages1/8. Prefixes, ages and representations share splits. No new proof-cost measurement is inferred. Width is relative to old exact cost; the unresolved PCA regression requires paid Full.','tab:confirmation-quality','rlrrrrrrr',r'$N$ & Basis & Agr. & Worst & Label cert. & Decisions & Regr. & Width & Rows',rows,True),encoding='utf-8')
    pilots=[];pilot_quality=[];results=[]
    config=read(campaign/'refinement_extension.json')
    for case,variant in ((case,variant) for case in read(campaign/'plan.json')['cases'] for variant in ['head79','pca79']):
        base=campaign/case['id']/variant;root=Path(config['root'])/case['id']/variant
        if not (root/'checked_comparison.json').exists():continue
        r=checked_refinement(root);q=refined_quality(base,root,quality);evidence(root/'checked_comparison.json');evidence(root/'protocol.json')
        tag='MNIST mature' if 'mature' in case['id'] else ('Fashion early' if 'fashion' in case['id'] else 'MNIST scale')
        tag+=' (head)' if variant=='head79' else ' (PCA)'
        proofs=r['base_public_proofs']+r['new_refinement_proofs']+r['fresh_fallback_proofs']
        pilots.append(f"{tag} & {case['n']} & {r['conclusive_without_fallback']}/{r['proposal_updates']} & {r['new_refinement_proofs']} & {r['fallback_updates']} & {r['active_phase_seconds_including_fitting_and_fallback']/60:.3f} & {r['full_reference_active_seconds']/60:.3f} & {r['full_over_group_active_phase']:.3f} & {proofs}")
        pilot_quality.append(f"{tag} & {q['mean_agreement']:.3f} & {q['minimum_agreement']:.3f} & {q['mean_assignment_certified']:.3f} & {q['mean_cost_error_percent']:.4f} & {q['mean_change_width']:.4f} & {q['certified_switching_regressions']}/{q['distinct_switching_regressions']}")
        results.append(dict(case=case['id'],variant=variant,comparison=r,quality=q))
    (out/'refinement_pilots.tex').write_text(table('Completed selected-method/Full pairs with eight workers. Selected-method cost sums shared measured compact components and new bounded-refinement work; it is not a separately repeated complete execution. Failed attempts, lazy key setup and fresh replay are charged. Rows denotes new exact-row proofs. These pairs do not imply completion of every other baseline.','tab:refinement-pilot','lrrrrrrrr',r'Cohort & $N$ & Decisions & Rows & FB & Method min. & Full min. & Ratio & Proofs',pilots,True)+table('Quality on the same selected-method/Full pairs. Predictor agreement, label certificates and approximate cost error remain native metrics; the aggregate refined cost-change intervals are proved. The final column restricts regression coverage to candidates with strictly switching rows. All percentage columns concern different quantities.','tab:refinement-pilot-quality','lrrrrrr',r'Cohort & Agr. & Worst & Label cert. & Cost error & Width & Switch regr.',pilot_quality,True),encoding='utf-8')
    return results
