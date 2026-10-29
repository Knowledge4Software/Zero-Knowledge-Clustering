"""Export paper tables from checked completed evidence, with no pending ratios."""
from pathlib import Path
import argparse,json
from commitjl.pilot_io import save,sha

REPO=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())


def quality_for_group(root):
    """Same-catalog native labels plus the actually proved change intervals."""
    import numpy as np
    from commitjl.relative_certificates import relative_certificate
    inp=read(root/'input.json');model=read(root/'frozen_prediction_model.json');task=read(root/'approved.json');native=read(root/'native_checks.json');x=np.asarray(inp['x'],dtype=np.int64)
    def exact(centers):
        c=np.asarray(centers,dtype=np.int64);result=np.empty((len(x),len(c)),dtype=np.int64)
        for start in range(0,len(x),64):result[start:start+64]=((x[start:start+64,None,:]-c[None,:,:])**2).sum(2)
        return result
    old_cost=int(exact(task['comparison_base']).min(1).sum());agreements=[];coverages=[];widths=[];regressions=detected=0
    assert len(task['reuse_centers'])==len(native['decisions'])
    for centers,r in zip(task['reuse_centers'],native['decisions']):
        distance=exact(centers);truth=distance.argmin(1);change=int(distance.min(1).sum()-old_cost);assert change==r['truth']
        assignment=relative_certificate(model,x,centers);labels=np.asarray(assignment['labels']);flags=np.asarray(assignment['certified_flags'],dtype=bool)
        assert not np.any(flags&(labels!=truth));agreements.append(100*float(np.mean(labels==truth)));coverages.append(100*float(flags.mean()));widths.append(100*(int(r['upper'])-int(r['lower']))/int(r['denominator'])/old_cost)
        regressions+=change>0;detected+=change>0 and r['decision']=='worsening'
    return dict(mean_agreement=float(np.mean(agreements)),minimum_agreement=min(agreements),mean_assignment_certified=float(np.mean(coverages)),mean_change_width=float(np.mean(widths)),distinct_regressions=regressions,regressions_certified_without_fallback=detected)


def table(caption,label,columns,heading,rows,wide=False):
    env='table*' if wide else 'table'
    return '\n'.join([r'\begin{'+env+'}[t]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+columns+'}',r'\toprule',heading+r'\\',r'\midrule',*[row+r'\\' for row in rows],r'\bottomrule',r'\end{tabular}',r'\end{'+env+'}',''])


def export(out,replication,scale,preview=False):
    out.mkdir(parents=True,exist_ok=True);sources={}
    def evidence(path):
        path=Path(path);sources[path.relative_to(REPO).as_posix()]=sha(path);return read(path)
    from campaigns.main_rank79 import verify_group
    old63=verify_group(REPO/'code/validation/outputs/group_compact_control_n128_20261025');old79=verify_group(REPO/'code/validation/outputs/group_compact_auxiliary_r79_n128_20261025/control')
    for path in ['paper_results/review_controls/compact_optimization_20261025/summary.json','paper_results/review_controls/compact_auxiliary_rank79_20261025/summary.json']:assert evidence(REPO/path)['complete']
    family=evidence(REPO/'code/validation/outputs/known_family_affine_n128_20261024/checked_comparison.json');assert family['complete'] and family['all_outputs_match_full'] and not family['open_future_centers_supported']
    native=evidence(REPO/'paper_results/review_controls/rank79_independent_replication_20261025/distinct_summary.json');assert native['complete']
    for name,digest in native['source_sha256'].items():assert sha(REPO/'paper_results/review_controls/rank79_independent_replication_20261025'/name)==digest
    development=evidence(REPO/'paper_results/review_controls/residual_certificate_rank_confirmation_20261025/distinct_summary.json');assert development['complete']
    assert development['source_results_sha256']==sha(REPO/'paper_results/review_controls/residual_certificate_rank_confirmation_20261025/results.csv')
    for path,digest in development['source_archive_sha256'].items():assert sha(REPO/path)==digest
    h=next(r for r in native['results'] if r['n']==1024 and r['representation']=='ordered_head' and r['rank']==79);p=next(r for r in native['results'] if r['n']==1024 and r['representation']=='pca' and r['rank']==79)
    macros=dict(GroupSixtyThreeMinutes=f"{old63['active_phase_seconds_including_fitting_and_fallback']/60:.3f}",GroupSeventyNineMinutes=f"{old79['active_phase_seconds_including_fitting_and_fallback']/60:.3f}",GroupFullMinutes=f"{old79['full_reference_active_seconds']/60:.3f}",GroupSixtyThreeSpeedup=f"{old63['full_over_group_active_phase']:.3f}",GroupSeventyNineSpeedup=f"{old79['full_over_group_active_phase']:.3f}",FreshUniqueHeadConclusive=h['conclusive'],FreshUniqueRequests=h['requests'],FreshUniquePcaConclusive=p['conclusive'],FreshUniqueRegressions=h['regressions'],FreshUniqueHeadDetected=h['detected'],FreshUniquePcaDetected=p['detected'],FreshNativeHeadAgreement=f"{h['mean_agreement_percent']:.3f}",FreshNativePcaAgreement=f"{p['mean_agreement_percent']:.3f}")
    for n,tag in [(1024,'Large'),(128,'Small')]:
        for method,label in [('ordered_head','Head'),('pca','Pca')]:
            r=next(v for v in native['results'] if v['n']==n and v['representation']==method and v['rank']==79);macros[f'FreshUnique{tag}{label}Decisions']=f"{r['conclusive']}/{r['requests']}";macros[f'FreshUnique{tag}{label}Regressions']=f"{r['detected']}/{r['regressions']}"
    rows=[]
    for r in native['results']:
        method='Head' if r['representation']=='ordered_head' else 'PCA';rows.append(f"{r['n']} & {method} & {r['rank']} & {r['conclusive']}/{r['requests']} & {r['detected']}/{r['regressions']} & {r['mean_agreement_percent']:.3f} & {r['mean_width_percent_old_cost']:.4f}")
    (out/'native_distinct.tex').write_text(table('Fresh-fit native results. Each distinct nonidentity quantized candidate is counted once within a deployed fit. Decision and regression columns give certified/total; agreement and width are percentages. Prediction rank stays 63. The sole distinct worsening set at $N=1024$ was submitted three times; it is not three regressions.','tab:group-fresh-native','rlrrrrr',r'$N$ & Basis & Cert. rank & Decisions & Regressions & Agreement & Width',rows,True),encoding='utf-8')
    rows=[]
    for rank,r in development['results_by_rank'].items():rows.append(f"{rank} & {r['conclusive_distinct_updates']}/{r['distinct_nonidentity_updates']} & {r['distinct_natural_regressions_detected']}/{r['distinct_natural_regressions']} & {r['original_proposal_requests']} & {r['repeated_center_requests_removed']}")
    (out/'development_distinct.tex').write_text(table('Stored-fit numerical development, distinct updates as primary units. Last columns retain proposal count and repeats removed; no new proof generation is inferred.','tab:group-development-distinct','rrrrr',r'Rank & Decisions & Regressions & Proposals & Repeats',rows),encoding='utf-8')
    full=old79['full_reference_active_seconds'];assert full==old63['full_reference_active_seconds']==family['full']['active_phase_wall_seconds']
    rows=[f"Exact public-center Full & {full/60:.3f} & 1.000 & {family['full']['public_proofs']}",f"Compact head 63 & {old63['active_phase_seconds_including_fitting_and_fallback']/60:.3f} & {old63['full_over_group_active_phase']:.3f} & {old63['group_public_proofs']}",f"Compact auxiliary 79 & {old79['active_phase_seconds_including_fitting_and_fallback']/60:.3f} & {old79['full_over_group_active_phase']:.3f} & {old79['group_public_proofs']}",f"Exact known-family affine & {family['family']['active_phase_wall_seconds']/60:.3f} & {family['full_over_family_active_phase']:.3f} & {family['family']['public_proofs']}"]
    (out/'completed_control.tex').write_text(table('Completed original one-fit $N=128$ control. Active minutes include cold setup and fresh public replay; all 19 distinct updates improve and require no fallback. Known-family affine assumes all 85 centers preapproved before preprocessing.','tab:group-proof-frontier','lrrr',r'Method & Minutes & Full/method & Proofs',rows),encoding='utf-8')
    main=[]
    if not preview:
        rsummary=evidence(replication/'summary.json');ssummary=evidence(scale/'summary.json');assert rsummary['complete'] and ssummary['complete']
        for r in rsummary['results']:
            checked=verify_group(replication/r['fit']/r['variant']);assert all(checked[k]==r[k] for k in checked);main.append(dict(case=r['fit'],n=128,root=replication/r['fit']/r['variant'],**r))
        for r in ssummary['results']:
            checked=verify_group(scale/r['case']/r['variant']);assert all(checked[k]==r[k] for k in checked);main.append(dict(root=scale/r['case']/r['variant'],**r))
        head=[r for r in main if r['variant']=='head79'];assert len(head)==4 and len(main)==11
        macros.update(MainHeadRatioMin=f"{min(r['full_over_group_active_phase'] for r in head):.2f}",MainHeadRatioMax=f"{max(r['full_over_group_active_phase'] for r in head):.2f}",MainLargestCatalog=max(r['n'] for r in main),MainHeadFallbackTasks=sum(r['fallback_updates'] for r in head),MainHeadTasks=sum(r['proposal_updates'] for r in head))
        rows=[];quality_rows=[];quality_records=[]
        for r in main:
            label=('Fashion' if 'fashion' in r['case'] else 'MNIST');name={'head79':'Head 79','head63':'Head 63','pca79':'PCA 79'}[r['variant']];rows.append(f"{label} & {r['n']} & {name} & {r['conclusive_without_fallback']}/{r['proposal_updates']} & {r['fallback_updates']} & {r['active_phase_seconds_including_fitting_and_fallback']/60:.2f} & {r['full_reference_active_seconds']/60:.2f} & {r['full_over_group_active_phase']:.2f}")
            q=quality_for_group(r['root']);quality_records.append(dict(case=r['case'],n=r['n'],variant=r['variant'],**q));reg=f"{q['regressions_certified_without_fallback']}/{q['distinct_regressions']}" if q['distinct_regressions'] else '--'
            quality_rows.append(f"{label} & {r['n']} & {name} & {q['mean_agreement']:.3f} & {q['minimum_agreement']:.3f} & {q['mean_assignment_certified']:.2f} & {q['mean_change_width']:.4f} & {reg}")
            evidence(r['root']/'checked_comparison.json');evidence(r['root']/'public/public_manifest.json');evidence(r['root']/'public/group_verify.json');evidence(r['root']/'native_checks.json')
        save(out/'main_quality.json',dict(complete=True,scope='Same catalogs as the complete proof timings; prediction/assignment certificates are independently recomputed native metrics, not outputs proved by the cost-change circuit.',results=quality_records))
        (out/'main_proofs.tex').write_text(table('Complete matched proof workloads, including prospectively generated Full fallback and fresh public replay. Resolved is without fallback; every remaining task receives exact fallback. Minutes are active deployment costs with all attempts charged. Different catalog sizes reuse the same two model fits and are scale tests, not independent fits.','tab:group-main-proofs','lrlrrrrr',r'Data & $N$ & Profile & Resolved & Fallback & Method min & Full min & Ratio',rows,True)+r'\noindent The head 79 Full/method ratios range from \MainHeadRatioMin{} to \MainHeadRatioMax{} across these workloads. Values below one denote a slower composed method. Across the four head 79 catalog workloads, \MainHeadFallbackTasks{} of \MainHeadTasks{} tasks require fallback; those tasks share two fitted models and must not be interpreted as independent trials.'+'\n'+table('Quality on the same catalogs as Table~\\ref{tab:group-main-proofs}. Mean/minimum assignment agreement and assignment-certified fractions are native checks of the frozen predictor, not outputs of the group proof. Change width is the proved pre-fallback interval relative to old cost. Regression counts concern distinct worsening tasks certified before fallback; a dash indicates no worsening candidate. All percentage columns retain failures.','tab:group-main-quality','lrlrrrrr',r'Data & $N$ & Profile & Mean agree & Min agree & Assign. cert. & Change width & Regressions',quality_rows,True),encoding='utf-8')
    else:(out/'main_proofs.tex').write_text('% Excluded in this preview: new main proofs are incomplete.\n')
    prefix='% Generated from checked evidence; do not edit constants by hand.\n'+('\\def\\MainRunsComplete{1}\n' if not preview else '')
    (out/'metrics.tex').write_text(prefix+'\n'.join('\\providecommand{\\'+k+'}{'+str(v)+'}' for k,v in macros.items())+'\n',encoding='utf-8')
    receipt=dict(complete=True,preview=preview,main_proof_rows=len(main),sources=sources,generated_sha256={p.name:sha(p) for p in out.glob('*.tex')});save(out/'export_receipt.json',receipt);return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['out','replication-root','scale-root']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--preview',action='store_true');a=p.parse_args();r=export(a.out.resolve(),a.replication_root.resolve(),a.scale_root.resolve(),a.preview);print(json.dumps(dict(complete=True,preview=r['preview'],main_proof_rows=r['main_proof_rows'])))
