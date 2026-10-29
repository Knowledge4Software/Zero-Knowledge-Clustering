"""Checked tables for one selected certificate and its relevant baselines."""
from pathlib import Path
import hashlib,json
import numpy as np
from campaigns.main_rank79 import verify_group
from campaigns.referee_final import CASES,verified_moments
from campaigns.local_fast_tradeoff import sha,save
from analysis.audit_local_fast_tradeoff import audit_case
from analysis.export_main_rank79_paper import table,quality_for_group
from analysis.refinement_paper_evidence import checked_refinement,refined_quality,export_extra

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'paper_results/referee_final_20261025/generated'
FIELD=21888242871839275222246405745257275088548364400416034343698204186575808495617


def read(path):return json.loads(Path(path).read_text())


def predictor_error(model,x,centers):
    from commitjl.reuse_certificates import integer_stats
    h,e=integer_stats(model,np.asarray(x,dtype=np.int64));hc,ec=integer_stats(model,np.asarray(centers,dtype=np.int64))
    distances=((np.asarray(h,dtype=object)[:,None,:]-np.asarray(hc,dtype=object)[None,:,:])**2).sum(2)+np.asarray(e,dtype=object)[:,None]+np.asarray(ec,dtype=object)[None,:]
    approx=sum(min(row) for row in distances)/model['basis_scale']**4
    a=np.asarray(x,dtype=np.int64);c=np.asarray(centers,dtype=np.int64);truth=int(((a[:,None,:]-c[None,:,:])**2).sum(2).min(1).sum())
    return 100*abs(approx-truth)/truth if truth else (0. if approx==0 else None)


def quality(root):
    result=quality_for_group(root);inp=read(root/'input.json');model=read(root/'frozen_prediction_model.json') if (root/'frozen_prediction_model.json').exists() else inp['model'];task=read(root/'approved.json')
    errors=[predictor_error(model,inp['x'],centers) for centers in task['reuse_centers']]
    x=np.asarray(inp['x'],dtype=np.int64);old=np.asarray(model['centers'],dtype=np.int64);old_distance=((x[:,None,:]-old[None,:,:])**2).sum(2);labels=old_distance.argmin(1);switching=[];native=read(root/'native_checks.json')['decisions']
    for index,centers in enumerate(task['reuse_centers']):
        c=np.asarray(centers,dtype=np.int64);dist=((x[:,None,:]-c[None,:,:])**2).sum(2);count=int((dist.min(1)<dist[np.arange(len(x)),labels]).sum());switching.append(dict(set=index,strict_switch_rows=count,truth=native[index]['truth'],decision=native[index]['decision']))
    result.update(mean_cost_error_percent=float(np.mean(errors)),maximum_cost_error_percent=max(errors),switching=switching,distinct_switching_regressions=sum(r['truth']>0 and r['strict_switch_rows']>0 for r in switching),certified_switching_regressions=sum(r['truth']>0 and r['strict_switch_rows']>0 and r['decision']=='worsening' for r in switching))
    return result


def exact_limits(model,q=16):
    b=np.asarray(model['basis'],dtype=np.int64);n,d,k=model['n'],model['d'],model['k'];t=model['basis_scale'];rank=len(b);x=1<<model['input_bits'];w=2*x
    l1=[sum(abs(int(v)) for v in row) for row in b];h=t*w*max(l1);r=w*max(t*t+sum(abs(int(b[j,a]))*l1[j] for j in range(rank)) for a in range(d));e=d*r*r
    from math import isqrt
    rho=isqrt(e)+(isqrt(e)**2<e);g=b@b.T-t*t*np.eye(rank,dtype=np.int64);gamma=max(sum(abs(int(v)) for v in row) for row in g)
    coeff=4*h*(gamma+t*t);constant=t*t*(2*rank*h*h+2*e)+2*gamma*rank*h*h;paired=rank*h*coeff+constant+4*t*t*rho*rho+1;guard=64*(n+q+k)*16*paired
    assert guard<FIELD
    return {key:str(value) for key,value in dict(field_modulus=FIELD,X=x,H_max=h,R_max=r,E_max=e,radius_max=rho,gram_max_row_sum=gamma,coefficient_max=coeff,constant_max=constant,paired_limit=paired,switch_limit=16*paired,field_guard=guard).items()}


def sample_seconds(s):return sum(s['prove_seconds'])+sum(s['verify_seconds'])+s['witness_seconds']+s.get('native_answer_seconds',0)


def original_cost_curve(out,group,family):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=REPO/'code/validation/outputs/local_fast_tradeoff_n128_20261024';full,_,_=audit_case(source,'full');full_samples=read(source/'full/bundle/full_reuse_8.json')['samples']
    group_records=read(group/'bundle/group_query_records.json');g=verify_group(group);group_cost=[sample_seconds(r['assigned'])+sum(sample_seconds(s) for s in r['batches']) for r in group_records]
    full_cost=[sum(sample_seconds(s) for s in full_samples if s.get('centre_set',0)==index) for index in range(20)]
    full_fixed=full['active_phase_wall_seconds']-sum(full_cost)+full_cost[0];group_fixed=g['active_phase_seconds_including_fitting_and_fallback']-sum(group_cost)
    assert full_fixed>=0 and group_fixed>=0
    x=np.arange(1,20);fc=full_fixed+np.cumsum(full_cost[1:]);gc=group_fixed+np.cumsum(group_cost)
    assert abs(fc[-1]-full['active_phase_wall_seconds'])<1e-5 and abs(gc[-1]-g['active_phase_seconds_including_fitting_and_fallback'])<1e-5
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.7))
    axes[0].plot(x,fc/60,label='Exact public-center Full');axes[0].plot(x,gc/60,label='Group certificate 79')
    axes[0].set_title('Fixed deployed model');axes[0].set_ylabel('Allocated active cost (min)');axes[0].legend(fontsize=7)
    mean_full=float(np.mean(full_cost[1:]));mean_group=float(np.mean(group_cost))
    for period in [1,4,19]:
        resets=np.ceil(x/period);axes[1].plot(x,(resets*group_fixed+x*mean_group)/60,label=f'Group: reset every {period}')
    axes[1].plot(x,(full_fixed+x*mean_full)/60,'k--',label='Full: setup retained');axes[1].set_title('Modeled baseline replacement');axes[1].legend(fontsize=6)
    for ax in axes:ax.set_xlabel('Distinct candidate requests');ax.grid(alpha=.25);ax.set_xlim(1,19)
    fig.tight_layout();path=out/'cost_frontier.pdf';fig.savefig(path,bbox_inches='tight');plt.close(fig)
    save(out/'cost_curve_components.json',dict(scope='Measured-component allocation on completed original two-worker MNIST19 control. All query/replay/checkpoint overhead allocated upfront. Prefixes and reset curves are modeled, not independent runs. No fallback occurs in this cohort.',full_fixed_seconds=full_fixed,group_fixed_seconds=group_fixed,full_request_seconds=full_cost[1:],group_request_seconds=group_cost))
    (out/'cost_frontier.tex').write_text(r'''\begin{figure*}[t]
\centering\includegraphics[width=.94\textwidth]{paper_results/referee_final_20261025/generated/cost_frontier.pdf}
\caption{Cost accounting from completed two-worker MNIST components. Left: fixed-baseline cumulative allocation, with measured replay, key-loading and other residual overhead charged upfront. The 19-request endpoints equal recorded active totals; intermediate prefixes are allocations, not fresh runs. Right: modeled repeated cold group preparation versus reusable Full setup, using measured mean query components. Reset curves are not measured rebase deployments and do not include newly fitted bases.}
\label{fig:cost-frontier}
\end{figure*}
''',encoding='utf-8')


def export(campaign,require_complete=False):
    out=DEST;out.mkdir(parents=True,exist_ok=True);sources={}
    def evidence(path):
        path=Path(path);sources[path.relative_to(REPO).as_posix()]=sha(path);return read(path)
    original=REPO/'code/validation/outputs/group_compact_auxiliary_r79_n128_20261025/control';fresh=REPO/'code/validation/outputs/rank79_independent_replication_20261025/fashion_mnist_k3_seed101_age8/head79'
    g=verify_group(original);f=verify_group(fresh);qg=quality(original);qf=quality(fresh)
    family=evidence(REPO/'code/validation/outputs/known_family_affine_n128_20261024/checked_comparison.json');assert family['complete'] and family['all_outputs_match_full']
    necessity=evidence(REPO/'code/validation/outputs/referee_necessity_check_20261025/summary.json');assert necessity['cases'][0]['simple_A_improvements']==necessity['cases'][0]['best_groupwise_improvements']==2
    rows=[f"MNIST, 19 & Exact Full & {g['full_reference_active_seconds']/60:.3f} & 1.000 & 320",f"MNIST, 19 & Group 79 & {g['active_phase_seconds_including_fitting_and_fallback']/60:.3f} & {g['full_over_group_active_phase']:.3f} & {g['group_public_proofs']}",f"MNIST, 19 & Known-family affine & {family['family']['active_phase_wall_seconds']/60:.3f} & {family['full_over_family_active_phase']:.3f} & {family['family']['public_proofs']}",f"Fashion, 2 & Exact Full & {f['full_reference_active_seconds']/60:.3f} & 1.000 & 48",f"Fashion, 2 & Group 79 & {f['active_phase_seconds_including_fitting_and_fallback']/60:.3f} & {f['full_over_group_active_phase']:.3f} & {f['group_public_proofs']}"]
    rows=[row for row in rows if 'Known-family affine' not in row]
    (out/'completed_proofs.tex').write_text(table('Completed real-proof controls, $N=128$, two Go workers and one repetition per workload. Active minutes include cold preparation, setup and fresh public replay. Neither cohort triggers fallback. Future candidate centers are not required by the preprocessing relation.','tab:completed-proofs','llrrr',r'Cohort, updates & Method & Minutes & Full/method & Proofs',rows,True),encoding='utf-8')
    closed_rows=[f"MNIST,19 & 128 & 2 & 85 & {family['query_center_sets']} & {family['family']['active_phase_wall_seconds']/60:.3f} & {family['family']['public_proofs']}"]
    rows=[]
    for name,q,count in [('MNIST',qg,19),('Fashion',qf,2)]:rows.append(f"{name} & {count}/{count} & {q['mean_agreement']:.3f} & {q['minimum_agreement']:.3f} & {q['mean_assignment_certified']:.3f} & {q['mean_cost_error_percent']:.4f} & {q['mean_change_width']:.4f}")
    (out/'completed_quality.tex').write_text(table('Selected group79 controls: conclusive cost-change decisions, native predictor agreement (mean/worst), native assignment-certified fraction, native mean absolute relative cost error, and accepted change-interval width. Percentage columns concern different quantities; prediction rank remains63.','tab:completed-quality','lrrrrrr',r'Cohort & Decisions & Agr. & Worst & Label cert. & Cost error & Width',rows,True),encoding='utf-8')
    native=evidence(REPO/'paper_results/review_controls/rank79_independent_replication_20261025/distinct_summary.json');rows=[]
    for r in native['results']:
        if r['rank']!=79:continue
        name='Head' if r['representation']=='ordered_head' else 'PCA';rows.append(f"{r['n']} & {name} & {r['mean_agreement_percent']:.3f} & {r['minimum_agreement_percent']:.3f} & {r['mean_assignment_certified_percent']:.3f} & {r['conclusive']}/{r['requests']} & {r['detected']}/{r['regressions']} & {r['mean_width_percent_old_cost']:.4f}")
    (out/'native_quality.tex').write_text(table('Fresh-fit native results for the selected79-dimensional certificate and frozen63-dimensional predictor. Each distinct nonidentity quantized update is counted once within a deployed fit. Agreement, label-certified fraction and width are percentages; decision and regression entries are certified/total. These are native quality observations, not additional proof timings.','tab:native-quality','rlrrrrrr',r'$N$ & Basis & Agr. & Worst & Label cert. & Decisions & Regressions & Width',rows,True),encoding='utf-8')
    matched=[];quality_rows=[];cost_rows=[];completed=[]
    parents=evidence(campaign/'parents.json') if (campaign/'parents.json').exists() else {}
    extension=read(campaign/'refinement_extension.json')
    for case in CASES:
        paths={name:campaign/case['id']/name for name in ['head79','pca79','moments_only','known_family_affine']}
        if not all((paths[name]/'checked_comparison.json').exists() and read(paths[name]/'checked_comparison.json').get('complete') for name in ['moments_only','known_family_affine']):continue
        refined={variant:Path(extension['root'])/case['id']/variant for variant in ['head79','pca79']}
        if not all((p/'checked_comparison.json').exists() and read(p/'checked_comparison.json').get('complete') for p in refined.values()):continue
        parent=Path(parents[case['id']]['ordered_head']);assert read(parent/'protocol.json')['proof_runtime_workers']==read(parent/'status.json')['proof_runtime_workers']==8
        for variant in ['head79','pca79','moments_only']:assert read(paths[variant]/'protocol.json')['proof_runtime_workers']==8
        assert read(paths['known_family_affine']/'protocol.json')['proof_runtime_workers']==8
        full,_,_=audit_case(parent,'full');completed.append(case['id']);tag=('MNIST mature' if 'mature' in case['id'] else ('Fashion early' if 'fashion' in case['id'] else 'MNIST scale'))
        methods=[('Full',None,dict(total=full['active_phase_wall_seconds'],proofs=full['public_proofs'],conclusive=len(read(parent/'full/input.json')['reuse_centers'])-1,fallback=0,ratio=1.))]
        for variant,label in [('head79','Group79'),('pca79','PCA79'),('moments_only','Moments-only')]:
            path=paths[variant];r=verified_moments(path) if variant=='moments_only' else checked_refinement(refined[variant]);evidence(path/('checked_comparison.json' if variant=='moments_only' else 'backbone_complete.json'))
            proof_count=r['group_public_proofs'] if variant=='moments_only' else r['base_public_proofs']+r['new_refinement_proofs']
            methods.append((label,path,dict(total=r['active_phase_seconds_including_fitting_and_fallback'],proofs=proof_count+r['fresh_fallback_proofs'],conclusive=r['conclusive_without_fallback'],fallback=r['fallback_updates'],ratio=r['full_over_group_active_phase'])))
        affine=evidence(paths['known_family_affine']/'checked_comparison.json');assert affine['all_outputs_match_full']
        closed_rows.append(f"{tag} & {case['n']} & 8 & {read(paths['known_family_affine']/'protocol.json')['bank_size']} & {affine['query_center_sets']} & {affine['family']['active_phase_wall_seconds']/60:.3f} & {affine['family']['public_proofs']}")
        for label,path,r in methods:
            matched.append(f"{tag} & {case['n']} & {label} & {r['total']/60:.3f} & {r['ratio']:.3f} & {r['conclusive']} & {r['fallback']} & {r['proofs']}")
            if label in ['Group79','PCA79']:
                variant='head79' if label=='Group79' else 'pca79';extra=refined[variant];q=refined_quality(path,extra,quality);quality_rows.append(f"{tag} & {label} & {q['mean_agreement']:.3f} & {q['minimum_agreement']:.3f} & {q['mean_assignment_certified']:.3f} & {q['mean_cost_error_percent']:.4f} & {q['mean_change_width']:.4f} & {q['regressions_certified_without_fallback']}/{q['distinct_regressions']}")
                state=read(path/'status.json');resources=read(extra/'status.json')['resources'];prep=sum(v['wall_seconds'] for k,v in state['resources'].items() if k not in ['group_query','group_verify'])+read(path/'protocol.json')['fitting_seconds']+resources.get('refinement_setup',{}).get('wall_seconds',0);query=state['resources']['group_query']['wall_seconds']+resources.get('refinement_prove',{}).get('wall_seconds',0)+read(extra/'selection.json')['wall_seconds'];replay=state['resources']['group_verify']['wall_seconds']+resources.get('refinement_verify',{}).get('wall_seconds',0);fallback=read(extra/'checked_comparison.json')['fallback_active_seconds'];cost_rows.append(f"{tag} & {label} & {prep/60:.3f} & {query/60:.3f} & {replay/60:.3f} & {fallback/60:.3f}")
            elif label in ['Full','Family affine']:
                quality_rows.append(f"{tag} & {label} & 100.000 & 100.000 & 100.000 & 0.0000 & 0.0000 & exact")
                owner=parent if label=='Full' else path;kind='full' if label=='Full' else 'family_affine';phase='full_reuse' if label=='Full' else 'family_affine';batch=8 if label=='Full' else 16
                resources={name:read(owner/kind/(name+'.resource.json'))['wall_seconds'] for name in ['prepare',phase,'verify']+([] if label=='Full' else ['stage1'])}
                cs=read(owner/kind/'bundle'/f'{phase}_{batch}.json');setup=cs['compile_seconds']+cs['setup_seconds'];prep=resources['prepare']+resources.get('stage1',0)+setup;query=resources[phase]-setup
                cost_rows.append(f"{tag} & {label} & {prep/60:.3f} & {query/60:.3f} & {resources['verify']/60:.3f} & 0.000")
            else:
                native=read(path/'native_checks.json');regressions=sum(v['truth']>0 for v in native['decisions']);inp=read(path/'input.json');a=np.asarray(inp['x'],dtype=np.int64);old=np.asarray(inp['model']['centers'],dtype=np.int64);oldcost=int(((a[:,None,:]-old[None,:,:])**2).sum(2).min(1).sum());records=read(path/'bundle/moments_query_records.json');width=float(np.mean([100*(int(v['Upper'])-int(v['Lower']))/oldcost for v in records]))
                quality_rows.append(f"{tag} & {label} & -- & -- & -- & -- & -- & 0/{regressions}")
                state=read(path/'status.json');prep=sum(v['wall_seconds'] for key,v in state['resources'].items() if key not in ['moments_query','moments_verify']);query=state['resources']['moments_query']['wall_seconds'];replay=state['resources']['moments_verify']['wall_seconds'];fallback=read(path/'checked_comparison.json')['fallback_active_seconds'];cost_rows.append(f"{tag} & {label} & {prep/60:.3f} & {query/60:.3f} & {replay/60:.3f} & {fallback/60:.3f}")
    if require_complete:assert len(completed)==len(CASES),'Incomplete matched cohort cannot become complete evidence'
    (out/'known_family.tex').write_text(table('Supplementary exact affine measurements with advance knowledge of every future center. The bank includes all evaluated center sets before preprocessing; sets include the deployed reference. These are a different information contract from the main unknown-future-center task. No online-affine speed comparison is inferred. Every assignment and cost is exact within the approved bank.','tab:known-family','lrrrrrr',r'Cohort & $N$ & Workers & Bank & Sets & Min. & Proofs',closed_rows,True),encoding='utf-8')
    if matched:
        (out/'matched_results.tex').write_text(table('Matched unknown-future-center controls against one fixed deployed reference, eight Go workers for every method and fresh fallback. Four distinct proposals per cohort. Preprocessing does not require future candidate coordinates. Diagnostic cases share previously observed fits. Conclusive counts are before fallback; proof counts include paid fallback.','tab:matched-results','lrlrrrrr',r'Cohort & $N$ & Method & Min. & Full/method & Concl. & FB & Proofs',matched,True),encoding='utf-8')
        (out/'matched_quality.tex').write_text(table('Quality in matched unknown-future-center controls. Percentage columns concern different quantities. Agreement, assignment-certified fraction and native cost error are separate from aggregate cost-change proof outputs. Width is the proved change-interval length as a percentage of old exact cost, before fallback. Moments-only has no label predictor or comparable two-sided width. A regression fraction0/0 means no regression was present.','tab:matched-quality','llrrrrrr',r'Cohort & Method & Agr. & Worst & Label cert. & Cost error & Width & Regr.',quality_rows,True),encoding='utf-8')
        (out/'matched_components.tex').write_text(table('Phase costs in minutes, including all recorded attempts. Prep includes fitting where used, commitments, cold key work and original-row proofs; Full/affine compile and setup time is separated from their query phase. Query retains native parameters and residual key I/O. Replay is fresh public-only checking; fallback is fresh Full generation/checking.','tab:matched-components','llrrrr',r'Cohort & Method & Prep & Query & Replay & FB',cost_rows,True),encoding='utf-8')
    else:
        (out/'matched_results.tex').write_text('At this evidence cutoff, no complete matched challenge-cohort comparison is included. The completed128-row controls above do not establish the cost of a nonzero-fallback service or large-catalog proof scaling.\n',encoding='utf-8')
        for name in ['matched_quality','matched_components']:(out/(name+'.tex')).write_text('% No incomplete comparison exported.\n',encoding='utf-8')
    addition=''
    if len(completed)==len(CASES):
        ratios=[checked_refinement(Path(extension['root'])/case['id']/'head79')['full_over_group_active_phase'] for case in CASES]
        addition=f"Matched challenge controls through 1,024 records give Full/method active-cost ratios of {min(ratios):.3f}--{max(ratios):.3f}, including newly generated fallback.\n"
    (out/'abstract_update.tex').write_text(addition,encoding='utf-8')
    if len(completed)==len(CASES):
        (out/'headline_values.tex').write_text('\\newcommand{\\MatchedMinRatio}{'+f'{min(ratios):.2f}'+'}\n\\newcommand{\\MatchedMaxRatio}{'+f'{max(ratios):.2f}'+'}\n',encoding='utf-8')
    models={}
    for name,path in [('completed_original',original),('completed_fashion',fresh)]:
        models[name]=exact_limits(read(path/'input.json')['model']);evidence(path/'checked_comparison.json');evidence(path/'native_checks.json');evidence(path/'public/public_manifest.json')
    original_cost_curve(out,original,family)
    pilots=export_extra(REPO,out,campaign,evidence,table,quality)
    repeat_path=campaign/'timing_repeat/checked_timing_repeat.json';timing_repeat=None
    if repeat_path.exists():
        timing_repeat=evidence(repeat_path);assert timing_repeat['complete'] and timing_repeat['proof_hash_sets_disjoint']
        second=checked_refinement(campaign/'timing_repeat/refined');assert second['active_phase_seconds_including_fitting_and_fallback']==timing_repeat['second_method_seconds']
        rows=[f"{name} & {timing_repeat[prefix+'_full_seconds']/60:.3f} & {timing_repeat[prefix+'_method_seconds']/60:.3f} & {timing_repeat[prefix+'_pair_ratio']:.3f}" for name,prefix in [('First','first'),('Second','second')]]
        (out/'timing_repeat.tex').write_text(table('Two cold proof-pipeline executions on the same fixed fitted MNIST mature workload, eight workers. Each execution generates fresh keys and proofs; proof-hash sets are disjoint (166 method and 80 Full proofs per execution). The representation-fitting component is recorded once and charged in each method total. The second Full execution also serves the paid moments fallback on byte-identical inputs. These are two descriptive ordered pairs, not independent fits or confidence intervals.','tab:timing-repeat','lrrr',r'Execution & Full min. & Head min. & Full/Head',rows),encoding='utf-8')
    else:(out/'timing_repeat.tex').write_text('% Timing repetition excluded until every fresh stage is complete.\n',encoding='utf-8')
    repeats=[];repeat_rows=[]
    for case in CASES:
        if case['id'] not in parents:continue
        parent=Path(parents[case['id']]['ordered_head']);moments=campaign/case['id']/'moments_only';fb=moments/'fallback'
        if not (moments/'checked_comparison.json').exists() or not read(moments/'checked_comparison.json').get('complete') or not (fb/'full/input.json').exists():continue
        if sha(parent/'full/input.json')!=sha(fb/'full/input.json'):continue
        first,_,_=audit_case(parent,'full');second,_,_=audit_case(fb,'full');evidence(moments/'checked_comparison.json')
        times=[first['active_phase_wall_seconds'],second['active_phase_wall_seconds']];ranges=[]
        for variant in ['head79','pca79']:
            r=checked_refinement(Path(extension['root'])/case['id']/variant);ratios=[t/r['active_phase_seconds_including_fitting_and_fallback'] for t in times];ranges.append(f"{min(ratios):.3f}--{max(ratios):.3f}")
        tag='MNIST mature' if 'mature' in case['id'] else ('Fashion early' if 'fashion' in case['id'] else 'MNIST scale')
        repeat_rows.append(f"{tag} & {times[0]/60:.3f} & {times[1]/60:.3f} & {ranges[0]} & {ranges[1]}")
        repeats.append(dict(case=case['id'],full_input_sha256=sha(fb/'full/input.json'),first_seconds=times[0],repeat_seconds=times[1],scope='Second cold Full execution is paid moments fallback on byte-identical inputs; selected method components are not repeated.'))
    (out/'full_repeat.tex').write_text(table('Sensitivity to observed Full timing variation on byte-identical integer task inputs. The second cold Full execution is the paid moments-only fallback, including fresh proofs and public replay; it is not an additional timing campaign. The ranges divide both observed Full totals by each single selected-method measured component sum. They are not confidence intervals or independently repeated paired comparisons.','tab:full-repeat','lrrrr',r'Cohort & First Full min. & Repeat min. & Head ratio range & PCA ratio range',repeat_rows,True) if repeat_rows else '% No complete identical-task Full repetition available at this cutoff.\n',encoding='utf-8')
    for item in pilots:
        base=campaign/item['case']/item['variant'];models[item['case']+'_'+item['variant']]=exact_limits(read(base/'input.json')['model'])
    save(out/'integer_limits.json',models)
    receipt=dict(complete=True,complete_matched_cohorts=completed,all_matched_complete=len(completed)==len(CASES),sources=sources,selected_certificate_rank=79,prediction_rank=63,old_variants_in_main_paper=False,quality_completed=dict(original=qg,fashion=qf),source_scope='Only complete verified comparisons enter tables; native and measured proof evidence are separate.')
    receipt['refinement_pairs']=pilots
    receipt['timing_repeat']=timing_repeat
    receipt['identical_task_full_repeats']=repeats
    receipt['timing_scope']='Selected refinement uses shared measured compact components plus new refinement and residual fallback; not an independent end-to-end repetition. Scheduling changed after the initial Full baseline; see memory_policy_amendment.json.'
    save(out/'export_receipt.json',receipt);return receipt
