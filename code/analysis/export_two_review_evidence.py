"""Completed-only exports for the two-review revision; no proof generation."""
from pathlib import Path
from collections import defaultdict
import argparse,json,math,re,statistics
from campaigns.local_fast_tradeoff import save,sha
from analysis.export_referee_final_paper import table

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'paper_results/referee_final_20261025/generated'
OLD=REPO/'code/validation/outputs/referee_optimized_20261026'


def read(path):return json.loads(Path(path).read_text())


def cleanup_timing_note(root,case):
    path=root/'filesystem_cleanup_interference_20261027.json'
    if path.exists() and case in read(path)['affected_proof_cases']:
        return 'Concurrent user-requested filesystem cleanup overlapped this comparison; raw times retained, excluded from clean speed-ratio conclusions.'
    return None


def quality(rows):
    regressions=[r for r in rows if r['true_change']>0]
    return dict(requests=len(rows),conclusive=sum(r['decision']!='inconclusive' for r in rows),regressions=len(regressions),
        detected=sum(r['decision']=='worsens' for r in regressions),mean_agreement=statistics.mean(r['agreement_percent'] for r in rows),
        worst_agreement=min(r['agreement_percent'] for r in rows),label_coverage=statistics.mean(r['assignment_certified_percent'] for r in rows),
        exact_rows=sum(r['exact_rows'] for r in rows),below99=sum(r['agreement_percent']<99 for r in rows))


def ratio_summary(values):
    values=list(values)
    result=dict(n=len(values),mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values),
        sample_sd=statistics.stdev(values) if len(values)>1 else None)
    if len(values)==3:
        logs=[math.log(v) for v in values];mean=statistics.mean(logs);half=4.302652729911275*statistics.stdev(logs)/math.sqrt(3)
        result.update(geometric_mean=math.exp(mean),conditional_t95=[math.exp(mean-half),math.exp(mean+half)],
            interpretation='Three serial timing repetitions of one fit; conditional log-ratio t interval, not a confidence interval across learning problems.')
    return result


def public_resources(directories):
    proofs=0;proof_bytes=0;key_bytes=0;replay=0.;files=[]
    for folder in directories:
        manifest=folder/'public_manifest.json'
        if not manifest.exists():continue
        values=read(manifest)['sha256'];files.append(manifest)
        for name in values:
            path=folder/name
            if not path.is_relative_to(folder) or not path.is_file() or sha(path)!=values[name]:
                raise ValueError('Public artifact differs from its manifest: '+str(path))
            if path.suffix=='.proof':proofs+=1;proof_bytes+=path.stat().st_size
            elif path.suffix=='.vk':key_bytes+=path.stat().st_size
        for path in folder.glob('*verify*.json'):
            value=read(path)
            if value.get('complete'):replay+=value.get('wall_seconds',0.)
    return dict(proofs=proofs,proof_bytes=proof_bytes,verifying_key_bytes=key_bytes,public_replay_seconds=replay,
        manifests=[dict(path=p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p),sha256=sha(p)) for p in files])


def check_exact_samples(source,samples,backend):
    """Recompute each integer batch and require full nonduplicated coverage."""
    import numpy as np
    x=np.asarray(source['x'],np.int64);n=len(x);seen=defaultdict(list)
    for sample in samples:
        bank=sample['set'] if backend=='PLONK' else int(re.search(r's(\d+)_',sample['file']).group(1))
        indices=sample['indices']
        if not 0<=bank<len(source['reuse_centers']) or any(type(i) is not int or not 0<=i<n for i in indices):
            raise ValueError('Exact control has invalid bank or row index')
        centers=np.asarray(source['reuse_centers'][bank],np.int64)
        truth=int(((x[indices,None,:]-centers[None,:,:])**2).sum(2).min(1).sum())
        if int(sample['cost' if backend=='PLONK' else 'psi'])!=truth:
            raise ValueError('Exact control differs from independently recomputed quantized Full')
        seen[bank].extend(indices)
    if set(seen)!=set(range(len(source['reuse_centers']))) or any(sorted(v)!=list(range(n)) for v in seen.values()):
        raise ValueError('Exact control omits or duplicates catalog coverage')


def completed_groth_control(parent):
    path=parent/'status.json';folder=parent/'full/public_bundle'
    if not path.exists():return None
    status=read(path);case=status.get('cases',{}).get('full',{})
    if not case.get('complete') or not (folder/'verify.json').exists() or not read(folder/'verify.json').get('complete'):return None
    source=read(parent/'full/input.json');q=len(source['queries'][0]);result=read(folder/f'full_reuse_{q}.json')
    check_exact_samples(source,result['samples'],'Groth16');resources=public_resources([folder])
    return dict(backend='Groth16',batch=q,repetition=parent.name,n=source['model']['n'],samples=resources['proofs'],
        active_seconds=case['wall_seconds'],public_replay_seconds=resources['public_replay_seconds'],proof_bytes=resources['proof_bytes'],
        setup_seconds=result['setup_seconds'],peak_rss_bytes=resource_peak(list((parent/'full').glob('*.resource.json'))),
        receipt=path.relative_to(REPO).as_posix(),receipt_sha256=sha(path),clean_timing=True)


def resource_peak(paths):
    peak=0
    for path in paths:
        if not path.exists():continue
        value=read(path)
        if 'resources' in value:values=value['resources'].values()
        else:values=[value]
        for row in values:peak=max(peak,row.get('peak_rss_bytes',0),row.get('peak_process_rss_bytes',0))
    return peak


def cohort_resources(parent,base,refined,label,n):
    result=read(refined/'checked_comparison.json');assert result['complete'] and result['no_wrong_decisions']
    values=public_resources([base/'public',refined/'public',refined/'fallback/full/public_bundle'])
    expected=result['base_public_proofs']+result['new_refinement_proofs']+result['fresh_fallback_proofs']
    if values['proofs']!=expected:raise ValueError(f'Public proof count differs for {label}: {values["proofs"]} != {expected}')
    values.update(label=label,n=n,peak_rss_bytes=resource_peak([base/'status.json',refined/'status.json']+list((refined/'fallback/full').glob('*.resource.json'))),
        seconds=result['active_phase_seconds_including_fitting_and_fallback'])
    full=public_resources([parent/'full/public_bundle']);full.update(label=label.split('/')[0]+'/Full',n=n,
        peak_rss_bytes=resource_peak(list((parent/'full').glob('*.resource.json'))),seconds=result['full_reference_active_seconds'])
    return values,full


def export(root):
    native=root/'learning_native';summary=read(native/'summary.json');assert summary['complete']
    raw=read(native/'records.json')
    assert all(int(v['lower'])<=v['true_change']*int(v['denominator'])<=int(v['upper']) for v in raw)
    assert all(v['decision']=='inconclusive' or v['decision']==('worsens' if v['true_change']>0 else 'improves' if v['true_change']<0 else 'equal') for v in raw)
    current=read(DEST/'optimized_evidence_receipt.json');rows=[];qualities=[]
    for v in current['pooled_confirmation']:
        q=dict(dataset=v['dataset'],basis=v['representation'],**{k:v[k] for k in ['requests','conclusive','mean_agreement','worst_agreement','label_coverage','exact_rows','below99']},regressions=v['regressions'],detected=v['regressions_certified'])
        qualities.append(q)
    groups=defaultdict(list)
    for v in raw:
        if v['n']==1024 and v['proposal']['family']=='new_training_minibatch':groups[v['dataset'],v['representation']].append(v)
    for (dataset,basis),values in groups.items():qualities.append(dict(dataset=dataset,basis=basis,**quality(values)))
    neural_activity=root/'neural_activity';activity_text='% Neural activity diagnostic is unfinished; no result is reported.\n'
    if (neural_activity/'summary.json').exists() and read(neural_activity/'summary.json').get('complete'):
        neural_rows=read(neural_activity/'records.json')
        for basis in ['ordered_head','pca']:
            values=[v for v in neural_rows if v['n']==1024 and v['representation']==basis]
            qualities.append(dict(dataset='har_neural256',basis=basis,**quality(values)))
        q=next(v for v in qualities if v['dataset']=='har_neural256' and v['basis']=='ordered_head')
        accuracy=statistics.mean(v['prototype_classification_accuracy'] for v in neural_rows if v['n']==1024 and v['representation']=='ordered_head')
        activity_text=(r'\paragraph{Neural activity prototype head} '
            'A further diagnostic freezes a561--256 ReLU--6 classifier trained on past500 HAR records '
            '(80epochs, seeds307/331). Class-conditional means of normalized256-dimensional activations '
            'form its prototype head. Incoming labeled minibatches supply six update rates. '
            'This addition follows the neural-text stress observation and is not independently selected confirmation. '
            f"Centroid resolves{q['conclusive']}/{q['requests']} updates and{q['detected']}/{q['regressions']} regressions, "
            f"with{q['mean_agreement']:.3f}\\% mean agreement and{q['worst_agreement']:.3f}\\% worst agreement. "
            f"The exact prototype classifier averages{accuracy:.3f}\\% activity accuracy across these requests. "
            'Activity accuracy is a separate native outcome; the proof concerns quantized nearest-prototype cost.\n')
    (DEST/'review_activity_neural.tex').write_text(activity_text,encoding='utf-8')
    names=dict(mnist='MNIST',fashion_mnist='Fashion',har561='HAR',text_minilm384='Neural text',har_neural256='Neural HAR')
    for q in qualities:
        rows.append(f"{names[q['dataset']]} & {'Centroid' if q['basis']=='ordered_head' else 'PCA'} & {q['conclusive']}/{q['requests']} & {q['detected']}/{q['regressions']} & {q['mean_agreement']:.3f} & {q['worst_agreement']:.3f} & {q['label_coverage']:.3f} & {q['exact_rows']} & {q['below99']}")
    (DEST/'review_learning_quality.tex').write_text(table('Native quality on 1,024-record catalogs. Image rows pool four new seeds and deployed ages1/8; HAR and neural text pool two row-disjoint fitted configurations with six new-training-minibatch proposals each. HAR uses six centers and neural text five. Coverage is before full-catalog fallback. Agreement, individual-label certificates and update decisions concern different outputs. Below99 counts requests missing the working agreement target; 0/0 means no regression occurred.','tab:review-quality','llrrrrrrr',r'Data & Basis & Updates & Regressions & Mean agr. & Worst & Label cert. & Exact rows & Below99',rows,True),encoding='utf-8')
    ablations=read(native/'ablations.json');rows=[]
    for dataset,basis in groups:
        rs=[v for v in ablations if v['fit'].startswith(dataset+'_') and v['representation']==basis]
        cells=[]
        for rank,cap in [(39,0),(63,0),(79,0),(79,4),(79,16),(79,32)]:
            subset=[v for v in rs if v['rank']==rank and v['cap']==cap]
            cells.append(f"{sum(v['decision']!='inconclusive' for v in subset)}/{len(subset)}")
        rows.append(f"{names[dataset]} & {'Centroid' if basis=='ordered_head' else 'PCA'} & "+' & '.join(cells))
    (DEST/'review_rank_cap.tex').write_text(table('Controlled native rank/cap ablation on the first declared fit per added corpus, N128, all six minibatch proposals. Prediction rank63 and precision14 stay fixed; only certificate rank and correction budget change. Each cell gives resolved updates. Ranks39/63/79 use prefixes of the same frozen training basis. These are numerical resolution counts, not measured proof-speed ratios.','tab:review-rank-cap','llrrrrrr',r'Data & Basis & 39/0 & 63/0 & 79/0 & 79/4 & 79/16 & 79/32',rows,True),encoding='utf-8')
    timing=read(DEST/'optimized_timing_receipt.json');ratios=defaultdict(list)
    for v in timing['records']:ratios[v['cohort'],v['method']].append(v['ratio'])
    statistics_rows=[];stats=[]
    for (cohort,method),values in ratios.items():
        if len(values)!=3:continue
        result=ratio_summary(values);stats.append(dict(cohort=cohort,method=method,**result))
        ci=result['conditional_t95'];label='MNIST3/101/8' if cohort=='matched_controls' else 'MNIST5/173/8'
        statistics_rows.append(f"{label} & {method} & 3 & {result['median']:.3f} & {result['sample_sd']:.3f} & {ci[0]:.3f}--{ci[1]:.3f}")
    (DEST/'review_timing_uncertainty.tex').write_text(table('Timing variability conditional on one fit and host. Ratios include all measured active components. SD is across three serial executions; the log-ratio Student t interval is descriptive under a normal-log-ratio assumption with only two degrees of freedom. It is not an interval across datasets or independent learning trials.','tab:review-uncertainty','llrrrr',r'Case & Method & Reps & Median ratio & SD & Conditional95\% interval',statistics_rows,True),encoding='utf-8')
    resources=[];new_timings=[]
    for name,label,n in [('matched_controls','MNIST128',128),('extended/fashion_mnist_k5_seed173_age8_n1024','Fashion1024',1024)]:
        case=OLD/name/'repetition_01';full_added=False
        for method,title in [('ordered_head','Centroid'),('pca','PCA')]:
            values,full=cohort_resources(case/method,case/(method+'_base'),case/(method+'_refined'),label+'/'+title,n)
            if not full_added:resources.append(full);full_added=True
            resources.append(values)
    for target in sorted(root.glob('proof_*'))+([root/'backend_groth_controls'] if (root/'backend_groth_controls').exists() else []):
        if not target.is_dir():continue
        if target.name=='backend_groth_controls':source=root/'backend_matched_inputs'
        else:
            case=next((v for v in read(root/'plan.json')['proof_cases'] if 'proof_'+v['dataset']==target.name),None)
            if case is None:
                if target.name!='proof_har_neural256':continue
                source=root/'neural_activity_inputs/har_neural256_k6_seed307_classmean'
            else:source=root/'learning_inputs'/f"{case['dataset']}_k{case['k']}_seed{case['seed']}_release"
        full_added=False
        for method,title in [('ordered_head','Centroid'),('pca','PCA')]:
            refined=target/(method+'_refined')
            if not (refined/'checked_comparison.json').exists() or not read(refined/'checked_comparison.json').get('complete'):continue
            values,full=cohort_resources(source/method,target/(method+'_base'),refined,target.name+'/'+title,read(source/method/'head_r63/input.json')['model']['n'])
            r=read(refined/'checked_comparison.json')
            timing_note=cleanup_timing_note(root,target.name)
            new_timings.append(dict(case=target.name,method=title,n=values['n'],seconds=values['seconds'],full_seconds=full['seconds'],
                ratio=r['full_over_group_active_phase'],decisions=r['conclusive_without_fallback'],requests=r['proposal_updates'],fallback=r['fallback_updates'],
                clean_timing=timing_note is None,timing_note=timing_note,
                receipt=(refined/'checked_comparison.json').relative_to(REPO).as_posix(),receipt_sha256=sha(refined/'checked_comparison.json')))
            if not full_added:resources.append(full);full_added=True
            resources.append(values)
    rows=[]
    names_case={'proof_har561':'HAR','proof_text_minilm384':'Neural text','proof_har_neural256':'Neural HAR','backend_groth_controls':'Neural text match'}
    for r in resources:
        cohort,method=r['label'].split('/',1);display=names_case.get(cohort,cohort)+'/'+method
        rows.append(f"{display} & {r['proofs']} & {r['proof_bytes']/2**20:.3f} & {r['verifying_key_bytes']/1024:.1f} & {r['public_replay_seconds']:.3f} & {r['peak_rss_bytes']/2**30:.3f}")
    (DEST/'review_resources.tex').write_text(table('Cold public artifacts and resource costs for completed comparisons (first execution per fit). Proof MiB counts cryptographic proof bytes; VK KiB counts exported verification keys. Replay is a fresh public-only verifier pass. RSS is the largest measured prover-process resident set across preprocessing, setup, queries, correction and paid fallback. Public witnesses, model metadata and transfer time are separate from proof bytes.','tab:review-resources','lrrrrr',r'Case/method & Proofs & Proof MiB & VK KiB & Replay s & Peak GiB',rows,True),encoding='utf-8')
    rows=[]
    for v in new_timings:
        label=names_case[v['case']]+(r'$^{\dagger}$' if not v['clean_timing'] else '')
        ratio=f"{v['ratio']:.3f}" if v['clean_timing'] else r'\emph{confounded}'
        rows.append(f"{label} & {v['n']} & {v['method']} & {v['full_seconds']/60:.3f} & {v['seconds']/60:.3f} & {ratio} & {v['decisions']}/{v['requests']} & {v['fallback']}")
    text=(table('Completed post-review learning-workload proof pairs, one execution per fit, matched frozen inputs, eight workers. Totals include preparation, keys, replay, corrections and newly generated fallback; Full uses batch8. Ratios below1 retain cases where direct Full is faster. A dagger marks overlap with concurrent filesystem cleanup: raw times remain, but speed ratios are not interpreted. These are separate from native1024-record quality.','tab:review-new-timings','lrlrrrrr',r'Data & N & Method & Full min. & Method min. & Ratio & Decisions & FB',rows,True)
        if rows else '% No new learning-workload proof pair has completed.\n')
    (DEST/'review_new_proof_timings.tex').write_text(text,encoding='utf-8')
    backends=[]
    interruptions=read(root/'backend_timing_interruption_amendment.json').get('interrupted_repetitions',[]) if (root/'backend_timing_interruption_amendment.json').exists() else []
    backend_paths=sorted((root/'backend').glob('repetition_*/result.json'))+sorted((root/'followups').glob('plonk_batch[248]/result.json'))
    for f in backend_paths:
        public=f.parent/'public_verify.json'
        if not public.exists() or not read(f).get('complete') or not read(public).get('complete'):continue
        v=read(f);samples=v['samples'];n=read(root/'backend_workload_selection.json')['n']
        source=read(root/'backend_matched_inputs/ordered_head/full/input.json')
        check_exact_samples(source,samples,'PLONK')
        for sample in samples:
            if sha(f.parent/sample['file'])!=sample['proof_sha256']:raise ValueError('PLONK proof hash changed')
        if read(public)['verified']!=len(samples):raise ValueError('Incomplete PLONK public replay')
        batch=max(len(s['indices']) for s in samples)
        backends.append(dict(backend='PLONK',batch=batch,repetition=f.parent.name,n=n,complete=True,samples=len(samples),wall_seconds=v['wall_seconds'],
            active_seconds=v['wall_seconds']+read(public)['wall_seconds'],
            clean_timing=f.parent.name not in interruptions,sleep_interrupted=f.parent.name in interruptions,
            proof_bytes=sum(s['proof_bytes'] for s in samples),public_replay_seconds=read(public)['wall_seconds'],
            srs_loading_seconds=v['srs_generation_or_loading_seconds'],circuit_setup_seconds=v['circuit_setup_seconds'],
            constraints=v['constraints'],benchmark_only_unsafe_srs=True,receipt=f.relative_to(REPO).as_posix(),receipt_sha256=sha(f)))
    if backends:
        rows=[f"PLONK & {v['batch']} & {v['repetition'].replace('_',r'\_')} & {v['samples']} & "+(f"{v['active_seconds']/60:.3f}" if v['clean_timing'] else r'\emph{interrupted}')+f" & {v['public_replay_seconds']:.3f} & {v['proof_bytes']/2**20:.3f} & {v['circuit_setup_seconds']:.3f}" for v in backends]
        caption=f"Completed exact Full controls on the same {backends[0]['n']}-record neural-text catalog and seven banks. Every exact batch and complete row coverage are independently checked; public proofs are replayed. Active time includes replay. Test-only cached KZG parameters are unsafe for deployment; creation is separate, while runs charge loading, circuit setup and proving. Interrupted timing is excluded from comparisons. Groth16 batch1 separates backend from batching; larger feasible PLONK batches are retained when completed. Repetitions share one fitted model."
        text=table(caption,'tab:review-backend','lrlrrrrr',r'Backend & Batch & Execution & Proofs & Active min. & Replay s & Proof MiB & Setup s',rows,True)
    else:text='% No real neural-feature PLONK workload has finished verification.\n'
    groth_controls=[]
    for parent in [root/'backend_matched_inputs/ordered_head']+sorted((root/'followups').glob('groth_batch1_r*')):
        value=completed_groth_control(parent)
        if value:groth_controls.append(value)
    if backends and groth_controls:
        extra=[f"Groth16 & {v['batch']} & {v['repetition'].replace('_',r'\_')} & {v['samples']} & {v['active_seconds']/60:.3f} & {v['public_replay_seconds']:.3f} & {v['proof_bytes']/2**20:.3f} & {v['setup_seconds']:.3f}" for v in groth_controls]
        text=table(caption,'tab:review-backend','lrlrrrrr',r'Backend & Batch & Execution & Proofs & Active min. & Replay s & Proof MiB & Setup s',rows+extra,True)
    (DEST/'review_backend.tex').write_text(text,encoding='utf-8')
    sources=[root/'plan.json',native/'summary.json',native/'records.json',native/'fit_records.json',native/'ablations.json']+list((root/'data').glob('*_source.json'))
    for extra in ['backend_timing_interruption_amendment.json','windows_sleep_events_20261027.json','neural_activity_amendment.json','bounded_followup_plan.json',
                  'neural_activity/summary.json','neural_activity/records.json','neural_activity/fit_records.json',
                  'filesystem_cleanup_interference_20261027.json']:
        if (root/extra).exists():sources.append(root/extra)
    result=dict(complete=True,native_study_complete=True,qualities=qualities,timing_uncertainty=stats,resources=resources,completed_plonk_runs=backends,completed_groth_backend_controls=groth_controls,new_proof_pairs=new_timings,
        source_sha256={p.relative_to(REPO).as_posix():sha(p) for p in sources},all_intervals_contain_truth=True,
        source_review_revision='Post-review additions; all new-training-minibatch requests retained, including failures. No universal99% or complete regression-detection claim.')
    for path in DEST.glob('review_*.tex'):
        text=path.read_text()
        text=re.sub(r'\b(resolves|and|with|past|on|batch|rank|cap|precision|working|all|at|Below|below|steps|ages|seed|seeds|N)(?=\d)', r'\1 ', text)
        text=text.replace('Groth 16','Groth16')
        text=re.sub(r'(?<=\d)(?=(?:epochs|records|updates|seconds)\b)', ' ', text)
        path.write_text(text,encoding='utf-8')
    save(DEST/'two_review_evidence_receipt.json',result);save(REPO/'output/two_review_results_20261027.json',result)
    print(json.dumps(dict(native_fits=summary['representation_fits'],quality_rows=len(qualities),completed_backend_runs=len(backends),resource_rows=len(resources))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=REPO/'code/validation/outputs/referee_two_reviews_20261027');a=p.parse_args();export(a.root.resolve())
