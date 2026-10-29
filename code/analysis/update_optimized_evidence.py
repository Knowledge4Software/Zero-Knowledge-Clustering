"""Export completed frozen-profile native and matched-proof evidence only."""
from collections import defaultdict
from pathlib import Path
import argparse,json,re,statistics
from campaigns.local_fast_tradeoff import save,sha
from analysis.export_main_rank79_paper import table

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'code/validation/outputs/referee_optimized_20261026'
DEST=REPO/'paper_results/referee_final_20261025/generated'


def read(path):return json.loads(path.read_text())


def case_label(name):
    if name=='matched_controls':return 'MNIST 3/101/8'
    match=re.fullmatch(r'(fashion_mnist|mnist)_k(\d+)_seed(\d+)_age(\d+)_n\d+',name)
    if match:
        dataset,k,seed,age=match.groups()
        return ('Fashion' if dataset=='fashion_mnist' else 'MNIST')+' '+k+'/'+seed+'/'+age
    return name.replace('_',r'\_')


def quality(rows):
    regressions=[v for v in rows if v['true_change']>0]
    return dict(requests=len(rows),initial=sum(v['initial_decision']!='inconclusive' for v in rows),
        conclusive=sum(v['decision']!='inconclusive' for v in rows),regressions=len(regressions),
        regressions_certified=sum(v['decision']=='worsens' for v in regressions),
        mean_agreement=statistics.mean(v['agreement_percent'] for v in rows),
        worst_agreement=min(v['agreement_percent'] for v in rows),
        label_coverage=statistics.mean(v['assignment_certified_percent'] for v in rows),
        mean_width=statistics.mean(v['width_percent_old_cost'] for v in rows),
        exact_rows=sum(v['exact_rows'] for v in rows),below99=sum(v['agreement_percent']<99 for v in rows))


def export_timings(root):
    cohorts=[root/'matched_controls']+sorted((root/'extended').glob('*')) if (root/'extended').exists() else [root/'matched_controls']
    records=[]
    for cohort in cohorts:
        if not (cohort/'plan.json').exists():continue
        for repetition in sorted(cohort.glob('repetition_*')):
            paths=[repetition/(m+'_refined/checked_comparison.json') for m in ['ordered_head','pca']]
            paths.append(repetition/'moments_only/checked_comparison.json')
            completed=[(method,read(path),path) for method,path in zip(['Centroid','PCA','Moments-only'],paths) if path.exists() and read(path).get('complete')]
            if not completed:continue
            seconds=completed[0][1]['full_reference_active_seconds']
            assert all(v['full_reference_active_seconds']==seconds for _,v,_ in completed)
            for method,v,path in completed:
                records.append(dict(cohort=cohort.name,repetition=repetition.name,method=method,
                    n=read(cohort/'plan.json')['catalog_records'],seconds=v['active_phase_seconds_including_fitting_and_fallback'],
                    full_seconds=seconds,ratio=v['full_over_group_active_phase'],decisions=v['conclusive_without_fallback'],requests=v['proposal_updates'],
                    fallback=v['fallback_updates'],receipt=path.relative_to(REPO).as_posix(),receipt_sha256=sha(path)))
    groups=defaultdict(list)
    for v in records:groups[v['cohort'],v['method']].append(v)
    rows=[]
    for (name,method),values in sorted(groups.items()):
        times=[v['seconds']/60 for v in values];full=[v['full_seconds']/60 for v in values]
        requests=sum(v['requests'] for v in values);decisions=sum(v['decisions'] for v in values);fallback=sum(v['fallback'] for v in values)
        label=case_label(name)
        rows.append(f"{label} & {values[0]['n']} & {method} & {len(values)} & {statistics.median(full):.3f} & {statistics.median(times):.3f} & {min(times):.3f}--{max(times):.3f} & {statistics.median(v['ratio'] for v in values):.3f} & {decisions}/{requests} & {fallback}")
    if rows:
        caption='Completed method--Full pairs, eight workers, frozen fitted models supplied as inputs; fitting is excluded consistently. Case identifies dataset and $k$/seed/reference Lloyd step. Reps counts completed pairs for that method, not completion of every baseline. Ratios include preprocessing, keys, recorded attempts, generation, public replay and newly paid fallback (FB). Repeated decisions share the same fit and proposals.'
        text=table(caption,'tab:optimized-proofs','lrlrrrrrrr',r'Case & $N$ & Method & Reps & Full min. & Method min. & Range & Full/method & Decisions & FB',rows,True)
    else:text='% No new matched repetition has completed; no provisional timing is exported.\n'
    (DEST/'optimized_proofs.tex').write_text(text,encoding='utf-8')
    sets=defaultdict(set)
    for v in records:sets[v['cohort'],v['repetition']].add(v['method'])
    complete=sum(methods=={'Centroid','PCA','Moments-only'} for methods in sets.values())
    save(DEST/'optimized_timing_receipt.json',dict(complete_comparisons_only=True,records=records,completed_method_pairs=len(records),completed_repetitions=complete))
    baselines=[];rows=[]
    for cohort in cohorts:
        case=cohort/'repetition_01'
        online=case/'online_affine/checked_comparison.json'
        if online.exists() and read(online).get('complete'):
            value=read(online);baselines.append(dict(cohort=cohort.name,method='Online affine',**value))
            rows.append(f"{case_label(cohort.name)} & Online affine & {value['full_seconds']/60:.3f} & {value['active_seconds']/60:.3f} & {value['full_over_affine']:.3f}")
        backend=case/'plonk_full/result.json'
        # The backend's exact result and public replay must both complete.
        replay=case/'plonk_full/public_verify.json'
        if backend.exists() and replay.exists() and read(backend).get('complete') and read(replay).get('complete'):
            queue=read(cohort/'queue.json');seconds=sum(v['wall_seconds'] for v in queue['attempts'] if v['phase'] in ['plonk_full','plonk_full_verify'])
            full=next((v['full_seconds'] for v in records if v['cohort']==cohort.name),None)
            if full is not None:
                baselines.append(dict(cohort=cohort.name,method='PLONK test SRS',seconds=seconds,full_seconds=full))
                rows.append(f"{case_label(cohort.name)} & PLONK test SRS & {full/60:.3f} & {seconds/60:.3f} & {full/seconds:.3f}")
    if rows:
        caption='Completed exact controls, one execution per fit and runtime setting. Online affine rebuilds for each currently disclosed bank with no future-center bank. Full/control compares matched active costs including fresh public replay.'
        if any(v['method']=='PLONK test SRS' for v in baselines):caption+=' PLONK rows use an unsafe test SRS for backend timing only.'
        (DEST/'optimized_exact_baselines.tex').write_text(table(caption,'tab:optimized-exact','llrrr',r'Case & Exact control & Full min. & Control min. & Full/control',rows,True),encoding='utf-8')
    else:(DEST/'optimized_exact_baselines.tex').write_text('% No additional exact control is complete.\n',encoding='utf-8')
    save(DEST/'optimized_exact_baseline_receipt.json',dict(completed_only=True,records=baselines))
    return records


def export(root=ROOT):
    summary=root/'native/summary.json'
    if not summary.exists() or not read(summary)['complete']:raise RuntimeError('Do not export unfinished native confirmation')
    raw=read(root/'native/records.json');fits=read(root/'native/fit_records.json');meta={(v['fit'],v['representation']):v for v in fits}
    assert all(int(v['lower'])<=v['true_change']*int(v['denominator'])<=int(v['upper']) for v in raw)
    assert all(v['decision']=='inconclusive' or v['decision']==('worsens' if v['true_change']>0 else 'improves' if v['true_change']<0 else 'equal') for v in raw)
    groups=defaultdict(list)
    for v in raw:
        f=meta[v['fit'],v['representation']]
        groups[f['dataset'],f['k'],v['n'],v['representation']].append(v)
    def label(dataset):return 'Fashion' if dataset=='fashion_mnist' else 'MNIST'
    def basis(method):return 'Centroid' if method=='ordered_head' else 'PCA'
    rows=[];pooled=[]
    for (d,k,n,m),values in sorted(groups.items()):
        if k!=5 or n!=1024:continue
        q=quality(values);pooled.append(dict(dataset=d,k=k,n=n,representation=m,**q))
        rows.append(f"{label(d)} & {basis(m)} & {q['initial']}/{q['requests']} & {q['conclusive']}/{q['requests']} & {q['regressions_certified']}/{q['regressions']} & {q['mean_agreement']:.3f} & {q['worst_agreement']:.3f} & {q['label_coverage']:.3f} & {q['exact_rows']}")
    (DEST/'optimized_confirmation.tex').write_text(table(
        'Frozen-profile native confirmation on four new seed/splits per dataset, $k=5$, $N=1024$, deployed ages 1/8. Initial and refined columns count update signs; regression coverage conditions on naturally positive exact change. Agreement and label coverage are percentages. Exact rows counts bounded corrections. These are native results, not new proof timings; queries and ages share fits.',
        'tab:optimized-confirmation','llrrrrrrr',r'Data & Basis & Initial & Refined & Regressions & Mean agr. & Worst agr. & Label cert. & Exact rows',rows,True),encoding='utf-8')
    rows=[];sweep=[]
    for (d,k,n,m),values in sorted(groups.items()):
        if n!=1024:continue
        q=quality(values);sweep.append(dict(dataset=d,k=k,n=n,representation=m,**q))
        rows.append(f"{label(d)} & {k} & {basis(m)} & {q['conclusive']}/{q['requests']} & {q['regressions_certified']}/{q['regressions']} & {q['worst_agreement']:.3f} & {q['label_coverage']:.3f} & {q['mean_width']:.3f} & {q['below99']}")
    (DEST/'optimized_centers.tex').write_text(table(
        'Native center-count sweep. Four seeds contribute to $k=5$; $k=3,10,20$ use seed 173. All declared distinct candidates are retained. Width is percent of deployed cost after bounded refinement; Below 99 counts requests missing the working agreement target. A 0/0 regression entry means no natural regression was present.',
        'tab:optimized-centers','lrlrrrrrr',r'Data & $k$ & Basis & Decisions & Regressions & Worst agr. & Label cert. & Width & Below 99',rows,True),encoding='utf-8')
    scale=defaultdict(list)
    for v in raw:
        f=meta[v['fit'],v['representation']]
        if f['dataset']=='mnist' and f['k']==5 and f['fit_seed']==173:scale[f['age'],v['n'],v['representation']].append(v)
    rows=[];scaling=[]
    for (age,n,m),values in sorted(scale.items()):
        q=quality(values);scaling.append(dict(age=age,n=n,representation=m,**q))
        rows.append(f"{age} & {n:,} & {basis(m)} & {q['conclusive']}/{q['requests']} & {q['mean_agreement']:.3f} & {q['worst_agreement']:.3f} & {q['label_coverage']:.3f} & {q['mean_width']:.3f} & {q['exact_rows']}")
    (DEST/'optimized_scale.tex').write_text(table(
        'Native scaling on new MNIST seed-173 fits, using real held-out prefixes disjoint from 3,000 training records. Age 1 is the primary predeclared scale series; the additional age-8 prefixes evaluated by the implementation are retained as paired diagnostics. Both bases share the catalog and updates; prefixes are correlated. This measures numerical quality up to 16,384 records, not completed large-catalog proof speed.',
        'tab:optimized-scale','rrlrrrrrr',r'Age & $N$ & Basis & Decisions & Mean agr. & Worst agr. & Label cert. & Width & Exact rows',rows,True),encoding='utf-8')
    kernels=read(root/'kernel_benchmark/checks.json');rows=[]
    for v in kernels['results']:
        name={'integer_stats':'Residual statistics','center_parameters':'Center parameters','complete_native_bounds':'Complete native bounds'}[v['kernel']]
        rows.append(f"{name} & {statistics.median(v['seconds']['baseline']):.4f} & {statistics.median(v['seconds']['optimized']):.4f} & {v['median_ratio']:.2f}")
    (DEST/'optimized_kernels.tex').write_text(table('Implementation-only paired native microbenchmarks, same 128-record catalog and exact integer outputs, three alternating-order repetitions. Times are median seconds. Ratios do not measure proof generation or deployment speed.','tab:optimized-kernels','lrrr',r'Kernel & Previous & Optimized & Ratio',rows),encoding='utf-8')
    sources=[summary,root/'native/records.json',root/'native/fit_records.json',root/'native/protocol.json',root/'study_plan.json',root/'plan.json',root/'kernel_benchmark/checks.json',root/'preflight_compact/checks.json',root/'preflight_refinement/checks.json']
    scope_path=root/'laptop_scope_amendment_20261027.json'
    if scope_path.exists():sources.append(scope_path)
    receipt=dict(complete=True,native_only=True,large_scale_proof_results_claimed=False,representation_fits=len(fits),split_configurations=8,
        pooled_confirmation=pooled,center_sweep=sweep,scaling=scaling,source_sha256={p.relative_to(REPO).as_posix():sha(p) for p in sources})
    timings=export_timings(root)
    receipt.update(native_only=not bool(timings),matched_timing_records=timings,
        large_scale_proof_results_claimed=any(v['n']>1024 for v in timings))
    if scope_path.exists():receipt['user_execution_scope']=read(scope_path)
    save(DEST/'optimized_evidence_receipt.json',receipt)
    report=['# Optimized frozen-profile evidence, 26 October 2026','',
        'The selected scientific profile is unchanged: predictor63, certificate79, precision14, at most16 width-ordered exact rows.',
        'Native quality is separated from actual proof performance. No superiority over unmeasured baselines is inferred.','',
        '| Dataset | Basis | Updates certified | Regressions certified | Mean agreement | Worst agreement | Label coverage | Below99 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for v in pooled:report.append(f"| {v['dataset']} | {v['representation']} | {v['conclusive']}/{v['requests']} | {v['regressions_certified']}/{v['regressions']} | {v['mean_agreement']:.3f}% | {v['worst_agreement']:.3f}% | {v['label_coverage']:.3f}% | {v['below99']} |")
    report+=['',f'New completed method--Full timing pairs: {len(timings)}. A complete repetition requires all three certificate/moments methods.',
        'Optimizations preserve mathematical outputs. Quality differences between old and new fitted workloads are not attributed to code optimization.',
        'Stress limits: Fashion k10 resolves6/8 updates and2/3 natural regressions with both bases; the other2 need Full. MNIST k20 includes1 centroid request below99% agreement.',
        'At16384 real records (primary MNIST age1), centroid resolves4/4 updates; mean agreement99.786%, worst99.762%, label coverage94.958%. These are native, not proof timing, results.',
        'All source records, inconclusive outcomes, below-target cases and resource exclusions remain retained.','Timing/control progress is in code/validation/outputs/referee_optimized_20261026/queue.json.']
    if scope_path.exists():report+=['','On27October the user limited further proof generation to at most1024 records. The4096/16384 proof benchmarks are deferred for laptop practicality, after partial results; their completed native quality results and the original study plan are retained. This is an execution-scope change, not a completed scale proof or an outcome-based deletion.']
    (REPO/'output/optimized_results_20261026.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    save(REPO/'output/optimized_results_20261026.json',receipt)
    print(json.dumps(dict(complete=True,representation_fits=len(fits),records=len(raw),confirmation=pooled),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args();export(a.root.resolve())
