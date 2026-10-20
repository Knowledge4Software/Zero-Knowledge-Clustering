"""Validate the completed campaign and add measured conclusions to its report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.laptop_comparison_report import report, table
from campaigns.laptop_comparison import atomic_json, fingerprint


def finalize(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['complete'], 'Campaign still incomplete'
    live=fingerprint(Path(manifest['plan'][0]['binary']))
    for key in ['version','binary_sha256','archive_index_sha256']:
        assert live[key]==manifest['identity'][key]
    for name,sha in manifest['identity']['sources'].items():
        candidates=[root/'sources'/name]
        if Path(name).name=='laptop_comparison.py' and Path(name).parent.name=='experiments':
            candidates.append(root/'source_revisions/memory_limit_fix/worker_after.py')
        assert any(p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sha for p in candidates), name
    assert all((root/'jobs'/j['id']/'done.json').exists() for j in manifest['plan'])
    report(root)
    geometry=pd.read_csv(root/'geometry_raw.csv')
    summary=pd.read_csv(root/'geometry_summary.csv')
    integer=pd.read_csv(root/'precision_raw.csv')
    circuits=pd.read_csv(root/'circuits_raw.csv')
    attempted_proofs=circuits[circuits.kind=='prove']
    proofs=attempted_proofs[attempted_proofs.status=='ok']
    assert not (circuits.status=='error').any(), 'Unresolved circuit error'
    assert summary.complete_cell.all(), 'Incomplete geometry cell'
    assert (geometry[geometry.method=='full'].saved_full_agreement==1).all()
    assert len(proofs) and attempted_proofs.status.isin(['ok','resource_limit']).all()
    assert proofs.verify_ok.all() and (proofs.successful_proofs==3).all()
    for path in (root/'jobs').glob('proof_*/records/*.json'):
        result=json.loads(path.read_text());raw=result['raw']
        if not result['ok']:
            assert result['failure_kind'] in ['timeout','memory_limit','oom']
            continue
        assert len(raw['prove_seconds_samples'])==3 and len(raw['verify_seconds_samples'])==3
        assert not raw['solve_only'] and not raw['diagnostic_prefix']
    exclusions=[]
    for folder in (root/'jobs').glob('proof_*'):
        job=json.loads((folder/'job.json').read_text())
        assert job['repetitions']==3
        inputs=[json.loads(p.read_text()) for p in (folder/'instances').glob('*.json')]
        for inst in inputs:
            assert not inst['solve_only'] and not inst['diagnostic_prefix']
            for key in ['x','centroids','c_theta']:
                assert inst[key]==inputs[0][key], f'Unmatched proof inputs: {key}'
        if len({(i['c_x'],i['r_x']) for i in inputs})!=1:
            exclusions.append(dict(job_id=folder.name,reason='commitment_opening_changed_on_resume'))
        conditions=set()
        for path in (folder/'rows').glob('*.json'):
            row=json.loads(path.read_text())
            conditions.add((row.get('rss_limit_gib','legacy'),row.get('go_memory_limit','legacy'),row.get('threads',4)))
        if len(conditions)>1:
            exclusions.append(dict(job_id=folder.name,reason='memory_or_thread_budget_changed_on_resume'))
    excluded_jobs={item['job_id'] for item in exclusions}
    pd.DataFrame(exclusions,columns=['job_id','reason']).to_csv(root/'comparison_exclusions.csv',index=False)
    native=pd.read_csv(root/'native_affine/measurements.csv')
    native_check=json.loads((root/'native_affine/validation.json').read_text())
    supplemental=Path(__file__).resolve().parents[1]/'experiments/laptop_affine_control.py'
    assert hashlib.sha256(supplemental.read_bytes()).hexdigest()==native_check['source_sha256']
    assert native_check['complete'] and len(native_check['replays'])==8
    assert native.cost_check_ok.all()
    validation=dict(complete=True,jobs=len(manifest['plan']),geometry_rows=len(geometry),
        integer_rows=len(integer),circuit_rows=len(circuits),
        verified_measured_proofs=int(proofs.successful_proofs.sum()),actual_proof_controls=len(proofs),
        resource_limited_proof_controls=int((attempted_proofs.status=='resource_limit').sum()),
        bounded_resource_controls=int((circuits.status=='resource_limit').sum()),
        unresolved_errors=int((circuits.status=='error').sum()),native_affine_controls=len(native),
        independent_dataset_replays=len(native_check['replays']),
        native_affine_label_mismatch_rows=native_check['label_mismatch_rows'],
        proof_groups_excluded_from_speed_ratios=sorted(excluded_jobs),
        default_binary_sha256=hashlib.sha256(Path('zk/bin/commitjl-prove.exe').read_bytes()).hexdigest(),
        finalizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(root/'validation.json',validation)
    snapshot=root/'supplemental_sources';snapshot.mkdir(exist_ok=True)
    (snapshot/'laptop_affine_control.py').write_bytes(supplemental.read_bytes())
    (snapshot/'finalize_laptop_comparison.py').write_bytes(Path(__file__).read_bytes())
    pairs=pd.read_csv(root/'pair_workload/measurements.csv')
    pair_check=json.loads((root/'pair_workload/validation.json').read_text())
    pair_source=Path(__file__).resolve().parents[1]/'experiments/laptop_pair_workload.py'
    assert pair_check['complete']
    assert hashlib.sha256(pair_source.read_bytes()).hexdigest()==pair_check['source_sha256']
    assert (pairs[pairs.method=='full'].pair_q95==0).all()
    (snapshot/'laptop_pair_workload.py').write_bytes(pair_source.read_bytes())
    validation['downstream_pair_controls']=len(pairs)
    atomic_json(root/'validation.json',validation)
    dims=geometry.groupby('dataset')[['n','d']].first()
    quality=[];cpu=[]
    for dataset in sorted(geometry.dataset.unique()):
        entry=dict(dataset=dataset,n=int(dims.loc[dataset,'n']),d=int(dims.loc[dataset,'d']))
        for method,prefix in [('hybrid_r4','hybrid'),('pca','pca')]:
            eligible=summary[(summary.dataset==dataset)&(summary.k==5)&
                (summary.method==method)&summary.selectable].sort_values(['total_m','q'])
            if len(eligible):
                row=eligible.iloc[0]
                entry[prefix+'_m']=int(row.total_m)
                entry[prefix+'_ari']=row.ari_mean
                entry[prefix+'_pair_q95']=row.pair_q95_mean
                entry[prefix+'_cost_error_worst']=row.projected_cost_relative_error_max
                if prefix=='hybrid':
                    n=native[(native.dataset==dataset)&(native.method==method)&
                        (native.total_m==row.total_m)&(native.q==row.q)]
                    if len(n)==2:
                        full=native[(native.dataset==dataset)&(native.method=='full')]
                        pca=native[(native.dataset==dataset)&(native.method=='pca')&(native.total_m==row.total_m)]
                        cpu.append(dict(dataset=dataset,total_m=int(row.total_m),q=int(row.q),
                            full_ms=1000*full.seconds_median.mean(),hybrid_ms=1000*n.seconds_median.mean(),
                            pca_same_m_ms=1000*pca.seconds_median.mean() if len(pca) else np.nan,
                            hybrid_speedup_vs_full=n.speedup_vs_full.mean()))
            else:
                for suffix in ['m','ari','pair_q95','cost_error_worst']:entry[prefix+'_'+suffix]=np.nan
        quality.append(entry)
    quality=pd.DataFrame(quality);cpu=pd.DataFrame(cpu)
    quality.to_csv(root/'quality_decision_k5.csv',index=False)
    cpu.to_csv(root/'qualified_native_comparison.csv',index=False)
    precision=integer[integer.method=='integer_hybrid'].groupby(['dataset','scale'])[
        ['ari','ari_integer_full','projected_cost_relative_error']].agg(['min','max'])
    precision.columns=['_'.join(c) for c in precision.columns]
    precision=precision.reset_index()
    precision.to_csv(root/'integer_hybrid_screen.csv',index=False)
    reduced=proofs[proofs.method.isin(['full','pca','hybrid_shared','hybrid_lookup'])][
        ['job_id','method','constraints','prove_seconds_median','single_run_total']]
    reduced=reduced.copy()
    reduced['timing_comparable']=~reduced.job_id.isin(excluded_jobs)
    speedups=pd.read_csv(root/'proof_speedups.csv')
    speedups['timing_comparable']=~speedups.job_id.isin(excluded_jobs)
    speedups.loc[~speedups.timing_comparable,
        [c for c in speedups.columns if 'speedup_vs_' in c]]=np.nan
    speedups.to_csv(root/'proof_speedups.csv',index=False)
    prefix=pd.read_csv(root/'constraint_breakdown.csv')
    full=prefix[prefix.method=='full'];total=int(full[full.stage=='complete'].cumulative_constraints.iloc[0])
    data=int(full[full.stage=='data'].increment.iloc[0]);centers=int(full[full.stage=='theta'].increment.iloc[0])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    stages=['range','theta','data','matrix','projection','complete']
    stack=prefix.pivot(index='method',columns='stage',values='increment').reindex(columns=stages)
    stack.plot.barh(stacked=True,ax=axes[0,0],width=.75)
    axes[0,0].set_title('Approximate circuit decomposition: n8,d32,k5')
    axes[0,0].set_xlabel('Constraints');axes[0,0].legend(fontsize=7,loc='lower right')
    plotted=proofs[proofs.method.isin(['full','pca','hybrid_shared','hybrid_lookup'])&
        (~proofs.job_id.isin(excluded_jobs))].pivot(
        index='job_id',columns='method',values='prove_seconds_median')
    plotted.index=plotted.index.str.removeprefix('proof_')
    plotted.plot.bar(ax=axes[0,1]);axes[0,1].set_title('Verified proofs: median of three')
    axes[0,1].set_xlabel('n_d_totalm_k_q');axes[0,1].set_ylabel('Prove seconds')
    axes[0,1].tick_params(axis='x',rotation=0);axes[0,1].legend(fontsize=7)
    if len(cpu):
        axes[1,0].barh(cpu.dataset,cpu.hybrid_speedup_vs_full)
        axes[1,0].axvline(1,color='black',ls='--')
    axes[1,0].set_title('Qualified k5 hybrid vs Full: BLAS native control')
    axes[1,0].set_xlabel('Full time / hybrid time; above 1 means hybrid faster')
    easy=geometry[(geometry.dataset=='gaussian_easy')&(geometry.k==5)&
        ((geometry.method.isin(['full','protected_only']))|
        ((geometry.total_m==20)&geometry.method.isin(['pca','hybrid_r4','sparse'])))].groupby('method')[
            ['ari','projected_cost_ratio']].mean()
    easy.plot.bar(ax=axes[1,1]);axes[1,1].set_title('Easy Gaussian: labels vs objective (m20)')
    axes[1,1].set_xlabel('Method; Full d256, protected-only r4');axes[1,1].tick_params(axis='x',rotation=20)
    axes[1,1].legend(fontsize=7)
    fig.tight_layout();fig.savefig(root/'diagnosis.png',dpi=160);plt.close(fig)
    intro=['## Measured outcome and next decision',
        '![Measured diagnosis](diagnosis.png)',
        f'The finite laptop plan completed: {len(geometry):,} whole-subset geometry rows, {len(integer):,} exact-integer screening rows and {validation["verified_measured_proofs"]} verified measured proofs. Resource-limited controls are recorded explicitly. These results support improved assignment and objective preservation, but do not establish simultaneous superiority over PCA in proof time and geometry.',
        'For k=5, the table selects the smallest tested compressed configuration passing every geometry threshold on both saved starts and all three fresh maps (PCA is deterministic). Blank cells mean no tested compressed configuration passed, or the archived PCA rank did not permit the required dimension. Errors below are for squared distances/costs. Pair Q95 is averaged over runs; cost error is the worst run. Qualification is exploratory on fixed subsets and is not a formal global JL guarantee.',
        table(quality),
        '## Finite-precision limitation',
        'Full-span protection preserves fixed-center decisions in exact real arithmetic. Rounded integer bases do not provide that unconditional guarantee. This screening table reports the minimum ARI over k=5/10, dimensions20/64/128 and three maps on the first min(128,n) points. It also separates agreement with quantized Full from agreement with the original float Full. In particular, T=4096 still has a Gisette case below ARI0.99; basis precision must be chosen from margin/rounding checks before claiming exact preservation in the proof implementation.',
        table(precision),
        f'At n=8,d=32,k=5, original-data authentication adds {data:,} constraints, about {100*data/total:.1f}% of the Full complete circuit; center authentication adds {centers:,}, about {100*centers/total:.1f}%. Prefix differences include deferred batching effects. This identifies authentication as the main measured bottleneck rather than final low-dimensional assignment.',
        'The real-shape n=2,d=784 MNIST controls hit the deliberately conservative ~3 GiB Go cap before a complete compiled count was returned. They are feasibility limits of this local budget, not successful proofs or evidence that more RAM would make the method faster. Other laptop applications remained protected by the watchdog.',
        '## Fairer native CPU control',
        'A supplementary experiment uses BLAS affine scores, first-index argmax and a directly computed objective for Full, PCA and the hybrid alike. Timings include projection plus assignment and objective, after a full-size warm-up, with five measurements and four threads. The table averages the two start-specific medians. Basis/map construction and cached PCA fitting are excluded; their original measurements are retained. Thus the earlier difference-based Full kernel is not used to infer a speed win over optimized Full.',
        'These supplementary CPU controls ran after available RAM declined substantially. They are implementation diagnostics under the observed laptop load, not publication-grade idle-machine performance estimates. Repeated warm samples do not remove memory pressure, execution-order effects or thermal drift.',
        table(cpu) if len(cpu) else 'No qualified hybrid configuration had a corresponding bounded native timing.',
        'PCA timings here use the same output dimension as the hybrid. A fast same-size PCA row may fail the geometry thresholds; compare the qualification table before treating it as an equally accurate alternative. Native controls use only map100, whereas quality qualification requires maps100/101/102. Five-neighbor recall is exported separately and can be poor even when sampled Q95 and cost pass; no neighborhood-quality guarantee is claimed.',
        '## Downstream workload: compression followed by many pair distances',
        'For each dataset with a qualified k5 full-rank hybrid, a deterministic sample min(1024,n) is projected once and all unordered pairs are evaluated using BLAS. Three warm timing samples include projection and the full pair matrix; map/basis fitting remains excluded. This measures an explicit workload where compression may repay its preprocessing, separately from cheap fixed-center assignment or proof generation. Squared pair-distance Q95 and the sum-of-pair-cost ratio are computed on every unordered pair in this sample. Saved indices, projected coordinates and condensed pair distances support additional plots. PCA uses the same dimension when archived rank permits; Arcene has no matched-rank PCA control. DBSCAN-margin is excluded because no tested compressed hybrid qualified. These are local native measurements, not proof speedups or full-corpus guarantees.',
        table(pairs[['dataset','method','n','total_m','seconds_median','speedup_vs_full','pair_q95','pair_cost_ratio']]),
        '## Verified proof comparisons',table(reduced),
        ('The n2,d128 group spans an execution-tool interruption and reduced available RAM: Full finished under the original 3 GiB cap, while continuation controls use the lower adaptive budget in job.json and a freshly generated commitment salt. The underlying data/model are identical, and each proof verifies its own correct opening. Its timing_comparable flag is false; speedup ratios are intentionally withheld. The two d32 groups completed with the original budget and matched openings. ' if 'proof_2_128_32_5_1' in excluded_jobs else '')+
        'Any groups with changed openings or recorded memory/thread budgets are listed in comparison_exclusions.csv and excluded from speed ratios. Resource-limited controls are retained separately. The current worker authenticates and reuses a saved opening on restart; archived measurements keep their original source fingerprints.',
        'Median prove time excludes compilation/setup; single_run_total includes both plus witness construction and median verification. All proof rows use identical input data and model within each shape, three measured proofs and one warm-up. The two d32 groups also use identical commitment openings. Synthetic fixtures are engineering controls, not proofs for the full archived datasets.',
        '## Changes justified by these measurements',
        'Keep full centroid-span protection for its exact-arithmetic fixed-center assignment property, and validate integer rounding against margins. Choose residual size using direct cost and geometry errors rather than ARI alone. The protected-only ablation can preserve labels while losing most of the objective and pair geometry, so it does not qualify as the requested compression.',
        'Prioritize preserving authentication while reducing its circuit cost, then investigate reuse of sparse routing and tighter component bounds. The current generic lookup path costs more constraints than the shared path on these fixtures. Changes to model/key binding or commitment architecture need explicit protocol design; dropping original-input authentication is not a valid speed optimization. Neither shared-routing reuse across points nor a commit-and-prove integration has been implemented in this campaign.',
        'Raw CSVs, lossless labels, sampled distances, saved bases, exact-integer inputs and complete backend proof records are retained. Eight independent replays reconstructed a full-rank hybrid from the archive and matched the saved distances and labels. The scheduling-only memory-limit correction and both source versions are documented in source_revisions/memory_limit_fix.']
    report_path=root/'report.md';original=report_path.read_text(encoding='utf-8')
    heading,body=original.split('\n',1)
    report_path.write_text(heading+'\n\n'+'\n\n'.join(intro)+'\n\n'+body,encoding='utf-8')
    print(json.dumps(validation,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/laptop_comparison_20261011'))
    finalize(parser.parse_args().out)
