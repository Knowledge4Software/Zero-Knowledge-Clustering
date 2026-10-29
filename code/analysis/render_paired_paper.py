"""Render the 22 October paper snapshot from completed, hashed evidence only."""
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
LABELS={'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','20ng_tfidf':'20NG TF--IDF'}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(out,name,caption,label,columns,rows,note='',wide=False):
    env='table*' if wide else 'table'
    text=[rf'\begin{{{env}}}[tp]',r'\centering',r'\small',
          rf'\caption{{{caption}}}\label{{{label}}}',
          r'\begin{tabular}{'+'l'+'r'*(len(columns)-1)+'}',r'\toprule',
          ' & '.join(columns)+r' \\',r'\midrule']
    text+=[' & '.join(str(x) for x in row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}']
    if note:text += [r'\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize '+note+r'\end{minipage}']
    text += [rf'\end{{{env}}}']
    (out/(name+'.tex')).write_text('\n'.join(text)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--out',type=Path,default=ROOT/'paper_results/review_controls/paired_20261022');a=p.parse_args()
    capture=a.capture.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    native=ROOT/'code/validation/outputs/review_update_decisions_20261021T184553Z/native_capture_20261022T0530Z'
    inputs=[];hash_checks=0
    for base,mname in [(capture,'capture.json'),(native,'capture_manifest.json')]:
        m=json.loads((base/mname).read_text())
        for e in m['files']:
            assert sha(base/e['path'])==e['sha256'],e['path'];hash_checks+=1
        inputs.append(dict(path=(base/mname).relative_to(ROOT).as_posix(),sha256=sha(base/mname)))
    cells=pd.read_csv(native/'native/cells.csv');assert len(cells)==2160
    assert not cells.false_certified_assignments.any()
    for row in cells.to_dict('records'):
        truth=int(row['true_change'])*int(row['cost_denominator'])
        for prefix in ['paired','independent']:
            assert int(row[prefix+'_lower_scaled'])<=truth<=int(row[prefix+'_upper_scaled'])
    f=cells[cells.kind=='lloyd_update'].copy();assert len(f)==1440
    assert not f.unchanged_centers.any()
    f['paired_conclusive']=f.paired_decision!='inconclusive'
    f['absolute_conclusive']=f.absolute_decision!='inconclusive'
    aggregate=f.groupby(['dataset','age','method','rank'],as_index=False).agg(
        requests=('set','count'),paired_rate=('paired_conclusive','mean'),
        independent_rate=('absolute_conclusive','mean'),agreement=('assignment_agreement','mean'),
        coverage=('certified_fraction','mean'),width=('paired_width_over_baseline','mean'))
    aggregate.to_csv(out/'native_groups.csv',index=False)
    get=lambda d,age,m,r:aggregate[(aggregate.dataset==d)&(aggregate.age==age)&(aggregate.method==m)&(aggregate['rank']==r)].iloc[0]
    rows=[]
    for d in LABELS:
        for age in (1,8):
            l,c=get(d,age,'learned',63),get(d,age,'pca',63)
            rows.append([LABELS[d],'Early' if age==1 else 'Mature',
                         f'{100*l.paired_rate:.2f}',f'{100*c.paired_rate:.2f}',
                         f'{100*l.coverage:.2f}',f'{100*c.coverage:.2f}',
                         f'{100*l.agreement:.2f}',f'{100*c.agreement:.2f}'])
    table(out,'native_decisions',
          'Independent-fit numerical grid at rank 63: update decisions, assignment coverage and agreement (percent).',
          'tab:paired-native',['Dataset','Age','Decision L','Decision P','Coverage L','Coverage P','Agreement L','Agreement P'],rows,
          'L: centroid-informed head plus residual PCA; P: PCA plus energy. Each row summarizes 60 dependent update requests from 15 fitted $k$/seed configurations. Both methods use the same held-out rows, updates, rank and precision. Decision rate is the fraction with a certified strict cost-change sign; assignment agreement is against quantized Full. These are numerical results, not proof timings.',True)
    records=[]
    for summary in sorted(capture.glob('*/summary.json')):
        name=summary.parent.name
        if name in ['server_smoke','portability_paired']:continue
        s=json.loads(summary.read_text());assert s['complete'] and not s['numerical_false_decisions']
        protocol=json.loads((summary.parent/'protocol.json').read_text())
        root=s['original_root'];decisions=pd.read_csv(summary.parent/'decisions.csv')
        for alt in s['alternatives']:
            phase=alt['phase'];key='verified_'+phase+'.json'
            replay=json.loads((summary.parent/key).read_text());assert replay['complete']
            assert alt['lloyd_requests']==4
            # Reconstruct every exported deployment component from raw phases.
            q=protocol['batch'];data=json.loads((summary.parent/'bundle'/f'{phase}_{q}.json').read_text())
            samples=data['samples']
            assert all(len(z['prove_seconds'])==protocol['reps'] for z in samples)
            assert len(samples)==alt['proves']
            shared=json.loads((summary.parent/'bundle/prepare.json').read_text())['native_prepare_seconds']
            stage=summary.parent/'bundle/stage1.json'
            stage_count=0
            if phase!='full_reuse':
                stage_data=json.loads(stage.read_text());stage_count=len(stage_data['samples'])
                assert stage_count==128
                shared+=stage_data['compile_seconds']+stage_data['setup_seconds']+sum(z['witness_seconds']+float(np.median(z['prove_seconds'])) for z in stage_data['samples'])
            query=data.get('native_prepare_seconds',0)+sum(z.get('native_answer_seconds',0)+z['witness_seconds']+float(np.median(z['prove_seconds'])) for z in samples)
            total=shared+data['compile_seconds']+data['setup_seconds']+query
            assert abs(total-alt['total_components_seconds'])<1e-5,(name,phase,total)
            assert replay['verified']==stage_count+len(samples)
            chosen=decisions[(decisions.phase==phase)&(decisions.update_kind=='lloyd_update')]
            assert len(chosen)==4
            assert int((chosen.decision!='inconclusive').sum())==alt['conclusive_lloyd_requests']
            for row in decisions[decisions.phase==phase].to_dict('records'):
                batch_samples=[z for z in samples if z.get('centre_set',0)==row['set']]
                assert len(batch_samples)==128//q
                if phase=='paired_change':
                    assert sum(int(z['change_lower_scaled']) for z in batch_samples)==int(row['change_lower_scaled'])
                    assert sum(int(z['change_upper_scaled']) for z in batch_samples)==int(row['change_upper_scaled'])
            meta=s['metadata']
            records.append(dict(case=name,dataset=meta['dataset'],k=meta['k'],seed=meta['fit_seed'],
                method='full' if phase=='full_reuse' else meta['method'],rank=0 if phase=='full_reuse' else meta['rank'],
                n=s['n'],batch=q,reps=protocol['reps'],original_root=root,
                workflow_wall_seconds=s['workflow_wall_seconds'],
                single_phase_deployment=s['actual_single_repetition_deployment'],**alt))
    proof=pd.DataFrame(records);proof.to_csv(out/'completed_alternatives.csv',index=False)
    family=proof[~proof.case.str.startswith('deployment_') & proof.phase.isin(['paired_change','full_reuse'])].copy()
    rows=[]
    for d in LABELS:
        part=family[family.dataset==d].copy()
        for _,r in part.sort_values(['method','rank'],ascending=[True,False]).iterrows():
            rows.append([LABELS[d],int(r.k),'Full' if r.method=='full' else ('Head' if r.method=='learned' else 'PCA'),
                '--' if r.method=='full' else int(r['rank']),int(r.batch),f'{r.components_plus_public_replay_seconds/60:.2f}',
                f'{int(r.conclusive_lloyd_requests)}/4'])
    table(out,'completed_costs',
          'Completed N=128 proof alternatives at the snapshot cutoff. Cost includes phase-specific public replay.',
          'tab:paired-costs',['Workload','$k$','Method','Rank','Batch','Cost (min)','Decisions'],rows,
          'Six sets comprise one null control, four actual Lloyd updates and one reverse-step diagnostic; all six are charged, but the last column counts only the four actual updates. Shared row binding is charged once to each reuse alternative. Query medians use three repetitions on the first MNIST fit and one elsewhere; Stage 1 uses one timing per row. All use $T=16384$ and matched batches within each fit. Reconstructed costs exclude fitting, benchmark warmups and process/I/O overhead; they include verifier replay and paired public-parameter preparation. A cheaper inconclusive service is not equivalent to exact Full.',True)
    resources=[]
    for _,r in family[family.dataset=='mnist'].sort_values(['method','rank'],ascending=[True,False]).iterrows():
        resources.append(['Full' if r.method=='full' else ('Head' if r.method=='learned' else 'PCA'),
            '--' if r.method=='full' else int(r['rank']),f'{r.proving_key_bytes/2**20:.2f}',
            f'{r.verifying_key_bytes/1024:.2f}',f'{r.proof_bytes/1024:.2f}',
            f'{r.public_witness_serialized_bytes/2**20:.2f}',int(r.stored_head_energy_scalars)])
    table(out,'resources','Serialized resources for the completed MNIST N=128/k=3 alternatives.',
          'tab:paired-resources',['Method','Rank','PK (MiB)','VK (KiB)','Proofs (KiB)','Public witnesses (MiB)','Head/energy scalars'],resources,
          'Reported resources include the Stage 1 and chosen query relation where applicable. PK/VK are proving/verification keys. Full needs no compressed catalog; original rows remain required by its prover. Public-witness bytes are serialized input volume, not automatically network traffic. Model/task JSON and fitting times are retained in the CSV export.',True)
    # Controlled rank study: all five completed proof cases, including negative outcomes.
    sens=ROOT/'code/validation/outputs/review_sensitivity_20261021T061346Z'
    sc=sens/'proofs_completed_20261021T1900Z'
    sm=json.loads((sc/'capture.json').read_text())
    for e in sm['files']:assert sha(sc/e['path'])==e['sha256'];hash_checks+=1
    for extra in [sc/'capture.json',sens/'prepared/protocol.json']:
        inputs.append(dict(path=extra.relative_to(ROOT).as_posix(),sha256=sha(extra)))
    protocol=json.loads((sens/'prepared/protocol.json').read_text())
    indices=[z['set'] for z in protocol['center_sets'] if float(z['requested_relative_drift'])==.01]
    if not indices:
        raise ValueError('missing predeclared 0.01 drift tasks')
    sr=[]
    for pth in sorted(sc.glob('mnist*/summary.json')):
        s=json.loads(pth.read_text());assert s['complete']
        if 'full_public' in pth.parent.name:
            sr.append(['Full','--','100.00','100.00','0.000',f'{s["total_seconds"]/60:.2f}']);continue
        w=pd.read_csv(pth.parent/'workload.csv');z=w[w['set'].isin(indices)];assert len(z)==2
        sr.append(['Head' if 'learned' in pth.parent.name else 'PCA',63 if 'r63' in pth.parent.name else 15,
             f'{100*z.agreement_quantized.mean():.2f}',f'{100*z.certificate_fraction.mean():.2f}',
             f'{100*z.certificate_interval_width.mean():.3f}',f'{s["certificate_total_seconds"]/60:.2f}'])
    table(out,'controlled_rank','Completed fixed-precision rank sensitivity on one MNIST fit (N=256).',
          'tab:controlled-rank',['Method','Rank','Agreement (\\%)','Coverage (\\%)','Width (\\%)','Cost (min)'],sr,
          'Quality averages the two requested relative-drift 0.01 directions. Cost charges all eleven center sets. Head/PCA use nested ranks 15/63, $T=16384$ and batch16; exact Full uses batch8 under the constraint guard. Component costs exclude verifier replay and fitting, unlike Table~\\ref{tab:paired-costs}. This is a one-fit control, not independent-model replication. Two smaller-drift directions round to unchanged centers and are retained as controls.',True)
    # A descriptive, controlled cost/decision frontier; no error bars from dependent requests.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.2,2.8),sharey=True)
    for ax,d in zip(axes,LABELS):
        part=family[family.dataset==d]
        for method,color,marker in [('learned','#0072B2','o'),('pca','#D55E00','s'),('full','#222222','^')]:
            z=part[part.method==method].sort_values('rank')
            if z.empty:continue
            xx=z.components_plus_public_replay_seconds/60;yy=z.conclusive_lloyd_requests/4
            ax.plot(xx,yy,color=color,marker=marker,label={'learned':'Head','pca':'PCA','full':'Exact Full'}[method],
                    linestyle='-' if len(z)>1 else 'None',markersize=6,alpha=.85)
            for (_,rr),x,y in zip(z.iterrows(),xx,yy):
                if method=='learned':ax.annotate('r'+str(int(rr['rank'])),(x,y),xytext=(4,-13 if y>.5 else 7),textcoords='offset points',fontsize=7,color=color)
        ax.set_title(LABELS[d].replace('--','-'),fontsize=10);ax.set_xlabel('Components + public replay (min)',fontsize=8)
        ax.set_ylim(-.08,1.12);ax.set_yticks([0,.5,1],labels=['0%','50%','100%']);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Actual updates with resolved sign',fontsize=8)
    axes[1].legend(loc='center right',fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(out/'cost_decisions.pdf');plt.close(fig)
    dep=proof[proof.case.str.startswith('deployment_')].set_index('case')
    assert all(dep.single_phase_deployment)
    assert dep.original_root.nunique()==1
    fit=f[(f.age==1)&(f['rank']==63)]
    snapshot=json.loads((capture/'capture.json').read_text())
    counts=dict(native_cells=len(cells),actual_native_requests=len(f),fitted_configurations=45,split_seed_replications=15,
        completed_main_cases=int(proof.case.nunique()),planned_main_cases=32,
        cutoff_utc=datetime.datetime.fromtimestamp(snapshot['captured_unix'],datetime.timezone.utc).isoformat(),
        deployment_paired_seconds=float(dep.loc['deployment_paired'].workflow_wall_seconds),
        deployment_full_seconds=float(dep.loc['deployment_full'].workflow_wall_seconds),
        deployment_ratio=float(dep.loc['deployment_full'].workflow_wall_seconds/dep.loc['deployment_paired'].workflow_wall_seconds),
        early_learned_absolute_rate=float(fit[fit.method=='learned'].absolute_conclusive.mean()),
        early_pca_absolute_rate=float(fit[fit.method=='pca'].absolute_conclusive.mean()),
        early_learned_paired_rate=float(fit[fit.method=='learned'].paired_conclusive.mean()),
        early_pca_paired_rate=float(fit[fit.method=='pca'].paired_conclusive.mean()),
        file_hashes_checked=hash_checks)
    (out/'snapshot.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    (out/'manifest.json').write_text(json.dumps(dict(inputs=inputs,files=[dict(path=q.name,sha256=sha(q)) for q in sorted(out.iterdir()) if q.is_file() and q.name!='manifest.json'],
        validation='PASS: capture hashes, complete cases, public replay counts, timing repetitions, reconstructed costs and retained outcomes.',
        boundary='Numerical native grid, worker-replayed proofs and fresh local artifact replays are distinct evidence cohorts.'),indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
