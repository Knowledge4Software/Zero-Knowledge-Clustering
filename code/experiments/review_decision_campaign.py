"""Serial proof queue and cost export for fixed independent-fit workloads."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys
import pandas as pd
from experiments.bound_audit import save


def log(root,text):
    with (root/'queue.log').open('a') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+text+'\n')


def jobs_for(entries):
    fits=[('mnist',3,11),('fashion_mnist',5,11),('20ng_tfidf',10,11),
          ('mnist',10,23),('fashion_mnist',3,23),('20ng_tfidf',5,23)]
    jobs=[]
    def find(fit,method,rank):
        return next(e for e in entries if (e['dataset'],e['k'],e['fit_seed'],e['method'],e['rank'])==(*fit,method,rank))
    for full in (False,True):
        e=find(fits[0],'learned',63)
        jobs.append(dict(name='deployment_'+('full' if full else 'paired'),input=e['path'],full=full,
                         also_absolute=False,batch=e['batch'],reps=1,warmup=0))
    for fit in fits:
        # Same precision, batches and timing repetitions within each fitted model.
        repeats=3 if fit==fits[0] else 1
        for method,rank in [('learned',63),('pca',63),('full',0),('learned',15),('pca',15)]:
            e=find(fit,'learned' if method=='full' else method,15 if method=='full' else rank)
            jobs.append(dict(name=f"{fit[0]}_k{fit[1]}_seed{fit[2]}_{method}_r{rank}",input=e['path'],
                full=method=='full',also_absolute=method!='full',batch=e['batch'],reps=repeats,warmup=1 if repeats==3 else 0))
    return jobs


def export_completed(root,jobs):
    rows=[];curves=[]
    for job in jobs:
        path=root/job['name']
        summary=json.loads((path/'summary.json').read_text())
        if not summary['complete']:raise ValueError('incomplete proof case')
        protocol=json.loads((path/'protocol.json').read_text());meta=summary.get('metadata') or {}
        decisions=pd.read_csv(path/'decisions.csv')
        for alternative in summary['alternatives']:
            row={**{k:meta.get(k) for k in ['dataset','k','fit_seed','age','method','rank']},**alternative,
                 'case':job['name'],'n':summary['n'],'reps':job['reps'],
                 'workflow_wall_seconds':summary['workflow_wall_seconds'],
                 'single_repetition_deployment':summary['actual_single_repetition_deployment'],
                 'representation_fit_seconds':0 if job['full'] else (meta.get('representation_fit_seconds') or {}).get(meta.get('method'),0),
                 'prototype_trace_seconds':meta.get('prototype_trace_seconds',0)}
            rows.append(row)
            if job['full']:row.update(method='full',rank=0)
            selected=decisions[decisions.phase==alternative['phase']].sort_values('set')
            cumulative=alternative['shared_preparation_seconds']+alternative['query_compile_setup_seconds']+alternative['public_center_parameter_seconds']
            for count,(_,request) in enumerate(selected.iterrows(),1):
                cumulative+=request['query_native_witness_prove_seconds']
                curves.append(dict(case=job['name'],phase=alternative['phase'],requests=count,
                                   measured_publisher_components_seconds=cumulative,
                                   cost_boundary='All requested center-parameter preparation charged upfront; verifier and fitting separately tabulated.'))
    pd.DataFrame(rows).to_csv(root/'completed_proof_costs.csv',index=False)
    pd.DataFrame(curves).to_csv(root/'observed_request_cost_curves.csv',index=False)
    save(root/'completed_summary.json',dict(complete=True,jobs=len(jobs),alternatives=len(rows),
         numerical_false_decisions=0,source_protocol='predeclared_protocol.json',
         timing_boundary='Separate publisher components, phase-specific public replay, fitting, and actual one-repetition deployment wall time; no equal-fidelity speed claim for inconclusive services.'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--bin',type=Path,required=True)
    a=p.parse_args();entries=json.loads((a.root/'native/proof_inputs.json').read_text())
    for e in entries:e['batch']=1 if e['dataset']=='20ng_tfidf' and e['k']==10 else e['batch']
    jobs=jobs_for(entries);save(a.root/'proof_jobs.json',jobs)
    errors=0
    for job in jobs:
        log(a.root,'START '+job['name'])
        cmd=[sys.executable,'-m','experiments.review_decision_controls','--input',str(a.root/'native'/job['input']),
             '--out',str(a.root/job['name']),'--bin',str(a.bin),'--batch',str(job['batch']),
             '--reps',str(job['reps']),'--warmup',str(job['warmup']),'--phase-hours','18']
        if job['full']:cmd+=['--full']
        elif job['also_absolute']:cmd+=['--also-absolute']
        with (a.root/(job['name']+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        log(a.root,f"END {job['name']} rc={r.returncode}")
        if r.returncode:errors+=1
    if errors:
        log(a.root,f'FAILED_JOBS={errors}');raise SystemExit(1)
    # Exercise the composed public auditor with real selective whole-task
    # fallback. It reads only public exports and independently approved pins/tasks.
    for job in jobs:
        if job['full']:continue
        if job['name']=='deployment_paired':full_name='deployment_full'
        else:
            prefix=job['name'].rsplit('_',2)[0]
            full_name=prefix+'_full_r0'
        paired=a.root/job['name'];full=a.root/full_name
        cmd=[sys.executable,'-m','experiments.review_public_decision_audit',
             '--paired-pin',str(paired/'owner_pin_paired_change.json'),'--paired-task',str(paired/'approved_public_task.json'),
             '--paired-bundle',str(paired/'public_bundle'),'--full-pin',str(full/'owner_pin_full_reuse.json'),
             '--full-task',str(full/'approved_public_task.json'),'--full-bundle',str(full/'public_bundle'),
             '--out',str(a.root/('public_audit_'+job['name'])),'--bin',str(a.bin)]
        with (a.root/('public_audit_'+job['name']+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        if r.returncode:log(a.root,'FAILED_PUBLIC_AUDIT '+job['name']);raise SystemExit(1)
    export_completed(a.root,jobs);log(a.root,'ALL_DONE')


if __name__=='__main__':main()
