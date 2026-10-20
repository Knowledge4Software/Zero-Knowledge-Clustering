"""Real tiny smoke of every campaign stage; deliberately not paper evidence."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['GOMAXPROCS']='2'
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import rq1_complete as rq1
from experiments import rq2345_complete as suite
from experiments.common import source_fingerprint


def main():
    started=time.perf_counter()
    root=Path('results/diagnostics/tiny_all_rq_'+time.strftime('%Y%m%d_%H%M%S'))
    root.mkdir(parents=True)
    dataset=dict(loader='gaussian_mixture',kwargs=dict(n=6,d=4,k=2,gamma=3.,seed=12345))
    rq1.DATASETS['tiny']=(dataset['loader'],dataset['kwargs'])
    cfg=dict(scope='paper',starts=2,seeds=2,k=[2],m=[2],q=[1],eps=.15,
             pairs=15,threads=1,bootstrap=30,tau_ari=.8,tau_phi=1.1,
             source_sha256=source_fingerprint(),methods=list(rq1.METHODS[:6]))
    frozen=root/'rq1/tiny'
    rq1.worker('tiny',cfg,frozen)
    from analysis.rq1_export import export as export_rq1
    export_rq1(root/'rq1')
    settings=dict(threads=1,proof_threads=2,proof_reps=1,proof_timeout=120,
                  proof_rss_gib=2,task_rss_gib=3,task_timeout=180,
                  bootstrap=30,seeds=2,starts=2,min_free_gib=.001)
    campaign=root/'campaign';campaign.mkdir()
    suite.write_json(campaign/'settings.json',settings)
    base=dict(dataset=dataset,k=2,m=2,q=1,seed=0,clustering_start=0,
              precision_bits=16,B=4096,backend='groth16',range_bits=20,
              projection='sparse_jl',variant='sparse')
    tasks=[];results=[]
    for stage in suite.STAGES:
        row=dict(base)
        if stage=='rq2_selected': row['variant']='full'
        if stage=='rq5_precision': row.update(normalization_reference_bits=24)
        if stage=='rq5_ablation': row.update(precision_bits=24,normalization_reference_bits=24)
        if stage=='rq4_phase': row.update(sweep_factor='gamma',sweep_value=3.)
        if stage in ['rq5_native','rq5_proof']: row.update(sweep_factor='n',sweep_value=6)
        task=dict(stage=stage,id=stage,row=row,name='tiny',scope='tiny_smoke_not_paper_evidence',
                  directory=str(frozen.resolve()),seeds=2,starts=2,
                  max_points=6,m_values=[2],q_values=[1],trials=16,source='nonce')
        tasks.append(task);out=campaign/stage/stage
        t=time.perf_counter()
        result=suite.worker(task,out,settings);result.setdefault('status','ok')
        result['elapsed_seconds']=time.perf_counter()-t
        suite.write_json(out/'result.json',result)
        assert result['status']=='ok',(stage,result)
        for arm in result.get('arms',[]):
            assert arm['ok'] and arm['verify_ok'] and arm['successful_proofs']==1,(stage,arm)
        results.append(dict(stage=stage,status=result['status'],seconds=result['elapsed_seconds'],
                            proof_arms=len(result.get('arms',[]))))
        print(json.dumps(results[-1]),flush=True)
    suite.write_json(campaign/'plan.json',tasks)
    suite.export(campaign)
    for path in root.rglob('*.npz'):
        with np.load(path,allow_pickle=False) as arrays:
            for key in arrays.files: arrays[key]
    summary=pd.read_csv(campaign/'summary.csv')
    assert summary.complete.all()
    raw=pd.read_csv(campaign/'all_rows.csv')
    assert raw['n_violations'].dropna().eq(0).all()
    report=dict(scope='tiny_smoke_not_full_campaign_validation',dataset=dataset,
                seeds=2,starts=2,proof_reps=1,warmup=1,stages=results,
                elapsed_seconds=time.perf_counter()-started,
                figures=len(list(root.rglob('*.png'))),npz_files=len(list(root.rglob('*.npz'))),
                verified_proof_arms=sum(r['proof_arms'] for r in results))
    suite.write_json(root/'check_report.json',report)
    print(json.dumps(dict(output=str(root.resolve()),**report)),flush=True)


if __name__=='__main__':
    main()
