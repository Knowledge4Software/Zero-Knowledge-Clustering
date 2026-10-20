"""Targeted follow-up to the completed paper campaign; original results stay immutable.

Run under systemd with --execute. Proofs are serial (16 Go threads), native
precision tasks use four workers with four BLAS threads. All tasks are bounded.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from campaigns.paper_queue import run_queue, atomic_json
from experiments import rq2345_complete as suite


def tasks_for(kind, original):
    tasks = []
    def add(stage, **kw):
        task = dict(stage=stage, **kw)
        tasks.append(dict(task, id=suite.digest(task)))
    base = dict(dataset=dict(loader='gaussian_mixture', kwargs=dict(
        n=96, d=128, k=5, gamma=3., seed=12345)), k=5, m=16, q=2,
        seed=0, clustering_start=0, precision_bits=24,
        normalization_reference_bits=24, B=4096, backend='groth16')
    if kind == 'sizing':
        # 96x128 is 50% above the previous maximum input size.
        # A larger 128x128 cell is allowed only by a recorded resource forecast.
        for n in [96, 128]:
            for variant in ['full', 'sparse', 'dense']:
                row = copy.deepcopy(base)
                row['dataset']['kwargs']['n'] = n
                row['variant'] = variant
                add('rq2_pilot', row=row, name=f'synthetic_{n}x128',
                    adaptive=(n == 128), scope='extended_synthetic_engineering_ladder')
    elif kind == 'real':
        import pandas as pd
        # Rows only are subsetted. All original features are retained.
        # Wide Arcene/Gisette are not silently feature-truncated to fit memory.
        for name, n in [('gaussian_easy',32), ('gaussian_hard',16),
                        ('mnist',8), ('fashion_mnist',8)]:
            points = pd.read_csv(original/name/'selected_operating_points.csv')
            selected = points[(points.method=='sparse_jl') & (points.k==5)]
            if len(selected) != 1:
                raise ValueError(f'Missing unique sparse k=5 operating point: {name}')
            selected = selected.iloc[0]
            row = {**copy.deepcopy(base), 'dataset':suite.spec(name),
                   'proof_n_max':n, 'm':int(selected.m), 'q':int(selected.q),
                   'precision_bits':24, 'normalization_reference_bits':24}
            # Geometric qualification and proof both use adaptive 24-bit
            # normalization here; the separate precision study freezes scale.
            for stage, variant in [('rq2_quality','sparse'),
                                   ('rq2_selected','full'), ('rq2_selected','sparse')]:
                kw=dict(name=name, directory=str((original/name).resolve()),
                        row={**row,'variant':variant}, scope='fixed_row_subset_all_features_requalified')
                if stage=='rq2_quality':
                    # Must match prove_measure's exact qualification task key.
                    kw.pop('scope')
                    kw.update(seeds=10, starts=5)
                add(stage, **kw)
    elif kind == 'precision':
        import pandas as pd
        for name in suite.DATASETS:
            points = pd.read_csv(original/name/'selected_operating_points.csv')
            sparse = points[points.method=='sparse_jl']
            selected = sparse[sparse.k==5]
            if selected.empty: selected = sparse[sparse.k==10]
            if selected.empty:
                k,m,q,scope=5,128,4,'exploratory_no_eligible_sparse_RQ1_point'
            else:
                selected=selected.iloc[0]
                k,m,q,scope=int(selected.k),int(selected.m),int(selected.q),'RQ1_selected_sparse_dimension_fixed_physical_scale'
            for bits in [16,24]:
                add('rq5_precision',name=name,row={**copy.deepcopy(base),
                    'dataset':suite.spec(name),'k':k,'m':m,'q':q,
                    'projection':'sparse_jl','precision_bits':bits},
                    seeds=10,starts=5,scope=scope)
    else:
        raise ValueError(kind)
    return tasks


def run_batch(kind, original, root, smoke=False):
    from experiments.common import source_fingerprint, capture_env, campaign_lock
    root.mkdir(parents=True,exist_ok=True)
    lock=campaign_lock(root)
    tasks=tasks_for(kind,original)
    if smoke:
        if kind!='sizing': raise ValueError('Smoke mode is only for sizing')
        tasks=copy.deepcopy(tasks[:3])
        for task in tasks:
            task['row']['dataset']['kwargs'].update(n=8,d=16)
            task['row']['m']=8
            task['name']='smoke_8x16'
            task['id']=suite.digest({k:v for k,v in task.items() if k!='id'})
    settings=dict(seeds=10,starts=5,jobs=4,threads=4,proof_threads=16,
        proof_reps=5,proof_rss_gib=42,task_rss_gib=46,proof_timeout=5400,
        task_timeout=7200,bootstrap=2000,max_proof_input_elements=16384,
        min_free_gib=20,source_sha256=source_fingerprint(),
        scope='followup_not_replacement_for_original_campaign')
    if smoke: settings['proof_reps']=1
    settings['original_inputs']={}
    for name in suite.DATASETS:
        for filename in ['manifest.json','selected_operating_points.csv']:
            p=original/name/filename
            settings['original_inputs'][f'{name}/{filename}']=hashlib.sha256(p.read_bytes()).hexdigest()
    for filename,value in [('settings.json',settings),('plan.json',tasks)]:
        path=root/filename
        if path.exists() and json.loads(path.read_text())!=value:
            raise ValueError(f'{filename} changed; choose a new output directory')
        suite.write_json(path,value)
    suite.write_json(root/'environment.json',capture_env())
    started=time.monotonic()
    state=dict(status='running',kind=kind,total=len(tasks),completed=0,results=[])
    def record(task,result):
        state['results'].append(dict(task_id=task['id'],status=result['status'],
            elapsed_seconds=result.get('elapsed_seconds'),peak_rss_bytes=result.get('monitored_peak_rss_bytes')))
        state['completed']=len(state['results'])
        atomic_json(root/'batch_state.json',state)
        print(json.dumps(state['results'][-1]),flush=True)
    if kind=='precision':
        # Each task receives a safe per-worker aggregate budget.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for task,result in zip(tasks,pool.map(lambda t:suite.execute(t,root,{**settings,'task_rss_gib':10}),tasks)):
                record(task,result)
    else:
        # Quality precedes proofs; other proof timings remain isolated.
        tasks=sorted(tasks,key=lambda t:0 if t['stage']=='rq2_quality' else 1)
        for task in tasks:
            if task.get('adaptive'):
                previous=[r for r in state['results'] if r['status']=='ok']
                # Forecast uses complete process-tree peak and a 20% buffer.
                # Progressing is optional; an exclusion is not measured OOM.
                largest=max((r.get('peak_rss_bytes') or 0 for r in previous),default=0)
                forecast=largest*128/96*1.2
                if not previous or any(r['status'] not in ['ok','excluded'] for r in state['results']) or forecast>38*2**30:
                    directory=root/task['stage']/task['id'];directory.mkdir(parents=True,exist_ok=True)
                    suite.write_json(directory/'task.json',task)
                    result=dict(status='excluded',reason='adaptive memory forecast exceeds 38 GiB admission threshold',
                                forecast_rss_bytes=forecast,scope=task['scope'])
                    suite.write_json(directory/'result.json',result);record(task,result);continue
            result=suite.execute(task,root,settings);record(task,result)
    suite.export(root)
    failures=[r for r in state['results'] if r['status'] not in ['ok','excluded']]
    state.update(status='completed' if not failures else 'completed_with_failures',
        elapsed_seconds=time.monotonic()-started,failed=len(failures))
    atomic_json(root/'batch_state.json',state)
    if failures: raise RuntimeError(f'{len(failures)} failed tasks; see preserved task logs')


def queue_plan(root, original):
    py=sys.executable
    def batch(kind,hours):
        return dict(name=kind,command=[py,'-u','campaigns/followup_server.py','--batch',kind,
            '--original',str(original),'--root',str(root/kind)],timeout_seconds=int(hours*3600),requires=[])
    return [batch('sizing',5),
        dict(name='rq1_targeted',command=[py,'-u','-m','experiments.rq1_complete','--run',
            '--datasets','20ng_tfidf','dbscan_margin','gaussian_easy',
            '--methods','sparse_jl','--m','64','128','256','512','1024','1536',
            '--q','4','8','--k','5','10','--seeds','10','--starts','5',
            '--jobs','3','--threads','5','--max-rss-gib','12',
            '--dataset-timeout-hours','2','--out',str(root/'rq1_targeted')],
            timeout_seconds=3*3600,requires=[]),
        dict(name='rq1_targeted_export',command=[py,'-u','-m','analysis.rq1_export',
            '--runs',str(root/'rq1_targeted')],timeout_seconds=3600,requires=['rq1_targeted']),
        batch('precision',2),batch('real',6)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--original',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--batch',choices=['sizing','real','precision'])
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if args.batch: return run_batch(args.batch,args.original.resolve(),args.root.resolve(),args.smoke)
    plan=queue_plan(args.root.resolve(),args.original.resolve())
    if args.execute: run_queue(CODE,args.root.resolve(),plan)
    else: print(json.dumps(plan,indent=2))


if __name__=='__main__': main()
