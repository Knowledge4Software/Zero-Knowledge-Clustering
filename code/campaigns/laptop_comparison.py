"""Sequential, resumable laptop campaign with process-tree RAM/time limits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time
import psutil

CODE=Path(__file__).resolve().parents[1]
VERSION='laptop_comparison_v1'
DATASETS=['gaussian_easy','gaussian_hard','dbscan_margin','mnist','fashion_mnist','gisette','arcene','20ng_tfidf']
METHODS=['full','pca','sparse_shared','sparse_lookup','hybrid_shared','hybrid_lookup','hybrid_specialized','protected_only']


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def plan(binary,smoke=False):
    jobs=[]
    selected=['gaussian_easy'] if smoke else DATASETS
    for name in selected:
        jobs.append(dict(id='geometry_'+name,kind='geometry',dataset=name,k=[5] if smoke else [5,10],
            starts=[3] if smoke else [3,4],seeds=[100] if smoke else [100,101,102],
            m=[20] if smoke else [20,64,128,256,512],q=[1] if smoke else [1,2],
            native_repetitions=1 if smoke else 3,smoke=smoke,timeout=600))
    for name in selected:
        jobs.append(dict(id='precision_'+name,kind='precision',dataset=name,n=16 if smoke else 128,
            k=[5] if smoke else [5,10],start=3,seeds=[100] if smoke else [100,101,102],
            m=[20] if smoke else [20,64,128],scales=[4096] if smoke else [1024,4096],timeout=240))
    jobs.append(dict(id='profile',kind='profile',shape=[2,4,3,2,1] if smoke else [8,32,16,5,1],
        methods=['full','hybrid_lookup'] if smoke else METHODS[:6],timeout=600,case_timeout=90))
    shapes=[[2,4,3,2,1]] if smoke else [[n,d,16 if d==32 else 32,k,1]
        for n,d,k in [(2,32,5),(8,32,5),(32,32,5),(2,128,5),(8,128,5),(32,128,5),(2,128,10),(8,128,10)]]+[[2,128,32,5,2]]
    for shape in shapes:
        jobs.append(dict(id='growth_'+'_'.join(map(str,shape)),kind='solve',shape=shape,
            methods=METHODS[:2]+['hybrid_shared','hybrid_lookup'] if smoke else METHODS,
            timeout=600,case_timeout=90))
    if not smoke:
        jobs.append(dict(id='real_mnist',kind='solve',dataset='mnist',shape=[2,784,64,5,1],
            methods=['full','pca','hybrid_shared','hybrid_lookup'],timeout=600,case_timeout=120))
    for shape in ([[2,4,3,2,1]] if smoke else [[2,32,16,5,1],[8,32,16,5,1],[2,128,32,5,1]]):
        jobs.append(dict(id='proof_'+'_'.join(map(str,shape)),kind='prove',shape=shape,
            methods=['full','hybrid_lookup'] if smoke else ['full','pca','sparse_shared','hybrid_shared','hybrid_lookup'],
            repetitions=1 if smoke else 3,case_timeout=240,timeout=1500))
    for job in jobs:
        job.update(binary=str(binary),threads=4,rss_gib=3.,scale=4096)
    return jobs


def fingerprint(binary):
    files=[Path(__file__),CODE/'experiments/laptop_comparison.py',CODE/'analysis/laptop_comparison_report.py',
        CODE/'commitjl/protected_jl.py',CODE/'commitjl/zk_bridge.py',CODE/'commitjl/sparse_gen.py',
        CODE/'commitjl/clustering.py',CODE/'zk/circuit/circuit.go',CODE/'zk/circuit/sparse_lookup.go',
        CODE/'zk/cmd/commitjl-prove/main.go']
    return dict(version=VERSION,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        archive_index_sha256=hashlib.sha256((CODE/'results/server_retirement_20261007/compact_arrays/array_index.json').read_bytes()).hexdigest(),
        sources={str(p.relative_to(CODE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def kill_tree(proc):
    try:
        parent=psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:child.kill()
            except psutil.Error:pass
        parent.kill()
    except psutil.Error:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill()


def execute(job,root,deadline,max_rss_gib):
    folder=root/'jobs'/job['id'];folder.mkdir(parents=True,exist_ok=True)
    job=dict(job,rss_gib=max_rss_gib)
    spec=folder/'job.json';atomic_json(spec,job)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','GOMAXPROCS']:
        env[key]=str(job['threads'])
    env['GOMEMLIMIT']=f'{max(512,int(max_rss_gib*1024*.65))}MiB'
    begin=time.monotonic();peak=0;status='running'
    with (folder/'worker.log').open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-u','-m','experiments.laptop_comparison',
            '--job',str(spec.resolve()),'--out',str(folder.resolve())],cwd=CODE,env=env,
            stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            while proc.poll() is None:
                now=time.monotonic()
                if now>=deadline or now-begin>job['timeout']:
                    status='campaign_budget' if now>=deadline else 'job_timeout';kill_tree(proc);break
                try:
                    parent=psutil.Process(proc.pid);processes=[parent]+parent.children(recursive=True)
                    rss=0
                    for p in processes:
                        try:rss+=p.memory_info().rss
                        except psutil.Error:pass
                    peak=max(peak,rss)
                    if rss>(max_rss_gib+.5)*2**30 or psutil.virtual_memory().available<768*2**20:
                        status='resource_limit';kill_tree(proc);break
                except psutil.Error:pass
                time.sleep(.1)
        except KeyboardInterrupt:
            kill_tree(proc);status='interrupted'
    if status=='running':status='ok' if proc.returncode==0 else 'error'
    record=dict(job_id=job['id'],status=status,wall_seconds=time.monotonic()-begin,
                peak_tree_rss_gib=peak/2**30,returncode=proc.returncode,rss_limit_gib=max_rss_gib)
    atomic_json(folder/'execution.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=CODE/'results/laptop_comparison_20261011')
    parser.add_argument('--binary',type=Path,default=CODE/'zk/bin/commitjl-experimental-v4.exe')
    parser.add_argument('--max-minutes',type=float,default=45)
    parser.add_argument('--max-rss-gib',type=float,default=3)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.max_minutes<=0 or not 1<=args.max_rss_gib<=4:parser.error('Use a positive deadline and 1-4 GiB cap')
    binary=args.binary.resolve()
    if not binary.exists():parser.error('Experimental v4 binary is missing; see the runbook build command')
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    identity=fingerprint(binary);jobs=plan(binary,args.smoke)
    manifest_path=root/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['identity']!=identity or manifest['plan']!=jobs:
            parser.error('Source, binary or plan changed. Use a new output folder.')
    else:
        import shutil
        for name in identity['sources']:
            target=root/'sources'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(CODE/name,target)
        manifest=dict(identity=identity,plan=jobs,complete=False,host=dict(total_ram_gib=psutil.virtual_memory().total/2**30,
            logical_cpus=os.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),processor=platform.processor(),
            platform=platform.platform(),python=sys.version),scope='Exploratory laptop diagnosis, not full-corpus proof performance',
            native='Exact preserved subsets, saved public centers, fresh native-simulation maps; no retraining',
            quality_thresholds=dict(ari=.99,centroid_q95=.2,pair_q95=.2,projected_cost_relative_error=.1),
            limits=dict(max_minutes=args.max_minutes,max_rss_gib=args.max_rss_gib),attempts=[])
    begin=time.monotonic();deadline=begin+60*args.max_minutes
    manifest['attempts'].append(dict(start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),max_minutes=args.max_minutes))
    atomic_json(manifest_path,manifest)
    try:
        for index,job in enumerate(jobs):
            folder=root/'jobs'/job['id']
            if (folder/'done.json').exists():continue
            if time.monotonic()>=deadline:break
            available=psutil.virtual_memory().available/2**30
            if available<1.5:
                print('Low available RAM: checkpoint saved; resume after closing other apps.',flush=True);break
            cap=min(args.max_rss_gib,max(1,available-1.3))
            if job['kind']=='prove' and not args.smoke:
                growth_id='growth_'+'_'.join(map(str,job['shape']))
                rows=root/'jobs'/growth_id/'rows'
                eligible=[]
                for method in job['methods']:
                    candidate=rows/(method+'_complete.json')
                    if candidate.exists():
                        row=json.loads(candidate.read_text())
                        if row['status']=='ok' and row['constraints']<=750000:eligible.append(method)
                job=dict(job,methods=eligible)
                if not eligible:
                    folder.mkdir(parents=True,exist_ok=True)
                    atomic_json(folder/'done.json',dict(complete=True,skipped=True,reason='No successfully solved control <=750000 constraints'))
                    continue
            print(f'[{index+1}/{len(jobs)}] {job["id"]} RAM cap {cap:.2f} GiB',flush=True)
            record=execute(job,root,deadline,cap)
            print(f'  {record["status"]}, {record["wall_seconds"]:.1f}s, peak {record["peak_tree_rss_gib"]:.2f} GiB',flush=True)
            if record['status'] in ['error','interrupted','campaign_budget']:
                manifest['stop_reason']=record['status'];break
    finally:
        manifest['complete']=all((root/'jobs'/job['id']/'done.json').exists() for job in jobs)
        manifest['last_attempt_wall_seconds']=time.monotonic()-begin
        atomic_json(manifest_path,manifest)
        from analysis.laptop_comparison_report import report
        report(root)
        print('Report: '+str(root/'report.md'),flush=True)


if __name__=='__main__':main()
