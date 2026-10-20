"""Finite low-cost additions: subset baselines, tiny matched backend pilot,
isolated verifier RSS, and protocol stage timings. Original runs are immutable.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from campaigns.paper_queue import run_queue
from experiments import rq2345_complete as suite


def baseline_tasks(original):
    tasks=[]
    for variant in ['dense','coord','pca-public']:
        row=dict(dataset=suite.spec('gaussian_easy'),k=5,m=16,q=1,seed=0,
            clustering_start=0,precision_bits=24,normalization_reference_bits=24,
            B=4096,backend='groth16',variant=variant,proof_n_max=32)
        quality=dict(stage='rq2_quality',name='gaussian_easy',directory=str(original/'gaussian_easy'),
                     row=row,seeds=10,starts=5)
        proof=dict(stage='rq2_selected',name='gaussian_easy',directory=str(original/'gaussian_easy'),
                   row=row,scope='same_frozen_32x256_subset_additional_baseline_requalified')
        tasks.extend([dict(t,id=suite.digest(t)) for t in [quality,proof]])
    return sorted(tasks,key=lambda t:0 if t['stage']=='rq2_quality' else 1)


def verify_export(folder):
    env={**os.environ,'COMMITJL_VERIFY_ONLY':str(folder)}
    started=time.perf_counter()
    binary=Path(os.environ['COMMITJL_ZK_BIN']).resolve()
    # Linux ru_maxrss can retain pre-exec launcher high-water memory. Read the
    # new address space's VmHWM only after /proc/exe confirms the Go verifier.
    process=subprocess.Popen([str(binary)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    peak=0;samples=0
    while process.poll() is None:
        if time.perf_counter()-started>60:
            process.kill();process.communicate();raise TimeoutError('isolated verification timed out')
        try:
            proc=Path('/proc')/str(process.pid)
            if (proc/'exe').resolve()==binary:
                for line in (proc/'status').read_text().splitlines():
                    if line.startswith('VmHWM:'):
                        peak=max(peak,int(line.split()[1])*1024);samples+=1
        except (OSError,ValueError):pass
        time.sleep(.001)
    stdout,stderr=process.communicate()
    if process.returncode:raise RuntimeError(stderr)
    value=json.loads(stdout)
    if not peak or not samples:raise RuntimeError('no post-exec verifier RSS sample captured')
    value['getrusage_peak_including_launcher_bytes']=value['peak_process_rss_bytes']
    value['peak_process_rss_bytes']=peak
    value['rss_measurement']='largest observed post-exec /proc/PID/status VmHWM; 1ms polling'
    value['rss_samples']=samples
    assert value['verify_ok'] and value['successful_verifications']==10
    value['fresh_process_wall_seconds']=time.perf_counter()-started
    value['files']={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in folder.iterdir() if p.suffix=='.bin'}
    suite.write_json(folder/'isolated_verifier.json',value)
    return value


def baselines(root,original):
    from experiments.common import source_fingerprint,capture_env,campaign_lock
    root.mkdir(parents=True,exist_ok=True);lock=campaign_lock(root)
    tasks=baseline_tasks(original)
    settings=dict(seeds=10,starts=5,threads=4,proof_threads=16,proof_reps=5,
        proof_rss_gib=40,task_rss_gib=46,proof_timeout=3600,task_timeout=3900,
        bootstrap=2000,max_proof_input_elements=8192,min_free_gib=20,
        source_sha256=source_fingerprint(),instrumented_binary_sha256=hashlib.sha256(Path(os.environ['COMMITJL_ZK_BIN']).read_bytes()).hexdigest())
    suite.write_json(root/'settings.json',settings);suite.write_json(root/'plan.json',tasks)
    suite.write_json(root/'environment.json',capture_env())
    results=[]
    for task in tasks:
        export=root/task['stage']/task['id']/'verifier_artifacts'
        os.environ['COMMITJL_VERIFIER_EXPORT_DIR']=str(export)
        r=suite.execute(task,root,settings)
        if r['status']=='ok' and task['stage']=='rq2_selected':
            v=verify_export(export)
            r['isolated_verifier_peak_rss_bytes']=v['peak_process_rss_bytes']
            suite.write_json(root/task['stage']/task['id']/'result.json',r)
        results.append(r)
        print(json.dumps(dict(task=task['id'],stage=task['stage'],status=r['status'])),flush=True)
    suite.export(root)
    suite.write_json(root/'completion.json',dict(status='completed',results=[r['status'] for r in results]))
    if any(r['status'] not in ['ok','excluded'] for r in results):raise RuntimeError('baseline task failed')


def backend_pilot(root):
    from experiments.rq2_proof_cost import build_zk_instance
    from commitjl import zk_bridge
    from experiments.common import capture_env,source_fingerprint
    root.mkdir(parents=True,exist_ok=True)
    suite.write_json(root/'environment.json',capture_env())
    results=[]
    row=dict(dataset=dict(loader='gaussian_mixture',kwargs=dict(n=8,d=16,k=5,gamma=3.,seed=12345)),
        k=5,m=8,q=2,seed=0,clustering_start=0,precision_bits=24,
        normalization_reference_bits=24,B=4096)
    for variant in ['full','sparse']:
        inst,*_=build_zk_instance({**row,'variant':variant},suite.PREREG)
        for backend in ['groth16','plonk']:
            folder=root/f'{variant}_{backend}';folder.mkdir(exist_ok=True)
            witness=copy.deepcopy(inst);witness.backend=backend
            suite.write_json(folder/'witness.json',witness.to_json_dict())
            export=folder/'verifier_artifacts'
            os.environ['COMMITJL_VERIFIER_EXPORT_DIR']=str(export)
            start=time.perf_counter()
            res=zk_bridge.run_prove(witness,timeout=900,repetitions=5,warmup=1,max_rss_bytes=8*2**30)
            record=dict(variant=variant,backend=backend,n=8,d=16,m=witness.m,k=5,q=2,
                ok=res.ok,raw=res.raw,error=res.error,peak_prover_rss_bytes=res.peak_rss_bytes_psutil,
                proof_call_seconds=time.perf_counter()-start,source_sha256=source_fingerprint(),
                circuit_scope='same_input_centroids_matrix_and_assignments_across_backends',
                plonk_srs='unsafe_test_only_KZG_not_deployment')
            suite.write_json(folder/'result.json',record)
            if not res.ok:raise RuntimeError(record)
            record['isolated_verifier']=verify_export(export)
            results.append(record);suite.write_json(root/'results.json',results)
            print(json.dumps(dict(variant=variant,backend=backend,status='ok')),flush=True)
    for variant in ['full','sparse']:
        a=json.loads((root/f'{variant}_groth16/witness.json').read_text())
        b=json.loads((root/f'{variant}_plonk/witness.json').read_text())
        a.pop('backend');b.pop('backend');assert a==b
    suite.write_json(root/'completion.json',dict(status='completed',matched_witnesses=True,measured_proofs=20))


def protocol(root,original_campaign):
    import pandas as pd
    from commitjl.randomness import TaskLog
    root.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in (original_campaign/'native/rq3_native').iterdir():
        task=json.loads((folder/'task.json').read_text())
        for r in json.loads((folder/'timings.json').read_text()):
            rows.append(dict(task_id=task['id'],source=task['source'],**r))
    frame=pd.DataFrame(rows)
    frame.to_csv(root/'ordering_stage_rows.csv',index=False)
    # Resample task means within source. Do not count repetitions as maps.
    import numpy as np
    rng=np.random.default_rng(20261007);summary=[]
    cols=['commit_X','record_theta','record_tasklog','issue_nonce',
          'substage_derive_seed','substage_expand_matrix','substage_hash_CA','project_and_cluster','total_seconds']
    for source,g in frame[frame.arm=='postcommit'].groupby('source'):
        for component in cols:
            means=g.groupby('task_id')[component].mean().to_numpy()
            boot=rng.choice(means,(2000,len(means)),replace=True).mean(axis=1)
            summary.append(dict(source=source,component=component,mean_ms=means.mean()*1000,
                ci_low_ms=np.quantile(boot,.025)*1000,ci_high_ms=np.quantile(boot,.975)*1000,
                tasks=len(means),observations=len(g)))
    pd.DataFrame(summary).to_csv(root/'ordering_stage_summary.csv',index=False)
    # Actual policy checks; expected rejections are not honest-run failures.
    checks=[]
    for i in range(10):
        log=TaskLog(Q=1,deadline=100.,_now=50.)
        log.record_commitment('p',i,i+1)
        for scenario,action,error in [
            ('budget',lambda:log.record_commitment('p',i+2,i+3),PermissionError),
            ('deadline',lambda:TaskLog(deadline=10.,_now=11.).record_commitment('p',i,i),PermissionError)]:
            start=time.perf_counter()
            try:action();raise AssertionError('policy accepted invalid request')
            except error:checks.append(dict(repetition=i,scenario=scenario,expected_rejected=True,seconds=time.perf_counter()-start))
        nonce=bytes([i])*32;log.consume_nonce(nonce)
        start=time.perf_counter()
        try:log.consume_nonce(nonce);raise AssertionError('nonce replay accepted')
        except ValueError:checks.append(dict(repetition=i,scenario='nonce_replay',expected_rejected=True,seconds=time.perf_counter()-start))
    pd.DataFrame(checks).to_csv(root/'policy_checks.csv',index=False)
    suite.write_json(root/'completion.json',dict(status='completed',ordering_rows=len(frame),
        expected_policy_rejections=len(checks),scope='local_tasklog_policy_and_existing_ordering_stage_timings',
        transcript_bytes_scope='existing serialized audit JSON; not deployed network wire traffic'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--original',type=Path,required=True)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--phase',choices=['baselines','backend','protocol'])
    args=ap.parse_args();root=args.root.resolve();original=args.original.resolve()
    if args.phase=='baselines':return baselines(root,original/'rq1')
    if args.phase=='backend':return backend_pilot(root)
    if args.phase=='protocol':return protocol(root,original)
    plan=[]
    for name,seconds in [('protocol',300),('backend',1800),('baselines',7200)]:
        plan.append(dict(name=name,command=[sys.executable,'-u','campaigns/quick_completion.py',
            '--phase',name,'--root',str(root/name),'--original',str(original)],timeout_seconds=seconds,requires=[]))
    if args.execute:run_queue(CODE,root,plan)
    else:print(json.dumps(plan,indent=2))


if __name__=='__main__':main()
