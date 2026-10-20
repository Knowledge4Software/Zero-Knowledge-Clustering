"""Fresh, fully covered changing-center proof controls; no catalog extrapolation.

Read an existing owner input without modifying it. Full uses public centers and
one setup. Every requested row/batch/set is proved, then reverified from a public
export against a separately supplied owner/setup pin and approved public task.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from experiments.bound_audit import bounded, save
from commitjl.reuse_certificates import certificate


def public_export(bundle: Path, out: Path):
    out.mkdir(exist_ok=False)
    metadata=['trusted.json','prepare.json','stage1.json']
    for prefix in ['reuse','reuse_certified','full_reuse']:
        metadata += [p.name for p in bundle.glob(prefix+'_*.json')
                     if p.stem.removeprefix(prefix+'_').isdigit()]
    files=[]
    for name in dict.fromkeys(metadata):
        p=bundle/name
        if not p.exists():continue
        data=json.loads(p.read_text())
        for sample in data.get('samples') or []:
            sample.pop('assign',None);sample.pop('certified_flags',None)
        save(out/name,data);files.append(name)
    for sub in ['keys','stage1','reuse','reuse_certified','full_reuse']:
        for p in (bundle/sub).glob('*.vk' if sub=='keys' else '*.proof'):
            target=out/p.relative_to(bundle);target.parent.mkdir(exist_ok=True)
            shutil.copy2(p,target);files.append(p.relative_to(bundle).as_posix())
    save(out/'public_manifest.json',dict(files=files,private_witness_files_included=False,
        sha256={f:hashlib.sha256((out/f).read_bytes()).hexdigest() for f in files}))


def run(a):
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    state=dict(complete=False,status='starting',started_unix=time.time(),phases=[])
    save(root/'status.json',state)
    try:
        source=a.input.resolve();original=json.loads(source.read_text())
        inp=copy.deepcopy(original);n=inp['model']['n'];d=inp['model']['d']
        if n!=len(inp['x']) or n%a.batch:raise ValueError('catalog must be complete and divisible by batch')
        if not inp.get('reuse_centers'):raise ValueError('source has no public updated center sets')
        if a.sets:inp['reuse_centers']=inp['reuse_centers'][:a.sets]
        inp['queries']=[list(range(s,s+a.batch)) for s in range(0,n,a.batch)]
        if a.full:
            inp['model'].update(method='full',basis=[],tail_rows=[],tail_signs=[],tail_width=0,seed='')
        phase='full_reuse' if a.full else 'reuse'
        task={k:inp[k] for k in ['reuse_centers','queries']}
        save(root/'input.json',inp);save(root/'approved_public_task.json',task)
        save(root/'protocol.json',dict(source_input=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            binary_sha256=hashlib.sha256(a.bin.read_bytes()).hexdigest(),n=n,d=d,k=len(task['reuse_centers'][0]),
            center_sets=len(task['reuse_centers']),batch=a.batch,reps=a.reps,warmup=a.warmup,
            all_catalog_rows_proved=True,full_catalog_extrapolated=False,setup_reused_across_centers=True,
            input_scale=inp['model']['input_scale'],basis_scale=inp['model']['basis_scale'],label=a.label,
            cost_policy='native prepare + compile + setup + native answer construction + witness + median measured prove; verify, warmup, fitting, I/O and public center preparation separate'))
        bundle=root/'bundle'

        def execute(kind, target=bundle):
            state.update(status='running',phase=kind);save(root/'status.json',state)
            cmd=[str(a.bin.resolve()),'-phase',kind,'-instance',str(root/'input.json'),'-out',str(target),
                 '-q',str(a.batch),'-reps',str(a.reps),'-warmup',str(a.warmup),'-max-constraints',str(a.limit)]
            if kind=='verify':cmd+=['-trust',str(root/'owner_setup_pin.json'),'-public-task',str(root/'approved_public_task.json')]
            stem=kind+('_public' if target!=bundle else '')
            resources=bounded(cmd,root/(stem+'.log'),seconds=a.phase_hours*3600)
            name=f'{kind}_{a.batch}.json' if kind in ['reuse','full_reuse','reuse_certified'] else kind+'.json'
            data=json.loads((target/name).read_text()) if (target/name).exists() else {}
            state['phases'].append(dict(phase=stem,resource=resources,status=data.get('status'),verified=data.get('verified')))
            save(root/'status.json',state)
            if resources['stop'] or resources['returncode']!=0 or not data.get('complete'):
                raise RuntimeError(f'{stem}: {resources} {data.get("error")} {data.get("status")}')
            return data

        prep=execute('prepare')
        source_prepare=source.parent/'prepare.json'
        if source_prepare.exists():
            owner_root=json.loads(source_prepare.read_text())['statement']['cx']
            if prep['statement']['cx']!=owner_root:raise RuntimeError('control changed the original committed catalog')
        stage1=None if a.full else execute('stage1')
        query=execute(phase)
        certified=execute('reuse_certified') if a.certify and not a.full else None
        shutil.copy2(bundle/'trusted.json',root/'owner_setup_pin.json')
        public_export(bundle,root/'public_bundle')
        verified=execute('verify',root/'public_bundle')
        xi=np.asarray(inp['x'],dtype=np.int64)
        rows=[]
        for s,centers in enumerate(task['reuse_centers']):
            ci=np.asarray(centers,dtype=np.int64)
            distances=((xi[:,None,:]-ci[None,:,:])**2).sum(2)
            exact_labels=distances.argmin(1);exact_phi=int(distances[np.arange(n),exact_labels].sum())
            samples=[x for x in query['samples'] if x.get('centre_set',0)==s]
            samples.sort(key=lambda x:x['id'])
            if [i for x in samples for i in x['indices']]!=list(range(n)):raise RuntimeError('missing/duplicate catalog row coverage')
            labels=np.asarray([i for x in samples for i in x['assign']])
            psi=sum(int(x['psi']) for x in samples)
            if a.full and (psi!=exact_phi or not np.array_equal(labels,exact_labels)):
                raise RuntimeError('dynamic Full differs from exact quantized reference')
            prove=sum(float(np.median(x['prove_seconds']))+x['witness_seconds']+x.get('native_answer_seconds',0.) for x in samples)
            preprocessing=0. if a.full else stage1['compile_seconds']+stage1['setup_seconds']+sum(x['witness_seconds']+np.median(x['prove_seconds']) for x in stage1['samples'])
            first=prep['native_prepare_seconds']+query['compile_seconds']+query['setup_seconds']+preprocessing if s==0 else 0.
            row=dict(method=a.label,set=s,n=n,batch=a.batch,proofs=len(samples),
                agreement_quantized=float(np.mean(labels==exact_labels)),
                witness_prove_seconds=prove,one_time_seconds=first,total_seconds=first+prove,
                query_constraints=query['constraints'],quantized_reference_cost=exact_phi,
                claimed_integer_cost=str(psi),cost_scale=1 if a.full else inp['model']['basis_scale']**4,
                independent_public_verified=verified['complete'])
            if certified:
                cs=sorted([x for x in certified['samples'] if x.get('centre_set',0)==s],key=lambda x:x['id'])
                native=certificate(inp['model'],inp['x'],centers)
                lo=sum(int(x['cost_lower_scaled']) for x in cs);hi=sum(int(x['cost_upper_scaled']) for x in cs)
                count=sum(x.get('certified_count',0) for x in cs)
                flags=[flag for x in cs for flag in x['certified_flags']]
                if (lo,hi,count,flags)!=(native['cost_lower_scaled'],native['cost_upper_scaled'],native['certified_count'],native['certified_flags']):
                    raise RuntimeError('proved certificate differs from independent integer implementation')
                if not lo<=exact_phi*inp['model']['basis_scale']**6<=hi:raise RuntimeError('interval missed exact quantized cost')
                if any(flag and label!=true for flag,label,true in zip(flags,labels,exact_labels)):
                    raise RuntimeError('certificate endorsed an incorrect assignment')
                cert_query=sum(np.median(x['prove_seconds'])+x['witness_seconds']+x.get('native_answer_seconds',0.) for x in cs)
                cert_first=prep['native_prepare_seconds']+preprocessing+certified['compile_seconds']+certified['setup_seconds'] if s==0 else 0.
                row.update(certificate_count=count,certificate_fraction=count/n,cost_lower_scaled=str(lo),cost_upper_scaled=str(hi),
                    certificate_interval_width=(hi-lo)/lo if lo else None,certificate_constraints=certified['constraints'],
                    certificate_query_seconds=float(cert_query),certificate_one_time_seconds=float(cert_first),
                    certificate_total_seconds=float(cert_query+cert_first),certificate_false_positives=0,
                    exact_label_policy_accept=count==n,
                    coverage99_cost1_policy_accept=(100*count>=99*n and lo>0 and 100*(hi-lo)<=lo))
            rows.append(row)
        pd.DataFrame(rows).to_csv(root/'workload.csv',index=False)
        summary=dict(complete=True,n=n,sets=len(task['reuse_centers']),method=a.label,
            original_root=prep['statement']['cx'],
            total_seconds=sum(x['total_seconds'] for x in rows),query_constraints=query['constraints'],
            verified_public_samples=verified['verified'],full_catalog_extrapolated=False,
            certificate_constraints=certified['constraints'] if certified else None,
            certificate_total_seconds=sum(x.get('certificate_total_seconds',0.) for x in rows) if certified else None,
            exact_label_policy_sets=sum(x.get('exact_label_policy_accept',False) for x in rows) if certified else None,
            coverage99_cost1_policy_sets=sum(x.get('coverage99_cost1_policy_accept',False) for x in rows) if certified else None,
            native_certificate_reference_matches=bool(certified),
            peak_rss_bytes=max(x['resource']['peak_rss_bytes'] for x in state['phases']))
        save(root/'summary.json',summary)
        state.update(complete=True,status='complete',finished_unix=time.time(),summary=summary)
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(root/'status.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--bin',type=Path,required=True);p.add_argument('--full',action='store_true')
    p.add_argument('--certify',action='store_true');p.add_argument('--label',default='full_public_centers')
    p.add_argument('--batch',type=int,default=8);p.add_argument('--reps',type=int,default=1)
    p.add_argument('--warmup',type=int,default=1);p.add_argument('--sets',type=int)
    p.add_argument('--limit',type=int,default=1_600_000);p.add_argument('--phase-hours',type=float,default=12)
    a=p.parse_args();run(a)


if __name__=='__main__':main()
