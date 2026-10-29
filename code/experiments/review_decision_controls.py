"""Fully covered, publicly replayed prototype-update decision controls.

Compare exact Full, independent absolute intervals, and paired change bounds.
All requests (including unchanged and inconclusive cases) are retained. No
accuracy threshold is used: only a strict sign of the certified cost change.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.bound_audit import bounded,save
from experiments.review_controls import public_export
from commitjl.paired_certificates import paired_change
from commitjl.reuse_certificates import certificate


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decision(lo,hi):return 'improves' if hi<0 else ('worsens' if lo>0 else 'inconclusive')
def exact_cost(rows,centers):
    x=np.asarray(rows,dtype=np.int64);c=np.asarray(centers,dtype=np.int64)
    if (int(np.abs(x).max())+int(np.abs(c).max()))**2*x.shape[1]*len(x)>=2**63:
        raise OverflowError('exact reference int64 bound')
    return int(((x[:,None,:]-c[None,:,:])**2).sum(2).min(1).sum())


def run(a):
    root=a.out.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();state=dict(complete=False,status='starting',phases=[])
    save(root/'status.json',state)
    try:
        owner=json.loads(a.input.read_text());inp=copy.deepcopy(owner);n=len(inp['x'])
        if inp['model']['n']!=n or n%a.batch:raise ValueError('complete divisible catalog required')
        inp['comparison_base']=copy.deepcopy(inp['model']['centers'])
        inp['queries']=[list(range(i,i+a.batch)) for i in range(0,n,a.batch)]
        if a.full:inp['model'].update(method='full',basis=[],tail_rows=[],tail_signs=[],tail_width=0,seed='')
        task={k:inp[k] for k in ['comparison_base','queries','reuse_centers']}
        save(root/'input.json',inp);save(root/'approved_public_task.json',task)
        save(root/'protocol.json',dict(source_sha256=sha(a.input),binary_sha256=sha(a.bin),
            n=n,batch=a.batch,reps=a.reps,warmup=a.warmup,full=a.full,also_absolute=a.also_absolute,
            alternative='exact Full' if a.full else 'paired original-space cost change',
            metadata=inp.get('experiment_metadata'),
            decision='strict sign of true quantized minimum-cost difference; inconclusive retained',
            timing='Query reps as requested; Stage1 timed once per row. Phase-specific public-only replay includes key loading and verifier center-parameter preparation.'))
        bundle=root/'bundle'
        def execute(phase,target=bundle,pin=None):
            state.update(status='running',phase=phase);save(root/'status.json',state)
            cmd=[str(a.bin.resolve()),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(target),
                 '-q',str(a.batch),'-reps',str(a.reps),'-warmup',str(a.warmup),'-max-constraints',str(a.limit)]
            if phase=='verify':cmd+=['-trust',str(pin),'-public-task',str(root/'approved_public_task.json')]
            stem=phase+('_'+pin.stem if pin else '')
            resources=bounded(cmd,root/(stem+'.log'),seconds=a.phase_hours*3600)
            name=f'{phase}_{a.batch}.json' if phase in ['full_reuse','reuse_certified','paired_change'] else phase+'.json'
            data=json.loads((target/name).read_text()) if (target/name).exists() else {}
            state['phases'].append(dict(phase=stem,resource=resources,complete=data.get('complete'),verified=data.get('verified')))
            save(root/'status.json',state)
            if resources['stop'] or resources['returncode'] or not data.get('complete'):
                raise RuntimeError(f'{stem}: {resources}; {data.get("error")}')
            return data,resources
        prep,_=execute('prepare')
        stage=None if a.full else execute('stage1')[0]
        phases=['full_reuse'] if a.full else (['reuse_certified','paired_change'] if a.also_absolute else ['paired_change'])
        measured={}
        for phase in phases:measured[phase]=execute(phase)[0]
        original_pin=json.loads((bundle/'trusted.json').read_text())
        public_export(bundle,root/'public_bundle')
        replays={}
        for phase in phases:
            pin=copy.deepcopy(original_pin)
            pin['verifying_key_sha256']={k:v for k,v in pin['verifying_key_sha256'].items() if k=='stage1' or k.startswith(phase+'_')}
            path=root/('owner_pin_'+phase+'.json');save(path,pin)
            verified,resource=execute('verify',root/'public_bundle',path)
            if verified['verified']!=(0 if a.full else n)+len(inp['reuse_centers'])*(n//a.batch):
                raise ValueError('phase-specific public replay lacks coverage')
            replays[phase]=resource['wall_seconds']
            shutil.copy2(root/'public_bundle/verify.json',root/('verified_'+phase+'.json'))
        workflow_wall=time.monotonic()-started  # Before private numerical cross-checks.
        base=exact_cost(inp['x'],inp['comparison_base']);den=int(inp['model']['basis_scale'])**6
        old=None if a.full or not a.also_absolute else certificate(inp['model'],inp['x'],inp['comparison_base'])
        rows=[]
        for set_id,centers in enumerate(inp['reuse_centers']):
            truth=exact_cost(inp['x'],centers)-base
            for phase,data in measured.items():
                samples=[s for s in data['samples'] if s.get('centre_set',0)==set_id]
                if [i for s in samples for i in s['indices']]!=list(range(n)):raise ValueError('missing or duplicate rows')
                if any(len(s['prove_seconds'])!=a.reps for s in samples):raise ValueError('query repetition mismatch')
                if phase=='full_reuse':
                    cost=sum(int(s['psi']) for s in samples)
                    if cost!=truth+base:raise ValueError('Full cost differs from integer oracle')
                    lo=hi=truth;scale=1
                elif phase=='paired_change':
                    lo=sum(int(s['change_lower_scaled']) for s in samples);hi=sum(int(s['change_upper_scaled']) for s in samples);scale=den
                    ref=paired_change(inp['model'],inp['x'],inp['comparison_base'],centers)
                    if (lo,hi)!=(ref['change_lower_scaled'],ref['change_upper_scaled']):raise ValueError('paired native/proved mismatch')
                else:
                    absolute=certificate(inp['model'],inp['x'],centers)
                    l=sum(int(s['cost_lower_scaled']) for s in samples);u=sum(int(s['cost_upper_scaled']) for s in samples)
                    if (l,u)!=(absolute['cost_lower_scaled'],absolute['cost_upper_scaled']):raise ValueError('absolute native/proved mismatch')
                    lo=l-old['cost_upper_scaled'];hi=u-old['cost_lower_scaled'];scale=den
                if not lo<=truth*scale<=hi:raise ValueError('change interval missed exact integer reference')
                verdict=decision(lo,hi)
                if (verdict=='improves' and truth>=0) or (verdict=='worsens' and truth<=0):raise ValueError('incorrect certified decision')
                roles=inp.get('experiment_metadata',{}).get('roles',[])
                kind=roles[set_id]['kind'] if roles else 'smoke'
                rows.append(dict(phase=phase,set=set_id,n=n,true_change=truth,update_kind=kind,
                    change_lower_scaled=str(lo),change_upper_scaled=str(hi),cost_denominator=str(scale),decision=verdict,
                    unchanged_centers=centers==inp['comparison_base'],
                    change_width_over_baseline=(hi-lo)/scale/base if base else None,
                    query_native_witness_prove_seconds=sum(s.get('native_answer_seconds',0)+s['witness_seconds']+float(np.median(s['prove_seconds'])) for s in samples),
                    query_verify_seconds=sum(float(np.median(s['verify_seconds'])) for s in samples)))
        pd.DataFrame(rows).to_csv(root/'decisions.csv',index=False)
        shared=prep['native_prepare_seconds']
        if stage:shared+=stage['compile_seconds']+stage['setup_seconds']+sum(s['witness_seconds']+float(np.median(s['prove_seconds'])) for s in stage['samples'])
        alternatives=[]
        for phase,data in measured.items():
            selected=[r for r in rows if r['phase']==phase]
            setup=data['compile_seconds']+data['setup_seconds']
            public_parameters=data.get('native_prepare_seconds',0.)
            query=public_parameters+sum(r['query_native_witness_prove_seconds'] for r in selected)
            proof_bytes=sum(p.stat().st_size for sub in ([phase] if a.full else ['stage1',phase]) for p in (bundle/sub).glob('*.proof'))
            key_bytes=sum((bundle/'keys'/(k+'.vk')).stat().st_size for k in original_pin['verifying_key_sha256'] if k=='stage1' or k.startswith(phase+'_'))
            alternatives.append(dict(phase=phase,shared_preparation_seconds=shared,query_compile_setup_seconds=setup,
                query_components_seconds=query,total_components_seconds=shared+setup+query,
                public_center_parameter_seconds=public_parameters,
                public_only_replay_wall_seconds=replays[phase],components_plus_public_replay_seconds=shared+setup+query+replays[phase],
                proves=len(data['samples']),proof_bytes=proof_bytes,verifying_key_bytes=key_bytes,
                proving_key_bytes=data.get('proving_key_bytes',0)+(stage.get('proving_key_bytes',0) if stage else 0),
                public_witness_serialized_bytes=sum(s.get('public_witness_bytes',0) for s in data['samples'])+(sum(s.get('public_witness_bytes',0) for s in stage['samples']) if stage else 0),
                model_json_bytes=len(json.dumps(inp['model']).encode()),approved_task_bytes=(root/'approved_public_task.json').stat().st_size,
                stored_head_energy_scalars=n*(len(inp['model']['basis'])+1) if not a.full else 0,
                genuine_requests=sum(not r['unchanged_centers'] for r in selected),
                conclusive_genuine_requests=sum(not r['unchanged_centers'] and r['decision']!='inconclusive' for r in selected),
                lloyd_requests=sum(r['update_kind']=='lloyd_update' for r in selected),
                conclusive_lloyd_requests=sum(r['update_kind']=='lloyd_update' and r['decision']!='inconclusive' for r in selected)))
        summary=dict(complete=True,n=n,sets=len(inp['reuse_centers']),alternatives=alternatives,
            original_root=prep['statement']['cx'],workflow_wall_seconds=workflow_wall,
            actual_single_repetition_deployment=(a.reps==1 and a.warmup==0 and len(phases)==1),
            private_reference_validation_excluded_from_workflow_wall=True,
            numerical_false_decisions=0,metadata=inp.get('experiment_metadata'))
        save(root/'summary.json',summary);state.update(complete=True,status='complete',summary=summary)
    except BaseException as e:
        state.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save(root/'status.json',state)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['input','out','bin']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--full',action='store_true');p.add_argument('--also-absolute',action='store_true')
    p.add_argument('--batch',type=int,default=4);p.add_argument('--reps',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--limit',type=int,default=1_600_000);p.add_argument('--phase-hours',type=float,default=12)
    run(p.parse_args())


if __name__=='__main__':main()
