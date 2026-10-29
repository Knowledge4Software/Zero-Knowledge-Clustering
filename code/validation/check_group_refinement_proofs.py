"""Real Groth16 refinement smoke, full public replay and transcript tampering."""
from pathlib import Path
import argparse, copy, json, os, shutil, subprocess, time
import numpy as np
from campaigns.local_fast_tradeoff import save, sha
from validation.check_group_fast_proofs import issue_challenge, export_public
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.jl_certificate_pilots import hint_order


def read(path): return json.loads(Path(path).read_text())


def run(root, binary):
    root.mkdir(parents=True, exist_ok=False)
    rows=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]]
    old=[[0,0,0],[8,8,0]]
    centers=[old,[[1,2,0],[5,6,1]],[[0,3,1],[8,5,3]],[[0,0,-1],[8,8,0]]]
    model=dict(version='bound-audit-v1',method='learned_energy',n=4,d=3,k=2,input_bits=4,
        basis_scale=8,input_scale='1',centers=old,basis=[[8,0,0]],tail_width=0,tail_rows=[],tail_signs=[],seed='')
    task=dict(queries=[[0,1],[2,3]],reuse_centers=centers,comparison_base=old)
    save(root/'input.json',dict(model=model,x=rows,salt_x='11',salt_z='13',**task))
    save(root/'approved.json',task)
    base=root/'base_private';extra=root/'refinement_private';public=root/'refinement_public'
    pin=root/'base_owner_pin.json';keypin=root/'refinement_owner_pin.json'
    env=dict(os.environ,COMMITJL_RESUME_PROOFS='1',GOMAXPROCS='2',GOMEMLIMIT='512MiB')
    logs=[]
    def invoke(phase,target,base_path=None,ok=True):
        args=[str(binary),'-phase',phase,'-out',str(target),'-instance',str(root/'input.json'),
            '-trust',str(pin),'-public-task',str(root/'approved.json'),'-q','2','-max-constraints','1600000']
        if base_path is not None:
            args+=['-refinement-base',str(base_path),'-refinement-trust',str(keypin),'-refinement-request',str(root/'requests.json')]
        start=time.monotonic();r=subprocess.run(args,capture_output=True,text=True,env=env,timeout=240)
        log=root/f'{len(logs):02d}_{phase}.log';log.write_text(r.stdout+r.stderr,encoding='utf-8')
        logs.append(dict(phase=phase,success=r.returncode==0,wall_seconds=time.monotonic()-start))
        assert (r.returncode==0)==ok,r.stdout+r.stderr
    invoke('group_compact_prepare',base)
    issue_challenge(base,pin,2);invoke('group_keys',base)
    owner=read(pin);owner['owner']=read(base/'trusted.json');save(pin,owner)
    invoke('group_rows',base);invoke('group_query',base)
    base_public=root/'base_public';export_public(base,base_public);invoke('group_verify',base_public)
    x=np.asarray(rows,dtype=np.int64);c=np.asarray(old,dtype=np.int64)
    od=((x[:,None,:]-c[None,:,:])**2).sum(2);groups=od.argmin(1)
    prepared=prepare_groups(x,groups.tolist(),len(old));orders=[]
    for new in centers:
        bounds=exact_old_correction_bounds(model,x,prepared,old,new)
        orders.append(hint_order(bounds,None,'width')[:16])
    assert any(r['decision']=='inconclusive' for r in read(base_public/'group_verified_decisions.json'))
    save(root/'requests.json',dict(sets=orders))
    invoke('refinement_setup',extra,base_public)
    save(keypin,read(extra/'trusted.json'))
    invoke('refinement_prove',extra,base)
    public.mkdir()
    files=['refinement_records.json','keys/group_refinement_1.vk']+[p.relative_to(extra).as_posix() for p in (extra/'refinement').glob('*.proof')]
    for name in files:
        target=public/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(extra/name,target)
    save(public/'public_manifest.json',dict(sha256={name:sha(public/name) for name in files}))
    invoke('refinement_verify',public,base_public)
    decisions=read(public/'refinement_verified_decisions.json')
    count=read(public/'refinement_verify.json')['verified']-read(base_public/'group_verify.json')['verified']
    assert count>0 and any(r['refined_rows']>0 and r['decision']!='inconclusive' for r in decisions)
    for r,new in zip(decisions,centers):
        cc=np.asarray(new,dtype=np.int64);truth=int(((x[:,None,:]-cc[None,:,:])**2).sum(2).min(1).sum()-od.min(1).sum())*int(r['denominator'])
        assert int(r['lower'])<=truth<=int(r['upper'])
    cases=[]
    def mutate(path,change):
        before=read(path);after=copy.deepcopy(before);change(after);save(path,after)
        invoke('refinement_verify',public,base_public,False);save(path,before);cases.append(str(path.relative_to(root)))
    selected=next(i for i,r in enumerate(read(public/'refinement_records.json')) if r['samples'])
    records=public/'refinement_records.json'
    mutate(records,lambda v:v[selected]['samples'][0].__setitem__('change_lower_scaled',str(int(v[selected]['samples'][0]['change_lower_scaled'])+1)))
    mutate(records,lambda v:v[selected]['samples'].append(copy.deepcopy(v[selected]['samples'][0])))
    mutate(records,lambda v:v.pop())
    mutate(records,lambda v:v[selected].__setitem__('Decision','improvement' if v[selected]['Decision']!='improvement' else 'worsening'))
    mutate(records,lambda v:v[selected]['samples'][0].__setitem__('file','../unapproved.proof'))
    mutate(root/'approved.json',lambda v:v['reuse_centers'][1][0].__setitem__(0,3))
    mutate(base_public/'group_public_rows.json',lambda v:v.pop())
    proof=public/read(records)[selected]['samples'][0]['file'];original=proof.read_bytes();proof.write_bytes(bytes([original[0]^1])+original[1:])
    invoke('refinement_verify',public,base_public,False);proof.write_bytes(original);cases.append('proof bytes')
    invoke('refinement_verify',public,base_public)
    assert not list(public.rglob('*.gob')) and not list(public.rglob('*.pk'))
    unchanged={name:sha(extra/name) for name in files if name.endswith('.proof')}
    before_records=read(extra/'refinement_records.json')
    invoke('refinement_setup',extra,base_public)
    invoke('refinement_prove',extra,base)
    assert all(sha(extra/name)==digest for name,digest in unchanged.items()),'Resume regenerated proof or changed the transcript'
    after_records=read(extra/'refinement_records.json')
    for records in [before_records,after_records]:
        for record in records:
            for sample in record['samples']:sample.pop('native_answer_seconds',None)
    assert before_records==after_records,'Resume changed the verified statement or original proof measurements'
    invoke('refinement_verify',public,base_public)
    result=dict(complete=True,binary_sha256=sha(binary),public_only=True,base_replayed=True,
        new_refinement_proofs=count,base_proofs=read(base_public/'group_verify.json')['verified'],
        tamper_checks=len(cases),tamper_cases=cases,all_intervals_contain_full=True,resume_keeps_proofs=True,decisions=decisions,phases=logs)
    save(root/'checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True)
    a=p.parse_args();run(a.root.resolve(),a.binary.resolve())
