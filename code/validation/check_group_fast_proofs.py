"""Interactive group-protocol smoke with public-only replay and tampering.

The auditor role runs outside the prover executable after its commit receipt.
This local simulation is not an authenticated network challenge service.
"""
from pathlib import Path
import argparse,copy,json,os,secrets,shutil,subprocess,time
import numpy as np
from campaigns.local_fast_tradeoff import save,sha

FIELD=21888242871839275222246405745257275088548364400416034343698204186575808495617


def issue_challenge(bundle,owner,q):
    receipt=json.loads((bundle/'group_commit_receipt.json').read_text())
    trust=json.loads((bundle/'trusted.json').read_text())
    assert trust['original_root']==receipt['statement']['cx']
    assert receipt['version'] in ['group-fingerprint-v1','group-exact-old-v2','group-compact-v3','group-moments-only-v1']
    pin=dict(version=receipt['version'],owner=trust,statement=receipt['statement'],
        header=dict(GroupRoot=receipt['GroupRoot'],StatsRoot=receipt['StatsRoot'],Challenge=secrets.randbelow(FIELD-1)+1),
        commit_sha256=sha(bundle/'group_commit_receipt.json'),q=q)
    # This file is independently retained by the simulated auditor. The prover
    # cannot replace it by passing a different pin to public replay.
    if owner.exists():raise FileExistsError(owner)
    save(owner,pin)
    save(owner.with_name('challenge_receipt.json'),dict(commit_sha256=pin['commit_sha256'],challenge=pin['header']['Challenge'],issued_unix=time.time(),role='local auditor simulation, post-commit independent randomness'))
    return pin


def export_public(bundle,target):
    target.mkdir(parents=True,exist_ok=False)
    receipt=json.loads((bundle/'group_commit_receipt.json').read_text())
    compact=receipt['version']=='group-compact-v3'
    files=['group_commit_receipt.json','group_public_rows.json','group_rows.json','group_query.json','group_query_records.json']
    if compact:files+=['keys/group_row_compact.vk','keys/group_linked_change.vk']
    else:
        files+=['group_link.json']
        row_key='group_row_exact' if receipt['version']=='group-exact-old-v2' else 'group_row'
        files+=['keys/'+name+'.vk' for name in [row_key,'group_link','group_moment']]
    # Copy only keys/proofs and explicitly public metadata. Never copy gob,
    # private H/E tables, membership paths, masks or setup/proof caches.
    files += [str(p.relative_to(bundle)).replace('\\','/') for p in (bundle/'keys').glob('group_switch_*.vk')]
    for phase in ['group_rows','group_link','group_query']:
        files += [str(p.relative_to(bundle)).replace('\\','/') for p in (bundle/phase).glob('*.proof')]
    for name in files:
        dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(bundle/name,dest)
    save(target/'public_manifest.json',dict(sha256={name:sha(target/name) for name in files}))


def run(root,binary,exact_old=False,compact=False):
    root.mkdir(parents=True,exist_ok=False);bundle=root/'bundle';q=2
    rows=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]]
    old=[[0,0,0],[8,8,0]]
    model=dict(version='bound-audit-v1',method='learned_energy',n=4,d=3,k=2,input_bits=4,basis_scale=8,input_scale='1',centers=old,basis=[[8,0,0],[0,8,0]],tail_width=0,tail_rows=[],tail_signs=[],seed='')
    task=dict(queries=[[0,1],[2,3]],reuse_centers=[old,[[1,0,0],[7,8,0]],[[0,0,8],[8,8,8]]],comparison_base=old)
    save(root/'input.json',dict(model=model,x=rows,salt_x='11',salt_z='13',**task));save(root/'approved.json',task)
    env=dict(os.environ,COMMITJL_RESUME_PROOFS='1',GOMAXPROCS='2',GOMEMLIMIT='1000MiB');counter=0
    def invoke(phase,target,success=True):
        nonlocal counter
        counter+=1
        command=[str(binary),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(target),'-q',str(q),'-reps','1','-warmup','0','-max-constraints','1600000','-trust',str(root/'owner_pin.json'),'-public-task',str(root/'approved.json')]
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
        (root/f'{counter:02d}_{phase}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        assert (result.returncode==0)==success,result.stdout+result.stderr
    if compact:invoke('group_compact_prepare',bundle)
    else:invoke('prepare',bundle);invoke('group_commit_exact' if exact_old else 'group_commit',bundle)
    pin=issue_challenge(bundle,root/'owner_pin.json',q)
    invoke('group_keys',bundle)
    # Trusted setup hashes are retained independently before the proof worker.
    pin['owner']=json.loads((bundle/'trusted.json').read_text());save(root/'owner_pin.json',pin)
    for phase in (['group_rows','group_query'] if compact else ['group_rows','group_link','group_query']):invoke(phase,bundle)
    public=root/'public';export_public(bundle,public);invoke('group_verify',public)
    expected_proofs=13 if compact else 10 if exact_old else 14
    result=json.loads((public/'group_verify.json').read_text());assert result['verified']==expected_proofs
    decisions=json.loads((public/'group_verified_decisions.json').read_text())
    x=np.asarray(rows,dtype=object)
    def cost(centers):return sum(min(sum((int(v)-int(c))**2 for v,c in zip(row,center)) for center in centers) for row in x)
    for record,centers in zip(decisions,task['reuse_centers']):
        truth=(cost(centers)-cost(old))*int(record['denominator'])
        assert int(record['lower'])<=truth<=int(record['upper'])
    cases=[]
    def mutate(name,change):
        path=public/name;before=json.loads(path.read_text());after=copy.deepcopy(before);change(after);save(path,after)
        invoke('group_verify',public,False);save(path,before);cases.append(name)
    mutate('group_public_rows.json',lambda v:v.pop())
    mutate('group_public_rows.json',lambda v:v.__setitem__(1,copy.deepcopy(v[0])))
    mutate('group_public_rows.json',lambda v:v[0]['Masked'].__setitem__(0,v[0]['Masked'][0]+1))
    mutate('group_query_records.json',lambda v:v[2 if exact_old and not compact else 0]['batches'].pop())
    mutate('group_query_records.json',lambda v:v[0].__setitem__('assigned_change',str(int(v[0]['assigned_change'])+1)))
    mutate('group_query_records.json',lambda v:v[0].__setitem__('Decision','worsening'))
    if exact_old and not compact:
        records=json.loads((public/'group_query_records.json').read_text())
        assert [v['mode'] for v in records]==['identity','assigned_upper_bound','switch']
        assert records[2]['Decision']=='worsening'
        mutate('group_query_records.json',lambda v:v[2].update(mode='assigned_upper_bound',batches=[]))
        mutate('group_query_records.json',lambda v:v[1].__setitem__('mode','identity'))
        changed=copy.deepcopy(pin);changed['version']='group-fingerprint-v1'
        save(root/'owner_pin.json',changed);invoke('group_verify',public,False);save(root/'owner_pin.json',pin)
    if compact:
        mutate('group_query_records.json',lambda v:v[0]['assigned'].__setitem__('file','unrelated.proof'))
        mutate('group_query.json',lambda v:v['samples'].pop())
        changed=copy.deepcopy(pin);changed['version']='group-exact-old-v2'
        save(root/'owner_pin.json',changed);invoke('group_verify',public,False);save(root/'owner_pin.json',pin)
    changed=copy.deepcopy(pin);changed['header']['Challenge']=1 if pin['header']['Challenge']!=1 else 2
    save(root/'owner_pin.json',changed);invoke('group_verify',public,False);save(root/'owner_pin.json',pin)
    changed=copy.deepcopy(task);changed['reuse_centers'][1][0][0]+=1
    save(root/'approved.json',changed);invoke('group_verify',public,False);save(root/'approved.json',task)
    changed=copy.deepcopy(task);changed['queries'][1]=changed['queries'][0]
    save(root/'approved.json',changed);invoke('group_verify',public,False);save(root/'approved.json',task)
    invoke('group_verify',public)
    assert not (public/'prepared.gob').exists() and not (public/'group_private.gob').exists()
    checks=dict(complete=True,fresh_public_proofs=expected_proofs,intervals_contain_full=True,public_only=True,tamper_checks=12 if exact_old or compact else 9,post_commit_random_auditor_challenge=True,binary_sha256=sha(binary))
    if exact_old:checks['exact_old_profile']=True
    if compact:checks['compact_profile']=True
    save(root/'checks.json',checks)
    print(json.dumps(dict(complete=True,proofs=expected_proofs,checks=str(root/'checks.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--exact-old',action='store_true');p.add_argument('--compact',action='store_true');a=p.parse_args();run(a.root.resolve(),a.binary.resolve(),a.exact_old,a.compact)
