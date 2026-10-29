"""Projection-free baseline smoke: complete public replay and substitutions."""
from pathlib import Path
import argparse,copy,json,os,shutil,subprocess
from campaigns.local_fast_tradeoff import save,sha
from validation.check_group_fast_proofs import issue_challenge


def export_public(bundle,target):
    target.mkdir(parents=True,exist_ok=False)
    files=['group_commit_receipt.json','group_public_rows.json','moments_rows.json','moments_query.json','moments_query_records.json','keys/moments_row.vk','keys/moments_upper.vk']
    files += [p.relative_to(bundle).as_posix() for phase in ['moments_rows','moments_query'] for p in (bundle/phase).glob('*.proof')]
    for name in files:
        destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(bundle/name,destination)
    save(target/'public_manifest.json',dict(sha256={name:sha(target/name) for name in files}))


def run(root,binary):
    root.mkdir(parents=True,exist_ok=False);bundle=root/'bundle';rows=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]];old=[[0,0,0],[8,8,0]]
    model=dict(version='bound-audit-v1',method='affine',n=4,d=3,k=2,input_bits=4,basis_scale=8,input_scale='1',centers=old,basis=[],tail_width=0,tail_rows=[],tail_signs=[],seed='')
    task=dict(queries=[[0,1],[2,3]],reuse_centers=[old,[[1,0,0],[7,8,0]],[[0,0,8],[8,8,8]]],comparison_base=old)
    save(root/'input.json',dict(model=model,x=rows,salt_x='11',salt_z='13',**task));save(root/'approved.json',task)
    env=dict(os.environ,COMMITJL_RESUME_PROOFS='1',GOMAXPROCS='2',GOMEMLIMIT='1200MiB');counter=0
    def invoke(phase,target,success=True):
        nonlocal counter
        counter+=1
        command=[str(binary),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(target),'-q','2','-max-constraints','1600000','-trust',str(root/'owner_pin.json'),'-public-task',str(root/'approved.json')]
        r=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
        (root/f'{counter:02d}_{phase}.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert (r.returncode==0)==success,r.stdout+r.stderr
    invoke('moments_prepare',bundle);pin=issue_challenge(bundle,root/'owner_pin.json',2);invoke('moments_keys',bundle)
    pin['owner']=json.loads((bundle/'trusted.json').read_text());save(root/'owner_pin.json',pin)
    for phase in ['moments_rows','moments_query']:invoke(phase,bundle)
    public=root/'public';export_public(bundle,public);invoke('moments_verify',public)
    result=json.loads((public/'moments_verify.json').read_text());assert result['verified']==7
    decisions=json.loads((public/'moments_verified_decisions.json').read_text());assert decisions[0]['decision']=='equal' and decisions[2]['decision']=='inconclusive'
    def mutate(name,change):
        before=json.loads((public/name).read_text());after=copy.deepcopy(before);change(after);save(public/name,after);invoke('moments_verify',public,False);save(public/name,before)
    mutate('group_public_rows.json',lambda rows:rows.pop())
    mutate('group_public_rows.json',lambda rows:rows.__setitem__(1,copy.deepcopy(rows[0])))
    mutate('group_public_rows.json',lambda rows:rows[0]['Masked'].__setitem__(0,rows[0]['Masked'][0]+1))
    mutate('moments_query_records.json',lambda v:v[1].__setitem__('assigned_change',str(int(v[1]['assigned_change'])-1)))
    mutate('moments_query_records.json',lambda v:v[2].__setitem__('Decision','worsening'))
    mutate('moments_query.json',lambda v:v['samples'].pop())
    changed=copy.deepcopy(task);changed['reuse_centers'][1][0][0]+=1;save(root/'approved.json',changed);invoke('moments_verify',public,False);save(root/'approved.json',task)
    changed=copy.deepcopy(pin);changed['header']['Challenge']+=1;save(root/'owner_pin.json',changed);invoke('moments_verify',public,False);save(root/'owner_pin.json',pin)
    invoke('moments_verify',public)
    assert not (public/'prepared.gob').exists() and not (public/'group_private.gob').exists()
    save(root/'checks.json',dict(complete=True,fresh_public_proofs=7,tamper_checks=8,public_only=True,projection_free=True,binary_sha256=sha(binary)))
    print(json.dumps(dict(complete=True,root=str(root))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.binary.resolve())
