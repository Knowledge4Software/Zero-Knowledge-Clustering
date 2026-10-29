"""Real known-family proof smoke, Full oracle, and public-only tamper checks."""
from pathlib import Path
import argparse,json,os,subprocess
import numpy as np
from campaigns.local_fast_tradeoff import save,sha
from experiments.review_controls import public_export


def run(root,binary):
    root.mkdir(parents=True,exist_ok=False)
    bank=[[0,0,0],[8,0,0],[0,8,0],[8,8,0]];rows=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]]
    model=dict(version='bound-audit-v1',method='affine',n=4,d=3,k=4,input_bits=4,basis_scale=8,input_scale='1',centers=bank,basis=[],tail_rows=[],tail_signs=[],tail_width=0,seed='')
    task=dict(queries=[[0,1],[2,3]],reuse_centers=[[bank[0],bank[1]],[bank[3],bank[2]]])
    save(root/'input.json',dict(model=model,x=rows,salt_x='11',salt_z='13',**task));save(root/'approved.json',task)
    env=dict(os.environ,COMMITJL_RESUME_PROOFS='1',GOMAXPROCS='2',GOMEMLIMIT='1000MiB');bundle=root/'bundle';counter=0
    def invoke(phase,target,success=True):
        nonlocal counter
        counter+=1
        command=[str(binary),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(target),'-q','2','-reps','1','-warmup','0','-max-constraints','1600000']
        if phase=='verify':command+=['-trust',str(root/'owner_pin.json'),'-public-task',str(root/'approved.json')]
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=240)
        (root/f'{counter:02d}_{phase}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        assert (result.returncode==0)==success,result.stdout+result.stderr
    for phase in ['prepare','stage1','family_affine']:invoke(phase,bundle)
    phase=json.loads((bundle/'family_affine_2.json').read_text());x=np.asarray(rows)
    for sample in phase['samples']:
        c=np.asarray(task['reuse_centers'][sample.get('centre_set',0)]);dist=((x[sample['indices'],None,:]-c[None,:,:])**2).sum(2)
        assert sample['assign']==dist.argmin(1).tolist() and int(sample['psi'])==int(dist.min(1).sum())
    save(root/'owner_pin.json',json.loads((bundle/'trusted.json').read_text()));public=root/'public';public_export(bundle,public);invoke('verify',public)
    replay=json.loads((public/'verify.json').read_text());assert replay['complete'] and replay['verified']==8
    original=json.loads((public/'family_affine_2.json').read_text());changed=json.loads(json.dumps(original));changed['samples'][0]['psi']=str(int(changed['samples'][0]['psi'])+1)
    save(public/'family_affine_2.json',changed);invoke('verify',public,False);save(public/'family_affine_2.json',original)
    outside=json.loads(json.dumps(task));outside['reuse_centers'][1][0]=[1,1,0];save(root/'approved.json',outside);invoke('verify',public,False)
    save(root/'approved.json',task);invoke('verify',public)
    assert not (public/'prepared.gob').exists() and not (public/'statistics.json').exists()
    save(root/'checks.json',dict(complete=True,fresh_public_proofs=8,all_costs_and_labels_match_full=True,modified_cost_rejected=True,out_of_family_center_rejected=True,private_rows_excluded=True,binary_sha256=sha(binary)))
    print(json.dumps(dict(complete=True,proofs=8,checks=str(root/'checks.json'))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);a=p.parse_args();run(a.root.resolve(),a.binary.resolve())
