"""Exercise real relative-certificate proofs, resume, and public-only replay."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

from campaigns.local_fast_tradeoff import resource_record
from commitjl.relative_certificates import relative_certificate
from experiments.review_controls import public_export


def save(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, binary):
    root.mkdir(parents=True, exist_ok=False)
    model=dict(version='bound-audit-v1', method='learned_energy', n=4, d=4,
        k=2, input_bits=4, basis_scale=8, input_scale='1',
        centers=[[0,0,0,0],[8,0,0,0]], basis=[[7,1,0,0],[0,7,1,0]],
        tail_width=0, tail_rows=[], tail_signs=[], seed='')
    rows=[[0,0,0,0],[1,1,0,0],[6,0,1,0],[-5,1,0,0]]
    task=dict(queries=[[0,1],[2,3]], reuse_centers=[model['centers'],[[0,1,0,0],[6,1,0,0]]])
    save(root/'input.json',dict(model=model,x=rows,salt_x='13',salt_z='17',**task))
    save(root/'approved.json',task)
    env=dict(os.environ, COMMITJL_RESUME_PROOFS='1',GOMAXPROCS='2',GOMEMLIMIT='1200MiB')
    timings=[]
    def invoke(phase, bundle, success=True):
        command=[str(binary),'-phase',phase,'-instance',str(root/'input.json'),'-out',str(bundle),'-q','2','-reps','1','-warmup','0','-max-constraints','1600000']
        if phase=='verify':command+=['-trust',str(root/'owner_pin.json'),'-public-task',str(root/'approved.json')]
        started=time.monotonic()
        result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=180)
        timings.append(dict(phase=phase,seconds=time.monotonic()-started,returncode=result.returncode,expected_success=success))
        (root/f'{len(timings):02d}_{phase}.log').write_text(result.stdout+result.stderr,encoding='utf-8')
        assert (result.returncode==0)==success, result.stdout+result.stderr
    bundle=root/'bundle'
    for phase in ['prepare','stage1','relative_reuse']:invoke(phase,bundle)
    originals={p.relative_to(bundle).as_posix():digest(p) for p in bundle.rglob('*') if p.suffix in ['.proof','.vk']}
    phase_data=json.loads((bundle/'relative_reuse_2.json').read_text())
    for sample in phase_data['samples']:
        subset=[rows[i] for i in sample['indices']]
        reference=relative_certificate(model,subset,task['reuse_centers'][sample.get('centre_set',0)])
        assert sample.get('relative_certified_count',0)==reference['certified_count']
        assert sample['assign']==reference['labels']
    for phase in ['stage1','relative_reuse']:
        invoke(phase,bundle)
        name=phase+'.json' if phase=='stage1' else phase+'_2.json'
        assert json.loads((bundle/name).read_text())['resumed_setup']
    assert originals=={p.relative_to(bundle).as_posix():digest(p) for p in bundle.rglob('*') if p.suffix in ['.proof','.vk']}
    save(root/'owner_pin.json',json.loads((bundle/'trusted.json').read_text()))
    public=root/'public';public_export(bundle,public)
    assert not any((public/name).exists() for name in ['prepared.gob','statistics.json','.resume'])
    exported=json.loads((public/'relative_reuse_2.json').read_text())
    assert all('assign' not in sample and 'certified_flags' not in sample for sample in exported['samples'])
    invoke('verify',public)
    receipt=json.loads((public/'verify.json').read_text())
    assert receipt['complete'] and receipt['verified']==8
    tampered=json.loads(json.dumps(exported));first=tampered['samples'][0]
    first['relative_certified_count']=(first.get('relative_certified_count',0)+1)%3
    save(public/'relative_reuse_2.json',tampered);invoke('verify',public,success=False)
    save(public/'relative_reuse_2.json',exported)
    changed=json.loads(json.dumps(task));changed['reuse_centers'][1][0][0]+=1
    save(root/'approved.json',changed);invoke('verify',public,success=False)
    save(root/'approved.json',task);invoke('verify',public)
    accounting=resource_record([dict(complete=False,running=False,wall_seconds=10,peak_rss_bytes=50),dict(complete=True,running=False,wall_seconds=7,peak_rss_bytes=40)])
    assert accounting['wall_seconds']==17 and accounting['resumed'] and accounting['complete']
    save(root/'checks.json',dict(complete=True,public_proofs=8,native_relative_counts_match=True,
        resumed_keys_and_proofs_unchanged=True,incorrect_public_count_rejected=True,
        changed_approved_centers_rejected=True,private_witness_excluded=True,
        resumed_timing_accumulates=True,timings=timings,binary_sha256=digest(binary)))
    print(json.dumps(dict(complete=True,public_proofs=8,checks=str(root/'checks.json'))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.binary.resolve())
