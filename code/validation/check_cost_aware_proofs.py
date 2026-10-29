"""Small fresh cryptographic test of the hybrid verifier and substitutions."""
from pathlib import Path
import argparse
import copy
import json
import os
import shutil
import subprocess

import numpy as np

from campaigns.published_baselines_laptop import GO, read, save, sha
from commitjl.cost_aware import Planner, ProofCosts
from validation.check_group_fast_proofs import issue_challenge, export_public
from validation.verify_cost_aware import verify


def run(root, binary):
    root.mkdir(parents=True, exist_ok=False)
    rows, old = [[0,0,1],[4,0,1],[7,7,2],[1,7,2]], [[0,0,0],[8,8,0]]
    model = dict(version='bound-audit-v1', method='learned_energy', n=4, d=3, k=2,
                 input_bits=4, basis_scale=8, input_scale='1', centers=old,
                 basis=[[8,0,0],[0,8,0]], tail_width=0, tail_rows=[], tail_signs=[], seed='')
    planner = Planner(model, rows, ProofCosts(100,1,1))
    rng = np.random.default_rng(2921)
    candidates = rng.integers(-6,14,(80,2,3)).tolist()
    pair = next((c, planner.request(c,1)) for c in candidates
                if planner.request(c,1)['path']=='certificate' and planner.request(c,1)['refinement_indices'])
    middle, planned = pair
    centers = [old, middle, [[0,0,8],[8,8,8]]]
    q = [[0,1],[2,3]]
    inp = dict(model=model, x=rows, salt_x='11', salt_z='13', reuse_centers=[old]+centers[:2], queries=q)
    cert = root/'cert_private'; full = root/'full_private'; ref = root/'ref_private'
    save(root/'input_cert.json', inp)
    save(root/'cert_task.json', dict(comparison_base=old, reuse_centers=centers[:2], queries=q))
    env = dict(os.environ, COMMITJL_RESUME_PROOFS='1', GOMAXPROCS='8', GOMEMLIMIT='1200MiB')
    number = 0
    def call(phase, out, source=None, extras=()):
        nonlocal number
        number += 1
        args = [binary,'-phase',phase,'-out',out,'-q','2','-reps','1','-warmup','0','-max-constraints','1600000']
        if source: args += ['-instance',source]
        result = subprocess.run(list(map(str, args+list(extras))), env=env, capture_output=True, text=True, timeout=180)
        (root/f'{number:02}_{phase}.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    call('group_compact_prepare', cert, root/'input_cert.json')
    pin = issue_challenge(cert, root/'cert_pin.json',2)
    common = ['-trust',root/'cert_pin.json','-public-task',root/'cert_task.json']
    call('group_keys', cert, root/'input_cert.json',common)
    pin['owner'] = read(cert/'trusted.json'); save(root/'cert_pin.json',pin)
    call('group_rows', cert, root/'input_cert.json',common)
    call('group_query', cert, root/'input_cert.json',common)
    refargs = [*common, '-refinement-base',cert,'-refinement-trust',root/'ref_pin.json','-refinement-request',root/'requests.json']
    save(root/'requests.json',dict(sets=[[],planned['refinement_indices']]))
    call('refinement_setup',ref,extras=refargs)
    save(root/'ref_pin.json',read(ref/'trusted.json'))
    call('refinement_prove',ref,extras=refargs)
    fin = copy.deepcopy(inp)
    fin['model'].update(method='full',basis=[],seed='')
    fin['reuse_centers'] = [old,centers[2]]
    save(root/'input_full.json',fin)
    call('prepare',full,root/'input_full.json')
    call('full_reuse',full,root/'input_full.json')
    pin = dict(binary_sha256=sha(binary),certificate=pin,full=read(full/'trusted.json'),refinement=read(root/'ref_pin.json'))
    save(root/'pin.json',pin)
    save(root/'task.json',dict(comparison_base=old,reuse_centers=centers))
    public = root/'public'; public.mkdir()
    export_public(cert,public/'certificate')
    # Export only the public Full relation files for q2.
    dest = public/'full'; dest.mkdir()
    for name in ['trusted.json','prepare.json','full_reuse_2.json']:
        value = read(full/name)
        for s in value.get('samples') or []: s.pop('assign',None)
        save(dest/name,value)
    for sub, suffix in [('keys','*.vk'),('full_reuse','*.proof')]:
        (dest/sub).mkdir()
        for p in (full/sub).glob(suffix): shutil.copy2(p,dest/sub/p.name)
    dest = public/'refinement';dest.mkdir()
    for name in ['refinement_records.json','keys/group_refinement_1.vk']+[p.relative_to(ref).as_posix() for p in (ref/'refinement').glob('*.proof')]:
        (dest/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ref/name,dest/name)
    save(public/'routes.json',dict(certificate=[0,1],full=[2]))
    save(public/'full_queries.json',q)
    def check(): return verify(public,root/'pin.json',root/'task.json',binary)
    honest = check()
    rejected = []
    def alter(name,path,mutate,binary_file=False):
        before = path.read_bytes()
        try:
            if binary_file: path.write_bytes(bytes([before[0]^1])+before[1:])
            else:
                value = read(path); mutate(value); save(path,value)
            try: check()
            except (ValueError,RuntimeError): rejected.append(name)
            else: raise AssertionError('accepted substitution: '+name)
        finally: path.write_bytes(before)
    alter('missing request',public/'routes.json',lambda v:v['full'].clear())
    alter('duplicate request',public/'routes.json',lambda v:v['full'].insert(0,1))
    alter('omitted row batch',public/'full_queries.json',lambda v:v.pop())
    alter('duplicated row batch',public/'full_queries.json',lambda v:v.__setitem__(1,v[0]))
    alter('different catalog root',root/'pin.json',lambda v:v['full'].__setitem__('original_root','19'))
    alter('wrong certificate key',root/'pin.json',lambda v:v['certificate']['owner']['verifying_key_sha256'].__setitem__('group_row_compact','0'*64))
    alter('wrong Full key',root/'pin.json',lambda v:v['full']['verifying_key_sha256'].__setitem__('full_reuse_2_k2','0'*64))
    alter('changed candidate',root/'task.json',lambda v:v['reuse_centers'][2][0].__setitem__(0,1))
    alter('changed reference',root/'task.json',lambda v:v['comparison_base'][0].__setitem__(0,1))
    alter('damaged certificate proof',next((public/'certificate/group_query').glob('*.proof')),None,True)
    alter('damaged refinement proof',next((public/'refinement/refinement').glob('*.proof')),None,True)
    alter('damaged Full proof',next((public/'full/full_reuse').glob('*.proof')),None,True)
    alter('false exact cost',public/'full/full_reuse_2.json',lambda v:v['samples'][0].__setitem__('psi',str(int(v['samples'][0]['psi'])+1)))
    alter('false certificate endpoint',public/'certificate/group_query_records.json',lambda v:v[0].__setitem__('assigned_change','1'))
    private = public/'prepared.gob'; private.write_bytes(b'private')
    try:
        try: check()
        except ValueError: rejected.append('private export')
        else: raise AssertionError('accepted private export')
    finally: private.unlink()
    # Stale prover-authored output must be overwritten by fresh verification.
    save(public/'verified.json',dict(complete=True,decisions=[]))
    restored = check()
    assert restored['decisions']==honest['decisions'] and len(restored['decisions'])==3
    x = np.asarray(rows,dtype=object)
    cost = lambda cs:sum(min(sum((int(a)-int(b))**2 for a,b in zip(row,c)) for c in cs) for row in x)
    for record, c in zip(honest['decisions'],centers):
        delta = (cost(c)-cost(old))*int(record['denominator'])
        assert int(record['lower'])<=delta<=int(record['upper'])
    result = dict(complete=True,fresh_real_proofs=True,public_only=True,requests=3,
                  refinement_proofs=len(list((ref/'refinement').glob('*.proof'))),
                  tamper_rejections=rejected,stale_output_ignored=True,all_intervals_contain_truth=True,
                  binary_sha256=sha(binary))
    save(root/'checks.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--binary',type=Path,default=GO);a=p.parse_args();run(a.root.resolve(),a.binary.resolve())
