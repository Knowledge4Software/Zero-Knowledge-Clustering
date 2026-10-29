"""Small real proof workflow checks paid fallback and nonduplicating resume."""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys
from campaigns.local_fast_tradeoff import save,sha,run


def check(root,binary,smoke):
    root.mkdir(parents=True,exist_ok=True);parent=root/'parent';target=root/'moments'
    old=[[0,0,0],[8,8,0]];rows=[[0,0,1],[4,0,1],[7,7,2],[1,7,2]]*4
    model=dict(version='bound-audit-v1',method='learned_energy',n=16,d=3,k=2,input_bits=4,basis_scale=8,input_scale='1',centers=old,basis=[[8,0,0],[0,8,0]],tail_width=0,tail_rows=[],tail_signs=[],seed='integration fixture')
    inp=dict(model=model,x=rows,salt_x='11',salt_z='13',queries=[list(range(16))],reuse_centers=[old,[[1,0,0],[7,8,0]],[[0,0,8],[8,8,8]]],comparison_base=old)
    full=copy.deepcopy(inp);full['model'].update(method='full',basis=[],seed='');full['queries']=[list(range(8)),list(range(8,16))]
    save(parent/'head_r63/input.json',inp);save(parent/'full/input.json',full);save(parent/'protocol.json',dict(fitting_seconds=0.,proof_runtime_workers=8))
    run(parent,binary,['full'])
    command=[sys.executable,'-m','campaigns.run_moments_control','--parent',str(parent),'--root',str(target),'--binary',str(binary),'--smoke-receipt',str(smoke)]
    if (target/'protocol.json').exists():command.append('--resume')
    subprocess.run(command,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    receipt=json.loads((target/'checked_comparison.json').read_text());assert receipt['fallback_updates']==1 and receipt['fresh_fallback_proofs']==4 and receipt['group_public_proofs']==18
    assert receipt['active_phase_seconds_including_fitting_and_fallback']>receipt['fallback_active_seconds']>0
    hashes={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*.proof')}
    subprocess.run(command+([] if '--resume' in command else ['--resume']),check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    resumed=json.loads((target/'checked_comparison.json').read_text());assert resumed['active_phase_seconds_including_fitting_and_fallback']==receipt['active_phase_seconds_including_fitting_and_fallback']
    assert hashes=={p.relative_to(target).as_posix():sha(p) for p in target.rglob('*.proof')}
    save(root/'checks.json',dict(complete=True,fresh_full_reference_proofs=6,moments_proofs=18,fresh_paid_fallback_proofs=4,resume_no_duplicate_generation=True,binary_sha256=sha(binary)))
    print(json.dumps(dict(complete=True,root=str(root))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','binary','smoke']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();check(a.root.resolve(),a.binary.resolve(),a.smoke.resolve())
