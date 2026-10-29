"""Small complete online-affine stream, fresh proofs against exact Full."""
from pathlib import Path
import argparse,json,subprocess,sys
from campaigns.local_fast_tradeoff import save,run,sha


def check(root,binary):
    if root.exists():raise RuntimeError('Preserve previous smoke')
    parent=root/'parent';rows=[[i%5-2,(3*i)%7-3,(i*i)%5-2] for i in range(16)]
    centers=[[[0,0,0],[2,1,0]],[[-1,1,0],[2,0,1]],[[0,-1,1],[1,2,0]]]
    model=dict(version='bound-audit-v1',method='full',n=16,d=3,k=2,input_bits=4,basis_scale=8,input_scale='1',centers=centers[0],basis=[],tail_width=0,tail_rows=[],tail_signs=[],seed='online affine smoke')
    save(parent/'full/input.json',dict(model=model,x=rows,salt_x='1701',salt_z='1703',queries=[list(range(i,i+8)) for i in range(0,16,8)],reuse_centers=centers,comparison_base=centers[0]))
    save(parent/'protocol.json',dict(fitting_seconds=0.,proof_runtime_workers=2,phase_wall_limit_seconds=120))
    run(parent,binary,['full'])
    subprocess.run([sys.executable,'-m','campaigns.run_online_affine_control','--parent',str(parent),'--root',str(root/'online'),'--binary',str(binary),'--run-proofs'],check=True)
    result=json.loads((root/'online/checked_comparison.json').read_text())
    exact=[]
    for candidate in centers:exact.append(sum(min(sum((a-b)**2 for a,b in zip(row,c)) for c in candidate) for row in rows))
    assert [r['exact_cost'] for r in result['rows']]==exact
    assert not result['future_centers_available']
    save(root/'checks.json',dict(complete=True,exact_costs_match_full=True,future_centers_unavailable=True,binary_sha256=sha(binary),requests=2,affine_proofs=result['proofs']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);a=p.parse_args();check(a.root.resolve(),a.binary.resolve())
