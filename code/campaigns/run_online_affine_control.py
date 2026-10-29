"""Exact affine rebuild control for centers disclosed one request at a time.

Preparation is read/write only. Proof execution requires --run-proofs and an
explicitly enabled experiment policy. Never substitute known-bank timings for
this control, and never infer that every possible affine prover needs a rebuild.
"""
from pathlib import Path
import argparse,copy,json,time
from campaigns.local_fast_tradeoff import save,sha,run as run_case
from campaigns.proof_runtime import proof_workers
from campaigns.execution_admission import require_execution

REPO=Path(__file__).resolve().parents[2]
POLICY=REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json'


def read(path):return json.loads(Path(path).read_text())


def bank_input(source,index):
    """Only the current request and the already deployed reference are visible."""
    centers=source['reuse_centers'][index]
    bank=list(dict.fromkeys(tuple(c) for c in centers))
    if not 2<=len(bank)<=128:raise ValueError('affine bank needs2..128 distinct centers')
    inp=copy.deepcopy(source)
    inp['model'].update(method='affine',k=len(bank),centers=[list(c) for c in bank],basis=[],tail_width=0,tail_rows=[],tail_signs=[],seed='')
    inp['reuse_centers']=[copy.deepcopy(centers)]
    n=inp['model']['n'];inp['queries']=[list(range(i,i+16)) for i in range(0,n,16)]
    if n%16:raise ValueError('full catalog must be divisible by16')
    return inp


def prepare(parent,root,binary):
    source=read(parent/'full/input.json');protocol=read(parent/'protocol.json')
    workers=proof_workers(protocol)
    if source['reuse_centers'][0]!=source['comparison_base']:raise ValueError('initial reference must be first')
    plan=dict(source_input_sha256=sha(parent/'full/input.json'),binary_sha256=sha(binary),requests=len(source['reuse_centers'])-1,
        n=source['model']['n'],workers=workers,reference_fixed=True,future_centers_available_to_preprocessing=False,
        contract='One initial exact reference cost; each previously unseen candidate gets its own current-request-only affine bank. Charge every native preparation, key family, authenticated derivation, query, fresh replay and failed attempt. Cache the initial cost once. No future-bank union.',
        limitations='Diagnostic replay of a predeclared stream, not a production deployment. This measures rebuilding the implemented affine baseline, not every possible exact affine/GKR/sum-check protocol. Replacing the deployed reference after acceptance is a separate, unmeasured experiment.')
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():assert read(root/'plan.json')==plan
    else:save(root/'plan.json',plan)
    for i in range(len(source['reuse_centers'])):
        target=root/f'bank_{i:03d}';inp=bank_input(source,i)
        meta=copy.deepcopy(protocol);meta.update(method='online_exact_affine_rebuild',fitting_seconds=0.,ranks=[],request_index=i,source_sha256=plan['source_input_sha256'],future_centers_available=False)
        for path,value in [(target/'family_affine/input.json',inp),(target/'protocol.json',meta)]:
            if path.exists():assert read(path)==value
            else:save(path,value)
    save(root/'preparation.json',dict(complete=True,proofs_generated=False,plan_sha256=sha(root/'plan.json'),prepared_unix=time.time()))
    return source,plan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['parent','root','binary']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--run-proofs',action='store_true');a=p.parse_args()
    parent,root,binary=a.parent.resolve(),a.root.resolve(),a.binary.resolve()
    source,plan=prepare(parent,root,binary)
    if not a.run_proofs:print(json.dumps(dict(prepared=True,proofs_generated=False)));return
    if not POLICY.exists():raise RuntimeError('Missing execution policy; no prover started')
    require_execution(read(POLICY))
    from analysis.audit_local_fast_tradeoff import audit_case
    rows=[];original_root=None;old_cost=None
    for i in range(plan['requests']+1):
        target=root/f'bank_{i:03d}'
        run_case(target,binary,['family_affine'])
        result,_,_=audit_case(target,'family_affine')
        trust=read(target/'family_affine/owner_pin.json')
        if original_root is None:original_root=trust['original_root']
        assert original_root==trust['original_root'],'banks must authenticate the same original catalog'
        phase=read(target/'family_affine/public_bundle/family_affine_16.json')
        exact_cost=sum(int(s['psi']) for s in phase['samples'])
        if old_cost is None:old_cost=exact_cost
        rows.append(dict(request=i,active_seconds=result['active_phase_wall_seconds'],proofs=result['public_proofs'],exact_cost=exact_cost,exact_change=exact_cost-old_cost))
        save(root/'progress.json',dict(complete=False,rows=rows))
    full,_,_=audit_case(parent,'full');seconds=sum(r['active_seconds'] for r in rows)
    save(root/'checked_comparison.json',dict(complete=True,plan_sha256=sha(root/'plan.json'),rows=rows,active_seconds=seconds,full_seconds=full['active_phase_wall_seconds'],full_over_affine=full['active_phase_wall_seconds']/seconds,proofs=sum(r['proofs'] for r in rows),future_centers_available=False,fixed_reference=True))


if __name__=='__main__':main()
