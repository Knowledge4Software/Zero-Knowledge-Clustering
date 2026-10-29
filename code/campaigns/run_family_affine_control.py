"""Measure exact authenticated affine scores for the preapproved center bank."""
from pathlib import Path
import argparse,hashlib,json,time
from campaigns.local_fast_tradeoff import save,sha,run
from analysis.audit_local_fast_tradeoff import audit_case
from validation.check_family_affine_proofs import run as smoke


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--smoke-receipt',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    parent=a.parent.resolve();root=a.root.resolve();binary=a.binary.resolve();root.mkdir(parents=True,exist_ok=a.resume)
    source=parent/'head_r63/input.json';inp=json.loads(source.read_text());bank=list(dict.fromkeys(tuple(c) for centers in inp['reuse_centers'] for c in centers))
    if not 2<=len(bank)<=128:raise ValueError('existing affine model supports a bank of2..128 centers')
    inp['model'].update(method='affine',k=len(bank),centers=[list(c) for c in bank],basis=[],tail_width=0,tail_rows=[],tail_signs=[],seed='')
    if a.resume:assert json.loads((root/'family_affine/input.json').read_text())==inp
    else:save(root/'family_affine/input.json',inp)
    protocol=json.loads((parent/'protocol.json').read_text());protocol.update(ranks=[],fitting_seconds=0.,method='exact_known_bank_affine',bank_size=len(bank),bank_sha256=hashlib.sha256(json.dumps(bank,separators=(',',':')).encode()).hexdigest(),source_sha256=sha(source),
        contract='The entire approved center bank is fixed before preprocessing. Query setups allow ordered subsets of that bank. Unknown future centers are rejected and require new preparation.',
        comparison=f"Same{inp['model']['n']} records and all{len(inp['reuse_centers'])} center sets as the completed public-center Full workload; contracts differ because Full also accepts centers outside this preapproved bank.",batches=dict(family_affine=16,full=8))
    if a.resume:assert json.loads((root/'protocol.json').read_text())==protocol
    else:save(root/'protocol.json',protocol)
    state=dict(complete=False,phase='smoke',started_unix=time.time(),bank_size=len(bank));save(root/'control_status.json',state)
    try:
        if a.smoke_receipt:
            checked=json.loads(a.smoke_receipt.read_text());assert checked['complete'] and checked['binary_sha256']==sha(binary)
            save(root/'prior_smoke_receipt.json',dict(source=str(a.smoke_receipt.resolve()),sha256=sha(a.smoke_receipt),**checked))
        elif (root/'smoke/checks.json').exists():
            checked=json.loads((root/'smoke/checks.json').read_text());assert checked['complete'] and checked['binary_sha256']==sha(binary)
        else:smoke(root/'smoke',binary)
        state['phase']='full_catalog_proofs';save(root/'control_status.json',state)
        run(root,binary,['family_affine'])
        state['phase']='independent_output_check';save(root/'control_status.json',state)
        family,quality,actual=audit_case(root,'family_affine');full,_,reference=audit_case(parent,'full')
        assert actual['x']==reference['x'] and actual['reuse_centers']==reference['reuse_centers']
        save(root/'checked_comparison.json',dict(complete=True,family=family,full=full,full_over_family=full['wall_seconds']/family['wall_seconds'] if family['uninterrupted_cold_wall_available'] else None,full_over_family_active_phase=full['active_phase_wall_seconds']/family['active_phase_wall_seconds'],all_outputs_match_full=True,query_center_sets=len(inp['reuse_centers']),contract=protocol['contract'],open_future_centers_supported=False))
        state.update(complete=True,phase='complete',finished_unix=time.time());save(root/'control_status.json',state)
    except BaseException as error:
        state.update(phase='failed',error=str(error));save(root/'control_status.json',state);raise


if __name__=='__main__':main()
