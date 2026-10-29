"""Checked optimized-certificate result; preserve unfavorable measurements."""
from pathlib import Path
import argparse,json,hashlib


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.out.resolve()
    result=read(root/'checked_comparison.json');state=read(root/'status.json');native=read(root/'native_checks.json');replay=read(root/'public/group_verify.json');model=read(root/'input.json')['model'];records=read(root/'bundle/group_query_records.json')
    assert result['complete'] and state['complete'] and result['profile']=='group-compact-v3' and replay['complete'] and native['complete']
    expected=model['n']+len(records)+sum(len(r.get('batches') or []) for r in records);assert replay['verified']==expected==result['group_public_proofs']
    assert all(int(r['lower'])<=int(r['truth'])*int(r['denominator'])<=int(r['upper']) for r in native['decisions'])
    manifest=read(root/'public/public_manifest.json')['sha256'];assert all(sha(root/'public'/name)==digest for name,digest in manifest.items())
    phases={k:v['wall_seconds'] for k,v in state['resources'].items()};summary=dict(**result,phase_seconds=phases,public_manifest_hashes_match=True,model_rows=model['n'],all_updates_are_improvements=all(r['truth']<0 for r in native['decisions']),natural_regression_proof_count=sum(r['truth']>0 for r in native['decisions']))
    names=['checked_comparison.json','status.json','protocol.json','native_checks.json','public/group_verify.json','public/public_manifest.json','owner_pin.json','frozen_prediction_model.json'];summary['source_sha256']={name:sha(root/name) for name in names}
    if (root/'auxiliary_basis_source.json').exists():summary['source_sha256']['auxiliary_basis_source.json']=sha(root/'auxiliary_basis_source.json')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=f'''# Optimized certificate: completed matched control

All{expected} group proofs passed fresh public-only verification. All exported manifest hashes match. The independent original-space check confirms every interval contains quantized Full truth. Resolved without fallback:{result['conclusive_without_fallback']}/{result['proposal_updates']}; fallback updates:{result['fallback_updates']}; freshly generated fallback proofs:{result['fresh_fallback_proofs']}.

Measured active time including fitting, native preparation, three cold setups, all proof attempts, fresh public replay and any fallback:{result['active_phase_seconds_including_fitting_and_fallback']/60:.2f}minutes. Matched reusable Full:{result['full_reference_active_seconds']/60:.2f}minutes. Full/method:{result['full_over_group_active_phase']:.3f}. Queued/RAM waits and between-phase export gaps are excluded. One supplied fit and one repetition under varying laptop load; not a general performance ranking.

The proof construction stores exact old membership in the same authenticated head/energy leaf and combines moment binding with each update aggregate. It retains the same algebra, complete row coverage, independently pinned post-commit challenge and public parameters. It does not batch different future updates together or restrict future centers to a known bank. Every proposed update receives switching bounds; the broad aggregate-only shortcut is disabled.

Auxiliary certificate basis precision:{result.get('certificate_scale_bits',14)}bits; rank{len(model['basis'])}. The frozen prediction model is unchanged and saved separately. An auxiliary precision is selected only after the stored-fit numerical screen preserves head coverage>=151/160 and natural-regression sensitivity>=30/36. A higher rank, if used, is fitted only to training rows and receives separate fresh preprocessing/setup/proofs. These screens are method development, not newly independent validation. The original prediction accuracy figures belong to their frozen model and workload, not all datasets. A standalone higher-rank proof cost does not establish cheap adaptive sharing across ranks.

Native parameter preparation uses exact int64 Gram calculations only inside mathematically checked bounds; unbounded callers retain big.Int. Tests compare both paths against an independent big-integer oracle. Public-center validation no longer constructs unused comparison coefficients. These are arithmetic/implementation optimizations, not relaxed mathematical guarantees. All legacy proof binaries/evidence are preserved.

Source `{root}`. The companion summary records source hashes and all phase costs. Passing tests/replay is not a complete security audit. No acceptance probability or absence of future referee objections is claimed. Artifacts remain private for supervisor review.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8');print(json.dumps(dict(complete=True,ratio=result['full_over_group_active_phase'],minutes=result['active_phase_seconds_including_fitting_and_fallback']/60,report=str(out/'REPORT.md'))))


if __name__=='__main__':main()
