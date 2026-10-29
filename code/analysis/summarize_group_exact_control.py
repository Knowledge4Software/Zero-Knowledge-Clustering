"""Summarize completed exact-old proofs and charge every resource attempt."""
from pathlib import Path
import argparse,hashlib,json


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','previous-group','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();parent=a.parent.resolve();previous=a.previous_group.resolve();out=a.out.resolve()
    comparison=read(root/'checked_comparison.json');state=read(root/'status.json');replay=read(root/'public/group_verify.json');native=read(root/'native_checks.json')
    assert comparison['complete'] and comparison['profile']=='group-exact-old-v2'
    assert state['complete'] and state['status']=='complete' and replay['complete'] and native['complete']
    assert not native['fallback_ids'] and comparison['fallback_updates']==0
    records=read(root/'bundle/group_query_records.json');decisions=native['decisions'];model=read(root/'input.json')['model']
    assert len(records)==len(decisions)==comparison['proposal_updates']==19
    assert all(d['truth']<0 and d['decision']=='improvement' for d in decisions)
    assert all(int(d['lower'])<=int(d['truth'])*int(d['denominator'])<=int(d['upper']) for d in decisions)
    switch_batches=sum(len(r.get('batches') or []) for r in records)
    expected=model['n']+1+len(records)+switch_batches
    assert replay['verified']==comparison['group_public_proofs']==expected==284
    manifest=read(root/'public/public_manifest.json')['sha256']
    assert all(sha(root/'public'/name)==digest for name,digest in manifest.items())
    assert sum(name.endswith('.proof') for name in manifest)==expected
    full=comparison['full_reference_active_seconds'];current=comparison['active_phase_seconds_including_fitting_and_fallback']
    old_head=read(parent/'status.json')['cases']['head_r63']['wall_seconds']
    old_group=read(previous/'checked_comparison.json')['active_phase_seconds_including_fitting_and_fallback']
    attempts=[attempt for resource in state['resources'].values() for attempt in resource['attempts']]
    ram_stops=[attempt for attempt in attempts if attempt.get('stop')=='RAM reserve']
    summary=dict(**comparison,expected_proof_count=expected,preprocessing_row_proofs=model['n'],link_proofs=1,aggregate_proofs=len(records),switching_batch_proofs=switch_batches,
        all_19_updates_are_improvements=True,real_proof_natural_regressions=0,public_manifest_hashes_match=True,
        previous_relative_active_seconds=old_head,previous_relative_full_ratio=full/old_head,
        previous_group_active_seconds=old_group,new_minus_previous_group_seconds=current-old_group,
        new_over_previous_relative_time_ratio=current/old_head,fivefold_speed_target_met=full/current>=5.,previous_relative_speed_preserved=current<=old_head,
        ram_stops=len(ram_stops),charged_ram_stopped_seconds=sum(x['wall_seconds'] for x in ram_stops),
        phase_seconds={k:v['wall_seconds'] for k,v in state['resources'].items()})
    paths=['checked_comparison.json','status.json','native_checks.json','protocol.json','owner_pin.json','approved.json','public/group_verify.json','public/group_verified_decisions.json','public/public_manifest.json','bundle/group_query_records.json','ram_retry_orchestration_20261024.json']
    summary['source_sha256']={name:sha(root/name) for name in paths}
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    report=f'''# Completed exact-old group certificate control

All {expected} proofs passed fresh public-only verification, and all {len(manifest)} exported manifest hashes still match. The proof count is {model['n']} exact-old row bindings, one group-moment link, {len(records)} update aggregates and {switch_batches} switching batches. All 19 approved updates have correct conclusive improvement certificates. Every interval contains independently computed quantized Full truth, with no fallback. This timing workload contains zero natural regressions; it does not authenticate the separate 36-regression numerical suite.

| Workflow | Active minutes | Full / workflow |
|---|---:|---:|
| Exact reusable public-center Full | {full/60:.2f} | 1.00 |
| Previous rank63 relative certificate | {old_head/60:.2f} | {full/old_head:.3f} |
| Previous group v1 certificate | {old_group/60:.2f} | {full/old_group:.3f} |
| Exact-old group v2 certificate | {current/60:.2f} | {full/current:.3f} |

The 5-fold target was not reached. V2's observed total differs from v1 by {current-old_group:.2f} seconds, so this run does not demonstrate an overall speed improvement. It is {100*(current/old_head-1):.1f}% slower than the previous relative workflow, which offered weaker cost-decision guarantees. These measurements are one fit and one repetition, taken serially under changing laptop load; they do not establish an equal-guarantee algorithm ranking.

Only {comparison['assigned_upper_fast_updates']}/19 updates take the aggregate-only fast path. The other {comparison['switch_updates']} require switching proofs. Relative to v1, this saves 16 proof files, while the total protocol still pays for four cold setups, full-catalog authenticated head/energy and exact-old labels, the group link and query proof work. The aggregate fast path certifies direction with a deliberately broad conservative interval; it does not promise a narrow cost estimate or certify every new individual assignment.

The total includes all active attempts, native preparation, fitting, cold setup, proof generation/checks and fresh public replay. Two RAM-reserve stops contributed {summary['charged_ram_stopped_seconds']:.2f} charged seconds, including any work subsequently repeated. Waiting for free RAM and between-phase orchestration/export gaps are excluded. This is an interrupted active-phase total, not an uninterrupted cold run. Do not subtract failed work and label the remainder a measured uninterrupted result.

The underlying rank63 predictions are unchanged: the separate completed assignment check on this same 128-record catalog had 99.794% mean and 99.219% worst agreement. The broader stored-fit numerical recheck retained head decision coverage151/160 and natural-regression sensitivity30/36, with no incorrect conclusive decisions. These numerical cases differ from this timing workload and are not additional real-proof independent replications.

No Full fallback was required here, so nonzero fallback-generation cost is still unmeasured. Authenticated natural-regression replication, selective exact correction, independent-fit timing, and original Mobin export recovery remain unfinished. Passing public proof replay and hash checks is not a complete security/privacy audit. All artifacts remain private for supervisor review.

Source: `{root}`. The companion summary pins the evidence hashes. No manuscript speed claim has been upgraded to 5-fold.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(complete=True,proofs=expected,updates=19,minutes=current/60,full_ratio=full/current,report=str(out/'REPORT.md'))))


if __name__=='__main__':main()
