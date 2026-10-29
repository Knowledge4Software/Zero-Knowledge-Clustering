"""Record the completed group/Full cost comparison without projected claims."""
from pathlib import Path
import argparse,hashlib,json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();parent=a.parent.resolve();out=a.out.resolve()
    comparison=json.loads((root/'checked_comparison.json').read_text());assert comparison['complete']
    state=json.loads((root/'status.json').read_text());assert state['complete'] and state['status']=='complete'
    replay=json.loads((root/'public/group_verify.json').read_text());assert replay['complete'] and replay['verified']==300
    native=json.loads((root/'native_checks.json').read_text());assert native['complete'] and native['fallback_ids']==[]
    decisions=native['decisions'];assert len(decisions)==19 and all(x['truth']<0 and x['decision']=='improvement' for x in decisions)
    assert all(int(x['lower'])<=int(x['truth'])*int(x['denominator'])<=int(x['upper']) for x in decisions)
    old=json.loads((parent/'status.json').read_text())['cases']['head_r63']['wall_seconds']
    group=comparison['active_phase_seconds_including_fitting_and_fallback'];full=comparison['full_reference_active_seconds']
    summary=dict(**comparison,previous_rank63_active_seconds=old,new_over_previous_rank63_time_ratio=group/old,
        previous_rank63_full_ratio=full/old,all_19_proof_workload_updates_are_improvements=True,
        no_natural_regression_replication_in_this_timing_workload=True,
        fivefold_speed_target_met=full/group>=5.,previous_speed_preserved=group<=old,
        phase_seconds={k:v['wall_seconds'] for k,v in state['resources'].items()})
    names=['checked_comparison.json','status.json','native_checks.json','public/group_verify.json','protocol.json','approved.json','owner_pin.json','bundle/group_query_records.json']
    summary['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    report=f'''# Completed group-certificate cost control

All300 proofs passed a fresh public-only replay:128 row bindings,one moment-link proof,19 assigned aggregates and152 switching-bound batches. All19 proposed updates have conclusive correct cost-change certificates, all intervals contain the independently computed quantized Full change, and no Full fallback was needed. Every update in this timing workload is an improvement relative to the deployed model. It does not constitute an authenticated replication of the separate natural-regression experiments.

| Workflow | Active phase minutes | Full / method |
|---|---:|---:|
| Exact public-center Full | {full/60:.2f} | 1.00 |
| Previous rank63 relative-certificate pipeline | {old/60:.2f} | {full/old:.2f} |
| New group cost-change-certificate pipeline | {group/60:.2f} | {full/group:.2f} |

The new workflow takes{100*(group/old-1):.1f}percent more time than the previous pipeline. It remains faster than Full on this workload, but it misses the5-fold speed target and does not preserve the previous speed level. Its output is a guaranteed interval/sign for the exact cost change; the earlier relative workflow's faster absolute intervals did not guarantee conclusive update decisions. These are different guarantee levels, not an equal-guarantee algorithm ranking.

The frozen MNIST workload has128 records,784 coordinates,k5,rank63 and19 nonidentical proposed center sets. Full's20 queries supply the old cost plus the same19 new costs. Head/group use16-row switching batches and Full uses8, under a1.6-million-constraint guard. No update was selected by its held-out outcome. Underlying rank63 assignment predictions are unchanged; the earlier checked minimum agreement on this catalog was99.21875percent. This is distinct from guaranteed cost-decision coverage.

The totals charge all active attempts, preparation, cold proof-key generation, row and moment binding, query generation/checks, fitting and fresh public replay. Offline/RAM queue waits and between-phase export/monitor gaps are excluded from this phase-sum metric. An initial RAM-guard stop is retained in the prepare resource record; do not call this an uninterrupted enclosing cold benchmark. It is one fit and one repetition measured serially under changing laptop clock/load conditions.

Measured components: key generation{state['resources']['group_keys']['wall_seconds']/60:.2f}minutes,row binding{state['resources']['group_rows']['wall_seconds']/60:.2f},moment link{state['resources']['group_link']['wall_seconds']/60:.2f},query generation/checks{state['resources']['group_query']['wall_seconds']/60:.2f},and public replay{state['resources']['group_verify']['wall_seconds']/60:.2f}. Key/setup, group aggregate and switching-query costs are concrete optimization targets.

No fallback was invoked, so this run does not measure prospective fallback generation costs. Authenticated selective corrections, real-proof natural regressions, independent-fit timing replication and original Mobin export recovery remain unfinished. The separate numerical suite resolves30/36 natural regressions directly versus0/36 with ordinary paired bounds; all36 resolve within52/1024 exact rows numerically. Do not combine that suite and this timing control into a claim of a fully authenticated5-fold improvement.

Source: `{root}`. The accompanying summary includes source hashes. Artifacts stay private for supervisor review. The manuscript has not been changed to claim the target was achieved.
'''
    (out/'REPORT.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(complete=True,proofs=replay['verified'],conclusive=19,active_minutes=group/60,full_ratio=full/group,slower_than_previous_percent=100*(group/old-1),report=str(out/'REPORT.md'))),flush=True)


if __name__=='__main__':main()
