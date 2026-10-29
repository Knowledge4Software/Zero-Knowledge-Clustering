"""Separate assignment correctness from sound cost-decision coverage."""
from pathlib import Path
import argparse,json
import pandas as pd
from campaigns.local_fast_tradeoff import save,sha

ROOT=Path(__file__).resolve().parents[2]
REPORTS=ROOT/'paper_results/review_controls'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--exact-confirmation',type=Path);p.add_argument('--exact-control',type=Path);a=p.parse_args()
    sources=[REPORTS/'group_change_bounds_20261024/results.csv',REPORTS/'group_change_confirmation_20261024/bounds/results.csv']
    frame=pd.concat([pd.read_csv(f) for f in sources],ignore_index=True)
    records=[]
    def add_rows(data,variants):
        for method in ['learned','pca']:
            for variant in variants:
                for budget in [0.,.05]:
                    subset=data[(data.method==method)&(data.variant==variant)&(data.budget_fraction==budget)]
                    if subset.empty:continue
                    assert len(subset)==160
                    assert not subset.duplicated(['fit','set']).any()
                    correct=(subset.decision==subset.truth);conclusive=(subset.decision!='inconclusive')
                    assert not (conclusive&~correct).any()
                    for row in subset.itertuples():assert int(row.lower)<=int(row.true_change)*16384**6<=int(row.upper)
                    reg=subset.truth=='worsens'
                    fast=subset.get('assigned_upper_bound_fast_path',pd.Series(False,index=subset.index)).eq(True)
                    records.append(dict(method=method,variant=variant,budget_fraction=budget,requests=len(subset),conclusive=int(conclusive.sum()),coverage_percent=100*float(conclusive.mean()),regressions=int(reg.sum()),regressions_detected=int((reg&conclusive).sum()),regression_sensitivity_percent=100*float(conclusive[reg].mean()),incorrect_conclusive=0,fast_assigned_upper_updates=int(fast.sum()) if variant=='group_exact_old' else None,native_only=True))
    add_rows(frame,['ordinary_paired','group_separate','group_paired'])
    if a.exact_confirmation:
        root=a.exact_confirmation.resolve();state=json.loads((root/'status.json').read_text());assert state['complete']
        exact_sources=[root/cohort/'results.csv' for cohort in ['historical','fresh']];sources+=exact_sources
        add_rows(pd.concat([pd.read_csv(f) for f in exact_sources],ignore_index=True),['group_exact_old'])
    report=a.out.resolve();report.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(report/'certificate_comparison.csv',index=False)
    baseline=REPORTS/'completed_fast_baseline_20261024/quality_checked_against_proofs.csv';sources.append(baseline)
    quality=pd.read_csv(baseline);assignment=[]
    for case in ['full','head_r39','head_r63']:
        part=quality[(quality['case']==case)&(quality['set']!=0)]
        assert len(part)==19
        assignment.append(dict(method=case,updates=19,mean_assignment_agreement_percent=float(part.agreement_percent.mean()),minimum_assignment_agreement_percent=float(part.agreement_percent.min()),mean_certified_assignment_coverage_percent=float(part.certified_coverage_percent.mean()),minimum_certified_assignment_coverage_percent=float(part.certified_coverage_percent.min())))
    summary=dict(complete=True,native_certificate_comparison=records,proof_workload_assignments=assignment,source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sources},limits=['The 160-update native suite is not the 19-update real-proof timing workload.','Group decision proofs do not prove every approximate assignment correct.','No incorrect conclusive native decisions is empirical validation of the bounds, not a complete security audit.','The new exact-old profile changes preprocessing and its aggregate fast path supplies a conservative cost interval.'])
    lines=['# Certificate comparison','', 'Assignment agreement and certificate coverage measure different things. A sound certificate may return inconclusive; this is not a wrong answer.','', '## All stored native fits: 160 updates per representation, 36 natural regressions','', '| Method | Numerical variant | Conclusive decisions | Natural regressions detected | Incorrect conclusive |','|---|---|---:|---:|---:|']
    for row in records:
        if row['budget_fraction']==0:lines.append(f"| {row['method']} | {row['variant']} | {row['conclusive']}/160 ({row['coverage_percent']:.2f}%) | {row['regressions_detected']}/36 ({row['regression_sensitivity_percent']:.2f}%) | 0 |")
    lines+=['','Full computes the exact reference and resolves every nonzero change. These native rows do not represent 160 authenticated proof executions. Every persisted fit/update is retained; this recheck is not a new independent fit cohort.','', '## Separate completed 128-record proof workload: 19 proposed updates','', '| Method | Mean assignment agreement | Worst agreement | Mean certified assignment coverage |','|---|---:|---:|---:|']
    for row in assignment:lines.append(f"| {row['method']} | {row['mean_assignment_agreement_percent']:.3f}% | {row['minimum_assignment_agreement_percent']:.3f}% | {row['mean_certified_assignment_coverage_percent']:.2f}% |")
    lines+=['','The group certificate changes the cost-change guarantee, not the rank63 predictions. Those unchanged predictions retain 99.219% minimum agreement on this workload; the group relation alone does not certify every assignment.','', 'Completed real-proof group v1: 19/19 correct conclusive decisions, all improvements, 300 fresh publicly replayed proofs, 68.88 active minutes and a 2.856-fold Full ratio. Previous relative rank63: 43.39 active minutes and a 4.534-fold ratio, with weaker cost-decision guarantees. Neither is evidence of a fivefold fully certified improvement.']
    if a.exact_control:
        control=a.exact_control.resolve();result=control/'checked_comparison.json';measured=json.loads(result.read_text());assert measured['complete'] and measured['profile']=='group-exact-old-v2'
        summary['completed_exact_control']=measured;summary['source_sha256'][str(result.relative_to(ROOT))]=sha(result)
        lines+=['',f"Exact-old v2 is complete: {measured['group_public_proofs']} proofs passed fresh public replay; {measured['conclusive_without_fallback']}/19 updates have correct conclusive certificates, all improvements, and no fallback. Active time including RAM retry attempts is {measured['active_phase_seconds_including_fitting_and_fallback']/60:.2f} minutes, Full ratio{measured['full_over_group_active_phase']:.3f}. The 5-fold target is not met. Only{measured['assigned_upper_fast_updates']}/19 updates skip switching proofs. The native regression suite remains separate from this all-improvement timing workload."]
    else:lines+=['','The exact-old optimization has a separate binary, smoke and timing queue. Its final speed remains unmeasured until the matched control finishes.']
    lines+=['','All results stay private for supervisor review.']
    save(report/'summary.json',summary)
    (report/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(report/'REPORT.md'),rows=len(records))),flush=True)


if __name__=='__main__':main()
