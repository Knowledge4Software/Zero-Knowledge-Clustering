"""Check and summarize small JL feasibility studies, retaining negative results."""
from pathlib import Path
import csv,json,math
from commitjl.pilot_io import save,sha


def main():
    repo=Path(__file__).resolve().parents[2];base=repo/'paper_results/review_controls';out=base/'jl_feasibility_20261025';out.mkdir(exist_ok=True)
    names={'quick':'jl_ideas_quick_20261025','matched':'jl_ideas_matched_20261025','prediction':'jl_assignment_pilot_20261025','confirmation':'jl_assignment_confirmation_20261025'}
    data={name:json.loads((base/folder/'summary.json').read_text()) for name,folder in names.items()};assert all(s['complete'] and s['native_only'] for s in data.values())
    checked=0
    for name in ['quick','matched']:
        for row in csv.DictReader((base/names[name]/'results.csv').open()):
            truth=int(row['true_change'])*16384**6;assert int(row['lower'])<=truth<=int(row['upper']);assert row['decision'] in ['inconclusive',row['truth']];checked+=1
    q=data['quick']['results'];width=next(r for r in q if r['idea']=='refinement' and r['variant']=='width');guides=[r for r in q if r['idea']=='refinement' and r['variant'].startswith('jl_m64_')];assert len(guides)==3 and all(r['conclusive']==6 for r in guides)
    assert all(r['conclusive']==0 for r in q if r['idea']=='basis' and r['variant'].startswith('jl_orthogonal_'))
    assert next(r for r in q if r['idea']=='basis' and r['variant']=='learned79')['conclusive']==4
    dev=data['prediction']['results'];original=next(r for r in dev if r['rank']==63 and r['kind']=='original_head_energy');candidate=next(r for r in dev if r['rank']==63 and r.get('width')==16 and r.get('alpha')==.125)
    assert candidate['passes_all_updates_all_seeds'];assert not any(r.get('passes_all_updates_all_seeds') for r in dev if r['rank']==39 and r.get('alpha',0)>0)
    confirmation=data['confirmation'];rows=list(csv.DictReader((base/names['confirmation']/'results.csv').open()));assert len(rows)==480;total=sum(int(r['n']) for r in rows);old=sum(int(r['original_correct']) for r in rows);new=sum(int(r['jl_correct']) for r in rows);assert abs(100*old/total-confirmation['original_agreement_percent'])<1e-10 and abs(100*new/total-confirmation['jl_agreement_percent'])<1e-10
    receipt=dict(complete=True,new_zk_proofs=0,production_method_unchanged=True,interval_rows_rechecked=checked,all_intervals_contain_truth=True,incorrect_conclusive=0,guide_exact_rows_without_jl=width['total_exact_rows'],guide_exact_rows_with_jl_each_seed=[r['total_exact_rows'] for r in guides],development_prediction_without_jl=original['mean_agreement'],development_prediction_with_jl=candidate['mean_agreement'],confirmation_prediction_without_jl=100*old/total,confirmation_prediction_with_jl=100*new/total,confirmation_extra_errors=old-new,confirmation_correct_assignment_difference=new-old,confirmation_error_count_note='Three sketch seeds are not independent learning problems.',confirmation_minimum_without_jl=confirmation['minimum_original_agreement_percent'],confirmation_minimum_with_jl=confirmation['minimum_jl_agreement_percent'],decision='No measured, consistent improvement justifying adding JL to the current production protocol or launching a large JL proof campaign.',source_sha256={})
    for key,folder in names.items():
        for name in ['protocol.json','summary.json','results.csv']:
            path=base/folder/name;receipt['source_sha256'][f'{folder}/{name}']=sha(path)
    for name in ['code/commitjl/jl_certificate_pilots.py','code/commitjl/pilot_io.py','code/experiments/pilot_jl_certificate_ideas.py','code/experiments/pilot_jl_assignment_shrinkage.py','code/experiments/confirm_jl_shrinkage.py','code/tests/test_jl_certificate_pilots.py']:
        receipt['source_sha256'][name]=sha(repo/name)
    # Illustration for ordinary Gaussian JL only, NOT a theorem for our q3
    # sparse hints and NOT a new acceptance policy. Fixed comparisons only.
    m=128*5*19;receipt['gaussian_jl_illustration']=[dict(comparisons=m,squared_norm_relative_error=eps,failure_budget=delta,sufficient_dimension=math.ceil(math.log(2*m/delta)/min((eps-math.log1p(eps))/2,(-eps-math.log1p(-eps))/2)),scope='Standard Gaussian chi-square Chernoff + union bound; sufficient, not necessary; fixed nonadaptive family; no claim for the sparse-q3 experimental maps.') for eps in [.25,.5] for delta in [1e-6,2**-40]]
    save(out/'summary.json',receipt)
    report=f'''# JL feasibility, 25 October 2026

The current production method and frozen proof binaries are unchanged. These
are small numerical research studies, not a new proof-speed benchmark. No new
ZK proofs were generated. The existing measured rank79 result remains61.637
active minutes and3.1913 times the matched reusable Full control.

| Idea | Small-test result | Interpretation |
|---|---|---|
| Sparse-JL measurements orthogonalized into16/32 residual basis directions | All six variants detect0/6 hard regressions; the trained SVD16-direction control detects4/6 | Not a replacement for the present certificate basis on this stress cohort |
| Untrusted JL64 hints for exact-refinement ordering | All three seeds need33 exact rows in total, versus34 for simple interval-width ordering; both resolve6/6 numerically | About2.94% fewer rows in this tiny cohort; fewer rows do not establish faster proofs |
| JL on group summaries, with exact residual-sum energies | All19 matched improvements resolve, but so do the no-JL compressed control and the SVD control; all six hard regressions require precise-moment fallback | No incremental JL decision benefit; any compression advantage needs its own binding proof and full cost measurement |
| Shrunken sparse-JL residual correction for predictions | Development mean {original['mean_agreement']:.6f}% -> {candidate['mean_agreement']:.6f}%; worst remains{candidate['minimum_agreement']:.6f}% | Small post-hoc development improvement, with16 additional stored sketch values |
| Same fixed predictor on all160 stored requests, with three new sketch seeds | Original {100*old/total:.6f}%, JL {100*new/total:.6f}%;{old-new} additional errors across{total} seed/record/update comparisons | The development improvement did not persist; do not promote it as a stronger predictor |

The confirmation has51 improved,49 worsened and380 equal seed/request pairs.
Its worst agreement falls from{confirmation['minimum_original_agreement_percent']:.6f}%
to{confirmation['minimum_jl_agreement_percent']:.6f}%. All are comparisons with
exact quantized Full labels, not semantic/classification accuracy. All160
requests remain present. The three new random-map seeds test seed robustness;
the fitted prediction models are stored previous cohorts, not new independent
learning problems. The development grid did not find a rank39+JL candidate
passing the user-selected99% floor on every19-update/three-seed combination.

The quick stress cohort consists of the six regressions previously unresolved
at rank63; it is deliberately post-hoc. The matched workload has N128 and19
improvements. They are separate workloads. All725 recorded numerical envelope
checks were rechecked for containment and conclusive correctness. Three
oracle-based Python tests passed: rounded-basis and mixed-basis aggregate
containment, exact recovery with an identity sketch, and sound refinement even
with deliberately misleading hints. These are not a security audit.

The random-basis branch uses QR on sparse measurements after removing the
frozen head span, followed by integer rounding. Its deterministic interval uses
the existing exact Gram correction. It must not be advertised as a new sparse
JL theorem. The hint branch never accepts a certificate from a probabilistic
distance estimate: only verified exact corrections could tighten the bounds.
That selective correction proof is not implemented here. The aggregate branch
also needs authenticated counts, means and residual-sum norms before deployment;
the present experiment is only a numerical reference.

An initial attempt stopped after one complete quick-case calculation when RAM
headroom dropped. That case and its original log are preserved. The resumed
implementation streamed the identical5000 sampled image rows from existing
local ARFF caches, verified their canonical float64 SHA256, and used chunked
Full checks. It completed all six cases without closing user applications or
changing the underlying dataset. `resume_receipt.json` records the runner
revision and prior checkpoint; no native orchestration time is claimed as
deployment performance.

Recommendation: retain the trained rank79 certificate as the current measured
candidate. Keep these JL experiments as honest ablations. There is no present
evidence for claiming a faster production method, a stronger certificate, or
substantive novelty merely from appending JL. No large JL proof run is queued.
The summary pins all source results and code. Artifacts remain private.
'''
    assert checked==725
    (out/'REPORT.md').write_text(report,encoding='utf-8');print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_sha256','gaussian_jl_illustration']}))


if __name__=='__main__':main()
