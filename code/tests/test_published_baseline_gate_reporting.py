"""Prevent statistical or approximate outputs becoming deterministic claims."""
from analysis.published_baseline_gate_report import summarize_native


def test_statistical_success_is_not_deterministic_coverage():
    record=dict(dataset='probe',n=4,method='rabitq_raw',fit_seed=1,case='probe_seed1',
        assignment_agreement=1.,point_sign_correct=True,true_change=3,estimated_change=2.9,
        conclusive=None,base_conclusive=None,corrections=None,statistical_conclusive=True)
    timing=dict(case='probe_seed1',n=4,method='rabitq_raw',phase='query',seconds=[[.01],[.02],[.03]])
    summary,_=summarize_native(dict(records=[record],timings=[timing]))
    assert summary[0]['statistical_coverage']==1.
    assert summary[0]['refined_coverage'] is None
    assert summary[0]['base_coverage'] is None
    assert summary[0]['fits']==1


def test_point_estimate_error_is_not_measured_certificate_failure():
    record=dict(dataset='probe',n=4,method='lightweight_coreset_m16',fit_seed=1,case='probe_seed1',
        assignment_agreement=None,point_sign_correct=False,true_change=3,estimated_change=-2,
        conclusive=None,base_conclusive=None,corrections=None)
    summary,_=summarize_native(dict(records=[record],timings=[]))
    assert summary[0]['wrong_estimated_signs']==1
    assert summary[0]['refined_coverage'] is None
    assert summary[0]['assignment_agreement'] is None


def test_majority_prediction_does_not_inflate_balanced_accuracy():
    rows=[]
    for delta in [3]*8+[-3]*2:
        rows.append(dict(dataset='probe',n=4,method='constant_worsens',fit_seed=1,case='probe_seed1',
            assignment_agreement=None,point_sign_correct=delta>0,true_change=delta,estimated_change=1,
            conclusive=None,base_conclusive=None,corrections=None))
    summary,_=summarize_native(dict(records=rows,timings=[]))
    assert summary[0]['point_sign_agreement']==.8
    assert summary[0]['balanced_sign_agreement']==.5
    assert summary[0]['missed_improvement_predictions']==2
