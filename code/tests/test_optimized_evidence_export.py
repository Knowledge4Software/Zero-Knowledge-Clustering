import pytest
from analysis import update_optimized_evidence as export
from campaigns.local_fast_tradeoff import save


def test_quality_retains_inconclusive_regressions_and_below_target_agreement():
    rows=[dict(true_change=4,initial_decision='inconclusive',decision='inconclusive',agreement_percent=98.5,assignment_certified_percent=70.,width_percent_old_cost=3.,exact_rows=16),
          dict(true_change=-3,initial_decision='inconclusive',decision='improves',agreement_percent=100.,assignment_certified_percent=90.,width_percent_old_cost=1.,exact_rows=2)]
    q=export.quality(rows)
    assert q['requests']==2 and q['conclusive']==1 and q['regressions_certified']==0
    assert q['below99']==1 and q['worst_agreement']==98.5 and q['exact_rows']==18


def test_timing_export_counts_completed_pairs_without_claiming_full_repetition(tmp_path,monkeypatch):
    monkeypatch.setattr(export,'REPO',tmp_path);dest=tmp_path/'tables';dest.mkdir();monkeypatch.setattr(export,'DEST',dest)
    root=tmp_path/'study';cohort=root/'matched_controls';save(cohort/'plan.json',dict(catalog_records=128))
    rep=cohort/'repetition_01'
    record=dict(complete=True,full_reference_active_seconds=120.,active_phase_seconds_including_fitting_and_fallback=60.,full_over_group_active_phase=2.,conclusive_without_fallback=3,proposal_updates=4,fallback_updates=1)
    for method in ['ordered_head','pca']:save(rep/(method+'_refined/checked_comparison.json'),record)
    assert len(export.export_timings(root))==2
    import json
    assert json.loads((dest/'optimized_timing_receipt.json').read_text())['completed_repetitions']==0
    save(rep/'moments_only/checked_comparison.json',record)
    rows=export.export_timings(root)
    assert len(rows)==3 and all(v['fallback']==1 for v in rows)
    assert json.loads((dest/'optimized_timing_receipt.json').read_text())['completed_repetitions']==1
    assert (dest/'optimized_proofs.tex').exists()
