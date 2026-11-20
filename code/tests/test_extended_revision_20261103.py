"""Archive namespacing and exact-cost repetition invariants on the server."""
import sys
import pytest
pytestmark=pytest.mark.skipif(sys.platform=='win32',reason='Linux coordinator uses flock')


def test_new_archives_cannot_overwrite_earlier_paired_or_scale_cells(monkeypatch,tmp_path):
    from campaigns import extended_revision_20261103 as q
    seen=[]
    monkeypatch.setattr(q,'PUBLISH',lambda root,name,paths:seen.append(name))
    q.publish(tmp_path,'paper_gap_checks_har_seed919_n128_q8_timing',[])
    q.publish(tmp_path,'paper_completion_fashion_seed907_n256_q1',[])
    assert seen==['extended_revision_paper_gap_checks_har_seed919_n128_q8_timing',
                  'extended_revision_paper_completion_fashion_seed907_n256_q1']


def test_fixed_fit_grid_is_twelve_and_excludes_new_workflow_fit():
    from campaigns import extended_revision_20261103 as q
    assert len(q.FITS)==len(set(q.FITS))==12
    assert {seed for _,seed in q.FITS}=={907,911,919}
    assert all(seed!=929 for _,seed in q.FITS)
