"""Bounded admission and fresh frozen-policy paired execution on Linux."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark=pytest.mark.skipif(sys.platform=='win32',reason='Server queue requires Linux flock')


def test_no_block_starts_without_full_time_headroom_and_disk_reserve():
    from campaigns import paper_gap_checks_20261103 as q
    cell=q.CELLS[0]
    assert q.admitted(cell,0,7000,32*2**30)
    assert not q.admitted(cell,0,6000,32*2**30)
    assert not q.admitted(cell,0,7000,31*2**30)
    assert [q.order_for(i,['pca','ordered_head']) for i in [1,2,3]]==[
        ['pca','ordered_head'],['ordered_head','pca'],['pca','ordered_head']]


def test_repetitions_generate_fresh_proofs_preserve_frozen_inputs_and_keep_negatives(tmp_path,monkeypatch):
    from campaigns import paper_gap_checks_20261103 as q
    main=tmp_path/'main';root=tmp_path/'new';original=main/'cases/har_seed919_n128_q8'
    original.mkdir(parents=True);root.mkdir()
    q.save(original/'selection.json',dict(choices={m:dict(rank=16) for m in ['pca','ordered_head']}))
    q.save(original/'provenance.json',dict(scale=1))
    np.savez(original/'arrays.npz',rows=np.zeros((128,3)),centers=np.zeros((8,2,3)))
    q.save(main/'cost_profiles.json',dict(profiles={'har':{'16':{'costs':{}}}}))
    before={p.name:q.sha(p) for p in original.iterdir()}
    monkeypatch.setattr(q,'MAIN',main);monkeypatch.setattr(q,'ROOT',root);monkeypatch.setattr(q,'REPO',tmp_path)
    monkeypatch.setattr(q.final,'write_inputs',lambda *a:None)
    def setup(campaign,case,method,profile):
        out=case/'templates'/method;out.mkdir(parents=True)
        for n,v in [('owner_pin.json',{}),('approved.json',{}),('planner.json',dict(certificate=list(range(8)),full=[],records=[dict(refinement_indices=[]) for _ in range(8)]))]:q.save(out/n,v)
        q.save(case/method/'input.json',dict(model={}))
        return case/method,out
    monkeypatch.setattr(q.final,'setup_method',setup)
    class Planner:
        def __init__(self,*a):pass
        def batch(self,*a):return dict(certificate=list(range(8)),full=[],records=[dict(refinement_indices=[]) for _ in range(8)])
    monkeypatch.setattr(q,'CalibratedPlanner',Planner);monkeypatch.setattr(q,'BatchCosts',lambda **k:k)
    calls=[]
    def prove(campaign,parent,case,source,run_baseline):
        assert not run_baseline
        assert case['name'] not in calls;calls.append(case['name'])
        q.save(parent/case['name']/'checked.json',dict(all_requests_verified=True,all_decisions_correct=True,
            measured_warm_routed_seconds=11 if source.name=='ordered_head' else 10,
            certificate_requests=8,selected_correction_proofs=0))
    monkeypatch.setattr(q,'prove_case',prove)
    monkeypatch.setattr(q.completion,'publish',lambda *a:None)
    class Campaign:
        def invoke(self,*a):return {}
    q.paired_repeats(Campaign(),root,'har',919)
    result=q.read(root/'cases/har_seed919_n128_q8_timing/result.json')
    assert len(calls)==6 and len(set(calls))==6
    assert result['summary']['ordered_head']['mean']==11
    assert result['summary']['pca']['mean']==10
    assert result['complete'] and result['repetitions']==3
    assert before=={p.name:q.sha(p) for p in original.iterdir()}
