"""Release-gate invariants and safe priority-coordinator recovery."""
import sys
import numpy as np
import pytest
from experiments import release_workflow_20261103 as work


def test_best_target_upper_contains_truth_for_improving_and_worsening_proposals():
    rows=np.array([[0,0],[1,0],[9,0],[10,0]],dtype=np.int64)
    old=np.array([[3,0],[7,0]],dtype=np.int64)
    for new in [old,np.array([[0,0],[10,0]]),np.array([[30,30],[-30,-30]])]:
        truth,upper,cost=work.moments_upper(rows,old,new)
        assert cost==26 and truth<=upper
        if upper<0:assert truth<0
    assert work.moments_upper(rows,old,old)[:2]==(0,0)


def test_native_workflow_keeps_every_step_and_rebuilds_only_after_strict_improvement(tmp_path,monkeypatch):
    source=tmp_path/'original';source.mkdir()
    rows=np.array([[0,0,0]]*64+[[10,0,0]]*64,dtype=np.int64)
    old=np.array([[4,0,0],[6,0,0]],dtype=np.int64)
    basis=np.array([[16384,0,0]],dtype=np.int64)
    np.savez(source/'arrays.npz',rows=rows,old=old,past=np.tile(rows,(4,1)),
         centroid=basis,pca=basis,test_incoming=np.repeat(rows,2,axis=0),
         validation_incoming=np.repeat(rows,2,axis=0),validation_rows=rows)
    choices={m:dict(rank=1) for m in ['ordered_head','pca']}
    work.save(source/'selection.json',dict(choices=choices))
    work.save(source/'provenance.json',dict(dataset='toy',seed=9,scale=1))
    pins={p.name:work.sha(p) for p in source.iterdir()};refits=[]
    def fit(past,reference,seed):
        refits.append(reference.copy());return {'centroid':basis,'pca':basis},{}
    monkeypatch.setattr(work,'fit_bases',fit)
    monkeypatch.setattr(work,'proposal_banks',lambda *a:(np.array([old]*8),[]))
    monkeypatch.setattr(work,'evaluate',lambda *a:[])
    monkeypatch.setattr(work,'validation_choice',lambda *a:dict(rank=1))
    def ablate(rows,reference,centers,*a,**kw):
        truth=work.moments_upper(rows,reference,centers[0])[0]
        return [dict(conclusive=True,corrections=0,lower=str(truth),upper=str(truth),denominator='1')]*2
    monkeypatch.setattr(work,'ablate',ablate)
    work.run(source,tmp_path/'out',{'toy':{}})
    result=work.read(tmp_path/'out/result.json');records=result['records']
    assert len(records)==8 and [r['step'] for r in records]==list(range(8))
    assert all(r['accepted']==(r['truth']<0) for r in records)
    assert len(refits)==sum(r['accepted'] for r in records[:-1])
    for previous,current in zip(records,records[1:]):
        assert current['episode']==previous['episode']+int(previous['accepted'])
    assert pins=={p.name:work.sha(p) for p in source.iterdir()}


@pytest.mark.skipif(sys.platform=='win32',reason='Linux signal coordinator')
def test_resume_is_limited_to_pinned_gap_coordinator(monkeypatch):
    from campaigns import review_completion_20261103 as q
    actions=[]
    class Process:
        def send_signal(self,sig):actions.append(sig)
    monkeypatch.setattr(q.psutil,'Process',lambda pid:Process())
    monkeypatch.setattr(q,'matches',lambda *a:False)
    state=dict(gap_paused=True,gap_owner=dict(pid=123,created=1,module='wrong'))
    assert not q.restore_gap(state) and not actions
    monkeypatch.setattr(q,'matches',lambda *a:True)
    assert q.restore_gap(state) and actions==[q.signal.SIGCONT]
    state['gap_paused']=False
    assert not q.restore_gap(state) and len(actions)==1
