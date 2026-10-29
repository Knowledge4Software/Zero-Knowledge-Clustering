"""Verify update construction and certificate references against exact oracles."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from commitjl.reuse_certificates import exact_reference
from experiments.review_sensitivity import center_frontier, nested_owner, native_summary, run_native, finalize, sha


def tiny_owner():
    return dict(model=dict(n=4,d=3,k=2,input_bits=4,basis_scale=8,
        centers=[[0,0,0],[4,0,0]],basis=[[8,0,0],[0,8,0]]),
        x=[[0,1,0],[1,2,0],[3,1,0],[4,2,0]],reuse_centers=[],queries=[])


def test_frontier_deployed_deterministic_and_realized_drift():
    centers=[[0]*40,[500]*40,[-500]*40]
    sets,meta,scale=center_frontier(centers,12,amplitudes=[.01,.1],seeds=[17,29])
    assert (sets,meta,scale)==center_frontier(centers,12,amplitudes=[.01,.1],seeds=[17,29])
    assert sets[0]==centers and len(sets)==5
    for entry,c in zip(meta[1:],sets[1:]):
        actual=np.sqrt(np.mean(np.sum((np.asarray(c)-centers)**2,axis=1)))/scale
        assert actual==entry['actual_relative_drift']
        assert abs(actual-entry['requested_relative_drift']) <= np.sqrt(40)*.5/scale


def test_clipping_and_quantization_duplicates_retained():
    sets,meta,_=center_frontier([[16]*5,[-16]*5],4,amplitudes=[1e-9,1.],seeds=[17])
    assert len(sets)==3 and meta[1]['unchanged_after_quantization']
    assert meta[2]['clipped_coordinates']>0
    assert np.abs(np.asarray(sets)).max()<=16


def test_nested_rank_and_catalog_leave_source_unchanged():
    source=tiny_owner();before=copy.deepcopy(source)
    result=nested_owner(source,source['x'],[[[0,0,0],[4,0,0]]],1,2,batch=1)
    assert source==before
    assert result['model']['basis']==source['model']['basis'][:1]
    assert result['model']['basis_scale']==8 and result['x']==source['x'][:2]
    assert result['queries']==[[0],[1]]
    with pytest.raises(ValueError):nested_owner(source,source['x'],[],3,4,batch=2)


def test_native_distance_matches_python_integer_oracle_and_exact_certificate():
    source=tiny_owner();centers=source['model']['centers']
    result=native_summary(source,centers)
    labels,cost=exact_reference(source['x'],centers)
    assert labels==[0,0,1,1] and result['quantized_true_cost']==cost
    assert result['certified_wrong_rows']==0
    assert result['coverage99_cost1_policy_accept']
    assert result['interval_relative_width']==0
    source['x'][0][0]=2**40
    with pytest.raises(OverflowError):native_summary(source,centers)


def test_native_rejects_changed_inputs_and_never_overwrites(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    path=prepared/'owner.json';path.write_text(json.dumps(tiny_owner()))
    (prepared/'protocol.json').write_text(json.dumps(dict(inputs=[dict(path=path.name,sha256=sha(path))])))
    path.write_text('{}')
    with pytest.raises(ValueError,match='hash changed'):run_native(prepared,tmp_path/'native')
    with pytest.raises(FileExistsError):run_native(prepared,tmp_path/'native')


def test_final_summary_requires_repetitions_and_public_replay_count(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    runs=tmp_path/'runs';(runs/'native').mkdir(parents=True)
    (runs/'native/cells.csv').write_text('cost_lower_scaled,cost_upper_scaled\n')
    job=dict(name='full',input='unused.json',batch=2,full=True,certified=False)
    (prepared/'protocol.json').write_text(json.dumps(dict(jobs=[job],center_sets=[dict(set=0)])))
    root=runs/'full';(root/'bundle').mkdir(parents=True)
    summary=dict(complete=True,sets=1,verified_public_samples=1)
    (root/'summary.json').write_text(json.dumps(summary))
    (root/'protocol.json').write_text(json.dumps(dict(reps=3,warmup=1)))
    (root/'workload.csv').write_text('set\n0\n')
    sample=dict(id=0,prove_seconds=[.1,.2])
    path=root/'bundle/full_reuse_2.json'
    path.write_text(json.dumps(dict(samples=[sample])))
    out=tmp_path/'completed.json'
    with pytest.raises(ValueError,match='three positive'):finalize(prepared,runs,out)
    assert not out.exists()
    sample['prove_seconds']=[.1,.2,.3]
    path.write_text(json.dumps(dict(samples=[sample])))
    summary['verified_public_samples']=2
    (root/'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='replay count'):finalize(prepared,runs,out)
    summary['verified_public_samples']=1
    (root/'summary.json').write_text(json.dumps(summary))
    finalize(prepared,runs,out)
    assert json.loads(out.read_text())['jobs'][0]['timing_samples'][0]['median']==.2


def test_final_summary_distinguishes_one_time_stage1_from_repeated_queries(tmp_path):
    prepared=tmp_path/'prepared';prepared.mkdir()
    runs=tmp_path/'runs';(runs/'native').mkdir(parents=True)
    (runs/'native/cells.csv').write_text('method,rank,n,set,certified_count,cost_lower_scaled,cost_upper_scaled\nlearned_energy,15,256,0,2,10,12\n')
    job=dict(name='reuse',input='owner.json',batch=2,full=False,certified=True)
    protocol=dict(jobs=[job],center_sets=[dict(set=0)],inputs=[dict(path='owner.json',method='learned_energy',rank=15)])
    (prepared/'protocol.json').write_text(json.dumps(protocol))
    root=runs/'reuse';(root/'bundle').mkdir(parents=True)
    (root/'summary.json').write_text(json.dumps(dict(complete=True,sets=1,verified_public_samples=3)))
    (root/'protocol.json').write_text(json.dumps(dict(reps=3,warmup=1)))
    (root/'workload.csv').write_text('set,certificate_count,cost_lower_scaled,cost_upper_scaled\n0,2,10,12\n')
    for phase,values in [('stage1',[.2]),('reuse_2',[.1,.2,.3]),('reuse_certified_2',[.2,.3,.4])]:
        (root/'bundle'/(phase+'.json')).write_text(json.dumps(dict(samples=[dict(id=0,prove_seconds=values)])))
    out=tmp_path/'completed.json'
    finalize(prepared,runs,out)
    samples=json.loads(out.read_text())['jobs'][0]['timing_samples']
    assert [s['expected_repetitions'] for s in samples]==[1,3,3]
    (root/'bundle/reuse_2.json').write_text(json.dumps(dict(samples=[dict(id=0,prove_seconds=[.1])])))
    with pytest.raises(ValueError,match='three positive'):finalize(prepared,runs,tmp_path/'bad.json')
