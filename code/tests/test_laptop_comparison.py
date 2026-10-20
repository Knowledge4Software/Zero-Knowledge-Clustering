import json
from pathlib import Path
import numpy as np
import pytest
from campaigns.laptop_comparison import plan
from experiments.laptop_comparison import metric_record, relative_errors, proof_cases


def test_exact_labels_survive_float_rounding():
    reference=np.array([[4.,1.]])
    # These unequal integer distances become tied in float64.
    exact=np.array([[2**60+1,2**60]],dtype=object)
    row,labels=metric_record(reference,exact,[1.],[1.],exact.argmin(axis=1))
    assert labels.tolist()==[1] and row['agreement']==1


def test_objective_metric_detects_cost_loss_with_perfect_assignments():
    original=np.array([[1.,4.],[9.,2.]])
    row,_=metric_record(original,original*.1,[4.,9.],[.4,.9])
    assert row['ari']==1 and row['original_assignment_cost_ratio']==1
    assert row['projected_cost_ratio']==pytest.approx(.1)
    assert not row['qualified_geometry']


def test_zero_distances_cannot_be_silently_excluded():
    errors=relative_errors([0.,1.],[.2,1.])
    assert errors['zero_reference_violations']==1


def test_all_proof_comparisons_use_same_input_and_matching_dimensions():
    instances=proof_cases(dict(shape=[2,8,6,3,1],scale=64))
    first=instances['full']
    for name,instance in instances.items():
        assert instance.X==first.X and instance.c_x==first.c_x and instance.r_x==first.r_x
        if name not in ['full','protected_only']:
            rank=0 if instance.protected_basis is None else len(instance.protected_basis)
            assert instance.m+rank==6


def test_campaign_bounded_and_fixed_confirmation_starts():
    jobs=plan(Path('binary.exe'))
    assert len({j['id'] for j in jobs})==len(jobs)
    assert all(j['rss_gib']==3 and j['threads']==4 and j['timeout']>0 for j in jobs)
    for job in jobs:
        if job['kind']=='geometry':
            assert job['starts']==[3,4] and job['seeds']==[100,101,102]


def test_memory_limit_records_resource_failure_and_continues(tmp_path, monkeypatch):
    from experiments.laptop_comparison import circuits
    from commitjl import zk_bridge
    monkeypatch.setattr(zk_bridge, 'run_prove', lambda *a, **kw: zk_bridge.ProveResult(
        ok=False, variant='full', backend='groth16', failure_kind='memory_limit',
        error='proof process exceeded RSS budget'))
    monkeypatch.setattr(zk_bridge, 'BIN_PATH', Path('unused.exe'))
    circuits(dict(kind='solve',shape=[2,8,6,3,1],scale=64,
        binary='unused.exe',methods=['full','pca'],case_timeout=1,rss_gib=1), tmp_path)
    assert json.loads((tmp_path/'done.json').read_text())['rows']==2
    for path in (tmp_path/'rows').glob('*.json'):
        row=json.loads(path.read_text())
        assert row['status']=='resource_limit' and not row['verify_ok']
        assert row['successful_proofs']==0


def test_affine_control_matches_distance_objective_and_first_ties():
    from experiments.laptop_affine_control import assign_cost
    from commitjl.clustering import _squared_distances
    x=np.array([[0.,0.],[1.,0.],[4.,-2.]])
    c=np.array([[0.,0.],[2.,0.],[0.,0.]])
    distances=_squared_distances(x,c)
    labels,cost=assign_cost(x,c)
    np.testing.assert_array_equal(labels,distances.argmin(axis=1))
    assert labels[1]==0 and cost==pytest.approx(distances.min(axis=1).sum())


def test_pair_workload_matches_direct_squared_distances():
    from experiments.laptop_pair_workload import all_pairs
    points=np.array([[0.,0.],[1.,3.],[1.,3.],[-2.,4.]])
    direct=np.sum((points[:,None,:]-points[None,:,:])**2,axis=-1)
    np.testing.assert_allclose(all_pairs(points),direct)


def test_resume_reuses_saved_opening_and_rejects_corruption(tmp_path):
    from experiments.laptop_comparison import restore_fixture_commitment, atomic_json
    job=dict(shape=[2,8,6,3,1],scale=64)
    first=proof_cases(job);saved=first['full'].to_json_dict()
    path=tmp_path/'instances/full_complete.json';atomic_json(path,saved)
    resumed=restore_fixture_commitment(proof_cases(job),tmp_path)
    assert all(i.c_x==first['full'].c_x and i.r_x==first['full'].r_x for i in resumed.values())
    saved['c_x']=str(int(saved['c_x'])+1);atomic_json(path,saved)
    with pytest.raises(ValueError,match='commitment opening is invalid'):
        restore_fixture_commitment(proof_cases(job),tmp_path)
