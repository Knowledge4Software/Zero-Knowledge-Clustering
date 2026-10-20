import copy
import json

import numpy as np
import pytest
from sklearn.cluster import DBSCAN

from campaigns.paper_compression import configuration,exclusive_run
from analysis.paper_compression_report import task_setup
from experiments.paper_artifacts import artifact_issues,job_complete,digest
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import NativeProjection
from experiments.paper_compression import (bootstrap_crossed, case_plan, distortion,
    measured_workloads, quality_row, reconstructed_centers, select_validation,
    split_indices, sqdist,safe_ratio,stable_topk)


def test_split_disjoint_complete_and_reproducible():
    folds=split_indices(100)
    assert [len(v) for v in folds.values()]==[60,20,20]
    np.testing.assert_array_equal(np.sort(np.concatenate(list(folds.values()))),np.arange(100))
    for key,value in folds.items(): np.testing.assert_array_equal(value,split_indices(100)[key])


def test_full_and_protected_only_have_different_geometric_claims():
    rng=np.random.default_rng(42); x=rng.normal(size=(80,8))
    centers=np.zeros((2,8)); centers[:,0]=[-1,1]
    train=x[:40]; evaluate=x[40:]; pairs=np.array([[i,i+1] for i in range(39)])
    ref=sqdist(evaluate,centers); original=ref.argmin(axis=1)
    train_labels=sqdist(train,centers).argmin(axis=1)
    phi=sqdist(evaluate,reconstructed_centers(evaluate,original,2)).min(axis=1).sum()
    train_phi=sqdist(evaluate,reconstructed_centers(train,train_labels,2)).min(axis=1).sum()
    pair_ref=((evaluate[pairs[:,0]]-evaluate[pairs[:,1]])**2).sum(axis=1)
    cfg=configuration('smoke')
    for method,model in [('full',NativeProjection('full',8)),
        ('protected_only',NativeProjection('pca',8,components=centroid_basis(centers).T,origin=centers[0]))]:
        case=dict(method=method,k=2,m=model.m,q=0,seed=-1,start=0,phase='test')
        row,arrays=quality_row(case,evaluate,np.array([]),model.transform(evaluate),model.transform(centers),
            model.transform(train),train,train_labels,ref,pairs,pair_ref,phi,train_phi,centers,cfg)
        assert row['ari']==1 and row['agreement']==1
        assert row['certified_violations']==0
        np.testing.assert_array_equal(arrays['labels'],original)
        if method=='full': assert row['pair_q95']<1e-12 and row['r_phi_train_reconstructed']==pytest.approx(1)
        else: assert row['pair_q95']>.5


def validation_fixture():
    rows=[]
    for m in (8,16):
        for seed in range(2):
            for start in range(2):
                rows.append(dict(phase='validation',k=5,method='protected_jl',m=m,q=1,seed=seed,start=start,
                    ari=1.,r_phi_train_reconstructed=1.,pair_q95=.4 if m==8 else .1,
                    centroid_q95=.1,projected_cost_relative_error=.05,pair_zero_violations=0,centroid_zero_violations=0))
    return rows


def test_selection_complete_quality_only_and_keeps_failed_operating_points():
    cfg=configuration('smoke'); rows=validation_fixture()
    selected,summary=select_validation(rows,cfg)
    json.dumps(dict(selected=selected,summaries=summary),allow_nan=False)
    assert selected[0]['m']==16
    changed=copy.deepcopy(rows)
    for r in changed: r.update(test_ari=0.,runtime=1e99,compression_seconds=-100.)
    assert select_validation(changed,cfg)==(selected,summary)
    missing=rows[:-1]
    assert select_validation(missing,cfg)[0][0]['selected'] is False
    violated=copy.deepcopy(rows); violated[-1]['pair_zero_violations']=1
    assert select_validation(violated,cfg)[0][0]['selected'] is False


def test_dimensional_budget_and_pca_training_rank():
    cfg=configuration('smoke'); cfg.update(m=[4,8,16,32],q=[1,2])
    cases=list(case_plan(cfg,cfg['m'],5,0,4,12,32,'validation'))
    assert all(c['m']<=11 for c in cases if c['method']=='pca')
    assert all(c['m']>4 for c in cases if c['method']=='protected_jl')
    assert all(c['m']<32 for c in cases if c['method'] not in ('full','protected_only'))


def test_zero_pairs_and_bootstrap_axes():
    error=distortion(np.array([0.,1.,4.]),np.array([0.,1.1,3.6]),.15,1e-12)
    assert error['zero_count']==1 and error['zero_violations']==0
    assert error['q95']==pytest.approx(.1)
    assert distortion(np.zeros(1),np.ones(1),.15,1e-12)['zero_violations']==1
    assert bootstrap_crossed(np.ones((2,3)),draws=100)==(1.,1.,1.)
    with pytest.raises(ValueError): bootstrap_crossed(np.array([[np.nan]]))


def test_workloads_replay_and_squared_dbscan_radius(tmp_path):
    rng=np.random.default_rng(67); x=rng.normal(size=(40,8)); centers=x[:5].copy()
    cfg=configuration('smoke'); case=dict(method='full',k=5,m=8,q=0,seed=-1,start=0,phase='test')
    timings=measured_workloads(case,NativeProjection('full',8),x,np.arange(30,40),centers,
        sqdist(x[30:40],centers),cfg,dict(radius=1.7,radii={'validation_q50':2.5}),tmp_path)
    assert timings['compression']['median']==0 and timings['workload_rows']==40
    assert timings['knn_recall']==1 and timings['kmeans_ari']==1
    assert timings['dbscan_certified_violations']==0
    assert 'heldout_assignment' in timings and 'numeric_preparation' in timings
    assert timings['measurement_version']==2 and timings['kmeans_refine_ari']==1
    assert timings['dbscan_profiles']['validation_q50']['graph_f1']==1
    with np.load(tmp_path/'queries/query0.npz') as saved:
        a,b=saved['left_indices'],saved['right_indices']
        expected=((x[a,None,:]-x[None,b,:])**2).sum(axis=2)
        np.testing.assert_allclose(saved['original'],expected,atol=1e-12)
        np.testing.assert_allclose(saved['projected'],expected,atol=1e-12)
    with np.load(tmp_path/'dbscan.npz') as saved:
        expected=DBSCAN(eps=1.7,min_samples=5,metric='precomputed').fit(np.sqrt(saved['original_sq'])).labels_
        np.testing.assert_array_equal(saved['original_labels'],expected)


def test_degenerate_objectives_are_undefined_and_not_selectable():
    assert safe_ratio(0,0) is None and safe_ratio(1,0) is None
    zeros=np.zeros((12,4));centers=np.zeros((2,4));pairs=np.array([[0,1],[1,2]])
    case=dict(method='full',phase='validation',k=2,m=4,q=0,seed=-1,start=0)
    row,_=quality_row(case,zeros[:6],np.array([]),zeros[:6],centers,zeros[6:],zeros[6:],
        np.zeros(6,dtype=int),np.zeros((6,2)),pairs,np.zeros(2),0.,0.,centers,configuration('smoke'))
    assert row['projected_cost_ratio'] is None and row['pair_q95'] is None
    assert row['r_phi'] is None and row['r_phi_train_reconstructed'] is None
    json.dumps(row,allow_nan=False)
    cfg=configuration('smoke');cfg['starts']=1
    assert not select_validation([row],cfg)[0][0]['selected']


def test_validation_rejects_test_rows_and_duplicate_cells():
    cfg=configuration('smoke');rows=validation_fixture();bad=copy.deepcopy(rows)
    bad[0]['phase']='test'
    with pytest.raises(ValueError,match='validation'):select_validation(bad,cfg)
    # Axes and count still look complete, but one Cartesian cell is duplicated.
    rows[-1]=copy.deepcopy(rows[-2])
    assert not select_validation(rows,cfg)[0][0]['selected']


def test_stable_topk_matches_stable_full_sort_at_duplicate_boundary():
    values=np.array([[1.,0.,0.,0.,2.],[2.,1.,0.,3.,4.]])
    np.testing.assert_array_equal(stable_topk(values,2),np.argsort(values,axis=1,kind='stable')[:,:2])


def test_task_totals_include_only_required_setup():
    row=dict(method='protected_jl',model_construction={'median':4.},centroid_fit={'median':20.},
        pca_fit=None,basis_fit={'median':1.})
    w={key:{'median':value} for key,value in dict(compression=5.,center_projection=3.,numeric_preparation=8.,
        full_numeric_preparation=7.,point_norm_preparation=2.,full_point_norm_preparation=1.).items()}
    distance=task_setup(row,w,'distance_queries');assert distance['conditional_setup_seconds']==11
    assert distance['method_model_fit_seconds']==21 and distance['full_model_fit_seconds']==0
    km=task_setup(row,w,'kmeans');assert km['conditional_setup_seconds']==9 and km['full_preparation_seconds']==0
    refine=task_setup(row,w,'kmeans_refine');assert refine['conditional_setup_seconds']==12 and refine['full_model_fit_seconds']==20
    assign=task_setup(row,w,'assignment');assert assign['conditional_setup_seconds']==20


def test_corrupt_checkpoint_is_never_complete(tmp_path):
    from experiments.laptop_comparison import atomic_json
    case=dict(method='full',phase='test',k=2,m=4,q=0,seed=-1,start=0)
    from experiments.paper_compression import case_key
    key=case_key(case);files=[tmp_path/'arrays'/f'{key}.npz',tmp_path/'models/model.npz']
    for path in files:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'initial')
    row=dict(case,model_artifact='models/model.npz',workloads=None,
        artifact_sha256={p.relative_to(tmp_path).as_posix():digest(p) for p in files})
    atomic_json(tmp_path/'rows'/f'{key}.json',row)
    atomic_json(tmp_path/'case_inventory.json',dict(cases=[dict(case,key=key)],expected_rows=1))
    atomic_json(tmp_path/'done.json',dict(complete=True))
    cache={};assert job_complete(tmp_path,cache)
    files[0].write_bytes(b'corrupt')
    assert not job_complete(tmp_path,cache) and artifact_issues(tmp_path,row,cache)


def test_exclusive_runner_lock_releases_and_blocks_duplicate(tmp_path):
    with exclusive_run(tmp_path):
        with pytest.raises(RuntimeError,match='active runner'):
            with exclusive_run(tmp_path):pass
    with exclusive_run(tmp_path):pass


def test_high_offset_geometry_does_not_silently_collapse():
    from experiments.paper_compression import assignment
    from commitjl.reusable_native import PreparedDataset
    x=np.array([[1e9,1e9],[1e9+1,1e9+2],[1e9+3,1e9-1]])
    expected=((x[:,None]-x[None,:])**2).sum(axis=2)
    np.testing.assert_array_equal(sqdist(x,x),expected)
    labels,cost=assignment(x,x[:2],np.einsum('ij,ij->i',x,x),np.einsum('ij,ij->i',x[:2],x[:2]))
    np.testing.assert_array_equal(labels,expected[:,:2].argmin(axis=1));assert cost==expected[:,:2].min(axis=1).sum()
    prepared=NativeProjection('full',2).prepare(x)
    np.testing.assert_array_equal(prepared.squared_distances(np.arange(3),np.arange(3)),expected)
    np.testing.assert_array_equal(prepared.precompute_distances(),expected)
