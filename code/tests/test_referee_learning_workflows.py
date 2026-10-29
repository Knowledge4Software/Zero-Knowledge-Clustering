import numpy as np
import pytest
from data.referee_revision_data import split_indices
from experiments.referee_learning_workflows import minibatch_proposals,proof_parent
from campaigns.local_fast_tradeoff import sha
import json


def test_sampling_has_real_distinct_rows_and_fails_instead_of_padding():
    result=split_indices(3000,2048,41)
    assert len(set(result))==2048 and np.array_equal(result,split_indices(3000,2048,41))
    with pytest.raises(ValueError):split_indices(1000,2048,41)


def test_training_minibatch_proposals_retain_both_batches_and_all_rates():
    old=np.asarray([[0,0],[20,20]],np.int64)
    incoming=np.asarray([[2,2],[18,18],[4,4],[16,16]],np.int64)
    proposed,meta=minibatch_proposals(old,incoming)
    assert len(proposed)==6 and [(m['batch'],m['rate']) for m in meta]==[(b,r) for b in [0,1] for r in [.25,.5,1.]]
    assert np.array_equal(proposed[2],[[2,2],[18,18]]) and np.array_equal(proposed[5],[[4,4],[16,16]])
    assert np.array_equal(old,[[0,0],[20,20]])


def test_empty_new_group_keeps_old_center():
    old=np.asarray([[0,0],[20,20]],np.int64);proposed,_=minibatch_proposals(old,np.ones((4,2),np.int64))
    assert all(np.array_equal(p[1],old[1]) for p in proposed)


def test_new_proof_inputs_share_exact_catalog_and_cache_only_duplicate_centers(tmp_path):
    old=np.zeros((2,80),np.int64);old[1]=10;basis=np.eye(80,dtype=np.int64)[:63]*16384
    model=dict(n=1024,d=80,k=2,centers=old.tolist(),basis=basis.tolist(),method='learned_energy')
    rows=np.zeros((1024,80),np.int64);sets=[old,old+1,old+1,old+2]
    meta=dict(training_sha256='training',audit_sha256='audit',prediction_fitting_seconds=1.,auxiliary_fitting_seconds=2.)
    proof_parent(tmp_path,model,np.eye(80,dtype=np.int64)[:79]*16384,rows,sets,meta,[{}]*4)
    head=json.loads((tmp_path/'head_r63/input.json').read_text());full=json.loads((tmp_path/'full/input.json').read_text())
    assert head['x']==full['x'] and len(head['x'])==128 and head['reuse_centers']==full['reuse_centers']
    assert head['reuse_centers']==[old.tolist(),(old+1).tolist(),(old+2).tolist()]
    protocol=json.loads((tmp_path/'protocol.json').read_text());assert protocol['duplicate_source_sets']==[0,2] and protocol['selected_source_sets']==[1,3]
    assert json.loads((tmp_path/'auxiliary_basis.json').read_text())['source_input_sha256']==sha(tmp_path/'head_r63/input.json')
