"""All-source data isolation, feature fidelity and native-baseline integration."""
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Linux-only server coordinator')


def test_wide_sources_retain_every_feature_and_disjoint_rows():
    from campaigns.server_all_sources_20261101 import source_rows
    for name, sizes, width in [('arcene',(64,4,32),10000), ('gisette',(512,512,1024),5000)]:
        past, incoming, audit, pin = source_rows(name,701)
        assert (len(past),len(incoming),len(audit)) == sizes
        assert past.shape[1] == incoming.shape[1] == audit.shape[1] == width
        sets = [set(pin[k]) for k in ['past_ids','incoming_ids','audit_ids']]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])


def test_tfidf_vocabulary_is_fitted_only_on_past_documents():
    from campaigns.server_all_sources_20261101 import source_rows,PACKAGE
    from campaigns.published_baselines_laptop import read
    past, incoming, audit, pin = source_rows('text_tfidf',709)
    prep = read(PACKAGE/'text_tfidf_seed709_preprocessing.json')
    assert past.shape == incoming.shape == (512,2000) and audit.shape == (1024,2000)
    assert prep['preprocessing_training_ids'] == pin['training_ids'][:512]
    assert not set(prep['preprocessing_training_ids']) & set(pin['audit_ids'])


def test_sample_limited_scale_uses_no_incoming_or_audit_information():
    from campaigns.server_all_sources_20261101 import quantize
    past = np.ones((64,84))
    incoming = np.full((4,84),1000.)
    audit = np.full((32,84),2000.)
    a,b,c,scale = quantize(past,incoming,audit)
    assert scale == 4095 and a.max() == b.max() == c.max() == 4095


def test_synthetic_controls_do_not_count_as_real_corpora():
    from campaigns.server_all_sources_20261101 import source_rows
    past,incoming,audit,pin = source_rows('dbscan_margin',719)
    assert (len(past),len(incoming),len(audit)) == (512,256,232)
    assert pin['synthetic_control'] is True
    assert len(set(pin['past_ids'])|set(pin['incoming_ids'])|set(pin['audit_ids'])) == 1000


def test_new_admission_rule_preserves_initial_study_and_recovery_budget():
    from campaigns.server_all_sources_20261101 import admissible
    assert admissible(31*3600,3600)
    assert not admissible(32*3600,3600)
    assert not admissible(32.5*3600,0)


def test_disk_cleanup_preserves_proofs_and_rejects_unowned_paths(tmp_path):
    from campaigns.server_all_sources_20261101 import retire_private_keys
    from campaigns.published_baselines_laptop import read
    root=tmp_path/'owned'
    case=root/'matched/case'
    private=case/'zkml_private';private.mkdir(parents=True)
    for name in ['pk','params','vk','proof','input.json']:
        (private/name).write_bytes(name.encode())
    go=case/'ordered_head/full/bundle/keys';go.mkdir(parents=True)
    (go/'full_8.pk').write_bytes(b'private')
    (go/'full_8.vk').write_bytes(b'public')
    retire_private_keys(root,case)
    assert not (private/'pk').exists() and not (go/'full_8.pk').exists()
    assert all((private/name).exists() for name in ['params','vk','proof','input.json'])
    assert (go/'full_8.vk').read_bytes()==b'public'
    assert len(read(case/'private_key_retirement.json')['files'])==2
    outside=tmp_path/'unowned';outside.mkdir()
    with pytest.raises(AssertionError):
        retire_private_keys(root,outside)


def test_resource_adapters_change_only_compiler_allowance_and_isolated_imports():
    import ast
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'campaigns'
    header='# Versioned all-source coordinator: compiler resource ceiling only; proof relations unchanged.\n'
    pairs={'local_fast_tradeoff':'server_all_sources_full_20261101',
           'run_group_fast_control':'server_all_sources_compact_20261101',
           'run_group_refinement_control':'server_all_sources_refinement_20261101',
           'cost_aware_pilot':'server_all_sources_cost_aware_20261101'}
    for old,new in pairs.items():
        expected=(root/(old+'.py')).read_text().replace('1600000','24000000')
        if old in ['run_group_fast_control','run_group_refinement_control']:
            expected=expected.replace('from campaigns.local_fast_tradeoff import',
                                      'from campaigns.server_all_sources_full_20261101 import')
        assert (root/(new+'.py')).read_text()==header+expected
    source=(root/'server_campaign_20261101.py').read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='warm')
    function='\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'
    function=function.replace('from campaigns.cost_aware_pilot import calibrate, prove_case',
                              'from campaigns.server_all_sources_cost_aware_20261101 import calibrate, prove_case')
    assert (root/'server_all_sources_warm_20261101.py').read_text()==(
        header+'import os\nfrom campaigns.server_campaign_20261101 import REPO, GO, RUST, read, save, sha\n\n\n'+function)


def test_native_baseline_gate_and_paired_proof_inputs_integrate(tmp_path,monkeypatch):
    import campaigns.server_all_sources_20261101 as module
    from campaigns.published_baselines_laptop import read,save,sha
    from commitjl.pilot_io import array_sha
    monkeypatch.setattr(module,'server_profile',lambda:{'workers':12})
    rng = np.random.default_rng(27)
    dataset,seed = 'har_neural256',701
    fitted = tmp_path/'fitted'/f'{dataset}_seed{seed}'
    fitted.mkdir(parents=True)
    arrays = dict(training=rng.integers(-10,10,(64,84)),audit=rng.integers(-10,10,(128,84)),
                  centers=rng.integers(-10,10,(5,6,84)))
    for method in ['ordered_head','pca']:
        basis=np.rint(np.linalg.qr(rng.normal(size=(84,79)))[0].T*16384).astype(np.int64)
        arrays[method+'_basis']=basis[:63]
        arrays[method+'_auxiliary']=basis
    np.savez_compressed(fitted/'arrays.npz',**arrays)
    save(fitted/'pin.json',dict(dataset=dataset,seed=seed,k=6,scale=1.,
        arrays_sha256=sha(fitted/'arrays.npz'),training_sha256=array_sha(arrays['training']),
        basis_fitting_seconds={'ordered_head':0.,'pca':0.},proposal_recipe=[{'id':i} for i in range(4)]))
    save(tmp_path/'plan.json',dict(selection='Frozen training-only candidates'))
    module.native_case(tmp_path,dataset,seed)
    result=read(tmp_path/'native_cases'/f'{dataset}_seed{seed}'/'native_results.json')
    assert result['complete'] and result['new_proofs']==0 and result['all_deterministic_intervals_valid']
    methods={r['method'] for r in result['records']}
    assert {'exact_full','ordered_head_certificate','pca_certificate','sign_rp_raw','sparse_rp_raw',
            'lightweight_coreset_m16','uniform_coreset_m32'} <= methods
    case=module.materialize(tmp_path,dataset,seed)
    left,right=[read(case/m/'full/input.json') for m in ['ordered_head','pca']]
    assert left==right and left['model']['n']==32 and left['model']['k']==6
    assert len(left['x'])==32 and len(left['reuse_centers'])==5
    left['x'][0][0]+=1
    save(case/'ordered_head/full/input.json',left)
    with pytest.raises(AssertionError):
        module.materialize(tmp_path,dataset,seed)
