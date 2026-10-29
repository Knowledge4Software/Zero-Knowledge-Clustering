import json
import numpy as np
import pytest
from analysis.export_two_review_evidence import ratio_summary,public_resources,quality,check_exact_samples,cleanup_timing_note
from campaigns.referee_two_reviews_local import Runner
from campaigns.local_fast_tradeoff import save,sha
from experiments.neural_activity_prototypes import class_updates


def test_timing_interval_is_conditional_on_three_repetitions_not_learning_units():
    result=ratio_summary([1.,2.,4.])
    assert result['n']==3 and result['geometric_mean']==pytest.approx(2.)
    assert result['conditional_t95'][0]<2<result['conditional_t95'][1]
    assert 'one fit' in result['interpretation'] and 'conditional_t95' not in ratio_summary([1.,2.])


def test_cleanup_marks_only_comparisons_that_overlap_not_unrelated_runs(tmp_path):
    assert cleanup_timing_note(tmp_path,'proof_har561') is None
    save(tmp_path/'filesystem_cleanup_interference_20261027.json',dict(affected_proof_cases=['proof_har561']))
    assert 'excluded from clean speed' in cleanup_timing_note(tmp_path,'proof_har561')
    assert cleanup_timing_note(tmp_path,'proof_text_minilm384') is None


def test_public_resource_counts_use_actual_proof_files_and_verifier_receipts(tmp_path):
    (tmp_path/'a.proof').write_bytes(b'proof');(tmp_path/'a.vk').write_bytes(b'key')
    save(tmp_path/'meta.json',dict(not_a_proof=True))
    save(tmp_path/'public_manifest.json',dict(sha256={n:sha(tmp_path/n) for n in ['a.proof','a.vk','meta.json']}))
    save(tmp_path/'group_verify.json',dict(complete=True,wall_seconds=.5))
    result=public_resources([tmp_path])
    assert result['proofs']==1 and result['proof_bytes']==5 and result['verifying_key_bytes']==3
    assert result['public_replay_seconds']==.5 and len(result['manifests'])==1
    (tmp_path/'a.proof').write_bytes(b'changed')
    with pytest.raises(ValueError,match='manifest'):public_resources([tmp_path])


def test_backend_check_rejects_duplicate_rows_even_with_matching_total_cost():
    source=dict(x=[[1],[1]],reuse_centers=[[[0],[3]]])
    rows=[dict(set=0,indices=[0],cost='1'),dict(set=0,indices=[1],cost='1')]
    check_exact_samples(source,rows,'PLONK')
    rows[1]['indices']=[0]
    with pytest.raises(ValueError,match='coverage'):check_exact_samples(source,rows,'PLONK')


def test_quality_retains_unresolved_regressions_and_below_target_assignments():
    rows=[dict(true_change=1,decision='inconclusive',agreement_percent=97.,assignment_certified_percent=20.,exact_rows=16),
          dict(true_change=2,decision='worsens',agreement_percent=100.,assignment_certified_percent=90.,exact_rows=0)]
    result=quality(rows)
    assert result['requests']==2 and result['conclusive']==1 and result['regressions']==2 and result['detected']==1
    assert result['below99']==1 and result['exact_rows']==16


def test_subset_resume_does_not_change_or_pad_catalog_inputs(tmp_path):
    source=tmp_path/'source';target=tmp_path/'target';runner=Runner.__new__(Runner)
    for method in ['ordered_head','pca']:
        for file in ['head_r63/input.json','full/input.json']:
            save(source/method/file,dict(model=dict(n=32),x=[[i,2*i] for i in range(32)],reuse_centers=[[[0,0],[1,1]],[[2,2],[3,3]]]))
        save(source/method/'protocol.json',dict(selected_source_sets=[0]))
        save(source/method/'auxiliary_basis.json',dict(source_input_sha256=sha(source/method/'head_r63/input.json')))
    runner.subset(source,target,16);before=sha(target/'ordered_head/full/input.json')
    runner.subset(source,target,16)
    assert sha(target/'ordered_head/full/input.json')==before
    value=json.loads((target/'ordered_head/full/input.json').read_text())
    assert value['x']==[[i,2*i] for i in range(16)] and value['queries']==[list(range(8)),list(range(8,16))]
    with pytest.raises(RuntimeError):runner.subset(source,target,32)


def test_labeled_neural_proposals_keep_class_semantics_and_fixed_rates():
    old=np.asarray([[0,0],[20,20]],np.int64);incoming=np.asarray([[2,2],[18,18],[4,4],[16,16]],np.int64)
    proposed,metadata=class_updates(old,incoming,np.asarray([0,1,0,1]))
    assert len(proposed)==6 and [v['rate'] for v in metadata]==[.25,.5,1.,.25,.5,1.]
    assert np.array_equal(proposed[2],[[2,2],[18,18]]) and np.array_equal(proposed[5],[[4,4],[16,16]])
