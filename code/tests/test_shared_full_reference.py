import json
from pathlib import Path
import pytest
from campaigns.shared_full_reference import share_full_reference
from commitjl.pilot_io import save,sha


@pytest.fixture
def references(tmp_path):
    source=tmp_path/'source';target=tmp_path/'target'
    inp=dict(model=dict(method='full',n=1),x=[[1]],queries=[[0]],reuse_centers=[[[0]]])
    save(source/'full/input.json',inp);save(target/'full/input.json',inp)
    save(source/'status.json',dict(cases=dict(full=dict(complete=True,wall_seconds=321.)),
        input_sha256=dict(full=sha(source/'full/input.json'))))
    for phase in ['prepare','full_reuse','verify']:save(source/'full'/(phase+'.resource.json'),dict(complete=True))
    save(source/'full/bundle/full_reuse_1.json',dict(complete=True,verified=1,samples=[dict(id=0)]))
    save(source/'full/public_bundle/verify.json',dict(complete=True,verified=1))
    proof=source/'full/public_bundle/full_reuse/one.proof';proof.parent.mkdir(parents=True);proof.write_bytes(b'checked-proof')
    save(source/'full/public_bundle/public_manifest.json',dict(sha256={'full_reuse/one.proof':sha(proof)}))
    key=source/'full/bundle/keys/key.bin';key.parent.mkdir(parents=True);key.write_bytes(b'complete-key')
    return source,target


def assert_shared(source,target):
    for path in (source/'full').rglob('*'):
        if path.is_file():assert sha(target/'full'/path.relative_to(source/'full'))==sha(path)
    assert json.loads((target/'status.json').read_text())['cases']['full']['wall_seconds']==321.
    provenance=json.loads((target/'full_reference_provenance.json').read_text())
    assert provenance['copy_attempts'][-1]['new_proof_generation'] is False


def test_interrupted_directory_copy_is_completed_without_removing_evidence(references):
    source,target=references
    partial=target/'full/bundle/keys';partial.mkdir(parents=True)
    evidence=partial/'interrupted.marker';evidence.write_bytes(b'retain')
    share_full_reference(source,target)
    assert evidence.read_bytes()==b'retain';assert_shared(source,target)


def test_missing_file_is_restored_on_resume(references):
    source,target=references;share_full_reference(source,target)
    key=target/'full/bundle/keys/key.bin';key.rename(key.with_name('retained-key'))
    receipt=share_full_reference(source,target)
    assert 'full/bundle/keys/key.bin' in receipt['copied_files'];assert_shared(source,target)


def test_corrupt_file_is_archived_before_recopy(references):
    source,target=references;share_full_reference(source,target)
    key=target/'full/bundle/keys/key.bin';key.write_bytes(b'interrupted')
    receipt=share_full_reference(source,target)
    assert len(receipt['archived_paths'])==1
    assert (target/receipt['archived_paths'][0]).read_bytes()==b'interrupted'
    assert_shared(source,target)


def test_complete_copy_is_not_rewritten_and_prior_provenance_is_retained(references):
    source,target=references;save(target/'full_reference_provenance.json',dict(supervisor_note='retain'))
    share_full_reference(source,target);proof=target/'full/public_bundle/full_reuse/one.proof';before=proof.stat().st_mtime_ns
    receipt=share_full_reference(source,target)
    assert receipt['copied_files']==[] and receipt['archived_paths']==[]
    assert proof.stat().st_mtime_ns==before
    provenance=json.loads((target/'full_reference_provenance.json').read_text())
    assert provenance['supervisor_note']=='retain' and len(provenance['copy_attempts'])==2
    assert_shared(source,target)


def test_input_mismatch_stops_before_copying_any_files(references):
    source,target=references;inp=json.loads((target/'full/input.json').read_text());inp['x']=[[2]];save(target/'full/input.json',inp)
    with pytest.raises(ValueError,match='inputs differ'):share_full_reference(source,target)
    assert list((target/'full').iterdir())==[target/'full/input.json']
    assert not (target/'status.json').exists()


def test_incomplete_source_stops_before_copying(references):
    source,target=references;save(source/'full/public_bundle/verify.json',dict(complete=False,verified=0))
    with pytest.raises(ValueError,match='public replay is incomplete'):share_full_reference(source,target)
    assert not (target/'status.json').exists()


def test_file_blocking_an_interrupted_directory_is_preserved(references):
    source,target=references;(target/'full/bundle').write_bytes(b'partial-directory-conflict')
    receipt=share_full_reference(source,target)
    assert any((target/path).read_bytes()==b'partial-directory-conflict' for path in receipt['archived_paths'])
    assert_shared(source,target)


def test_full_reference_with_different_parallelism_is_rejected_before_copy(references):
    source,target=references
    save(source/'protocol.json',dict(proof_runtime_workers=2))
    save(target/'protocol.json',dict(proof_runtime_workers=8))
    with pytest.raises(ValueError,match='different proof worker counts'):
        share_full_reference(source,target)
    assert not (target/'status.json').exists()
    assert not (target/'full/bundle').exists()
