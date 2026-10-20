"""One narrowly verified migration for the v1 memory_limit classification bug."""
import argparse
import hashlib
import json
from pathlib import Path
from campaigns.laptop_comparison import CODE, atomic_json, fingerprint


def migrate(root):
    root=Path(root).resolve()
    path=root/'manifest.json'
    manifest=json.loads(path.read_text())
    old=manifest['identity']
    binary=Path(manifest['plan'][0]['binary'])
    current=fingerprint(binary)
    name=str(Path('experiments/laptop_comparison.py'))
    changes=[key for key in old['sources'] if old['sources'][key]!=current['sources'][key]]
    assert changes==[name], f'Unexpected source changes: {changes}'
    assert {k:v for k,v in old.items() if k!='sources'}=={k:v for k,v in current.items() if k!='sources'}
    for key,sha in old['sources'].items():
        assert hashlib.sha256((root/'sources'/key).read_bytes()).hexdigest()==sha, key
    previous=(root/'sources'/name).read_text()
    expected=previous.replace(
        "['timeout','rss_limit','oom','oom_budget','memory_budget']",
        "['timeout','rss_limit','memory_limit','oom','oom_budget','memory_budget']").replace(
        '# The bridge records memory budget failures as rss_budget_exceeded.',
        '# Recognize alternate RSS/budget failure names without masking solver errors.')
    assert expected==(CODE/name).read_text(), 'Migration changes numerical logic or unexpected text'
    row_path=root/'jobs/real_mnist/rows/full_complete.json'
    row=json.loads(row_path.read_text())
    assert row['failure_kind']=='memory_limit' and row['status']=='error'
    assert not row['verify_ok'] and row['successful_proofs']==0
    revision=root/'source_revisions/memory_limit_fix'
    assert not revision.exists(), 'Migration already applied'
    revision.mkdir(parents=True)
    (revision/'manifest_before.json').write_bytes(path.read_bytes())
    (revision/'full_row_before.json').write_bytes(row_path.read_bytes())
    (revision/'worker_after.py').write_bytes((CODE/name).read_bytes())
    execution=root/'jobs/real_mnist/execution.json'
    if execution.exists():
        (revision/'execution_before.json').write_bytes(execution.read_bytes())
    receipt=dict(reason='Scheduling-only correction: memory_limit is a bounded resource failure, not a solver error.',
        previous_worker_sha256=old['sources'][name],updated_worker_sha256=current['sources'][name],
        numerical_changes=False,raw_proof_record_changed=False,
        row_status_correction=dict(record=str(row_path.relative_to(root)),before='error',after='resource_limit'),
        preserved_original_sources='sources',revised_source='source_revisions/memory_limit_fix/worker_after.py')
    atomic_json(revision/'receipt.json',receipt)
    row['status']='resource_limit'
    atomic_json(row_path,row)
    manifest['identity']=current
    manifest.setdefault('source_revisions',[]).append(receipt)
    manifest.pop('stop_reason',None)
    atomic_json(path,manifest)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    migrate(parser.parse_args().out)
