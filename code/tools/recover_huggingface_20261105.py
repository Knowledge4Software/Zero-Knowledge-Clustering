"""Read-only recovery from the user's public evidence snapshot, pinned by commit."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'output/server_run_20261101/huggingface_recovery_20261105'
REPO = 'Parsagh1383/zk-audit-results'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    info = json.loads((DEST / 'dataset_info.json').read_text())
    revision = info['sha']
    paths = {x['rfilename'] for x in info['siblings']}
    tree = {x['path']: x for x in json.loads((DEST / 'archive_tree.json').read_text())}
    wanted = {f'extended_revision_20261103/{name}' for name in
              ['coverage.json', 'matched_queue.json', 'finished.json', 'sealed.json', 'plan.json', 'plan_pin.json']}
    wanted |= {p for p in paths if p.startswith('extended_revision_20261103/') and
               ('/exclusions/' in p or p.endswith('/events.json') or p.endswith('/control_repeats.json'))}
    wanted |= {p for p in paths if p.startswith('extended_revision_20261103/cases/') and
               p.count('/') == 3 and p.endswith('/result.json')}
    for cohort in ['review_completion_20261103', 'extended_revision_20261103']:
        for d, seed in [(d, s) for s in [907, 911, 919] for d in ['fashion', 'har', 'mnist', 'text']]:
            if (cohort == 'review_completion_20261103') != (seed == 919 and d in ['fashion', 'har']):
                continue
            for phase in ['prepare', 'keys', 'rows', 'query', 'replay']:
                name = f'{cohort}/resources/{d}_seed{seed}_n128_q8_{phase}.attempt_01.json'
                if name in paths:
                    wanted.add(name)
                else:
                    # These ten earlier receipts remain in the already downloaded,
                    # SHA-pinned review_completion_complete_evidence.tar.gz.
                    assert cohort == 'review_completion_20261103', name
    for case in ['mnist_seed907_n128_q8_controls', 'fashion_seed929_n128_q4_checkpoint0',
                 'fashion_seed929_n128_q4_checkpoint1']:
        for suffix in ['.sha256.json', '.tar.gz']:
            wanted.add('extended_revision_20261103/archives/' + case + suffix)
    assert wanted <= paths, sorted(wanted - paths)

    def fetch(path):
        out = DEST / 'raw' / path
        assert out.resolve().is_relative_to(DEST.resolve())
        out.parent.mkdir(parents=True, exist_ok=True)
        metadata = tree.get(path, {})
        expected = metadata.get('lfs', {}).get('oid')
        if out.exists() and (expected is None or sha(out) == expected):
            return dict(path=path, bytes=out.stat().st_size, sha256=sha(out), expected_lfs_sha256=expected)
        temporary = out.with_suffix(out.suffix + '.part')
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{path}'
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as target:
            while chunk := response.read(2**20):
                target.write(chunk)
        actual = sha(temporary)
        if expected:
            assert actual == expected, path
        if metadata:
            assert temporary.stat().st_size == metadata['size'], path
        temporary.replace(out)
        return dict(path=path, bytes=out.stat().st_size, sha256=actual, expected_lfs_sha256=expected)

    completed, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, path): path for path in sorted(wanted)}
        for future in concurrent.futures.as_completed(futures):
            path = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failures.append(dict(path=path, error=str(error)))
    receipt = dict(repository=REPO, revision=revision, recovered_unix=time.time(),
                   files=completed, failures=failures, upload_performed=False)
    (DEST / 'recovery_receipt.json').write_text(json.dumps(receipt, indent=2))
    assert not failures, failures
    for entry in completed:
        if not entry['path'].endswith('.tar.gz'):
            continue
        archive = DEST / 'raw' / entry['path']
        pin = json.loads(archive.with_name(archive.name[:-7] + '.sha256.json').read_text())
        assert pin['sha256'] == sha(archive) and pin['bytes'] == archive.stat().st_size
    status = json.loads((DEST / 'raw/extended_revision_20261103/coverage.json').read_text())
    finished = json.loads((DEST / 'raw/extended_revision_20261103/finished.json').read_text())
    assert finished['queue_complete'] and not status['failures']
    print(json.dumps(dict(revision=revision, files=len(completed), queue_complete=True,
                          failures=status['failures'], exclusions=status['exclusions']), indent=2))


if __name__ == '__main__':
    main()
