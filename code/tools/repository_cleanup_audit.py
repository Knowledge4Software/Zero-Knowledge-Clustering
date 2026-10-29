"""Read-only inventory for the source-cleanup commit; no file contents printed."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit, urlunsplit

REPO = Path(__file__).resolve().parents[2]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=REPO)


def paths(*args):
    return [x.decode('utf-8') for x in git(*args).split(b'\0') if x]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--build-paper', action='store_true')
    args = parser.parse_args()
    if args.build_paper:
        return build_staged_paper()
    if args.staged:
        return staged_check()
    tracked = paths('ls-files', '-z')
    ignored_tracked = paths('ls-files', '-ci', '--exclude-standard', '-z')
    untracked = paths('ls-files', '--others', '--exclude-standard', '-z')
    status = [s.decode('utf-8') for s in git('status', '--porcelain=v1', '-z').split(b'\0') if s]
    def size(name):
        p = REPO / name
        return p.stat().st_size if p.is_file() else 0
    candidate = set(tracked + untracked)
    largest = sorted(((size(name), name) for name in candidate), reverse=True)[:25]
    try:
        remote = git('remote', 'get-url', 'origin').decode().strip()
    except subprocess.CalledProcessError:
        remote = None
    if remote and '://' in remote:
        parsed = urlsplit(remote)
        remote = urlunsplit((parsed.scheme, parsed.netloc.rsplit('@', 1)[-1], parsed.path, '', ''))
    result = dict(branch=git('branch', '--show-current').decode().strip(), origin=remote,
        upstream=git('for-each-ref', '--format=%(upstream:short)', 'refs/heads/' + git('branch', '--show-current').decode().strip()).decode().strip(),
        tracked_files=len(tracked), status_counts=dict(Counter(s[:2] for s in status)),
        tracked_ignored_count=len(ignored_tracked), tracked_ignored=ignored_tracked,
        untracked_count=len(untracked), untracked_by_root=dict(Counter(p.split('/')[0] for p in untracked)),
        untracked_files=untracked, largest_candidate_files=[dict(path=p, bytes=n) for n,p in largest],
        suspicious_path_names=[p for p in sorted(candidate) if re.search(r'(^|/)(\.env($|\.)|id_(rsa|ed25519)|credentials|secrets?)(/|\.|$)|\.(pem|pfx|p12|key)$',p,re.I)])
    out = REPO / 'tmp/repository_cleanup_20261029'; out.mkdir(parents=True, exist_ok=True)
    (out / 'inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['tracked_ignored','untracked_files']}, indent=2))


def staged_check():
    """Verify protected bytes, credential patterns and the paper dependency closure."""
    def indexed(name):
        return git('show', ':' + name)
    staged = set(paths('ls-files', '-z'))
    changed = paths('diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z')
    before = json.loads((REPO / 'tmp/pdfs/latest_baselines_20261029/protected_before.json').read_text())
    preserved = []
    for name, digest in before['protected_sha256'].items():
        assert name in staged, name
        assert hashlib.sha256(indexed(name)).hexdigest() == digest, name
        preserved.append(name)
    assert not {'Sparse_JL_for_ZK_Clustering.zip', 'Spectral_CCD.zip'} & staged
    assert not [p for p in staged if p.startswith(('tmp/', 'code/.cache/'))]
    patterns = [
        re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
        re.compile(r'\b(?:ghp_|gho_|ghu_|ghs_|github_pat_)[A-Za-z0-9_]{25,}\b'),
        re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
        re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b'),
    ]
    findings = []
    binary_suffixes = {'.pdf', '.png', '.jpg', '.jpeg', '.gob', '.proof', '.vk', '.zip', '.parquet', '.npz', '.npy'}
    for name in changed:
        if Path(name).suffix.lower() in binary_suffixes:
            continue
        content = indexed(name).decode('utf-8', errors='replace')
        for i,line in enumerate(content.splitlines(),1):
            if any(pattern.search(line) for pattern in patterns):
                findings.append(dict(path=name, line=i))
    assert not findings, findings
    # Render inputs in the active manuscript closure must survive the cleanup.
    closure = set()
    fls = REPO / 'tmp/pdfs/latest_baselines_20261029/manuscript.fls'
    for line in fls.read_text().splitlines():
        if not line.startswith('INPUT '): continue
        path = Path(line[6:].strip()); path = path if path.is_absolute() else REPO / path
        path = path.resolve()
        if path.is_relative_to(REPO) and not path.is_relative_to(REPO / 'tmp'):
            if path.suffix.lower() in {'.tex', '.bib', '.pdf', '.png', '.svg'}:
                closure.add(path.relative_to(REPO).as_posix())
    closure.add('references.bib')
    assert closure and not closure - staged, sorted(closure - staged)
    out = REPO / 'tmp/repository_cleanup_20261029'
    export = out / 'staged_paper'; export.mkdir(exist_ok=True)
    for name in closure:
        target = export / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(indexed(name))
    result = dict(complete=True, protected_files_byte_identical=len(preserved),
        staged_changed_files_scanned=len(changed), credential_pattern_findings=findings,
        root_reference_archives_ignored=True, temporary_and_dataset_cache_paths_not_tracked=True,
        manuscript_dependencies_tracked=len(closure), staged_paper_export=str(export.relative_to(REPO)),
        inventory_only_no_files_deleted=True,
        secret_scan_scope='Changed staged text files; checks specific private-key and credential token patterns, not a comprehensive security audit.')
    (out / 'staged_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


def build_staged_paper():
    from pypdf import PdfReader
    out = REPO / 'tmp/repository_cleanup_20261029'
    export = out / 'staged_paper'
    with (out / 'staged_paper_build.log').open('wb') as stream:
        result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
            '-outdir=build', 'manuscript.tex'], cwd=export, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, out / 'staged_paper_build.log'
    rebuilt = PdfReader(export / 'build/manuscript.pdf')
    reviewed = PdfReader(REPO / 'output/pdf/prototype_audit_latest_baselines_20261029.pdf')
    normalize = lambda pages: re.sub(r'\s+', '', '\n'.join(p.extract_text() for p in pages))
    assert len(rebuilt.pages) == len(reviewed.pages) == 20
    assert normalize(rebuilt.pages) == normalize(reviewed.pages)
    value = dict(complete=True, returncode=0, pages=20, extracted_text_identical_to_reviewed_pdf=True,
        scope='Built entirely from staged manuscript dependencies plus installed LaTeX packages; root source ZIPs, private data and proof caches were not used.')
    (out / 'staged_paper_build.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
