"""Check the active paper, portable evidence and primary project navigation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_flags(data, current_source, recorded_source, measured_sha, recorded_sha):
    """Retain the historical byte/provenance regression interface."""
    return {
        'source_matches': bool(data['_source_sha256'].eq(current_source).all()),
        'recorded_source_matches': bool(data['_source_sha256'].eq(recorded_source).all()),
        'recorded_bytes_match': measured_sha == recorded_sha,
    }


def main():
    problems = []
    dependencies = set()
    tex_sources = []

    def visit(path):
        if path in dependencies:
            return
        if not path.is_file():
            problems.append('Missing manuscript dependency: '+str(path.relative_to(ROOT)))
            return
        dependencies.add(path)
        if path.suffix != '.tex':
            return
        source = path.read_text(encoding='utf-8-sig')
        tex_sources.append(source)
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', source):
            if name == 'glyphtounicode':
                continue  # TeX-distribution dependency, not a project file.
            visit(ROOT/(name if Path(name).suffix else name+'.tex'))
        for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', source):
            visit(ROOT/(name if Path(name).suffix else name+'.pdf'))

    visit(ROOT/'manuscript.tex')
    bib = (ROOT/'references.bib').read_text(encoding='utf-8')
    keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib))
    cited = {x.strip() for text in tex_sources for group in re.findall(r'\\cite\w*\{([^}]+)\}', text) for x in group.split(',')}
    if cited-keys:
        problems.append('Missing bibliography keys: '+str(sorted(cited-keys)))
    joined = '\n'.join(tex_sources)
    for phrase in ['larger-catalog results remain pending', '1,024-record learned run is active',
                   'Matched 1,024-record proof measurements remain pending']:
        if phrase in joined:
            problems.append('Obsolete campaign status in active paper: '+phrase)
    expected_original = '4f06491ac2e89b747083d42f9983289b278de6835dc86d2bcac7fd2294f93a69'
    original_hash = sha256(ROOT/'paper_history/supervisor_draft.tex')
    if original_hash != expected_original:
        problems.append('Supervisor original bytes changed')

    export = ROOT/'paper_results/review_controls/generated_completed_20261020/manifest.json'
    manifest = json.loads(export.read_text())
    files_checked = completed = replayed = 0
    for relative, expected in zip(manifest['source_captures'], manifest['source_capture_sha256']):
        folder = ROOT/relative
        if sha256(folder/'capture.json') != expected:
            problems.append('Capture manifest changed: '+relative)
        capture = json.loads((folder/'capture.json').read_text())
        for record in capture['files']:
            if sha256(folder/record['path']) != record['sha256']:
                problems.append('Captured bytes changed: '+relative+'/'+record['path'])
            files_checked += 1
        for name in capture['completed_cases']:
            case = folder/name
            status = json.loads((case/'status.json').read_text())
            replay = json.loads((case/'public_bundle/verify.json').read_text())
            if not status['complete'] or not replay['complete']:
                problems.append('Incomplete captured case: '+name)
            completed += 1
            replayed += replay['verified']
    if (files_checked, completed, replayed) != (208,14,6848):
        problems.append('Unexpected campaign coverage: '+str((files_checked,completed,replayed)))
    if manifest['quality_record_set_evaluations'] != 33280 or manifest['updated_strict_policy_accepted_sets'] != 0:
        problems.append('Certificate evaluation totals differ')
    for row in manifest['completed_matched_methods']:
        source = ROOT/row['source_local'].replace('\\','/')
        if sha256(source) != row['source_sha256']:
            problems.append('Owner input bytes changed: '+row['source_local'])

    docs = [ROOT/name for name in [
        'README.md','output/README.md','output/mobin_completed_results_20261020.md',
        'code/README.md','code/analysis/README.md','code/results/README.md',
        'code/validation/README.md','code/docs/review_controls_20261020.md',
        'paper_history/README.md','paper_results/review_controls/README.md',
        'paper_results/reuse_revision/README.md','paper_versions/README.md']]
    links = 0
    for doc in docs:
        text = doc.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if '://' in target or target.startswith(('#','mailto:')):
                continue
            target = target.strip('<>').split('#',1)[0]
            if target and not (doc.parent/target).exists():
                problems.append('Broken link in '+doc.relative_to(ROOT).as_posix()+': '+target)
            links += 1
    receipt = ROOT/'code/validation/outputs/cleanup_publication_20261020.json'
    cleanup = json.loads(receipt.read_text(encoding='utf-8-sig'))
    report = {
        'checked_utc':datetime.now(timezone.utc).isoformat(),
        'status':'PASS' if not problems else 'FAIL',
        'scope':'Active manuscript dependencies/citations, primary navigation, selected captured bytes, completed worker verification receipts and owner input identity. Historical local-only campaigns are outside this check.',
        'supervisor_original_sha256':original_hash,
        'manuscript_dependencies_checked':len(dependencies),
        'bibliography_keys_cited':len(cited),
        'captured_files_hash_checked':files_checked,
        'completed_cases':completed,
        'worker_public_replayed_proofs':replayed,
        'quality_record_set_evaluations':manifest['quality_record_set_evaluations'],
        'documents_checked':len(docs),'local_links_checked':links,
        'superseded_files_archived':len(cleanup['files']),
        'problems':problems,
    }
    out = ROOT/'code/validation/outputs/layout_verification.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    raise SystemExit(bool(problems))


if __name__ == '__main__':
    main()
