"""Check a paper PDF/source release against project bytes and review evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', default='neurocomputing_completed_20261020')
    p.add_argument('--visual-review-complete', action='store_true',
                   help='Record completed assistant inspection of all final rendered pages.')
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('invalid release tag')
    from pypdf import PdfReader
    pdf = ROOT/'output/pdf'/(args.tag+'.pdf')
    archive = ROOT/'output'/(args.tag+'_sources.zip')
    receipt = ROOT/'output'/(args.tag+'_validation.json')
    status = json.loads(receipt.read_text())
    assert status['build_returncode'] == 0
    assert not status['latex_warnings_requiring_review']
    assert status['pdf_sha256'] == sha(pdf)
    matched = []
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist()) == status['source_files']
        for name in z.namelist():
            path = ROOT/name
            if path.is_file():
                assert z.read(name) == path.read_bytes(), name
                matched.append(name)
    reader = PdfReader(pdf)
    texts = [page.extract_text() for page in reader.pages]
    assert all(len(text.strip()) > 100 for text in texts), 'unexpected near-empty page'
    text = '\n'.join(texts)
    for token in ['24,681.3','6,848','99.74','85.20','5.08']:
        assert token in text, 'missing completed result: '+token
    for token in ['larger-catalog results remain pending','1,024-record learned run is active']:
        assert token not in text, 'obsolete campaign status'
    preserved = {
        '01_supervisor_original.pdf':'8b88b1d2b469c891773d3200fdae8831f444745b5ec7a67d16b811656e4a44d3',
        '02_changes_red_green.pdf':'dd1c901696f70b2f6a4b2583308f19b95f7941357c2e06fc8b83e1772c85462d',
        '03_final_revised.pdf':'b82dfa56fa363096ef01489b45e563cf96584fb452ecee780768c32a9904ca68',
    }
    for name, expected in preserved.items():
        assert sha(ROOT/'output/pdf'/name) == expected, name+' changed'
    layout = json.loads((ROOT/'code/validation/outputs/layout_verification.json').read_text())
    assert layout['status'] == 'PASS'
    cleanup = json.loads((ROOT/'code/validation/outputs/cleanup_publication_20261020.json').read_text(encoding='utf-8-sig'))
    archive_root = Path(cleanup['archive'])
    archived_verified = 0
    if archive_root.exists():
        for record in cleanup['files']:
            assert sha(archive_root/record['path']) == record['sha256'], record['path']
            archived_verified += 1
    rendered = sorted((ROOT/'tmp/pdfs'/args.tag).glob('final-page-*.png'))
    if args.visual_review_complete:
        assert len(rendered) == len(reader.pages), 'final render count differs from PDF pages'
        status['visual_review'] = f'Assistant inspected all {len(reader.pages)} final rendered pages; no clipping or overlap observed.'
    status.update(page_count=len(reader.pages),source_project_files_byte_checked=len(matched),
                  source_zip_sha256=sha(archive),preserved_earlier_pdf_hashes_match=True,
                  layout_evidence_check=layout['status'],archived_file_hashes_checked=archived_verified,
                  author_metadata_and_declarations='Pending author confirmation; no submission approval implied')
    receipt.write_text(json.dumps(status,indent=2),encoding='utf-8')
    checks = {
        'status':'PASS','release':args.tag,'pdf_sha256':sha(pdf),
        'source_zip_sha256':sha(archive),'pdf_pages':len(reader.pages),
        'source_zip_project_files_byte_checked':len(matched),
        'all_completed_result_tokens_present':True,'preserved_earlier_pdfs_unchanged':True,
        'captured_file_hashes_checked':layout['captured_files_hash_checked'],
        'completed_cases':layout['completed_cases'],
        'worker_public_replayed_proofs':layout['worker_public_replayed_proofs'],
        'independent_integer_record_set_evaluations':layout['quality_record_set_evaluations'],
        'archived_file_hashes_checked':archived_verified,
        'visual_review':status['visual_review'],
        'validation_boundary':'Numerical/capture/PDF checks are local. Full-campaign cryptographic replay counts come from the completed public-only worker receipts.',
    }
    check_name = ('publication_checks_20261020.json' if args.tag=='neurocomputing_completed_20261020'
                  else args.tag+'_checks.json')
    (ROOT/'code/validation/outputs'/check_name).write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__ == '__main__':
    main()
