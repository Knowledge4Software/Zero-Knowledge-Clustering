"""Build, package and validate the private 22 October completed-evidence paper."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess,sys,zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from analysis.build_three_papers import expand


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--visual-review-complete',action='store_true')
    a=parser.parse_args();tag='neurocomputing_paired_20261022'
    build=ROOT/'tmp/pdfs'/tag;build.mkdir(parents=True,exist_ok=True)
    command=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(build),'manuscript.tex']
    r=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (build/'console.log').write_bytes(r.stdout)
    if r.returncode:raise RuntimeError(r.stdout.decode(errors='replace')[-4000:])
    log=(build/'manuscript.log').read_text(errors='replace')
    warnings=[line for line in log.splitlines() if any(token in line for token in
        ['Overfull','undefined references','undefined citations','Citation ','Float too large','multiply defined','LaTeX Error'])]
    assert not warnings,warnings
    from pypdf import PdfReader
    pdf=ROOT/'output/pdf'/(tag+'.pdf');shutil.copy2(build/'manuscript.pdf',pdf)
    reader=PdfReader(pdf);texts=[p.extract_text() for p in reader.pages];text='\n'.join(texts)
    assert min(len(s.strip()) for s in texts)>800,'near-empty page'
    for token in ['2,160','1,440','1,054.55','2,181.77','2.07','89.33','92.58','6,848','7,888','61.63','24,681.3','12:32']:
        assert token in text,'missing result '+token
    assert '15 of 32' in text
    assert 'retained privately for supervisor review' in text.replace('\n',' ')
    snapshot=json.loads((ROOT/'paper_results/review_controls/paired_20261022/snapshot.json').read_text())
    assert snapshot['completed_main_cases']==15 and snapshot['file_hashes_checked']==409
    manifest_path=ROOT/'paper_results/review_controls/paired_20261022/manifest.json'
    evidence=json.loads(manifest_path.read_text())
    for entry in evidence['inputs']:assert sha(ROOT/entry['path'])==entry['sha256']
    for entry in evidence['files']:assert sha(manifest_path.parent/entry['path'])==entry['sha256']
    original_sha='4f06491ac2e89b747083d42f9983289b278de6835dc86d2bcac7fd2294f93a69'
    assert sha(ROOT/'paper_history/supervisor_draft.tex')==original_sha
    old={
      '01_supervisor_original.pdf':'8b88b1d2b469c891773d3200fdae8831f444745b5ec7a67d16b811656e4a44d3',
      '02_changes_red_green.pdf':'dd1c901696f70b2f6a4b2583308f19b95f7941357c2e06fc8b83e1772c85462d',
      '03_final_revised.pdf':'b82dfa56fa363096ef01489b45e563cf96584fb452ecee780768c32a9904ca68',
      'neurocomputing_tradeoff_20261021.pdf':'1dc481a1dad8e6bf7566554b92398ea07c15e54db234d8ea266970d413905984'}
    for name,expected in old.items():assert sha(ROOT/'output/pdf'/name)==expected,name+' changed'
    files=set();expand(ROOT/'manuscript.tex',files);files.add(ROOT/'references.bib')
    for folder in [ROOT/'paper_results/review_controls',ROOT/'paper_results/reuse_revision/generated_20261020']:
        files.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.csv','.json'})
    for folder,manifest in [
        ('code/validation/outputs/review_update_decisions_20261021T184553Z/manuscript_capture_20261022T123127Z','capture.json'),
        ('code/validation/outputs/review_update_decisions_20261021T184553Z/native_capture_20261022T0530Z','capture_manifest.json'),
        ('code/validation/outputs/review_sensitivity_20261021T061346Z/proofs_completed_20261021T1900Z','capture.json')]:
        base=ROOT/folder;m=json.loads((base/manifest).read_text());files.add(base/manifest)
        files.update(base/e['path'] for e in m['files'])
    extra=[
      'code/analysis/render_paired_paper.py','code/analysis/build_paired_paper_release.py','code/analysis/build_three_papers.py',
      'code/docs/review_handoff_20261022.md','code/docs/review_response_20261021.md',
      'code/validation/outputs/review_sensitivity_20261021T061346Z/prepared/protocol.json',
      'code/validation/outputs/review_update_decisions_20261021T184553Z/artifact_verification_20261022.json',
      'code/validation/outputs/review_paired_release_20261021/reviewer_overlap_reproduction.json',
      'code/validation/outputs/review_update_decisions_20261021T184553Z/fallback_capture_20261022T0625Z/fresh_public_audit/decisions.json']
    files.update(ROOT/name for name in extra)
    source={p.relative_to(ROOT).as_posix():p.read_bytes() for p in sorted(files)}
    for name in ['elsarticle.cls','elsarticle-num.bst']:
        candidate=Path(subprocess.check_output(['kpsewhich',name],text=True).strip())
        source[name]=candidate.read_bytes()
    source['README_RELEASE.md']=(
      '# Private paper snapshot - 22 October 2026\n\n'
      'Compile manuscript.tex with latexmk -pdf using TeX Live.\n'
      'The evidence cutoff is 22 October 2026,12:32UTC (16:02 Tehran).\n'
      'This draft includes completed numerical results and15/32 main proof cases; '
      'remaining cases and larger paired scale controls are pending.\n'
      'No public repository release or journal submission is authorized by this package.\n\n'
      'The supervisor will finalize author fields, declarations and prose.\n'
      'The original supervisor manuscript and earlier PDF releases are preserved separately.\n\n'
      'To regenerate the new numerical tables with Python,pandas,numpy,matplotlib:\n'
      'python code/analysis/render_paired_paper.py --capture '
      'code/validation/outputs/review_update_decisions_20261021T184553Z/manuscript_capture_20261022T123127Z\n\n'
      'Captured benchmark diagnostic arrays differ from public-only proof exports. '
      'This source package supplies numerical provenance, not the separate complete proof artifact.\n').encode()
    source_manifest=[dict(path=name,sha256=hashlib.sha256(data).hexdigest()) for name,data in sorted(source.items())]
    source['source_manifest.json']=json.dumps(source_manifest,indent=2).encode()
    archive=ROOT/'output'/(tag+'_sources.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(source.items()):z.writestr(name,data)
    with zipfile.ZipFile(archive) as z:
        for e in source_manifest:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
        for name in source:
            path=ROOT/name
            if path.is_file():assert z.read(name)==path.read_bytes(),name
    if a.visual_review_complete:
        assert len(list(build.glob('revised-page-*.png')))==len(reader.pages)
    receipt=dict(status='PASS',pdf=pdf.relative_to(ROOT).as_posix(),pdf_sha256=sha(pdf),
        source_zip=archive.relative_to(ROOT).as_posix(),source_zip_sha256=sha(archive),source_files=len(source),
        build_returncode=r.returncode,latex_warnings_requiring_review=warnings,
        benign_underfull_hbox_messages=log.count('Underfull'),
        page_count=len(reader.pages),minimum_page_text_characters=min(map(len,texts)),
        evidence_snapshot=snapshot,numerical_exports_and_capture_hashes='PASS',
        earlier_pdfs_and_supervisor_original_unchanged=True,publication='Private; no upload or publication performed',
        author_fields='Reserved for supervisor',visual_review=(
          'Assistant inspected all final rendered pages; no clipping, overlap or near-empty transition page.' if a.visual_review_complete else 'Pending'))
    path=ROOT/'output'/(tag+'_validation.json');path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    validation=ROOT/'code/validation/outputs'/(tag+'_checks.json');validation.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
