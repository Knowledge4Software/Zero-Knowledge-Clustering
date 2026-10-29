"""Preserve received sources and package a separately reviewable methodology."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import argparse,hashlib,json,shutil

REPO=Path(__file__).resolve().parents[2]
DEST=REPO/'paper_results/supervisor_methodology_20261027'


def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    for archive,folder in [('Sparse_JL_for_ZK_Clustering.zip','received'),('Spectral_CCD.zip','style_reference')]:
        root=(DEST/folder).resolve();root.mkdir(parents=True,exist_ok=True)
        with ZipFile(REPO/archive) as z:
            for info in z.infolist():
                path=(root/info.filename).resolve()
                if not path.is_relative_to(root):raise ValueError('Archive path escapes intended directory')
                if info.is_dir():path.mkdir(parents=True,exist_ok=True);continue
                value=z.read(info)
                if path.exists() and path.read_bytes()!=value:raise RuntimeError('Preserved source changed: '+str(path))
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.exists():path.write_bytes(value)
    # The supplied main manuscript uses Elsevier with named authors. Preserve
    # it byte-for-byte; the README's Springer description is out of date.
    original=(DEST/'received/manuscript.tex').read_text(encoding='utf-8-sig')
    supplied=original[original.index(r'\begin{abstract}'):original.index(r'\section{Methodology}')]
    record=dict(received_archives={n:digest(REPO/n) for n in ['Sparse_JL_for_ZK_Clustering.zip','Spectral_CCD.zip']},
        supplied_abstract_introduction_related_sha256=hashlib.sha256(supplied.encode()).hexdigest(),
        sources={str(p.relative_to(DEST)).replace('\\','/'):digest(p) for folder in ['received','style_reference'] for p in sorted((DEST/folder).rglob('*')) if p.is_file()},
        methodology_scope='Current implemented prototype-update certificate; no sparse-JL transform.',
        supervisor_sections_modified=False,original_archives_modified=False)
    (DEST/'received_sources_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    # Use the current checked bibliography for the independent section preview.
    shutil.copyfile(REPO/'references.bib',DEST/'methodology_references.bib')
    print(json.dumps(dict(preserved_files=len(record['sources']),destination=str(DEST))))


def package():
    required=['methodology.tex','methodology_appendix.tex','methodology_pipeline.tex','methodology_preview.tex',
              'methodology_references.bib','README.md','received_sources_manifest.json','methodology_checks.json']
    files={name:DEST/name for name in required}
    for name in ['Sparse_JL_for_ZK_Clustering.zip','Spectral_CCD.zip']:files['received_archives/'+name]=REPO/name
    pdf=REPO/'output/pdf/prototype_update_methodology_20261027.pdf'
    if not pdf.exists():raise RuntimeError('Build and inspect the methodology preview first')
    validation=json.loads((DEST/'methodology_checks.json').read_text())
    if validation.get('visual_review')!='PASS' or validation['pdf_sha256']!=digest(pdf):raise RuntimeError('Preview needs matching visual review')
    if any(digest(DEST/name)!=value for name,value in validation['source_sha256'].items()):raise RuntimeError('Methodology sources changed after reviewed build')
    received=json.loads((DEST/'received_sources_manifest.json').read_text())
    if any(digest(REPO/name)!=value for name,value in received['received_archives'].items()):raise RuntimeError('Received archive changed')
    files['prototype_update_methodology_20261027.pdf']=pdf
    manifest={name:digest(path) for name,path in files.items()}
    path=REPO/'output/supervisor_methodology_20261027.zip'
    with ZipFile(path,'w',ZIP_DEFLATED) as z:
        for name,file in files.items():z.write(file,name)
        z.writestr('package_manifest.json',json.dumps(manifest,indent=2))
    with ZipFile(path) as z:
        assert all(hashlib.sha256(z.read(name)).hexdigest()==h for name,h in manifest.items())
    print(json.dumps(dict(package=str(path),sha256=digest(path),files=len(files))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',action='store_true');a=p.parse_args()
    package() if a.package else prepare()
