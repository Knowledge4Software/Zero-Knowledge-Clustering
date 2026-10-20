"""Compile the revised manuscript and make a minimal portable source archive."""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    package = ROOT/'Sparse_JL_for_ZK_Clustering'
    build = ROOT/'tmp/pdfs/manuscript_bound_revision'
    build.mkdir(parents=True, exist_ok=True)
    # Follow the canonical manuscript, including the original input structure.
    # This avoids packaging unused rewrite sections or omitting retained figures.
    found=set()
    def dependencies(p):
        if p in found:return
        if not p.exists():raise FileNotFoundError(p)
        found.add(p)
        if p.suffix!='.tex':return
        text=p.read_text(encoding='utf-8')
        text='\n'.join(x for x in text.splitlines() if not x.lstrip().startswith('%'))
        for m in re.finditer(r'\\(?:input|include)\{([^}]+)\}',text):
            rel=Path(m[1]);dependencies(ROOT/(m[1] if rel.suffix else m[1]+'.tex'))
        for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
            rel=Path(m[1]);dependencies(ROOT/(m[1] if rel.suffix else m[1]+'.pdf'))
    dependencies(ROOT/'manuscript.tex')
    files = sorted(found|{ROOT/'references.bib'})
    files += [ROOT/'paper_results/manifest.json',ROOT/'paper_results/README.md',ROOT/'paper_results/scaling_actual_queries.csv']
    files += [p for p in (ROOT/'paper_results/bound_revision').rglob('*') if p.is_file()]
    files=list(dict.fromkeys(files))
    for src in files:
        dst = package/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    env = dict(os.environ, LC_ALL='C', LANG='C')
    command = ['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
               '-outdir='+str(build), 'manuscript.tex']
    with (build/'build_console.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=package, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print((build/'build_console.log').read_text(encoding='utf-8',errors='replace')[-8000:])
        raise SystemExit(result.returncode)
    log=(build/'manuscript.log').read_text(encoding='utf-8',errors='replace')
    bad=[line for line in log.splitlines() if any(s in line for s in
         ['Overfull','undefined references','Citation','LaTeX Error','Float too large'])]
    target = ROOT/'output/pdf/commitjl_manuscript_revised.pdf'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build/'manuscript.pdf',target)
    shutil.copy2(build/'manuscript.pdf',package/'manuscript.pdf')
    sources=ROOT/'output/commitjl_manuscript_revised_sources.zip'
    archivefiles=files+[package/'sn-jnl.cls',package/'sn-mathphys-num.bst']
    with zipfile.ZipFile(sources,'w',zipfile.ZIP_DEFLATED) as z:
        for f in archivefiles:
            name=f.name if f.parent==package else f.relative_to(ROOT).as_posix()
            z.write(f,name)
        z.writestr('README.txt','Compile manuscript.tex with pdflatex, bibtex, pdflatex, pdflatex.\n'
                   'All required sections, generated tables and PDF figures are included.\n'
                   'The author metadata/declarations require confirmation before submission.\n'
                   'CSV and manifest files document numerical provenance; raw experiments are separate.\n')
    status={'pdf':str(target.relative_to(ROOT)), 'source_zip':str(sources.relative_to(ROOT)),
            'pdf_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'latex_warnings_requiring_review':bad,
            'source_files':len(archivefiles), 'build_returncode':result.returncode,
            'visual_review':'Pending rendered-page inspection'}
    (ROOT/'output/manuscript_revision_validation.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))


if __name__=='__main__':
    main()
