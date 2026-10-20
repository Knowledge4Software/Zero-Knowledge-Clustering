"""Stage and compile the reusable-statistics manuscript in an isolated directory."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='reusable_statistics_20261020')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('tag must be a simple artifact name')
    build = ROOT/'tmp/pdfs'/args.tag
    stage = build/'source'
    stage.mkdir(parents=True, exist_ok=True)
    found = set()

    def dependencies(path):
        if path in found:
            return
        if not path.is_file():
            distribution_input = subprocess.check_output(['kpsewhich', path.name], text=True).strip()
            if distribution_input:
                return
            raise FileNotFoundError(path)
        found.add(path)
        if path.suffix != '.tex':
            return
        text = path.read_text(encoding='utf-8')
        text = '\n'.join(x for x in text.splitlines() if not x.lstrip().startswith('%'))
        for match in re.finditer(r'\\(?:input|include)\{([^}]+)\}', text):
            rel = Path(match[1])
            dependencies(ROOT/(match[1] if rel.suffix else match[1]+'.tex'))
        for match in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', text):
            rel = Path(match[1])
            dependencies(ROOT/(match[1] if rel.suffix else match[1]+'.pdf'))

    dependencies(ROOT/'manuscript.tex')
    found.add(ROOT/'references.bib')
    found.update(p for p in (ROOT/'paper_results/reuse_revision').rglob('*')
                 if p.is_file() and 'generated' not in p.relative_to(ROOT/'paper_results/reuse_revision').parts)
    found.update(p for p in (ROOT/'paper_results/review_controls').rglob('*') if p.is_file())
    for name in ['full_accuracy.csv','capped_accuracy.csv','query64.csv','preprocessing_source.csv','manifest.json']:
        found.add(ROOT/'paper_results/bound_revision'/name)
    found.add(ROOT/'paper_results/attack_controls.csv')
    for src in sorted(found):
        dst = stage/src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    class_name = 'elsarticle' if '{elsarticle}' in (ROOT/'manuscript.tex').read_text() else 'sn-jnl'
    template_names = ['elsarticle.cls','elsarticle-num.bst'] if class_name == 'elsarticle' else ['sn-jnl.cls','sn-mathphys-num.bst']
    for name in template_names:
        template = ROOT/'paper_assets'/name
        if not template.exists():
            template = Path(subprocess.check_output(['kpsewhich', name], text=True).strip())
        shutil.copy2(template, stage/name)
    (stage/'README.txt').write_text(
        'Compile manuscript.tex with latexmk -pdf, or pdflatex/bibtex/pdflatex/pdflatex.\n'
        'The manuscript uses an Elsevier elsarticle working format. Check the target journal guide before submission.\n'
        'Author information and declarations are pending; this is a research draft.\n'
        'Generated CSVs and manifests retain numerical provenance. Raw proof bundles are separate.\n', encoding='utf-8')
    command = ['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
               '-outdir='+str(build), 'manuscript.tex']
    with (build/'build_console.log').open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=stage, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print((build/'build_console.log').read_text(encoding='utf-8',errors='replace')[-6500:])
        raise SystemExit(result.returncode)
    log = (build/'manuscript.log').read_text(encoding='utf-8',errors='replace')
    bad = [line for line in log.splitlines() if any(s in line for s in
           ['Overfull','undefined references','Citation','LaTeX Error','Float too large','multiply defined'])]
    target = ROOT/'output/pdf'/(args.tag+'.pdf')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build/'manuscript.pdf', target)
    sources = ROOT/'output'/(args.tag+'_sources.zip')
    with zipfile.ZipFile(sources, 'w', zipfile.ZIP_DEFLATED) as archive:
        current = [stage/src.relative_to(ROOT) for src in sorted(found)]
        current += [stage/name for name in template_names+['README.txt']]
        for src in current:
            archive.write(src, src.relative_to(stage).as_posix())
    status = dict(pdf=str(target.relative_to(ROOT)), source_zip=str(sources.relative_to(ROOT)),
                  pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), build_returncode=result.returncode,
                  latex_warnings_requiring_review=bad, source_files=len(found)+3,
                  visual_review='Pending rendered-page inspection')
    (ROOT/'output'/(args.tag+'_validation.json')).write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))


if __name__ == '__main__':
    main()
