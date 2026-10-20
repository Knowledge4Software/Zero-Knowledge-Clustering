"""Build preserved original, editorial color comparison and clean paper sources.

The comparison is explicitly at paragraph/environment level. It uses an
editorial section map and exact normalized matches, not a claimed word diff.
"""
from __future__ import annotations

import argparse
import csv
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'paper_history/supervisor_draft.tex'
DATE = '20261020'
NAMES = ['01_supervisor_original', '02_changes_red_green', '03_final_revised']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def norm(text):
    return re.sub(r'\s+', ' ', uncomment(text)).strip()


def expand(path, dependencies):
    dependencies.add(path)
    text = path.read_text(encoding='utf-8')
    def include(match):
        name = match[1]
        child = ROOT/(name if Path(name).suffix else name+'.tex')
        if not child.exists():
            # Standard TeX distribution inputs (e.g. glyphtounicode) are
            # resolved by TeX; they are not project-owned source files.
            found=subprocess.check_output(['kpsewhich',name],text=True).strip()
            if found:
                return match[0]
            raise FileNotFoundError(child)
        return ('\n% REVISION_SOURCE_BEGIN '+child.relative_to(ROOT).as_posix()+'\n'
                +expand(child, dependencies)+'\n% REVISION_SOURCE_END\n')
    text = re.sub(r'\\(?:input|include)\{([^}]+)\}', include, text)
    for name in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', uncomment(text)):
        child = ROOT/(name if Path(name).suffix else name+'.pdf')
        if not child.exists():
            raise FileNotFoundError(child)
        dependencies.add(child)
    return text


def blocks(text):
    """Split at paragraphs/heads without crossing LaTeX environment boundaries."""
    buffer, stack, sources = [], [], ['manuscript.tex']
    source = sources[-1]
    for line in text.splitlines(keepends=True):
        if line.startswith('% REVISION_SOURCE_'):
            if buffer and not stack:
                yield source, ''.join(buffer)
                buffer=[]
            if line.startswith('% REVISION_SOURCE_BEGIN '):
                sources.append(line.strip().split(' ',2)[2])
            else:
                sources.pop()
            source=sources[-1]
            continue
        is_heading = re.match(r'\\(?:section|subsection|subsubsection|paragraph)\*?(?:\[|\{)', line)
        if is_heading and buffer and not stack:
            yield source, ''.join(buffer)
            buffer=[]
        buffer.append(line)
        for match in re.finditer(r'\\(begin|end)\{([^}]+)\}', line):
            if match[1]=='begin':
                stack.append(match[2])
            elif stack:
                expected=stack.pop()
                if expected!=match[2]:
                    raise ValueError((expected, match[2], line))
        if not stack and (not line.strip() or is_heading):
            yield source, ''.join(buffer)
            buffer=[]
    if buffer:
        yield source, ''.join(buffer)
    if stack:
        raise ValueError(stack)


def comparison(expanded, original_text):
    preamble, rest = expanded.split('\\begin{document}',1)
    front, body = rest.split('\\end{frontmatter}',1)
    body, references = body.split('\\bibliographystyle',1)
    old_body=original_text.split('\\begin{document}',1)[1].split('\\end{document}',1)[0]
    old_blocks=[b for _,b in blocks(old_body) if norm(b)]
    old_normal=norm(old_body)
    preamble += ('\n\\definecolor{RevisionRed}{HTML}{B31B1B}\n'
                 '\\definecolor{RevisionGreen}{HTML}{11723A}\n')
    # Bibliography entries already cited in the baseline remain black; entries
    # introduced by this revision are green. Do not alter citation numbering.
    original_keys=set()
    for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}', original_text):
        original_keys.update(k.strip() for k in group.split(','))
    preamble+='\n'.join('\\expandafter\\def\\csname RevisionOldKey'+key+'\\endcsname{1}' for key in sorted(original_keys))+'\n'
    preamble+=r'''
\newcommand{\RevisionReferenceColor}[1]{\ifcsname RevisionOldKey#1\endcsname\color{black}\else\color{RevisionGreen}\fi}
\makeatletter
\AtBeginDocument{%
\let\revision@bibitem\@bibitem
\def\@bibitem#1{\RevisionReferenceColor{#1}\revision@bibitem{#1}}%
\let\revision@lbibitem\@lbibitem
\def\@lbibitem[#1]#2{\RevisionReferenceColor{#2}\revision@lbibitem[#1]{#2}}%
}
\makeatother
'''
    front=re.sub(r'\\title\{([^\n]+)\}', r'\\title{\\textcolor{RevisionRed}{\1}}', front)
    front=re.sub(r'\\author\{([^\n]+)\}', r'\\author{\\textcolor{RevisionRed}{\1}}', front)
    front=front.replace('\\begin{abstract}', '\\begin{abstract}\\color{RevisionRed}')
    front=front.replace('\\begin{keyword}', '\\begin{keyword}\\color{RevisionRed}')
    legend=(r'\noindent{\small\textbf{Comparison with the preserved first draft.} '
            r'\textcolor{RevisionRed}{Red: revised passages.} '
            r'\textcolor{RevisionGreen}{Green: new passages.} Black: unchanged material. '
            r'Colors apply to paragraphs, equations and tables, not individual words. '
            r'The source package records the classification and original passages that were rewritten or removed.}'+'\n\\medskip\n')
    output, records, original_used=[],[],set()
    security_new=True
    experiment_new=False
    appendix_new=True
    title=re.compile(r'\\(?:section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}')
    for source,block in blocks(body):
        normalized=norm(block)
        if not normalized:
            output.append(block)
            continue
        heading=title.search(block)
        heading_text=heading[1] if heading else ''
        if source.endswith('security.tex') and heading:
            security_new=heading_text not in ['Threat model and two-stage binding', 'Public outputs and verification boundary', 'Commit-then-sample JL option']
        if source.endswith('experiments.tex') and heading:
            if heading_text=='Fixed-center fidelity and the cached baseline':
                experiment_new=True
        if source.endswith('appendix.tex') and heading:
            appendix_new=heading_text=='Additional fixed-center evidence'
        if normalized in old_normal:
            category='unchanged'
        elif (source.endswith('method.tex') and heading_text!='Reusable statistics and proof relations'):
            category='added'
        elif 'generated_20261020/' in source or 'bound_revision/' in source or source.endswith(('fixed_supplement.tex','attack_supplement.tex')):
            category='added'
        elif source.endswith('experiments.tex') and experiment_new:
            category='added'
        elif source.endswith('security.tex') and security_new:
            category='added'
        elif source.endswith('appendix.tex') and appendix_new:
            category='added'
        elif heading_text=='Privacy-preserving prototype models' or ('Schubert et al.' in block):
            category='added'
        elif any(x in block for x in ['We therefore study audits under changing public centers.', '\\begin{table', '\\begin{figure']):
            category='added'
        else:
            category='revised'
        ratios=[SequenceMatcher(None, norm(old).split(), normalized.split(), autojunk=False).ratio() for old in old_blocks]
        best=max(range(len(ratios)), key=ratios.__getitem__)
        if category=='unchanged':
            original_used.add(best)
        records.append(dict(block=len(records)+1, source=source, category=category,
                            nearest_original_block=best+1, lexical_similarity=round(ratios[best],4),
                            revised_text=block.strip(), nearest_original_text=old_blocks[best].strip()))
        structural = re.match(r'^\\(?:FloatBarrier|onecolumn|appendix|twocolumn|numberwithin|renewcommand|Needspace)\b', normalized)
        if category=='unchanged' or structural:
            output.append(block)
        else:
            color='RevisionRed' if category=='revised' else 'RevisionGreen'
            if heading:
                # Keep the section command's no-break behavior intact. Ending
                # an outer color group after a heading can orphan it at a foot.
                block=re.sub(r'(\\(?:section|subsection|subsubsection|paragraph)\*?\{)([^}]+)(\})',
                             lambda m:m[1]+'\\textcolor{'+color+'}{'+m[2]+'}'+m[3],block)
                output.append(block)
                continue
            # Make float contents explicitly colored; TeX can reset color while
            # moving a float out of the surrounding paragraph group.
            block=re.sub(r'(\\begin\{(?:table|figure)\*?\}(?:\[[^\]]*\])?)', r'\1\\color{'+color+'}', block)
            output.append('\\begingroup\\color{'+color+'}\n'+block+'\n\\par\\endgroup\n')
    changed=preamble+'\\begin{document}'+front+'\\end{frontmatter}\n'+legend+''.join(output)+'\\bibliographystyle'+references
    return changed, records, [b for i,b in enumerate(old_blocks) if i not in original_used]


def copy_asset(path, dest):
    target=dest/path.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path,target)


def template(name, dest):
    path=ROOT/'paper_assets'/name
    if not path.exists():
        found=subprocess.check_output(['kpsewhich',name],text=True).strip()
        if not found:
            raise FileNotFoundError(name)
        path=Path(found)
    shutil.copy2(path,dest/name)


def build_one(name, source, build_root, pdf_root):
    build=build_root/name
    build.mkdir(parents=True, exist_ok=True)
    env=dict(os.environ, LC_ALL='C', LANG='C')
    with (build/'console.log').open('w',encoding='utf-8') as stream:
        result=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
                               '-outdir='+str(build),'manuscript.tex'],cwd=source,env=env,
                              stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        print((build/'console.log').read_text(encoding='utf-8',errors='replace')[-7000:])
        raise RuntimeError('Build failed: '+name)
    log=(build/'manuscript.log').read_text(encoding='utf-8',errors='replace')
    warnings=[line for line in log.splitlines() if any(x in line for x in
              ['Overfull','undefined references','undefined citations','Citation `','Float too large','multiply defined','LaTeX Error'])]
    dest=pdf_root/(name+'.pdf')
    shutil.copy2(build/'manuscript.pdf',dest)
    return dict(name=name,pdf=dest.relative_to(ROOT).as_posix(),sha256=sha(dest),
                warnings=warnings,source=source.relative_to(ROOT).as_posix(),visual_review='pending')


def package_release(release):
    archive_path=ROOT/'output/three_paper_versions_20261020_sources.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(release.rglob('*')):
            if path.is_file():
                archive.write(path,path.relative_to(release).as_posix())
        path=ROOT/'output/Neurocomputing_readiness_20261020.md'
        archive.write(path,path.name)
    print(json.dumps(dict(source_zip=str(archive_path),sha256=sha(archive_path)),indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--only', choices=['original','changes','final','all'],default='all')
    ap.add_argument('--package',action='store_true')
    ap.add_argument('--package-only',action='store_true',help='Package reviewed sources without regenerating or compiling PDFs')
    args=ap.parse_args()
    release=ROOT/'paper_versions'/DATE
    release.mkdir(parents=True,exist_ok=True)
    if args.package_only:
        package_release(release)
        return
    for name in ['checks.json','fidelity_validation.json']:
        shutil.copy2(ROOT/'code/validation/outputs/completed_reuse_20261020'/name,
                     release/('evidence_'+name))
    shutil.copy2(ROOT/'paper_results/reuse_revision/source_provenance.json',
                 release/'earlier_native_source_provenance.json')
    pdf_root=ROOT/'output/pdf'
    pdf_root.mkdir(parents=True,exist_ok=True)
    build_root=ROOT/'tmp/pdfs'/('three_papers_'+DATE)
    original_hash=sha(ORIGINAL)
    dependencies=set()
    expanded=expand(ROOT/'manuscript.tex',dependencies)
    old_text=ORIGINAL.read_text(encoding='utf-8')
    changed,records,unmatched=comparison(expanded,old_text)
    all_sources=[release/name for name in NAMES]
    for source in all_sources:
        source.mkdir(parents=True,exist_ok=True)
    orig,marked,final=all_sources
    shutil.copy2(ORIGINAL,orig/'manuscript.tex')
    old_assets=set()
    expand(ORIGINAL,old_assets)
    for path in old_assets-{ORIGINAL}:
        copy_asset(path,orig)
    shutil.copy2(ROOT/'paper_history/before_reuse_revision_20261019/references.bib',orig/'references.bib')
    for name in ['sn-jnl.cls','sn-mathphys-num.bst']:
        template(name,orig)
    for source in [marked,final]:
        for path in dependencies-{ROOT/'manuscript.tex'}:
            copy_asset(path,source)
        shutil.copy2(ROOT/'references.bib',source/'references.bib')
        for name in ['elsarticle.cls','elsarticle-num.bst']:
            template(name,source)
        for folder in ['paper_results/reuse_revision/generated_20261020']:
            for path in (ROOT/folder).iterdir():
                if path.is_file():
                    copy_asset(path,source)
        for name in ['full_accuracy.csv','capped_accuracy.csv','query64.csv','preprocessing_source.csv','manifest.json']:
            copy_asset(ROOT/'paper_results/bound_revision'/name,source)
        copy_asset(ROOT/'paper_results/attack_controls.csv',source)
    shutil.copy2(ROOT/'manuscript.tex',final/'manuscript.tex')
    (marked/'manuscript.tex').write_text(changed,encoding='utf-8')
    with (release/'change_map.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (release/'ORIGINAL_PASSAGES_REWRITTEN_OR_REMOVED.txt').write_text(
        'Original blocks not retained verbatim. Some are rewritten, others removed.\n'
        'This is a source-text audit, not an assertion that every block was deleted.\n\n'
        +'\n\n--------------------\n\n'.join(unmatched),encoding='utf-8')
    for source,name in zip(all_sources,NAMES):
        (source/'README.txt').write_text(
            name+'\nCompile manuscript.tex with latexmk -pdf (TeX Live).\n'
            'The original preserves historical wording, colors and placeholders.\n'
            'The comparison uses red for rewritten blocks, green for new blocks and black for unchanged material.\n'
            'This is an editorial paragraph/environment comparison, not a word-level tracked deletion view.\n'
            'The final is the clean AI-assisted research revision, pending human author approval.\n'
            'Author declarations, release identifiers and independent reuse-bundle replay remain pre-submission tasks.\n',encoding='utf-8')
    assert original_hash==sha(ORIGINAL)==sha(orig/'manuscript.tex')
    statuses=[]
    for key,name,source in zip(['original','changes','final'],NAMES,all_sources):
        if args.only not in ('all',key):
            continue
        status=build_one(name,source,build_root,pdf_root)
        statuses.append(status)
        print(json.dumps(status,indent=2),flush=True)
    receipt=release/'validation.json'
    previous=json.loads(receipt.read_text()) if receipt.exists() else {}
    prior={x['name']:x for x in previous.get('builds',[])}
    prior.update({x['name']:x for x in statuses})
    counts={c:sum(x['category']==c for x in records) for c in ['unchanged','revised','added']}
    validation=dict(date='2026-10-20',original_source_sha256=original_hash,
                    original_source_unchanged=True,comparison_level='paragraph/environment; editorial section mapping',
                    comparison_counts=counts,builds=list(prior.values()))
    receipt.write_text(json.dumps(validation,indent=2),encoding='utf-8')
    if args.package:
        package_release(release)


if __name__=='__main__':
    main()
