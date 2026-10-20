"""Check source preservation and that comparison markup changes no paper text."""
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import pymupdf

ROOT=Path(__file__).resolve().parents[2]
RELEASE=ROOT/'paper_versions/20261020'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_editorial_markup(text):
    text=text.split('\\begin{document}',1)[1]
    legend='\\noindent{\\small\\textbf{Comparison with the preserved first draft.}'
    if legend in text:
        start=text.index(legend)
        end=text.index('\\medskip',start)+len('\\medskip')
        text=text[:start]+text[end:]
    pattern=re.compile(r'\\textcolor\{Revision(?:Red|Green)\}\{')
    while match:=pattern.search(text):
        start=match.end();depth=1;end=start
        while depth:
            if text[end]=='{' and (end==0 or text[end-1]!='\\'):depth+=1
            if text[end]=='}' and (end==0 or text[end-1]!='\\'):depth-=1
            end+=1
        text=text[:match.start()]+text[start:end-1]+text[end:]
    text=re.sub(r'\\color\{Revision(?:Red|Green)\}', '',text)
    text=re.sub(r'\\(?:begingroup|endgroup|par)\b','',text)
    text=re.sub(r'(?<!\\)%[^\n]*','',text)
    return re.sub(r'\s+','',text)


def main():
    spec=importlib.util.spec_from_file_location('paper_builder',ROOT/'code/analysis/build_three_papers.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    expanded=builder.expand(ROOT/'manuscript.tex',set())
    marked=(RELEASE/'02_changes_red_green/manuscript.tex').read_text(encoding='utf-8')
    clean=without_editorial_markup(expanded)
    comparison=without_editorial_markup(marked)
    if clean!=comparison:
        at=next((i for i,(a,b) in enumerate(zip(clean,comparison)) if a!=b),min(len(clean),len(comparison)))
        raise AssertionError(dict(offset=at,clean=clean[max(0,at-100):at+150],marked=comparison[max(0,at-100):at+150]))
    original=ROOT/'paper_history/supervisor_draft.tex'
    assert sha(original)==sha(RELEASE/'01_supervisor_original/manuscript.tex')
    assert sha(ROOT/'manuscript.tex')==sha(RELEASE/'03_final_revised/manuscript.tex')
    validation=json.loads((RELEASE/'validation.json').read_text())
    assert sha(original)==validation['original_source_sha256']
    pdfs=[]
    for row in validation['builds']:
        path=ROOT/row['pdf'];assert sha(path)==row['sha256']
        doc=pymupdf.open(path)
        colors=set();outside=[]
        for i,page in enumerate(doc):
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line.get('spans',[]):
                        colors.add(span['color'])
                        x0,y0,x1,y1=span['bbox']
                        if x0<0 or y0<0 or x1>page.rect.width+.5 or y1>page.rect.height+.5:
                            outside.append(dict(page=i+1,text=span['text']))
        if row['name']!='01_supervisor_original':
            assert not row['warnings'],row['warnings']
            assert not outside,outside
            text='\n'.join(page.get_text() for page in doc)
            assert '\ufffd' not in text and '??' not in text
            assert all(len(page.get_text())>400 for page in doc)
            for value in ['1,267.6','8,642.8','98.31','0.178','6.82','99.987']:
                assert value in text,value
        if row['name']=='02_changes_red_green':
            assert {0xb31b1b,0x11723a,0}.issubset(colors),colors
        pdfs.append(dict(name=row['name'],pages=len(doc),sha256=sha(path),outside_page_text=outside))
    result=dict(original_source_unchanged=True,clean_frozen_source_matches_working=True,
                comparison_has_identical_scientific_text=True,
                comparison_markup='Editorial paragraph/environment colors; not word-level tracked deletions',
                current_pdfs_have_no_unresolved_references_or_latex_overflow=True,
                original_baseline_warnings_retained=True,pdfs=pdfs)
    (RELEASE/'artifact_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
