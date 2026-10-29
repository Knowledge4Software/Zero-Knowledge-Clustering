"""Render every paper page and prepare contact sheets for visual review."""
from pathlib import Path
import json
import re
import subprocess

from PIL import Image, ImageDraw, ImageOps
from pypdf import PdfReader

from analysis.update_latest_paper_20261029 import BUILD, REPO, read, sha, write


def main():
    pdf = BUILD / 'manuscript.pdf'; reader = PdfReader(pdf)
    folder = BUILD / 'renders'; folder.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['pdftoppm', '-r', '105', '-png', str(pdf), str(folder / 'page')],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    images = sorted(folder.glob('page-*.png'))
    assert len(images) == len(reader.pages)
    contacts = []
    for start in range(0, len(images), 4):
        group = images[start:start+4]
        sheet = Image.new('RGB', (1500, 2040), '#d9dee5')
        draw = ImageDraw.Draw(sheet)
        for j, path in enumerate(group):
            with Image.open(path) as source:
                preview = ImageOps.contain(source.convert('RGB'), (730, 970))
            left = 10 + (j % 2)*750; top = 35 + (j // 2)*1020
            draw.text((left, top-22), f'Page {start+j+1}', fill='black')
            sheet.paste(preview, (left, top))
        target = folder / f'contact_{start+1:02}_{start+len(group):02}.png'
        sheet.save(target); contacts.append(target.relative_to(REPO).as_posix())
    texts = [page.extract_text() for page in reader.pages]
    all_text = '\n'.join(texts)
    for expected in ['97.62', '2.124', '4.730', '2.068', '6.690', '3.957', '11.046', '2.290', 'RaBitQ']:
        assert expected in all_text, expected
    log = (BUILD / 'manuscript.log').read_text(encoding='utf-8', errors='replace')
    problems = re.findall(r'Overfull .*|LaTeX Error.*|.*undefined.*|.*multiply defined.*|Float too large.*', log)
    assert not problems, problems
    write(BUILD / 'page_text.json', dict(pages=[dict(page=i+1,text=t) for i,t in enumerate(texts)]))
    write(BUILD / 'render_checks.json', dict(complete=True, pages=len(texts), pdf_sha256=sha(pdf),
        rendered_pages=[p.relative_to(REPO).as_posix() for p in images], contact_sheets=contacts,
        expected_result_strings_present=True, latex_errors_or_overflows=problems,
        visual_review_pending=True))
    print(json.dumps(dict(pages=len(texts), contact_sheets=contacts,
        result_pages=[i+1 for i,t in enumerate(texts) if any(s in t for s in ['2.124','97.62','11.046','Native Runtime of the Later'])])))


if __name__ == '__main__':
    main()
