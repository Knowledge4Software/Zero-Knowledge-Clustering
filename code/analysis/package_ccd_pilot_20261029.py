"""Validate and package the reviewed paper; never launch proof experiments."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper_results/ccd_pilot_20261029'
BUILD = REPO / 'tmp/pdfs/ccd_pilot_20261029'
TAG = 'prototype_audit_ccd_pilot_20261029'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    preservation = read(REPO / 'paper_history/ccd_pilot_before_20261029/preservation.json')
    prefix = bytes.fromhex(preservation['original_root_prefix_hex'])
    assert (REPO / 'manuscript.tex').read_bytes().startswith(prefix)
    for name, digest in preservation['protected_sha256'].items():
        assert sha(REPO / name) == digest, name
    for name in ['methodology.tex', 'methodology_appendix.tex']:
        assert (PAPER / name).read_bytes() == (
            REPO / 'paper_results/supervisor_methodology_20261027' / name).read_bytes()

    evidence = read(PAPER / 'evidence_manifest.json')
    figures = read(PAPER / 'figures/manifest.json')
    for manifest in [evidence, figures]:
        for name, digest in manifest['sources_sha256'].items():
            assert sha(REPO / name) == digest, name
    svg_checks = []
    for item in figures['figures']:
        svg = PAPER / 'figures' / item['svg']
        assert sha(svg) == item['svg_sha256']
        assert sha(PAPER / 'figures' / item['png']) == item['png_sha256']
        root = ET.parse(svg).getroot()
        nodes = list(root.iter())
        assert not any(n.tag.endswith('}image') for n in nodes), svg
        assert any(n.tag.endswith('}text') for n in nodes), svg
        assert len(PdfReader(svg.with_suffix('.pdf')).pages) == 1
        assert item['labels_within_canvas'] and not item['embedded_raster']
        svg_checks.append(svg.name)

    receipt = read(REPO / 'paper_results/referee_final_20261025/generated/optimized_timing_receipt.json')
    for row in receipt['records']:
        assert sha(REPO / row['receipt']) == row['receipt_sha256'], row['receipt']

    pdf = BUILD / 'manuscript.pdf'
    pages = len(PdfReader(pdf).pages)
    assert pages == 18
    visual = read(BUILD / 'visual_review.json')
    assert visual['pdf_sha256'] == sha(pdf) and visual['passed']
    assert visual['pages_inspected'] == list(range(1, pages + 1))
    log = (BUILD / 'manuscript.log').read_text(encoding='utf-8', errors='replace')
    errors = re.findall(r'Overfull .*|LaTeX Error.*|.*undefined.*|.*multiply defined.*|Float too large.*', log)
    assert not errors, errors
    for tex in PAPER.rglob('*.tex'):
        assert not any(b < 32 and b not in [9, 10, 13] for b in tex.read_bytes()), tex

    output = REPO / 'output'
    (output / 'pdf').mkdir(exist_ok=True)
    final_pdf = output / 'pdf' / (TAG + '.pdf')
    shutil.copy2(pdf, final_pdf)

    # Collect exactly the local TeX/figure dependencies recorded by the build.
    files = {}
    def add(path, name=None):
        path = path.resolve()
        assert path.is_file(), path
        name = name or path.relative_to(REPO).as_posix()
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        files[name] = path
    for line in (BUILD / 'manuscript.fls').read_text(encoding='utf-8').splitlines():
        if not line.startswith('INPUT '):
            continue
        path = Path(line[6:].strip())
        path = path if path.is_absolute() else REPO / path
        path = path.resolve()
        if path.is_relative_to(REPO) and not path.is_relative_to(REPO / 'tmp'):
            if path.suffix.lower() in ['.tex', '.bib', '.pdf', '.png', '.svg']:
                add(path)
    add(REPO / 'references.bib')
    add(BUILD / 'manuscript.bbl', 'manuscript.bbl')
    for path in PAPER.rglob('*'):
        if path.is_file():
            add(path)
    for name in evidence['sources_sha256']:
        add(REPO / name)
    for name in figures['sources_sha256']:
        add(REPO / name)
    for row in receipt['records']:
        add(REPO / row['receipt'])
    for name in [
        'code/analysis/prepare_ccd_pilot_20261029.py',
        'code/analysis/package_ccd_pilot_20261029.py',
        'code/figures/ccd_pilot_methodology_20261029.py',
        'code/figures/server_study_assets_20261029.py',
        'paper_history/ccd_pilot_before_20261029/preservation.json',
    ]:
        add(REPO / name)
    for name in ['elsarticle.cls', 'elsarticle-num.bst']:
        located = subprocess.check_output(['kpsewhich', name], text=True).strip()
        add(Path(located), name)
    readme = '''# Preliminary CCD-structured paper, 29 October 2026

Open manuscript.tex as the main document. Build with a TeX Live/MiKTeX
installation containing elsarticle, amsmath, amssymb, amsthm, booktabs,
graphicx, enumitem, url, xcolor, textcomp, tabularx, array, adjustbox,
placeins, dblfloatfix, TikZ and hyperref:

    latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex

The class, bibliography style, compiled bibliography, references and all
local LaTeX/figure inputs are included. LaTeX uses vector PDF figures;
the editable SVGs and PNG previews are in paper_results/ccd_pilot_20261029/figures/.

Six new tables are generated from the included completed numerical exports.
For regeneration use Python 3 with matplotlib, Pillow, numpy and pypdf,
then run the evidence and figure scripts listed in the paper README.
Do not rerun the one-time prepare integration against this edited master.
The packaging script's preservation checks require the original full private
workspace; it is not needed for ordinary paper compilation.

This package builds a manuscript. It does not include the complete private
proof archive, training data, model weights, proving keys or proof-campaign
software. Some evidence reports refer to artifacts in the main private
repository. It is not a standalone reproduction of cryptographic experiments.

Results are completed local pilots, including unfavorable cases. Proposed
server measurements remain future work. The completed methodology and
earlier abstract/introduction are preserved; later sections are a review
draft following the supplied CCD section structure. Author metadata and final
submission statements still need author review. No new experiments were run.
'''
    bundle_manifest = {
        'files_sha256': {name: sha(path) for name, path in sorted(files.items())},
        'main': 'manuscript.tex', 'new_proofs_run': False,
        'status': 'preliminary local-results paper; server evaluation prospective',
    }
    source_zip = output / (TAG + '_sources.zip')
    with zipfile.ZipFile(source_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('README_BUILD.md', readme)
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(bundle_manifest, indent=2))
    with zipfile.ZipFile(source_zip) as archive:
        assert archive.testzip() is None
        for name, digest in bundle_manifest['files_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name

    svg_zip = output / 'methodology_svg_20261029.zip'
    with zipfile.ZipFile(svg_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['methodology_four_stages', 'methodology_interval_refinement']:
            for suffix in ['.svg', '.pdf', '.png']:
                archive.write(PAPER / 'figures' / (name + suffix), name + suffix)
        archive.writestr('README.txt', 'Editable vector methodology figures with matching PDFs and PNG previews.\nThe interval-refinement endpoints are illustrative, not experimental results.\n')

    validation = {
        'complete': True, 'pdf': final_pdf.relative_to(REPO).as_posix(),
        'pages': pages, 'pdf_sha256': sha(final_pdf),
        'source_zip': source_zip.relative_to(REPO).as_posix(),
        'source_zip_sha256': sha(source_zip), 'source_files': len(files),
        'methodology_svg_zip_sha256': sha(svg_zip),
        'protected_files_unchanged': len(preservation['protected_sha256']),
        'early_master_prefix_unchanged': True,
        'completed_methodology_and_appendix_byte_identical': True,
        'original_master_backup_sha256': preservation['original_root_sha256'],
        'evidence_sources_unchanged': len(evidence['sources_sha256']),
        'archived_comparison_receipts_verified': len(receipt['records']),
        'svg_figures_validated': svg_checks,
        'visual_review': visual, 'latex_errors_or_overflows': errors,
        'new_proofs_run': False, 'server_experiments_run': False,
        'results_status': 'preliminary completed local evidence',
        'submission_ready': False,
    }
    path = output / (TAG + '_validation.json')
    path.write_text(json.dumps(validation, indent=2), encoding='utf-8')
    print(json.dumps({'complete': True, 'pages': pages, 'source_files': len(files),
                      'source_zip_bytes': source_zip.stat().st_size,
                      'receipts_verified': len(receipt['records'])}))


if __name__ == '__main__':
    main()
