"""Evidence-only manuscript refresh; preserve the selected method and early prose."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

from pypdf import PdfReader

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper_results/ccd_pilot_20261029'
BUILD = REPO / 'tmp/pdfs/latest_baselines_20261029'
TAG = 'prototype_audit_latest_baselines_20261029'
PIN = BUILD / 'protected_before.json'
GATE = REPO / 'code/validation/outputs/published_baseline_gate_20261029'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, indent=2)
    path.write_text(text, encoding='utf-8', newline='\n')


def capture():
    assert not PIN.exists(), 'Preservation baseline already captured'
    protected = [REPO / 'manuscript.tex',
        REPO / 'paper_results/referee_final_20261025/abstract.tex',
        REPO / 'paper_results/referee_final_20261025/introduction.tex']
    protected += [PAPER / name for name in ['methodology.tex', 'methodology_appendix.tex', 'methodology_figures.tex']]
    protected += [p for p in (PAPER / 'figures').glob('methodology_*') if p.is_file()]
    protected += [REPO / 'code/commitjl' / name for name in ['group_moment_reference.py',
        'group_change_bounds.py', 'reuse_certificates.py', 'paired_certificates.py',
        'cost_aware.py', 'enriched_jl.py', 'protected_jl.py']]
    protected += sorted((REPO / 'code/zk').rglob('*.go'))
    protected += [REPO / 'code/zk/go.mod', REPO / 'code/zk/go.sum']
    bib = (REPO / 'references.bib').read_bytes()
    write(PIN, dict(protected_sha256={p.relative_to(REPO).as_posix(): sha(p) for p in protected},
        bibliography_original_bytes=len(bib), bibliography_original_sha256=hashlib.sha256(bib).hexdigest(),
        scope='Post-methodology prose/results and appended bibliography entries only. No core code or protected authoring changes.'))
    print(json.dumps(dict(protected_files=len(protected))))


def verify_preservation():
    pins = read(PIN)
    for name, digest in pins['protected_sha256'].items():
        assert sha(REPO / name) == digest, name
    bib = (REPO / 'references.bib').read_bytes()[:pins['bibliography_original_bytes']]
    assert hashlib.sha256(bib).hexdigest() == pins['bibliography_original_sha256']
    for name in ['methodology.tex', 'methodology_appendix.tex']:
        assert (PAPER / name).read_bytes() == (REPO / 'paper_results/supervisor_methodology_20261027' / name).read_bytes()
    return pins


def table(name, caption, label, columns, header, rows):
    text = '\\begin{table*}[tbp]\n\\centering\n\\small\n'
    text += '\\caption{' + caption + '}\n\\label{' + label + '}\n'
    text += '\\begin{adjustbox}{max width=\\textwidth}\n'
    text += '\\begin{tabular*}{.98\\textwidth}{@{\\extracolsep{\\fill}}' + columns + '@{}}\n'
    text += '\\toprule\n' + header + '\\\\\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + '\\\\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular*}\n\\end{adjustbox}\n\\end{table*}\n'
    write(PAPER / 'generated' / name, text)


def evidence():
    verify_preservation()
    source = REPO / 'output/published_baseline_gate_20261029.json'
    receipt = read(REPO / 'output/published_baseline_gate_20261029_receipt.json')
    assert receipt['complete']
    for group in ['source_sha256', 'report_sha256', 'proof_attempt_receipts_sha256', 'warm_attempt_receipts_sha256']:
        for name, digest in receipt[group].items():
            assert sha(REPO / name) == digest, name
    gate = read(source)
    assert gate['complete'] and gate['proof_campaign_complete'] and gate['warm_comparison_complete']
    native = {(r['dataset'], r['method']): r for r in gate['native_summary'] if r['n'] == 128}
    methods = [('ordered_head_certificate', 'Centroid'), ('pca_certificate', 'PCA'),
        ('sign_rp_raw', 'Sign RP'), ('sparse_rp_raw', 'Sparse RP'), ('rabitq_raw', 'RaBitQ'),
        ('lightweight_coreset_m32', 'Lightweight32'), ('uniform_coreset_m32', 'Uniform32')]
    percent = lambda x: '--' if x is None else f'{100*x:.2f}'
    rows = []
    for method, title in methods:
        values = []
        for dataset in ['fashion', 'text', 'har']:
            r = native[dataset, method]
            assert r['fits'] == 3 and r['requests'] == 24
            values += [percent(r['assignment_agreement']), percent(r['balanced_sign_agreement'])]
        rows.append([title, *values])
    table('latest_native_quality.tex',
        'Paper-derived numerical controls on 128 records. Each source pools eight fixed requests from each of three exploratory fits. A is nearest-center label agreement and B is balanced estimated cost-change sign agreement, in percent. These are different tasks; neither is class-label accuracy. Coresets do not predict unsampled labels (--). None of the raw numerical controls supplies a cryptographic certificate.',
        'tab:latest-native', 'lrrrrrr',
        'Method & Fashion A & Fashion B & Text A & Text B & HAR A & HAR B', rows)
    rows = []
    for method, title in methods[:2] + [('sign_rp_certificate', 'Sign RP + wrapper'), ('sparse_rp_certificate', 'Sparse RP + wrapper')]:
        values = []
        for dataset in ['fashion', 'text', 'har']:
            r = native[dataset, method]
            values += [percent(r['base_coverage']), percent(r['refined_coverage'])]
        rows.append([title, *values])
    table('latest_native_coverage.tex',
        'Deterministic numerical decision coverage (percent), before/after at most 16 row corrections, on the same three-fit 128-record cohorts. Every retained deterministic interval contains the exact integer change. The random-projection rows attach our interval relation; poor wrapper coverage is not a failure of the original randomized approximation theorems. These are native bound checks, not proof timings.',
        'tab:latest-coverage', 'lrrrrrr',
        'Basis in our relation & Fashion base & Refined & Text base & Refined & HAR base & Refined', rows)
    proof = {(r['dataset'], r['method']): r for r in gate['proof_rows']}
    proof_methods = [('Exact Full / Groth16', 'Full'), ('Current centroid certificate', 'Centroid'),
        ('PCA certificate', 'PCA'), ('Sign RP + our certificate adapter', 'Sign RP + wrapper'),
        ('Sparse RP + our certificate adapter', 'Sparse RP + wrapper'), ('ZKML primitive adapter / Halo2', 'ZKML adapter')]
    rows = []
    for key, title in proof_methods:
        values = []
        for dataset in ['fashion', 'text']:
            r = proof[dataset, key]
            assert r['complete'] and r['correct_decisions'] and r['requests'] == 4
            values += [f'{r["seconds"]/60:.3f}', f'{r["coverage"]}/4', r['fallback']]
        rows.append([title, *values])
    table('latest_cold_proofs.tex',
        'New matched cold proof cohort: 32 records, four requests, seed557, one timing execution. Minutes include enclosing invocation walls, validation, preparation, public replay and all newly generated fallback; certificate totals also charge basis fitting. ZKML diagnostic MockProver work is excluded. Cov. is conclusive compact coverage before Full, or exact responses for Full/ZKML; FB is paid Full fallback. Different backend/output contracts prevent an unqualified proving-system ranking.',
        'tab:latest-cold', 'lrrrrrr', 'Method & Fashion min. & Cov. & FB & Text min. & Cov. & FB', rows)
    rows = []
    for dataset in ['fashion', 'text']:
        for method in ['ordered_head', 'pca', 'sign_rp', 'sparse_rp']:
            r = next(r for r in gate['warm_rows'] if r['dataset'] == dataset and r['method'] == method)
            assert r['all_requests_verified'] and r['all_decisions_correct']
            title = dict(ordered_head='Centroid', pca='PCA', sign_rp='Sign RP + wrapper', sparse_rp='Sparse RP + wrapper')[method]
            rows.append([dataset.capitalize(), title, r['certificate_requests'], r['direct_full_requests'],
                f'{r["measured_warm_routed_seconds"]/60:.3f}', f'{r["shared_warm_full_seconds"]/60:.3f}',
                f'{r["shared_full_over_routed"]:.3f}'])
    table('latest_warm_routing.tex',
        'Equal-routing warm follow-up on the same 32-record, four-request cohort. The existing calibration and ten-percent activation margin apply to every basis. Setup, authenticated rows and applicable old evidence are reused equally; fresh request proofs, key loads, whole-batch native planning and public replay are charged. One fresh Full comparator is shared per dataset, not counted as four repetitions. Times are phase-wall minutes, excluding export/copy and independent numerical oracle checks.',
        'tab:latest-warm', 'llrrrrr', 'Data & Basis/policy & Compact & Full & Service min. & Full min. & Ratio', rows)
    rows = []
    for dataset in ['fashion', 'text', 'har']:
        for method, title in methods[:5] + methods[5:6]:
            r = native[dataset, method]
            phases = r['median_native_query_ms']
            prediction = phases.get('prediction', phases.get('query'))
            bound = phases.get('certificate')
            rows.append([dataset.capitalize(), title, f'{prediction:.3f}', '--' if bound is None else f'{bound:.3f}'])
    table('latest_native_runtime.tex',
        'Recurring native work on 128 records, in milliseconds: median complete-pass per-bank time within each fit, then median across three fits. Raw prediction/cost routines use floating-point arrays (coreset sampled distances use exact integers); our bound/refinement code uses conservative integer arithmetic. Reusable fitting/preparation and independent oracles are outside these columns. No native time is a cryptographic proving time.',
        'tab:latest-native-runtime', 'llrr', 'Data & Method & Cost/prediction ms & Bound/refine ms', rows)
    sources = [source, REPO / 'output/published_baseline_gate_20261029_receipt.json',
        GATE / 'native_results.json', GATE / 'rabitq_followup/results.json', GATE / 'warm_followup_v2/checks.json']
    write(PAPER / 'latest_evidence_manifest.json', dict(complete=True,
        sources_sha256={p.relative_to(REPO).as_posix(): sha(p) for p in sources},
        generated_tables=5, manuscript_update_new_benchmark_proofs=False,
        proof_campaign='Completed local published_baseline_gate_20261029; new manuscript uses retained outcomes.',
        independent_fit_count=3, proof_fit_count=1, proof_timing_repetitions=1,
        protected_authoring_unchanged=True, server_measurements_available=False))
    print(json.dumps(dict(generated_tables=5, protected_files=len(read(PIN)['protected_sha256']))))


def build():
    verify_preservation(); BUILD.mkdir(parents=True, exist_ok=True)
    with (BUILD / 'console.log').open('wb') as stream:
        r = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
            '-outdir=' + str(BUILD), 'manuscript.tex'], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps(dict(returncode=r.returncode, log=str(BUILD / 'console.log'))))
    if r.returncode:
        raise SystemExit(r.returncode)


def package():
    pins = verify_preservation(); evidence_manifest = read(PAPER / 'latest_evidence_manifest.json')
    for name, digest in evidence_manifest['sources_sha256'].items():
        assert sha(REPO / name) == digest, name
    pdf = BUILD / 'manuscript.pdf'; pages = len(PdfReader(pdf).pages)
    visual = read(BUILD / 'visual_review.json')
    assert visual['passed'] and visual['pdf_sha256'] == sha(pdf)
    assert visual['pages_inspected'] == list(range(1, pages + 1))
    log = (BUILD / 'manuscript.log').read_text(encoding='utf-8', errors='replace')
    errors = re.findall(r'Overfull .*|LaTeX Error.*|.*undefined.*|.*multiply defined.*|Float too large.*', log)
    assert not errors, errors
    files = {}
    def add(path, name=None):
        path = Path(path).resolve(); assert path.is_file(), path
        name = name or path.relative_to(REPO).as_posix()
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        files[name] = path
    for line in (BUILD / 'manuscript.fls').read_text(encoding='utf-8').splitlines():
        if not line.startswith('INPUT '):
            continue
        path = Path(line[6:].strip()); path = path if path.is_absolute() else REPO / path
        path = path.resolve()
        if path.is_relative_to(REPO) and not path.is_relative_to(REPO / 'tmp') and path.is_file():
            if path.suffix.lower() in ['.tex', '.bib', '.pdf', '.svg', '.png']:
                add(path)
    add(REPO / 'references.bib'); add(BUILD / 'manuscript.bbl', 'manuscript.bbl')
    for path in PAPER.rglob('*'):
        if path.is_file(): add(path)
    for name in evidence_manifest['sources_sha256']:
        add(REPO / name)
    for name in ['output/published_baseline_gate_20261029.md',
        'output/published_baseline_gate_20261029_native_per_fit.csv',
        'output/published_baseline_gate_20261029_native_requests.csv',
        'output/current_method_audit_20261029.md', 'output/current_method_audit_20261029.json',
        'code/analysis/update_latest_paper_20261029.py',
        'code/analysis/audit_current_method_20261029.py',
        'code/analysis/review_latest_paper_20261029.py',
        'code/analysis/prepare_ccd_pilot_20261029.py',
        'code/experiments/server_balanced_72h_20261029.json',
        'code/docs/published_baseline_gate_20261029.md']:
        add(REPO / name)
    add(PIN, 'AUTHORING_PRESERVATION.json')
    for filename in ['python_tests.resource.json', 'python_tests.log', 'go_tests.json', 'render_checks.json']:
        add(BUILD / filename, 'validation/latest_baselines_20261029/' + filename)
    # Retained earlier tables depend on their immutable numerical exports too.
    for manifest in [read(PAPER / 'evidence_manifest.json'), read(PAPER / 'figures/manifest.json')]:
        for name, digest in manifest['sources_sha256'].items():
            assert sha(REPO / name) == digest, name
            add(REPO / name)
    for name in ['elsarticle.cls', 'elsarticle-num.bst']:
        located = subprocess.check_output(['kpsewhich', name], text=True).strip(); add(located, name)
    readme = '''# Updated preliminary paper, 29 October 2026

Build the included manuscript.tex with latexmk -pdf -interaction=nonstopmode
-halt-on-error manuscript.tex. A TeX Live/MiKTeX installation with elsarticle,
TikZ, amsmath and the other packages listed in manuscript.tex is required.
The bibliography, compiled bibliography, class/style and local figures/tables
are included. Existing abstract, introduction and methodology are unchanged.
Only completed local results are reported; server measurements remain future.

The new tables can be regenerated by code/analysis/update_latest_paper_20261029.py
evidence in the complete private workspace. This manuscript build package is
not an independent reproduction of cryptographic experiments; proving keys,
full proof bundles, datasets and the complete campaign environment are not
included. Reports retain paths/hashes into the main private evidence archive.
Author metadata/declarations require author review. Submission-ready status
is not claimed. No server execution was performed for this paper update.
'''
    out = REPO / 'output'; final_pdf = out / 'pdf' / (TAG + '.pdf')
    shutil.copy2(pdf, final_pdf)
    source_zip = out / (TAG + '_sources.zip')
    manifest = dict(main='manuscript.tex', files_sha256={n: sha(p) for n, p in sorted(files.items())})
    with zipfile.ZipFile(source_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()): archive.write(path, name)
        archive.writestr('README_BUILD.md', readme)
        archive.writestr('SOURCE_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(source_zip) as archive:
        assert archive.testzip() is None
        for name, digest in manifest['files_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
    validation = dict(complete=True, pdf=final_pdf.relative_to(REPO).as_posix(), pages=pages,
        pdf_sha256=sha(final_pdf), source_zip=source_zip.relative_to(REPO).as_posix(),
        source_zip_sha256=sha(source_zip), source_files=len(files),
        protected_files_unchanged=len(pins['protected_sha256']), protected_sha256=pins['protected_sha256'],
        abstract_introduction_methodology_unchanged=True, original_bibliography_prefix_unchanged=True,
        evidence_sources_verified=True, latex_errors_or_overflows=errors, visual_review=visual,
        server_experiments_run=False, new_benchmark_proofs_for_paper_update=False,
        submission_ready=False, results_status='Completed local exploratory baselines, including negative outcomes.')
    write(out / (TAG + '_validation.json'), validation)
    print(json.dumps(dict(complete=True, pages=pages, source_files=len(files))))


def check_package():
    """Build the delivered dependency closure outside the working source tree."""
    archive_path = REPO / 'output' / (TAG + '_sources.zip')
    target = BUILD / 'independent_package_check'; target.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
        archive.extractall(target)
    with (target / 'build_check.log').open('wb') as stream:
        result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
            'manuscript.tex'], cwd=target, stdout=stream, stderr=subprocess.STDOUT)
    assert result.returncode == 0, target / 'build_check.log'
    delivered = PdfReader(REPO / 'output/pdf' / (TAG + '.pdf'))
    rebuilt = PdfReader(target / 'manuscript.pdf')
    normalize = lambda pages: re.sub(r'\s+', '', '\n'.join(p.extract_text() for p in pages))
    assert len(delivered.pages) == len(rebuilt.pages)
    assert normalize(delivered.pages) == normalize(rebuilt.pages)
    path = REPO / 'output' / (TAG + '_validation.json'); value = read(path)
    value['independent_source_package_build'] = dict(returncode=0, pages=len(rebuilt.pages),
        extracted_text_identical=True, log=(target / 'build_check.log').relative_to(REPO).as_posix())
    write(path, value)
    print(json.dumps(value['independent_source_package_build']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['capture', 'evidence', 'build', 'package', 'check_package'])
    args = parser.parse_args(); globals()[args.phase]()
