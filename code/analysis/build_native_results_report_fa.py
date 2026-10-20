"""Source-bound Persian explanation of Sparse JL, Protected JL and native results.

Uses XeLaTeX for proper Persian shaping and bidirectional mathematical layout.
Does not execute experiments or rewrite their evidence.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import unicodedata

import numpy as np
import pandas as pd
import pymupdf as fitz

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / 'code'
ROOT = CODE / 'results/paper_compression_balanced_audited'
TABLES = ROOT / 'paper_tables'
SUPPORT = ROOT / 'new_report_fa_20261012'
BUILD = REPO / 'tmp/pdfs/native_report_fa_build'
OUTPUT = REPO / 'output/pdf/native_compression_results_fa_20261012.pdf'
TEMPLATE = Path(__file__).with_name('native_results_report_fa.tex')
ENGINE = Path('C:/texlive/2026/bin/windows/xelatex.exe')
ORDER = ['gaussian_easy', 'gaussian_hard', 'dbscan_margin', 'mnist',
         'fashion_mnist', 'gisette', 'arcene', '20ng_tfidf']
NAMES = dict(gaussian_easy='Gaussian easy', gaussian_hard='Gaussian hard',
             dbscan_margin='DBSCAN margin', mnist='MNIST', fashion_mnist='Fashion-MNIST',
             gisette='Gisette', arcene='Arcene', **{'20ng_tfidf': '20 Newsgroups'})
METHODS = dict(full='Full', pca='PCA (SVD)', pca_randomized='PCA (randomized)',
               sparse_jl='Sparse JL', protected_jl='Protected JL', protected_only='Protected only')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def esc(value):
    return str(value).replace('\\', r'\textbackslash{}').replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')


def lr(value):
    return r'\lr{' + esc(value) + '}'


def n(value, digits=3):
    if pd.isna(value):
        return '--'
    value = 0.0 if abs(float(value)) < .5 * 10 ** (-digits) else float(value)
    return f'{value:.{digits}f}'


def label(row):
    return lr(NAMES[row.dataset] + (' (nonzero)' if row.variant == 'nonzero' else ''))


def ordered(frame):
    frame = frame.copy()
    frame['_o'] = frame.dataset.map({s: i for i, s in enumerate(ORDER)})
    return frame.sort_values(['_o', 'variant', 'k'] if 'k' in frame else ['_o', 'variant']).drop(columns='_o')


def table(title, headers, rows, spec=None, small=True):
    # Native RTL tables: the first logical column appears at the right edge.
    spec = spec or ('@{}l' + 'r' * (len(headers) - 1) + '@{}')
    return '\n'.join([
        r'\begin{center}', r'\refstepcounter{table}',
        r'\textbf{جدول \thetable. ' + title + r'}\par\vspace{3pt}',
        r'{\small' if small else '{', r'\renewcommand{\arraystretch}{1.35}',
        r'\setlength{\tabcolsep}{5pt}', r'\begin{tabular}{' + spec + '}', r'\toprule',
        ' & '.join(headers) + r' \\', r'\midrule',
        *[' & '.join(str(c) for c in row) + r' \\' for row in rows],
        r'\bottomrule', r'\end{tabular}}', r'\end{center}',
    ])


def build():
    SUPPORT.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    validation = json.loads((ROOT / 'report/validation.json').read_text())
    replay = json.loads((ROOT / 'report/replay_validation.json').read_text())
    config = json.loads((ROOT / 'manifest.json').read_text())['identity']['configuration']
    assert config['eps'] == .15 and config['thresholds']['distortion'] == .20
    assert (config['threads'], config['seeds'], config['starts'], config['timing_repetitions']) == (4, 5, 3, 5)
    english = json.loads((ROOT / 'new_report_20261012/report_manifest.json').read_text())
    assert validation['complete'] and validation['valid_observed_results'] and replay['all_passed']
    for path, digest in english['source_csv_sha256'].items():
        assert sha(ROOT / path) == digest, path
    frames = {path.name: pd.read_csv(path) for path in sorted(TABLES.glob('*.csv'))}
    ops = frames['11_selected_operating_points.csv']
    hjl = ordered(ops[ops.method.eq('protected_jl')])
    assert len(hjl) == 16 and hjl.test_fidelity_rule_pass.sum() == 15
    replace = {}
    saved_tables = {}

    def add(key, title, headers, rows):
        replace[key] = table(title, headers, rows)
        saved_tables[key] = dict(title=title, headers=headers, rows=rows)

    # Educational examples: independent tiny algebra checks, not study results.
    x = np.array([2., 1., 3., 4.])
    a = np.array([[1., 0., -1., 0.], [0., 1., 0., 1.]])
    z = a @ x
    assert np.array_equal(z, [-1., 5.]) and x @ x == 30 and z @ z == 26
    x2 = np.array([1.4, 3., 4.])
    centers = np.array([[0., 0., 0.], [2., 0., 0.]])
    u = np.array([[1.], [0.], [0.]])
    s = np.ones((1, 3))
    def protected(x):
        h = x @ u
        return np.column_stack((h, x @ s.T - h @ (s @ u).T))
    orig = ((x2 - centers) ** 2).sum(axis=1)
    compressed = ((protected(x2[None, :]) - protected(centers)) ** 2).sum(axis=1)
    assert np.allclose(orig, [26.96, 25.36]) and np.allclose(compressed, [50.96, 49.36])
    assert np.allclose(compressed - orig, [24., 24.]) and orig.argmin() == compressed.argmin() == 1
    toys = dict(sparse=dict(original_squared_norm=float(x @ x), projected_squared_norm=float(z @ z), z=z.tolist()),
                protected=dict(original_squared_distances=orig.tolist(), compressed_squared_distances=compressed.tolist(), common_offset=24))
    (SUPPORT / 'educational_examples.json').write_text(json.dumps(toys, indent=2), encoding='utf-8')

    add('DATASETS', 'داده‌ها و تقسیم عددی ثابت',
        ['مجموعه', r'$n$', r'$d$', 'آموزش', 'اعتبارسنجی', 'آزمون'],
        [[label(r), *[lr(int(r[v])) for v in ['n', 'd', 'train_rows', 'validation_rows', 'test_rows']]]
         for _, r in ordered(frames['01_datasets.csv']).iterrows()])
    sel = frames['02_selection.csv']
    add('QUALIFICATION', 'عبور از معیار مشترک در هجده تنظیم داده و تعداد مرکز',
        ['روش', 'انتخاب اعتبارسنجی', 'قبول آزمون انتخاب‌شده'],
        [[lr(METHODS[m]), lr(f'{int(sel[sel.method.eq(m)].selected.sum())}/18'),
          lr(f'{int(sel[sel.method.eq(m)].test_fidelity_rule_pass.sum())}/{int(sel[sel.method.eq(m)].selected.sum())}')
          if sel[sel.method.eq(m)].selected.sum() else '--']
         for m in ['full', 'pca', 'pca_randomized', 'sparse_jl', 'protected_jl', 'protected_only']])
    add('SELECTED', 'ابعاد منتخب ' + lr('Protected JL') + ' و سقف اطمینان خطای هندسی در آزمون',
        ['مجموعه', r'$k$', r'$m$', r'$q$', r'$100m/d$', r'$U_{95}(Q^{pair}_{95})$', 'نتیجه'],
        [[label(r), lr(int(r.k)), lr(int(r.m)), lr(int(r.q)), lr(n(100 * r.retained_dimension_fraction, 2)),
          lr(n(r.pair_q95_ci_high)), 'قبول' if r.test_fidelity_rule_pass else 'رد'] for _, r in hjl.iterrows()])
    ref = frames['04_fidelity_m128_k5.csv']
    matched = []
    for ds in ['mnist', 'gisette', '20ng_tfidf']:
        for method in ['full', 'pca', 'pca_randomized', 'sparse_jl', 'protected_jl', 'protected_only']:
            r = ref[(ref.dataset == ds) & (ref.variant == 'primary') & (ref.method == method)].iloc[0]
            matched.append([lr(NAMES[ds]), lr(METHODS[method]), lr(int(r.m)),
                            lr(n(r.ari_mean)), lr(n(r.pair_q95_mean)), lr(n(r.projected_cost_relative_error_mean))])
    add('MATCHED', 'مقایسه در بعد مشترک ۱۲۸، پنج مرکز و تنکی یک',
        ['داده', 'روش', r'$m$', r'$ARI$', r'$Q^{pair}_{95}$', r'$E_{cost}$'], matched)
    sparse_comparison = []
    for _, meta in ordered(frames['01_datasets.csv']).iterrows():
        plain = ref[(ref.dataset == meta.dataset) & (ref.variant == meta.variant) & ref.method.eq('sparse_jl')].iloc[0]
        hybrid = ref[(ref.dataset == meta.dataset) & (ref.variant == meta.variant) & ref.method.eq('protected_jl')].iloc[0]
        sparse_comparison.append([label(meta), lr(n(plain.ari_mean)), lr(n(hybrid.ari_mean)),
                                  lr(n(plain.pair_q95_mean)), lr(n(hybrid.pair_q95_mean))])
    add('SPARSE_COMPARISON', 'دو روش JL در تمام داده‌ها، بعد ۱۲۸ و پنج مرکز',
        ['داده', r'$ARI_{JL}$', r'$ARI_H$', r'$Q_{95,JL}$', r'$Q_{95,H}$'], sparse_comparison)
    plain_choices = ordered(ops[ops.method.eq('sparse_jl')])
    assert len(plain_choices) == 2
    add('SPARSE_CHOICES', 'انتخاب‌های نسخه ساده JL و هزینه سی‌ودو پرس‌وجو',
        ['داده', r'$k$', r'$m$', 'آزمون', 'هسته', 'استفاده مجدد', 'شروع سرد'],
        [[label(r), lr(int(r.k)), lr(int(r.m)), 'قبول' if r.test_fidelity_rule_pass else 'رد',
          *[lr(n(r[c])) for c in ['query_only_speedup_mean', 'conditional_reuse_speedup_mean', 'cold_speedup_mean']]]
         for _, r in plain_choices.iterrows()])
    add('RUNTIME', 'نسبت زمان Full به روش ما برای سی‌ودو پرس‌وجو',
        ['مجموعه', r'$k$', r'$m$', 'هسته', 'استفاده مجدد', 'شروع سرد'],
        [[label(r), lr(int(r.k)), lr(int(r.m)), *[lr(n(r[c])) for c in
          ['query_only_speedup_mean', 'conditional_reuse_speedup_mean', 'cold_speedup_mean']]] for _, r in hjl.iterrows()])
    gis = hjl[(hjl.dataset == 'gisette') & (hjl.k == 5)].iloc[0]
    add('GIS_CI', 'Gisette، پنج مرکز، بعد ۲۵۶: میانگین نسبت و بازه اطمینان',
        ['مرز زمان', 'نسبت میانگین', 'بازه اطمینان ۹۵ درصد'],
        [[t, lr(n(gis[p + '_mean'])), lr(f"[{n(gis[p+'_ci_low'])}, {n(gis[p+'_ci_high'])}]")]
         for t, p in [('فقط هسته پرس‌وجو', 'query_only_speedup'), ('استفاده مجدد با آماده‌سازی', 'conditional_reuse_speedup'),
                      ('شروع سرد با آموزش', 'cold_speedup')]])
    add('DOWNSTREAM', 'کیفیت کاربردهای جدید در تنظیم‌های منتخب روش ما',
        ['مجموعه', r'$k$', r'$m$', 'بازیابی پنج همسایه', 'خوشه‌بندی تازه', 'بازتنظیم مراکز'],
        [[label(r), lr(int(r.k)), lr(int(r.m)), *[lr(n(r[c])) for c in
          ['knn_recall_mean', 'kmeans_ari_mean', 'kmeans_refine_ari_mean']]] for _, r in hjl.iterrows()])
    add('CERTIFICATES', 'گواهی عمومی حاشیه در انتخاب‌های پنج‌مرکزی Protected JL'.replace('Protected JL', lr('Protected JL')),
        ['داده', r'$m$', 'توافق برچسب', 'شرط حاشیه', 'شرط فاصله', 'پوشش گواهی'],
        [[label(r), lr(int(r.m)), *[lr(n(100 * r[c], 2)) for c in
          ['agreement_mean', 'margin_coverage_mean', 'centroid_embedding_point_fraction_mean', 'certificate_coverage_mean']]]
         for _, r in hjl[hjl.k.eq(5)].iterrows()])
    setup = frames['05_compression_setup.csv']
    training = frames['06_model_training.csv']
    stages = []
    for _, r in hjl[hjl.k.eq(5)].iterrows():
        cell = setup[(setup.dataset == r.dataset) & (setup.variant == r.variant) & (setup.k == r.k)
                     & (setup.method == r.method) & (setup.m == r.m) & (setup.q == r.q)].iloc[0]
        fit = training[(training.dataset == r.dataset) & (training.variant == r.variant)
                       & training.stage.eq('centroids') & training.fit_key.str.startswith('k5_', na=False)].median_seconds.median()
        stages.append([label(r), lr(int(r.m)), lr(n(fit, 4)), lr(n(cell.model_construction_seconds, 4)),
                       lr(n(cell.compression_seconds, 4)), lr(n(cell.point_norm_preparation_seconds, 6))])
    add('STAGES', 'تفکیک مراحل در پنج مرکز؛ ثانیه، میانه مقادیر ثبت‌شده',
        ['داده', r'$m$', 'آموزش مراکز', 'ساخت نگاشت', 'فشرده‌سازی', 'آماده‌سازی نرم'], stages)
    # DBSCAN is reported at a matched workload dimension, not the selected geometry m.
    wl = frames['07_downstream_quality.csv']
    db = frames['10_dbscan_profiles.csv']
    dbrows = []
    for ds in ['dbscan_margin', 'mnist', 'gisette', '20ng_tfidf']:
        for method in ['pca', 'sparse_jl', 'protected_jl']:
            q = 0 if method == 'pca' else 1
            rr = wl[(wl.dataset == ds) & wl.variant.eq('primary') & wl.k.eq(5) & wl.method.eq(method) & wl.m.eq(64) & wl.q.eq(q)].iloc[0]
            extra = db[(db.dataset == ds) & db.variant.eq('primary') & db.k.eq(5) & db.method.eq(method) & db.m.eq(64) & db.q.eq(q) & db.profile.eq('validation_q75')]
            if len(extra):
                assert len(extra) == 1
                qq = extra.iloc[0]
            else:
                radii = json.loads((ROOT / 'jobs' / f'{ds}_primary' / 'selection.json').read_text())
                assert np.isclose(radii['radii']['validation_q75'], radii['radius'], rtol=1e-12, atol=0)
                qq = rr  # Q75 equals primary: the campaign intentionally does not repeat it.
            dbrows.append([lr(NAMES[ds]), lr(METHODS[method]), lr(n(rr.dbscan_ari_mean)), lr(n(rr.graph_f1_mean)),
                           lr(n(qq.dbscan_ari_mean)), lr(n(qq.graph_f1_mean))])
    add('DBSCAN', 'کنترل DBSCAN در بعد ۶۴ و پنج مرکز: شعاع اصلی و صدک ۷۵',
        ['داده', 'روش', r'$ARI_0$', r'$F1_0$', r'$ARI_{75}$', r'$F1_{75}$'], dbrows)

    # Plots use mathematical axis labels and original dataset/method proper names;
    # all interpretive labels, keys and captions are Persian in the surrounding text.
    quality = frames['03_fidelity_all.csv']
    charts = []
    for ds in ['mnist', 'gisette', '20ng_tfidf']:
        lines = []
        for method, color, extra in [('protected_jl', 'teal', ''), ('sparse_jl', 'orange', ''),
                                      ('pca', 'blue', ''), ('pca_randomized', 'gray', 'dashed,')]:
            q = 1 if method in ['protected_jl', 'sparse_jl'] else 0
            cells = quality[(quality.dataset == ds) & quality.variant.eq('primary') & quality.k.eq(5)
                            & quality.method.eq(method) & quality.q.eq(q)].sort_values('m')
            pts = ' '.join(f'({int(r.m)},{r.pair_q95_mean:.8f})' for _, r in cells.iterrows())
            lines.append(r'\addplot[' + extra + color + r',thick,mark=*,mark size=1pt] coordinates {' + pts + '};')
        charts.append(r'\begin{tikzpicture}\begin{axis}[width=.32\textwidth,height=5.5cm,'
                      r'xmode=log,log basis x=2,xmin=15,xmax=530,ymin=0,ymax=1.04,'
                      r'xtick={16,64,256,512},xticklabels={$16$,$64$,$256$,$512$},ytick={0,.2,.6,1},'
                      r'xlabel={$m$},ylabel={$Q_{95}$},title={\lr{' + NAMES[ds] + r'}},'
                      r'tick label style={font=\scriptsize},grid=major,grid style={gray!15}]' + '\n' + '\n'.join(lines) +
                      r'\addplot[black,dotted,domain=16:512] {.2};\end{axis}\end{tikzpicture}')
    replace['GEOMETRY_PLOT'] = r'\begin{center}\begin{LTR}' + '\n'.join(charts) + r'\end{LTR}\end{center}'
    bars = []
    timing_names = ['mnist', 'fashion_mnist', 'gisette', 'arcene']
    for prefix, color in [('query_only_speedup', 'teal'), ('conditional_reuse_speedup', 'blue'), ('cold_speedup', 'orange')]:
        pts = []
        for ds in timing_names:
            r = hjl[(hjl.dataset == ds) & hjl.k.eq(5)].iloc[0]
            pts.append(f'({NAMES[ds]},{r[prefix+"_mean"]:.8f})')
        bars.append(r'\addplot[fill=' + color + r',draw=' + color + '] coordinates {' + ' '.join(pts) + '};')
    replace['TIMING_PLOT'] = (r'\begin{center}\begin{LTR}\begin{tikzpicture}\begin{axis}['
        r'width=.94\textwidth,height=6cm,ybar,bar width=10pt,ymode=log,ymin=.07,ymax=30,'
        r'symbolic x coords={MNIST,Fashion-MNIST,Gisette,Arcene},xtick=data,'
        r'ylabel={$T_{\mathrm{Full}}/T_{\mathrm{method}}$},ytick={.1,.3,1,3,10,30},'
        r'yticklabels={$0.1$,$0.3$,$1$,$3$,$10$,$30$},grid=major,grid style={gray!15}]' + '\n'.join(bars) +
        r'\end{axis}\end{tikzpicture}\end{LTR}\end{center}')

    tex = TEMPLATE.read_text(encoding='utf-8')
    for key, value in replace.items():
        token = '@@' + key + '@@'
        assert token in tex, key
        tex = tex.replace(token, value)
    assert '@@' not in tex
    tex_path = BUILD / 'report.tex'
    tex_path.write_text(tex, encoding='utf-8')
    (SUPPORT / 'report_tables.json').write_text(json.dumps(saved_tables, ensure_ascii=False, indent=2), encoding='utf-8')
    for pass_index in range(3):
        result = subprocess.run([str(ENGINE), '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                                cwd=BUILD, capture_output=True, timeout=180)
        (SUPPORT / f'xelatex_pass_{pass_index+1}.log').write_bytes(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stdout.decode('utf-8', errors='replace')[-6000:])
    shutil.copyfile(BUILD / 'report.pdf', OUTPUT)
    shutil.copyfile(tex_path, SUPPORT / 'native_results_report_fa.tex')
    doc = fitz.open(OUTPUT)
    text = unicodedata.normalize('NFKC', '\n'.join(p.get_text() for p in doc))
    persian_chars = sum('\u0600' <= c <= '\u06ff' for c in text)
    assert persian_chars > 10000, persian_chars
    assert 17 <= len(doc) <= 26, len(doc)
    for word in ['Sparse', 'Protected', 'MNIST', 'Gisette', 'Arcene', 'PCA', 'SHAKE256']:
        assert word in text, word
    manifest = dict(date='2026-10-12', pdf=str(OUTPUT), pdf_sha256=sha(OUTPUT), pages=len(doc),
                    persian_character_count=persian_chars, tables=len(saved_tables),
                    source_csv_sha256={p.name: sha(p) for p in sorted(TABLES.glob('*.csv'))},
                    source_code_sha256={str(p.relative_to(CODE)): sha(p) for p in
                        [CODE/'commitjl/reusable_native.py', CODE/'commitjl/sparse_gen.py', CODE/'experiments/paper_compression.py']},
                    validation_sha256=sha(ROOT/'report/validation.json'),
                    replay_sha256=sha(ROOT/'report/replay_validation.json'),
                    generator_sha256=sha(Path(__file__)), template_sha256=sha(TEMPLATE),
                    validation_choices=16, protected_test_passes=15,
                    representative_replay_cases=replay['checked_cases'],
                    new_experiments_run=False, educational_examples_verified=True,
                    scope='Completed native numerical study; not new cryptographic proof evidence')
    (SUPPORT / 'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(pdf=str(OUTPUT), pages=len(doc), tables=len(saved_tables), persian_chars=persian_chars)))


if __name__ == '__main__':
    build()
