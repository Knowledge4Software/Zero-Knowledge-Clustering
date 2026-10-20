"""Build and validate the Persian educational guide from executed examples."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'code/results/overview/education/persian_guide'
BUILD = ROOT / 'code/.cache/persian_guide/book'
DEST = ROOT / 'output/pdf/commitjl_complete_guide_fa.pdf'
SOURCE = Path(__file__).with_name('commitjl_guide_fa.tex')
ANALYSIS = ROOT / 'code/results/diagnostics/local_v1/analysis'


def dump(name, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
    (BUILD / name).write_text(text, encoding='utf-8')


def figure(fig, name):
    fig.savefig(ASSETS / name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    examples = json.loads((ASSETS / 'examples.json').read_text(encoding='utf-8'))
    for relative_path, expected in examples['source_fingerprints'].items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected, relative_path
    for dataset, values in examples['datasets'].items():
        excerpt = {k: values[k] for k in ('shape', 'dtype', 'first_label', 'first_12_features') if k in values}
        excerpt['first_12_features'] = [round(v, 5) for v in values['first_12_features']]
        excerpt['first_nonzero_index_value'] = [[i, round(v, 5)] for i,v in values['first_8_nonzero_index_value'][:4]]
        dump(dataset + '_sample.txt', '\n'.join([
            f"shape = {tuple(values['shape'])}", f"dtype = {values['dtype']}",
            f"first_label = {values['first_label']}",
            'first_12_features =',
            np.array2string(np.array(excerpt['first_12_features']),precision=5,max_line_width=70),
            'first_nonzero_index_value =',
            repr(excerpt['first_nonzero_index_value']),
        ]))
    toy=examples['toy_pca']
    dump('toy_pca_output.txt', 'mean = [2., 2.]\n'+
         'direction = '+repr(np.round(toy['components'][0],6).tolist())+'\n'+
         'scores = '+repr(np.round(np.array(toy['Z']).ravel(),6).tolist())+'\n'+
         'retained_variance = '+repr(toy['explained_variance_ratio'][0]))
    pca = examples['mnist_pca']
    dump('mnist_pca_output.txt', f"X_shape = ({pca['n']}, {pca['d']})\nZ_shape = {tuple(pca['Z_shape'])}\n"+
         'first_scores = '+repr(np.round(pca['row0_first6_scores'],3).tolist())+'\n'+
         f"variance_retained = {pca['variance_retained']:.6f}\nactual_pixel_patch =\n"+
         np.array2string(np.array(pca['pixel_row0_patch']),precision=1,max_line_width=75))
    dump('assignments_output.txt', '\n'.join(f'{key} = {pca[key]}' for key in (
        'true_digits_first12', 'full_labels_first12', 'pca_labels_first12',
        'sparse_labels_first12', 'pca_ari_to_full', 'sparse_ari_to_full')))
    tfidf = examples['tfidf_toy']
    dump('tfidf_output.txt', 'features = '+repr(tfidf['features'])+'\nmatrix =\n'+
         np.array2string(np.array(tfidf['X']),precision=3,max_line_width=80))
    proof = examples['tiny_protocol']
    projected=np.array(proof['X'])@np.array(proof['A']).T
    centers=np.array(proof['C'])@np.array(proof['A']).T
    costs=np.sum((projected-centers[np.array(proof['assignments'])])**2,axis=1)
    def vectors(rows):
        return ','.join('('+','.join(str(int(v)) for v in row)+')' for row in rows)
    dump('proof_math.tex', '\\[Z=\\begin{bmatrix}'+r'\\'.join('&'.join(str(v) for v in row) for row in projected)+
         '\\end{bmatrix},\\quad C_Z=\\begin{bmatrix}'+r'\\'.join('&'.join(str(v) for v in row) for row in centers)+
         '\\end{bmatrix},\\quad \\widehat\\Psi='+ '+'.join(str(v) for v in costs)+'='+str(proof['psi_hat'])+'\\]\n')
    dump('proof_macros.tex', f"\\newcommand{{\\ProofWall}}{{{proof['wall_s']:.2f}}}\n"+
         f"\\newcommand{{\\ProofConstraints}}{{{proof['constraints']}}}\n"+
         f"\\newcommand{{\\ProofBytes}}{{{proof['proof_bytes']}}}\n")
    dump('proof_output.txt', '\n'.join(f'{key} =\n{np.array2string(np.array(proof[key]),max_line_width=75)}'
         for key in ('X','C','A'))+'\n'+ '\n'.join(f'{key} = {proof[key]}' for key in (
             'assignments','psi_hat','proof_ok','verify_ok','constraints','proof_bytes','backend','wall_s')))
    dump('transcript_output.txt', {k: proof[k] for k in ('C_X','C_theta','C_A','rho_hex','seed_hex')})
    rq1 = pd.read_csv(ANALYSIS / 'rq1_summary.csv')
    names = ['gaussian_mixture','arcene','fashion_mnist','mnist','gisette','20ng_tfidf','dbscan_margin']
    labels = ['Gaussian','Arcene','Fashion','MNIST','Gisette','Text','DBSCAN synth.']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(8,3.6))
    loc = np.arange(len(names))
    for shift, method, color, caption in [(-.19,'sparse_jl','#008A8A','SparseJL q=2'),(.19,'pca','#D37835','PCA')]:
        selected = rq1[(rq1.method==method)&(rq1.m==64)]
        if method == 'sparse_jl': selected = selected[selected.q==2]
        selected = selected.set_index('dataset_name').loc[names]
        means = selected.ari_mean.to_numpy()
        errors = np.array([means-selected.ari_ci95_low.to_numpy(),selected.ari_ci95_high.to_numpy()-means])
        ax.bar(loc+shift,means,.36,color=color,label=caption,yerr=errors,capsize=3)
    ax.set(xticks=loc,xticklabels=labels,ylabel='ARI to fixed full-space assignments',
           ylim=(0,1.23),yticks=np.arange(0,1.01,.2))
    ax.tick_params(axis='x',labelrotation=20);ax.legend(loc='upper right')
    ax.set_title('Matched m=64; ten seeds; 95% confidence intervals')
    figure(fig,'quality_matched.png')
    rq5 = pd.read_csv(ANALYSIS / 'rq5_summary.csv')
    fig,ax=plt.subplots(figsize=(7,3.5))
    for m,color in [(16,'#D37835'),(64,'#008A8A'),(128,'#193047')]:
        rows=rq5[(rq5.test=='separation')&(rq5.m==m)].sort_values('gamma')
        if rows.empty: rows=rq5[rq5.gamma.notna()&(rq5.m==m)].sort_values('gamma')
        ax.errorbar(rows.gamma,rows.ari_mean,yerr=[rows.ari_mean-rows.ari_ci95_low,rows.ari_ci95_high-rows.ari_mean],
                    marker='o',color=color,label=f'm={m}',capsize=3)
    ax.set(xlabel='Cluster separation parameter gamma',ylabel='ARI to full-space assignments',ylim=(-.03,1.08))
    ax.legend();figure(fig,'separation_clean.png')
    fig,ax=plt.subplots(figsize=(7,3.6))
    for name,caption,color in [('mnist','MNIST','#193047'),('20ng_tfidf','Text','#D37835'),('arcene','Arcene','#008A8A')]:
        rows=rq5[(rq5.test=='precision')&(rq5.policy=='fixed_reference_24')&(rq5.dataset_name==name)].sort_values('bits')
        ax.plot(rows.bits,rows.ari_mean,'o-',label=caption,color=color)
    ax.set(xlabel='Precision label (bits), fixed 24-bit physical reference',ylabel='Assignment ARI',xticks=[8,12,16,20,24],ylim=(-.03,1.08))
    ax.legend();figure(fig,'precision_clean.png')
    measurements=pd.read_parquet(ROOT/'code/results/diagnostics/local_v1/measurements.parquet')
    timing=measurements[(measurements.rq=='RQ3')&(measurements.test=='overhead')]
    averages=timing.groupby('n')[['total_s','stage_commit_X_s','after_commit_s']].mean()
    fig,axes=plt.subplots(1,2,figsize=(8,3.2))
    x=np.arange(len(averages)); ns=averages.index.astype(str).tolist()
    axes[0].bar(x,averages.stage_commit_X_s,color='#193047',label='Commit input')
    axes[0].bar(x,averages.total_s-averages.stage_commit_X_s,bottom=averages.stage_commit_X_s,
                color='#D37835',label='Other host stages')
    axes[0].set(xticks=x,xticklabels=ns,xlabel='Number of rows n',ylabel='Mean host time (seconds)')
    axes[0].legend(fontsize=8)
    shares=100*averages.after_commit_s/averages.total_s
    axes[1].bar(x,shares,color='#008A8A')
    axes[1].set(xticks=x,xticklabels=ns,xlabel='Number of rows n',ylabel='Post-commit share (%)',ylim=(0,2.1))
    for i,val in enumerate(shares): axes[1].text(i,val+.05,f'{val:.2f}%',ha='center',fontsize=9)
    fig.suptitle('Native host measurements; beacon = simulated local trace',fontsize=10)
    fig.tight_layout();figure(fig,'overhead_clean.png')
    rq4=pd.read_csv(ANALYSIS/'rq4_summary.csv')
    eligible=rq4[(rq4.test=='centroid')&(rq4.m==256)&(rq4.q==2)&(rq4.eps==.15)]
    keys=['arcene','fashion_mnist','mnist','gaussian_mixture','20ng_tfidf','gisette']
    eligible=eligible.set_index('dataset_name').loc[keys]
    fig,ax=plt.subplots(figsize=(7,3.5))
    vals=eligible.eligible_fraction_mean.to_numpy()
    ax.bar(np.arange(6),vals,color='#008A8A',yerr=[vals-eligible.eligible_fraction_ci95_low,
        eligible.eligible_fraction_ci95_high-vals],capsize=3)
    ax.set(xticks=np.arange(6),xticklabels=['Arcene','Fashion','MNIST','Gaussian','Text','Gisette'],
           ylabel='Eligible point fraction',ylim=(min(-.05,eligible.eligible_fraction_ci95_low.min()-.025),1.08),
           title='Certificate coverage: m=256, q=2, epsilon=0.15')
    figure(fig,'coverage_clean.png')
    for image in ASSETS.glob('*.png'): shutil.copy2(image, BUILD / image.name)
    shutil.copy2(SOURCE, BUILD / 'guide.tex')
    engine = Path('C:/texlive/2026/bin/windows/xelatex.exe')
    for pass_no in range(1,4):
        run = subprocess.run([str(engine),'-interaction=nonstopmode','-halt-on-error','guide.tex'],
                             cwd=BUILD,capture_output=True,timeout=150)
        (BUILD / f'compile_{pass_no}.txt').write_bytes(run.stdout+run.stderr)
        if run.returncode:
            raise RuntimeError(run.stdout.decode('utf-8',errors='replace')[-6000:])
    pdf = BUILD / 'guide.pdf'
    doc = fitz.open(pdf)
    text = '\n'.join(page.get_text() for page in doc)
    persian_chars = sum('\u0600' <= c <= '\u06ff' or '\ufb50' <= c <= '\ufeff' for c in text)
    assert persian_chars > 10000, 'Persian text extraction is unexpectedly missing.'
    assert len(doc) >= 25, 'The comprehensive guide is unexpectedly short.'
    outside = []
    for number,page in enumerate(doc,1):
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0: continue
            for line in block['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0 < 12 or x1 > page.rect.width-12 or y0 < 12 or y1 > page.rect.height-12:
                        outside.append({'page':number,'text':span['text'],'bbox':span['bbox']})
    shutil.copy2(pdf,DEST)
    manifest={'pdf':str(DEST.relative_to(ROOT)),'pages':len(doc),'persian_characters':persian_chars,
              'outside_page_safety_margin':outside,'examples_sha256':hashlib.sha256((ASSETS/'examples.json').read_bytes()).hexdigest(),
              'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              'benchmark_sha256':hashlib.sha256((ROOT/'code/results/diagnostics/local_v1/measurements.parquet').read_bytes()).hexdigest(),
              'tiny_proof_verified':proof['proof_ok'] and proof['verify_ok']}
    (ASSETS/'document_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
