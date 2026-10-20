"""Export the paired completed native comparison; never rerun experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS={'gaussian_easy':'Gaussian easy','gaussian_hard':'Gaussian hard','dbscan_margin':'DBSCAN-margin synthetic',
        'mnist':'MNIST','fashion_mnist':'Fashion-MNIST','gisette':'Gisette','arcene':'Arcene','20ng_tfidf_nonzero':'20 Newsgroups'}
ORDER=list(LABELS)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def compare(root):
    root=Path(root);out=root/'comparison';out.mkdir(exist_ok=True)
    quality=pd.read_csv(root/'native_quality.csv');geometry=pd.read_csv(root/'native_geometry.csv')
    assert len(quality)==240 and len(geometry)==192,'Complete native cohort required'
    q=quality[quality.method.isin(['pca','protected_tail'])].copy()
    assert q.groupby(['dataset','k','method']).size().eq(3).all()
    q['absolute_original_cost_error_percent']=q.claimed_original_cost_error.abs()*100
    q['agreement_quantized_full_percent']=q.agreement_quantized_full*100
    q['agreement_float_full_percent']=q.agreement*100
    for col in ['model_fit','compression','downstream']:q[col+'_ms']=q[col+'_seconds']*1000
    cols=['ari_quantized_full','agreement_quantized_full_percent','ari','agreement_float_full_percent','r_phi',
          'absolute_original_cost_error_percent','model_fit_ms','compression_ms','downstream_ms','n_audit','d','stored_columns']
    means=q.groupby(['dataset','k','method'])[cols].mean()
    spread=q.groupby(['dataset','k','method'])[cols].std().add_suffix('_std_across_starts')
    summary=means.join(spread).reset_index()
    summary.to_csv(out/'commitjl_vs_pca_by_dataset_k.csv',index=False)
    gg=geometry[geometry.method.isin(['pca','protected_tail'])].groupby(['dataset','k','method'])[['q95','mean_error','exceed_015','pairs']].mean()
    gg.reset_index().to_csv(out/'commitjl_vs_pca_geometry.csv',index=False)
    lines=['# CommitJL versus PCA-16: every dataset and center count','',
           'This report uses the completed native part of the 17 October bound-audit cohort. '
           'It does not include partial proof-campaign timings. Ours is protected_tail: head + energy + sparse residual JL.',
           '', 'Both representations store 16 values per point, excluding shared centers/model parameters. '
           'At full center rank, ours allocates 4+1+11 at k=5 and 9+1+6 at k=10; PCA stores 16 projected coordinates.',
           'All cells average three clustering starts. Centers are trained on 60% of each corpus and the remaining 40% is evaluated. '
           'ARI/agreement below compare with Full on the same quantized inputs; they do not measure agreement with class labels. '
           'Values against floating Full are exported separately because source quantization changes some text assignments.',
           '', 'Lower values are better for time and distortion. ARI/agreement are better when higher. '
           'R_phi is original-space cost using the method\'s assignments divided by Full\'s optimal fixed-center assignment cost. '
           'Reported-cost error instead compares the method\'s reported cost with floating original-space cost; '
           'PCA proves a projected-space objective, so this discrepancy does not invalidate its proof.',
           '', 'All timing values are milliseconds for the entire held-out batch. Parameter construction excludes shared KMeans training. '
           'Compression excludes parameter construction. Downstream answering uses already-compressed data. '
           'These short native timings are descriptive; differences near timer/system noise are not established speed advantages.']
    ds=json.loads((root/'datasets.json').read_text())
    lines+=['','## Dataset coverage','',table(['Dataset','Total rows','Original dimensions','Held-out rows'],
        [[LABELS[x['dataset']],x['n'],x['d'],int(means.loc[(x['dataset'],5,'pca'),'n_audit'])] for x in ds])]
    allpairs=[]
    for k in [5,10]:
        accuracy=[];timing=[];cost=[];geo=[];floats=[]
        for name in ORDER:
            p=means.loc[name,k,'pca'];o=means.loc[name,k,'protected_tail'];gp=gg.loc[name,k,'pca'];go=gg.loc[name,k,'protected_tail']
            accuracy.append([LABELS[name],f'{p.ari_quantized_full:.6f}',f'{o.ari_quantized_full:.6f}',
                f'{p.agreement_quantized_full_percent:.3f}%',f'{o.agreement_quantized_full_percent:.3f}%'])
            timing.append([LABELS[name]]+[f'{x:.3f}' for x in [p.model_fit_ms,o.model_fit_ms,p.compression_ms,o.compression_ms,p.downstream_ms,o.downstream_ms]])
            cost.append([LABELS[name],f'{p.r_phi:.9f}',f'{o.r_phi:.9f}',f'{p.absolute_original_cost_error_percent:.6f}%',f'{o.absolute_original_cost_error_percent:.6f}%'])
            geo.append([LABELS[name],f'{gp.q95*100:.2f}%',f'{go.q95*100:.2f}%',f'{gp.exceed_015*100:.2f}%',f'{go.exceed_015*100:.2f}%'])
            floats.append([LABELS[name],f'{p.ari:.6f}',f'{o.ari:.6f}',f'{p.agreement_float_full_percent:.3f}%',f'{o.agreement_float_full_percent:.3f}%'])
            allpairs.append(dict(dataset=name,k=k,ari_gain=o.ari_quantized_full-p.ari_quantized_full,
                parameter_speedup=p.model_fit_ms/o.model_fit_ms,compression_speedup=p.compression_ms/o.compression_ms,
                downstream_speedup=p.downstream_ms/o.downstream_ms,
                construct_and_compress_speedup=(p.model_fit_ms+p.compression_ms)/(o.model_fit_ms+o.compression_ms)))
        lines+=['',f'## k={k}: assignment quality against quantized Full','',
                table(['Dataset','PCA ARI','Ours ARI','PCA agreement','Ours agreement'],accuracy),
                '',f'## k={k}: native time in milliseconds','',
                table(['Dataset','PCA construct','Ours construct','PCA compress','Ours compress','PCA answer','Ours answer'],timing),
                '',f'## k={k}: original-space cost diagnostics','',
                table(['Dataset','PCA R_phi','Ours R_phi','PCA reported-cost abs. error','Ours reported-cost abs. error'],cost),
                '',f'## k={k}: pair-distance geometry','',
                'Numbers average the per-start statistics over reproducible maps, with about 1000 sampled pairs per start. '
                'Energy is excluded from Euclidean coordinates: ours therefore uses 15 Euclidean coordinates within its 16-value storage budget. '
                'Proof bundles use separate post-commit maps; these geometry rows do not evaluate those exact maps.',
                '',table(['Dataset','PCA q95 error','Ours q95 error','PCA pairs >15% error','Ours pairs >15% error'],geo),
                '',f'## k={k}: assignment quality against floating original Full','',
                table(['Dataset','PCA ARI','Ours ARI','PCA agreement','Ours agreement'],floats)]
    pairs=pd.DataFrame(allpairs);pairs.to_csv(out/'paired_speedups.csv',index=False)
    wins=dict(cells=len(pairs),ari_higher=int((pairs.ari_gain>1e-12).sum()),ari_ties=int((pairs.ari_gain.abs()<=1e-12).sum()),
              parameter_construction_faster=int((pairs.parameter_speedup>1).sum()),compression_faster=int((pairs.compression_speedup>1).sum()),
              downstream_faster=int((pairs.downstream_speedup>1).sum()))
    (out/'comparison_summary.json').write_text(json.dumps(wins,indent=2))
    lines+=['','## Interpretation','',
        f'Across {wins["cells"]} dataset/k cells, ours has higher mean ARI in {wins["ari_higher"]}, ties in {wins["ari_ties"]}, '
        f'faster parameter construction in {wins["parameter_construction_faster"]}, faster compression application in {wins["compression_faster"]}, '
        f'and faster downstream answering in {wins["downstream_faster"]}. Timing counts describe observed means, not significance tests.',
        'The JL tail is unused for these fixed-center assignments/costs; its role is the separate pair-distance experiment. '
        'Small original-space R_phi penalties show that assignment disagreements can have little effect on total original-space cost.',
        'This is a PCA comparison, not evidence of superiority to the exact affine baseline. Actual proof, setup and complete-workload timings are a separate experiment.',
        '', 'CSV exports include unrounded means and standard deviations across starts. Raw native CSV files and the snapshot receipt are preserved one directory above.']
    (out/'commitjl_vs_pca_complete.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(wins));print(str((out/'commitjl_vs_pca_complete.md').resolve()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('results',type=Path)
    compare(parser.parse_args().results)
