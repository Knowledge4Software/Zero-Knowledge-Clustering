"""Update result tables without rewriting the original manuscript's prose."""
from pathlib import Path
import hashlib,json,re,shutil
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'paper_results/bound_revision'
OUT=ROOT/'paper_results'
DATA=ROOT/'code/results/final_paper_review_1792295389'


def table(path,caption,label,headers,rows,notes):
    s=[r'\begin{table*}[tp]',r'\caption{'+caption+r'}\label{'+label+'}',r'\centering\small',
       r'\begin{adjustbox}{max width=\textwidth}',r'\begin{tabular}{@{}'+'l'+'r'*(len(headers)-1)+r'@{}}',
       r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
    s+=[' & '.join(map(str,row))+r' \\' for row in rows]
    s+=[r'\bottomrule',r'\end{tabular}',r'\end{adjustbox}',
        r'\par\vspace{3pt}\begin{minipage}{0.98\textwidth}\footnotesize '+notes+r'\end{minipage}',r'\end{table*}']
    (OUT/path).write_text('\n'.join(s)+'\n',encoding='utf-8')


def main():
    aliases={'datasets':('datasets','tab:datasets-a'), 'fidelity':('capped_accuracy','tab:fidelity-a'),
             'proof':('query8','tab:zkperf-a'),'ablation':('ablation','tab:ablation-a')}
    for dest,(source,label) in aliases.items():
        s=(SRC/(source+'.tex')).read_text(encoding='utf-8')
        s=re.sub(r'\\label\{[^}]+\}',lambda _:r'\label{'+label+'}',s,count=1)
        s=s.replace('$q=','$b=').replace('q=','b=')
        (OUT/(dest+'.tex')).write_text(s,encoding='utf-8')
    pq=pd.read_csv(DATA/'main/analysis/proof_instance_quality.csv')
    pq=pq[(pq.q>=8)&pq.method.isin(['pca','protected_energy','protected_tail','affine'])]
    rows=[];names={'pca':'PCA-16','protected_energy':'Head + energy','protected_tail':'CommitJL-16','affine':'Exact affine'}
    for method in names:
        z=pq[pq.method==method];assert len(z)==13
        rows.append([names[method],13,f'{100*z.agreement_mean.mean():.3f}',
                     f'{100*z.claimed_original_cost_error_median.median():.6f}',
                     f'{100*z.claimed_original_cost_abs_error_max.max():.6f}'])
    table('cost_error.tex','Original-space diagnostics for matched actual-proof configurations.','tab:cost-error',
          ['Method','Configurations','Mean agree. (\\%)','Median signed cost error (\\%)','Max abs. cost error (\\%)'],rows,
          'Floating Full reference; 13 matched catalog/query-size configurations per method with batch size at least eight. '
          'Agreement is averaged within each configuration before aggregation. The signed-error column is the median of per-configuration signed medians; '
          'the maximum uses every recorded subset. It is not a mean absolute error. One-record ARI is uninformative. Assignment diagnostics are private harness metadata.')
    q=pd.read_csv(DATA/'corrected_handoff/main_actual_queries.csv')
    e=pd.read_csv(DATA/'completion/full_budget_control/full_q8_controls.csv')
    rows=[]
    for ds,k in [('gaussian_hard',5),('gaussian_hard',10)]:
        a=q[(q.dataset==ds)&(q.k==k)&(q.method=='full')&(q.q==8)].iloc[0]
        rows.append(['Gaussian-hard',k,32,8,f'{a.constraints:,}',f'{a.prove_median_seconds:.3f}'])
    for a in e.itertuples():
        rows.append([{'mnist':'MNIST','fashion_mnist':'Fashion-MNIST'}[a.dataset],a.k,a.n_catalog,a.q,
                     f'{a.constraints:,}',f'{a.prove_median_seconds:.3f}'])
    table('earlier_full.tex','Full-dimensional controls in the same completed Poseidon2 cohort.','tab:earlier-full',
          ['Corpus','$k$','Catalog $N$','Batch $b$','Constraints','Proving (s)'],rows,
          'All rows are actual independently verified proofs, not projected constraint estimates. '
          'Image controls use two distinct subsets at a 1.5M ceiling; Gaussian controls use eight. '
          'Other larger Full shapes remain unmeasured. A preflight exclusion is not an observed out-of-memory failure.')
    cp=pd.read_csv(DATA/'completion/campaign/proofs.csv');large=cp[cp.large&(cp.status=='complete')]
    large[large.phase.str.match(r'query_\d+$')].to_csv(OUT/'scaling_actual_queries.csv',index=False)
    rows=[]
    for b in [1,8,32,64]:
        for method in ['pca','protected_energy','protected_tail','affine','cached']:
            a=large[(large.variant==method)&(large.phase==f'query_{b}')].iloc[0]
            rows.append([b,dict(names,cached='Cached exact')[method],f'{a.constraints:,}',f'{a.prove_median_seconds:.3f}'])
    table('scaling.tex','Measured query-size scaling on the 64-record MNIST catalog, $k=5$.','tab:sensitivity-a',
          ['Batch $b$','Method','Constraints','Proving time (s)'],rows,
          'Actual proofs at every displayed size, with three timed repetitions after warmup. '
          'Batches 8 and 32 have four genuinely different subsets; batch one has two; the full-catalog batch has one. '
          'Setup and Stage 1 are excluded. Full is measured only at batch one for this catalog.')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for method,color,marker in [('pca','#285c9b','o'),('protected_energy','#168274','s'),
                                 ('protected_tail','#43a99c','^'),('affine','#79619b','D'),('cached','#627282','x')]:
        z=large[(large.variant==method)&large.phase.str.match(r'query_\d+$')].copy()
        z['batch']=z.phase.str.removeprefix('query_').astype(int);z=z.sort_values('batch')
        axes[0].plot(z.batch,z.constraints/1000,marker=marker,color=color,label=dict(names,cached='Cached exact')[method])
        axes[1].plot(z.batch,z.prove_median_seconds,marker=marker,color=color)
    axes[0].set_ylabel('Constraints (thousands)');axes[1].set_ylabel('Proving time (s)')
    for ax in axes:ax.set_xlabel('Requested records per query');ax.set_xticks([1,8,32,64]);ax.grid(alpha=.17)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=[0,.18,1,1]);fig.savefig(OUT/'figures/scaling.pdf',bbox_inches='tight');plt.close(fig)
    # Keep archived qualification labels/data, explicitly identify their cohort.
    old=ROOT/'tmp/manuscript_before_bound_revision_20261018/paper_results'
    s=(old/'eligibility.tex').read_text(encoding='utf-8').replace('[!htbp]','[tp]')
    (OUT/'eligibility.tex').write_text(s,encoding='utf-8')
    source_manifest=json.loads((SRC/'manifest.json').read_text())
    record={'revision':'targeted update preserving original manuscript structure','current_cohort':'bound-audit-v1',
            'table_exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scaling_figure_sha256':hashlib.sha256((OUT/'figures/scaling.pdf').read_bytes()).hexdigest(),
            'current_numeric_manifest':'paper_results/bound_revision/manifest.json',
            'current_manifest_sha256':hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest(),
            'archived_tables':['eligibility.tex','ordering.tex','attacks.tex','certificates.tex'],
            'source_inputs':source_manifest['inputs'],
            'updated_tables':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in
                              ['datasets.tex','fidelity.tex','proof.tex','cost_error.tex','earlier_full.tex','scaling.tex','ablation.tex']}}
    (OUT/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Updated seven result tables and scaling figure; archived evidence kept separate.')


if __name__=='__main__':main()
