"""Validate and consolidate completed research stages into a reviewable report."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
from experiments.protected_jl_study import summarize,DATASETS,bootstrap_crossed
from tools.rebuild_protected_jl import rebuild

CODE=Path(__file__).resolve().parents[1]
RESULTS=CODE/'results'
OUT=RESULTS/'protected_jl_20261010'


def table(frame):
    # Keep offline report generation independent of pandas' optional tabulate.
    def cell(value):
        if isinstance(value,(float,np.floating)):
            return f'{value:.4f}'
        return str(value).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(map(cell,frame.columns))+' |',
           '| '+' | '.join(['---']*len(frame.columns))+' |']
    lines.extend('| '+' | '.join(map(cell,row))+' |' for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def add_ablation():
    summary=pd.read_csv(OUT/'native/summary.csv')
    entries=[]
    for dataset in DATASETS:
        part=summary[(summary.dataset==dataset)&(summary.k==5)]
        row=dict(dataset=dataset)
        for method in ['sparse','hybrid_r1','hybrid_r2','hybrid_r4']:
            selected=part[(part.method==method)&(part.residual_m==16)&(part.q==1)].iloc[0]
            row[method+'_ari']=selected.ari_mean
        head=part[(part.method=='protected_only')&(part.protected_rank==4)].iloc[0]
        row['protected_only_ari']=head.ari_mean
        row['protected_only_distortion_q95']=head.distortion_q95_mean
        entries.append(row)
    ablation=pd.DataFrame(entries)
    ablation.to_csv(OUT/'protected_rank_ablation.csv',index=False)
    heading='## Protected-rank ablation'
    report=OUT/'report.md';contents=report.read_text(encoding='utf-8')
    if heading not in contents:
        text=heading+'\n\n'+table(ablation)+'\n\n'
        text+='Random variants all use residual m=16,q=1; total dimensions are 16,17,18,20 respectively. Protected-only has four dimensions and no random residual. These comparisons isolate the effect of additional protection but are not equal-dimension competitions. Its large sampled distance distortion despite perfect label fidelity shows why assignment quality alone is insufficient to justify geometry preservation.\n\n'
        contents=contents.replace('## Exact-integer fidelity on full current paper subsets',text+'## Exact-integer fidelity on full current paper subsets')
        report.write_text(contents,encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    native=OUT/'native'
    if not native.exists():
        native.mkdir()
        primary=RESULTS/'protected_jl_20261010_native_v3'
        text=RESULTS/'protected_jl_20261010_native_text'
        if not json.loads((text/'manifest.json').read_text())['complete']:raise ValueError('Text stage incomplete')
        frames=[pd.read_csv(primary/'rows.csv').query("dataset != '20ng_tfidf'"),pd.read_csv(text/'rows.csv')]
        all_rows=pd.concat(frames,ignore_index=True)
        if set(all_rows.dataset)!=set(DATASETS):raise ValueError('Missing dataset')
        key=['dataset','k','method','protected_rank','residual_m','q','total_m']
        for _,g in all_rows.groupby(key):
            deterministic=g.method.iloc[0] in ['pca','protected_only']
            expected=5 if deterministic else 25
            if len(g)!=expected or g.duplicated(['start','seed']).any():raise ValueError('Incomplete cell')
            if set(g.start)!=set(range(5)):raise ValueError('Missing original start')
            if set(g.seed)!=({-1} if deterministic else set(range(5))):raise ValueError('Missing map seed')
        all_rows.to_csv(native/'rows.csv',index=False)
        for dataset in DATASETS:
            source=text if dataset=='20ng_tfidf' else primary
            shutil.copytree(source/'bases'/dataset,native/'bases'/dataset)
            shutil.copy2(source/f'{dataset}_labels.u16.gz',native/f'{dataset}_labels.u16.gz')
            g=all_rows[all_rows.dataset==dataset]
            n=int(g.n.iloc[0]);offsets=g.labels_byte_offset.to_numpy()
            if not np.array_equal(offsets,np.arange(len(g))*2*n):raise ValueError('Label offsets invalid')
            with gzip.open(native/f'{dataset}_labels.u16.gz','rb') as f: size=len(f.read())
            if size!=len(g)*2*n:raise ValueError('Assignment stream length mismatch')
        (native/'manifest.json').write_text(json.dumps(dict(complete=True,rows=len(all_rows),
            configuration=dict(seeds=5,starts=5,k=[5,10],m=[16,32,64,128],q=[1,2]),
            source_stages=[str(primary),str(text)],
            note='First stage reached 1800s budget after seven complete datasets. The entire text dataset was completed separately with identical float computations; partial text from the first stage is excluded.'),indent=2))
    if not (native/'summary.csv').exists() or not (native/'quality.png').exists():
        summarize(native,500)
    raw=pd.read_csv(native/'rows.csv');summary=pd.read_csv(native/'summary.csv')
    matches=[]
    for dataset in DATASETS:
        for method in ['pca','sparse','hybrid_r1','hybrid_r2','hybrid_r4','protected_only']:
            candidate=raw[(raw.dataset==dataset)&(raw.k==5)&(raw.method==method)]
            if method=='protected_only':candidate=candidate[candidate.protected_rank==4]
            elif method!='pca':candidate=candidate[(candidate.residual_m==16)&(candidate.q==1)]
            else:candidate=candidate[candidate.total_m==16]
            if len(candidate):
                row_index=int(candidate.index[0]);rebuild(native,row_index)
                matches.append(dict(dataset=dataset,method=method,row_index=row_index,labels_match=True))
    (OUT/'replay_validation.json').write_text(json.dumps(matches,indent=2))
    matched=pd.read_csv(RESULTS/'protected_jl_20261010_matched_pca/rows.csv')
    quality=[]
    for dataset in DATASETS:
        original=summary[(summary.dataset==dataset)&(summary.k==5)&(summary.method=='sparse')&(summary.residual_m==16)&(summary.q==1)].iloc[0]
        hybrid=summary[(summary.dataset==dataset)&(summary.k==5)&(summary.method=='hybrid_r4')&(summary.residual_m==16)&(summary.q==1)].iloc[0]
        control=matched[(matched.dataset==dataset)&(matched.k==5)]
        pca=float(control.ari.mean())
        quality.append(dict(dataset=dataset,total_m=20,pca_ari=pca,hybrid_ari=hybrid.ari_mean,
                            sparse16_ari=original.ari_mean,hybrid_distortion_q95=hybrid.distortion_q95_mean))
    quality=pd.DataFrame(quality);quality.to_csv(OUT/'quality_matched_dimensions.csv',index=False)
    precision=pd.read_csv(RESULTS/'protected_jl_20261010_precision_full_final/rows.csv')
    physical=[]
    for dataset in DATASETS:
        h=precision[(precision.dataset==dataset)&(precision.method=='integer_hybrid')&(precision.scale==4096)]
        p=precision[(precision.dataset==dataset)&(precision.method=='integer_pca20')&(precision.scale==4096)]
        means={}
        for metric in ['ari_original','r_phi']:
            matrix=h.pivot(index='seed',columns='start',values=metric).to_numpy()
            means[metric]=bootstrap_crossed(matrix,1000)
        physical.append(dict(dataset=dataset,n=int(h.n.iloc[0]),pca_ari=float(p.ari_original.mean()),
            hybrid_ari=means['ari_original'][0],hybrid_ari_ci_low=means['ari_original'][1],
            hybrid_ari_ci_high=means['ari_original'][2],r_phi=means['r_phi'][0],r_phi_ci_high=means['r_phi'][2]))
    physical=pd.DataFrame(physical);physical.to_csv(OUT/'physical_quality.csv',index=False)
    compile_data=pd.read_csv(RESULTS/'protected_jl_20261010_compile_final/timings.csv')
    routing=pd.read_csv(RESULTS/'protected_jl_20261010_routing_scaling/timings.csv')
    timings=pd.read_csv(RESULTS/'protected_jl_20261010_timed/timings.csv')
    routing_rows=[]
    for shape,g in routing.groupby('shape'):
        old=int(g[g.method=='sparse_shared_scores'].constraints.iloc[0]);new=int(g[g.method=='sparse_lookup_scores'].constraints.iloc[0])
        routing_rows.append(dict(shape=shape,shared_constraints=old,lookup_constraints=new,reduction_pct=100*(1-new/old)))
    routing_rows=pd.DataFrame(routing_rows);routing_rows.to_csv(OUT/'routing_savings.csv',index=False)
    for folder in ['protected_jl_20261010_compile_final','protected_jl_20261010_routing_scaling','protected_jl_20261010_backend_plonk',
                   'protected_jl_20261010_lookup_groth16','protected_jl_20261010_high_precision_proofs','protected_jl_20261010_timed']:
        manifest=json.loads((RESULTS/folder/'manifest.json').read_text())
        if not manifest['complete']:raise ValueError('Incomplete required stage: '+folder)
        if not pd.read_csv(RESULTS/folder/'timings.csv').passed.all():raise ValueError('Failed stage: '+folder)
    parts=[
        '# Protected-coordinate Sparse JL: research evaluation',
        'The hybrid substantially improves fixed-centroid fidelity. Simultaneous superiority over optimized PCA is **not established**. The strongest quality variant protects the complete centroid-difference span, and the protected-only ablation also preserves labels. JL therefore needs a separate geometry/objective justification.',
        '## Evaluation scope',
        f'{len(raw):,} native rows cover all eight exact server subsets, k=5/10, five original center initializations, five fresh maps, residual m=16/32/64/128 and q=1/2. No dataset was resampled. Gaussian easy 2000x256, hard 2000x512, DBSCAN margin 1000x128, MNIST/Fashion 5000x784 each, Gisette 3000x5000, Arcene 100x10000, and 20NG 3000x2000. Models are fixed-centroid assignment, not projected KMeans training or clinical ground truth.',
        'The initial 30-minute native budget completed seven corpora; text was completed in a separate bounded stage. The validated combined dataset excludes its interrupted partial stage. All archived inputs, bases and lossless labels are local; a replay check passed for '+str(len(matches))+' selected method/dataset cells.',
        '## Floating-point fidelity: matched total dimension',
        'Both PCA and full-rank hybrid have 20 output dimensions for k=5. Hybrid uses four protected and sixteen random residual coordinates. Sparse16 is an additional reference, not dimension-matched. ARI is agreement with the saved Full labels. Q95 describes sampled pairwise distance distortion, not a global JL guarantee.',
        table(quality),
        '![Native quality curves](native/quality.png)',
        'Protecting only one or two directions often improves on Sparse JL but does not consistently reach PCA. Full-rank protection preserves the fixed-center decision in exact arithmetic; the protected-only control shows that the random residual is not responsible for this perfect ARI.',
        '## Exact-integer fidelity on full current paper subsets',
        'These controls use every row of each current paper subset, k=5, two saved starts, three fresh maps, residual m=16,q=1, coefficient scale T=4096 and a common per-dataset input scaling into [-4096,4096]. PCA has the same 20 dimensions and coefficient scale. Intervals are pointwise crossed bootstrap on just three maps and two starts; they are exploratory. The integer Full control separates input rounding from basis rounding. Results at T=1024 are retained in raw CSV.',
        table(physical),
        'These numbers do not establish k=10 integer fidelity on full subsets, independent-test-set generalization, or full-corpus proof performance. The smaller initial precision screen covers k=5/10 on only min(128,n) rows.',
        '## Circuit feasibility and actual proof timings',
        'All '+str(len(compile_data))+' final solver/compile controls passed; these generated zero proofs. Real Groth16 and PLONK controls also passed. Solver-only diagnostics use a test-only commitment substitution; real proof backends use their actual commitment computation. PLONK uses an unsafe test SRS.',
        'The following measured run uses fixed synthetic n=4,d=16,m=8,k=3,q=2, five timed Groth16 proofs plus one warm-up per configuration, four Go threads and a 4 GiB process limit. PCA has two dimensions; hybrids have additional protected coordinates. This is an implementation comparison, not a fidelity-selected corpus comparison or statistical PCA-beating claim.',
        table(timings[['method','constraints','prove_median','setup_seconds','single_run_total','peak_rss_gib']]),
        '## Generic sparse routing',
        table(routing_rows),
        'Lookup routing reduces constraints for larger m, but costs more for tiny m. Fewer constraints alone do not guarantee faster proving. Compile-time specialized routing retains authentication, but every changed row pattern needs new compilation/setup/key; this cost is not free.',
        '## Decision and remaining work',
        'Retain the hybrid and lookup paths as opt-in research variants. Do not replace existing paper claims or label the old server results as optimized. The quality gain is promising; the current proof measurements do not justify a joint speed/quality win over PCA. Confirm the candidate on an independent split, test k=10 integer fidelity on full subsets, and benchmark fidelity-qualified real proof shapes on sufficient RAM. The adaptive basis and lookup permutation argument require an explicit composed security analysis and external review before deployment.',
        'The original commit-then-fresh-map idea is retained, but the hybrid basis is public/fixed in its circuit. A production verifier must validate and fix it from the declared public centroids before the challenge, pin the verification relation/key, and authenticate the fresh A. The experimental CLI does not perform the full deployment protocol.',
        '## Artifacts and reproduction',
        'See [implementation notes](../../docs/protected_jl_research.md), `native/rows.csv`, `native/summary.csv`, lossless label streams, saved bases, `quality_matched_dimensions.csv`, `physical_quality.csv`, `routing_savings.csv`, and the source stage manifests. `tools.rebuild_protected_jl` reconstructs Z/centroids/labels for any native row without the server. Proof stages retain complete instance JSON and raw timing/verification records.',
    ]
    (OUT/'report.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    add_ablation()
    (OUT/'validation.json').write_text(json.dumps(dict(complete=True,native_rows=len(raw),native_datasets=DATASETS,
        replay_checks=len(matches),compile_configs=len(compile_data),timed_proofs=int(timings.successful_proofs.sum()),
        physical_full_subset_rows=len(precision),limitations='Exploratory; no joint PCA superiority claim'),indent=2))
    print(OUT/'report.md')

if __name__=='__main__':main()
