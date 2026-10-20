"""Explain identical-map speed optimization and the separate new-draw replication."""
import argparse
import json
from pathlib import Path

import pandas as pd

from commitjl.reusable_native import file_hash
from experiments.laptop_comparison import atomic_json


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
        '| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


def report(root, benchmark):
    manifest=json.loads((root/'manifest.json').read_text())
    review=json.loads((root/'review_manifest.json').read_text())
    bench=json.loads(benchmark.read_text())
    replay_path=root/'preserved_previous_maps.json'
    replay=json.loads(replay_path.read_text())
    assert manifest['complete'] and review['complete'] and bench['complete']
    assert replay['complete'] and replay['all_csr_arrays_and_fingerprints_bitwise_equal']
    for name,digest in review['files'].items():
        assert file_hash(root/name)==digest, name
    assert all(r['records_csr_fingerprint_and_projection_bitwise_equal'] for r in bench['rows'])
    code=Path(__file__).resolve().parents[1]
    for name,digest in bench['source_hashes'].items():
        assert file_hash(code/name)==digest, name
    rows=pd.read_csv(root/'rows.csv')
    summary=pd.read_csv(root/'summary.csv')
    comp=pd.read_csv(root/'reviewed_comparisons.csv')
    config=manifest['identity']['configuration']
    stages=rows.groupby(['dataset','method','m'],as_index=False)[[
        'model_fit_seconds','construction_seconds','compression_seconds',
        'fresh_seconds','cold_fresh_seconds']].mean()
    stages.to_csv(root/'stage_means.csv',index=False)
    pairs=[]
    ours_rows=rows[rows.method.eq('protected_plus12')]
    for competitor in ['pca_full','pca_randomized','pca_randomized_fast','protected_jl']:
        base=rows[rows.method.eq(competitor)]
        if competitor=='pca_full':
            matched=ours_rows.merge(base.drop(columns='seed'),on=['dataset','m','start'],
                suffixes=('_ours','_base'),validate='many_to_one')
        else:
            matched=ours_rows.merge(base,on=['dataset','m','start','seed'],
                suffixes=('_ours','_base'),validate='one_to_one')
        for _,r in matched.iterrows():
            pairs.append(dict(dataset=r.dataset,m=r.m,start=r.start,seed=r.seed,competitor=competitor,
                ari_ours=r.fresh_ari_ours,ari_base=r.fresh_ari_base,
                ari_difference=r.fresh_ari_ours-r.fresh_ari_base,
                cold_seconds_ours=r.cold_fresh_seconds_ours,cold_seconds_base=r.cold_fresh_seconds_base,
                cold_speedup=r.cold_fresh_seconds_base/r.cold_fresh_seconds_ours,
                test_ari_ours=r.test_fresh_ari_ours,test_ari_base=r.test_fresh_ari_base,
                knn_ours=r.knn_recall_ours,knn_base=r.knn_recall_base,
                pair_q95_ours=r.pair_q95_ours,pair_q95_base=r.pair_q95_base))
    pd.DataFrame(pairs).to_csv(root/'paired_cases.csv',index=False)
    focused=[]
    primary=[]
    for dataset in config['datasets']:
        for m in config['m']:
            part=summary[summary.dataset.eq(dataset)&summary.m.eq(m)]
            ours=part[part.method.eq('protected_plus12')].iloc[0]
            fast=part[part.method.eq('pca_randomized_fast')].iloc[0]
            c=comp[comp.dataset.eq(dataset)&comp.m.eq(m)&comp.competitor.eq('pca_randomized_fast')].iloc[0]
            focused.append([dataset,m,f'{fast.fresh_ari_mean:.4f}',f'{ours.fresh_ari_mean:.4f}',
                f'{fast.cold_fresh_seconds_mean:.4f}',f'{ours.cold_fresh_seconds_mean:.4f}',
                f'{c.cold_fresh_speedup_mean:.2f}',str(c.ci_speed_and_fresh_ari_win)])
            primary.append([dataset,m,f'{c.fresh_ari_difference_mean:+.4f}',
                f'[{c.fresh_ari_difference_ci_low:+.4f}, {c.fresh_ari_difference_ci_high:+.4f}]',
                f'[{c.cold_fresh_speedup_ci_low:.2f}, {c.cold_fresh_speedup_ci_high:.2f}]'])
    btable=[[r['d'],r['residual_m'],r['q'],f'{r["before_median_seconds"]:.4f}',
        f'{r["after_median_seconds"]:.4f}',f'{r["speedup"]:.2f}'] for r in bench['rows']]
    select=stages[stages.method.isin(['protected_plus12','pca_randomized_fast'])]
    stagelist=[[r.dataset,r.method,r.m,f'{r.model_fit_seconds:.4f}',f'{r.construction_seconds:.4f}',
        f'{r.compression_seconds:.4f}',f'{r.fresh_seconds:.4f}',f'{r.cold_fresh_seconds:.4f}'] for r in select.itertuples()]
    geo=summary[summary.method.isin(['protected_plus12','pca_randomized_fast'])]
    geolist=[[r.dataset,r.method,r.m,f'{r.knn_recall_mean:.4f}',f'{r.pair_q95_mean:.4f}',
        f'{r.original_partition_sse_ratio_mean:.4f}'] for r in geo.itertuples()]
    joint=comp[comp.competitor.eq('pca_randomized_fast')&comp.ci_speed_and_fresh_ari_win.eq(True)]
    named=', '.join(f'{r.dataset} at m={r.m}' for r in joint.itertuples()) or 'none'
    lines=['# Identical-map optimization and fresh-seed results', '',
        f'Completed {manifest["completed_cases"]} cases in {manifest["elapsed_seconds"]/60:.1f} minutes. '+
        f'{config["seeds"]} new draw IDs ({config["seed_offset"]}--{config["seed_offset"]+config["seeds"]-1}) '+
        f'crossed with {config["starts"]} new center/clustering start IDs ({config["start_offset"]}--{config["start_offset"]+config["starts"]-1}). '+
        'All four saved corpus subsets and both target dimensions are retained. The new results are not pooled with the previous screen.', '',
        '## What changed', '',
        'The SHAKE stream previously expanded 4096 bytes before returning its first four-byte word. '+
        'It now expands a 64-byte prefix first and grows refill size up to 4096 bytes as needed. '+
        'The unbounded XOF bytes, rejection decisions, independent sign domain, sparse matrix and model outputs are identical. '+
        'This implementation change affects construction time, not the algorithm or its numerical quality. '+
        'No new model, random family, direction count, or dataset subset was selected.', '',
        f'All {replay["hybrid_maps_checked"]} hybrid maps from the previous completed screen were also regenerated '+
        'from their original seed payloads. Every CSR array and complete model fingerprint matches bitwise. '+
        'The old results and reports were preserved.', '',
        '## Paired implementation benchmark', '',
        'Seven identical-seed old/new comparisons per configuration, with excluded warmup and alternating order. '+
        'Both construct the same CSR map and complete NativeProjection with a fixed rank-16 learned basis. '+
        'Every record, CSR array, model fingerprint and projected output is checked bitwise. '+
        'These timings exclude model fitting and corpus conversion; the next section includes them.', '',
        table(['Input d','JL tail rows','q','Before s','After s','Construction speedup'],btable), '',
        'The high-density q=65 control is retained even when the adaptive expansion offers no improvement. '+
        'No claim of universal speedup over all q is made.', '',
        '## Complete workload versus fast dedicated-rank PCA', '',
        'The method is the original centroid span plus 12 training-residual PCA directions and Sparse JL of its orthogonal residual. '+
        'Its total output width is 128 (16 protected + 112 JL) or 256 (16 protected + 240 JL). '+
        'The fast PCA control is fitted separately at exactly 128 or 256 components with two power iterations. '+
        'Full-SVD and default randomized PCA are also retained in the detailed comparisons.', '',
        '**Total time includes learning, map construction, compression and one fresh whole-corpus KMeans run.** '+
        'Fresh ARI measures agreement with matched Full clustering, not class-label accuracy. '+
        'Speedup is the mean of paired PCA/method ratios, not the ratio of rounded table means.', '',
        table(['Dataset','m','PCA ARI','Our ARI','PCA total s','Our total s','Speedup','Both lower CIs pass'],focused), '',
        '## Uncertainty and joint evidence', '',
        table(['Dataset','m','ARI difference','ARI difference 95% CI','Speedup 95% CI'],primary), '',
        f'Both pointwise lower bounds favor the method against fast PCA in {len(joint)} of '+
        f'{len(config["datasets"])*len(config["m"])} cells: {named}. '+
        'These intervals resample the new draw/start axes conditional on the same data, with no multiplicity correction. '+
        'They are exploratory replication evidence, not an independent-data or general superiority claim. '+
        'The small number of draws/starts limits precision, and nonlinear KMeans can reach different partitions after projection.', '',
        'The previous and current ARI means can differ because new centers, projection draws and clustering starts were used. '+
        'Those changes must not be attributed to the byte-identical implementation optimization. '+
        'The paired implementation benchmark above isolates that optimization.', '',
        '## Where total time goes', '',
        table(['Dataset','Method','m','Learning s','Construction s','Compression s','Fresh KMeans s','Total s'],stagelist), '',
        'A cached fit is charged its full measured cost in every cold use. Totals sum stage medians; '+
        'archive reading, Python startup and result-file writes are excluded consistently. '+
        'Four CPU threads, float64, equal original-point initialization IDs, bounded iterations and randomized method order are shared.', '',
        '## Geometry and partition cost', '',
        table(['Dataset','Method','m','5-NN recall (higher)','Pair squared-distance Q95 error (lower)',
               'Original-space partition SSE / Full (near 1)'],geolist), '',
        'Pair error summarizes the same saved sampled test pairs; it is not a worst-case all-pairs guarantee. '+
        'Partition SSE uses the learned labels evaluated in the original space. '+
        'Fixed-center ARI=1 is a structural property and cannot establish success on fresh clustering. '+
        'A numerical sketch benchmark does not demonstrate cryptographic proof speed or privacy of a training-derived basis.', '',
        '## Reproducibility', '',
        '- `protocol.md`: plan written before the replication.',
        '- `manifest.json`: frozen configuration, source hashes and completion status.',
        '- `rows/`, `artifacts/`, `datasets/`: per-case measurements, labels, model parameters and fit records.',
        '- `audit.json`: artifact integrity, dimension/cost checks and saved-result numerical replay.',
        '- `preserved_previous_maps.json`: bitwise regeneration of every previous hybrid map.',
        '- `reviewed_comparisons.csv`: paired comparisons against every PCA control and original Protected JL.',
        '- `stage_means.csv`: separate cost components for every method.',
        '- `paired_cases.csv`: matched per-draw/start quality and speed values for scatter plots and stability analysis.',
        '- `index.html`: complete method summaries and figures.', '']
    destination=root/'optimization_findings.md'
    destination.write_text('\n'.join(lines),encoding='utf-8')
    atomic_json(root/'optimization_report_manifest.json',dict(complete=True,
        builder_sha256=file_hash(Path(__file__)),benchmark=str(benchmark.resolve()),
        benchmark_sha256=file_hash(benchmark),inputs=dict(review_manifest=file_hash(root/'review_manifest.json')),
        previous_map_replay_sha256=file_hash(replay_path),
        outputs={p.name:file_hash(p) for p in [destination,root/'stage_means.csv',root/'paired_cases.csv']},
        primary_joint_fast_pca_cells=len(joint)))
    print(json.dumps(dict(report=str(destination),joint_fast_pca_cells=len(joint))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--benchmark',type=Path,required=True)
    args=parser.parse_args()
    report(args.root,args.benchmark)


if __name__=='__main__':
    main()
