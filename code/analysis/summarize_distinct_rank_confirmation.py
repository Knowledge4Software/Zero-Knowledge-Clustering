"""Deduplicate stored rank-screen requests by their authenticated center arrays.

The original numerical exports are retained. A distinct update is keyed by
cohort, deployed fit and the quantized proposed center array, and excludes the
public identity with the deployed centers.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np
from commitjl.pilot_io import save,sha,array_sha


def run(root,repo):
    sources={'historical':repo/'paper_results/review_controls/local_group_moments_extended_20261023',
             'fresh':repo/'paper_results/review_controls/group_change_confirmation_20261024/source'}
    original=json.loads((root/'summary.json').read_text())
    records=list(csv.DictReader((root/'results.csv').open()))
    archives={};groups={};hashes={}
    for row in records:
        fit=row['fit'];cohort=row['cohort'];key=(cohort,fit)
        if key not in archives:
            path=sources[cohort]/(fit+'.npz');digest=sha(path)
            assert digest==original['source_checks'][fit]['source_archive_sha256']
            with np.load(path) as data:archives[key]=data['centers'].copy()
            hashes[path.relative_to(repo).as_posix()]=digest
        centers=archives[key];index=int(row['set']);rank=int(row['rank'])
        digest=array_sha(centers[index]);identity=bool(np.array_equal(centers[index],centers[0]))
        truth=int(row['true_change']);den=int(row['denominator'])
        assert int(row['lower'])<=truth*den<=int(row['upper'])
        assert row['decision'] in ['inconclusive',row['truth']]
        group=groups.setdefault((cohort,fit,digest,rank),dict(representative=row,source_sets=[],identity=identity))
        for field in ['true_change','truth','lower','upper','denominator','decision','combined_with_63']:
            assert row[field]==group['representative'][field],f'inconsistent duplicate {key}: {field}'
        assert identity==group['identity'];group['source_sets'].append(index)
    results={}
    for rank in sorted({int(r['rank']) for r in records}):
        raw=[r for r in records if int(r['rank'])==rank]
        unique=[g for key,g in groups.items() if key[3]==rank]
        active=[g['representative'] for g in unique if not g['identity']]
        regressions=[r for r in active if r['truth']=='worsens']
        results[str(rank)]=dict(original_proposal_requests=len(raw),
            distinct_center_sets_including_public_identity=len(unique),
            repeated_center_requests_removed=len(raw)-len(unique),
            distinct_public_identity_sets=sum(g['identity'] for g in unique),
            original_public_identity_requests=sum(len(g['source_sets']) for g in unique if g['identity']),
            distinct_nonidentity_updates=len(active),
            conclusive_distinct_updates=sum(r['decision']!='inconclusive' for r in active),
            distinct_natural_regressions=len(regressions),
            distinct_natural_regressions_detected=sum(r['decision']=='worsens' for r in regressions),
            combined_conclusive_distinct_updates=sum(r['combined_with_63']!='inconclusive' for r in active),
            combined_distinct_natural_regressions_detected=sum(r['combined_with_63']=='worsens' for r in regressions))
        assert len(raw)==original['results_by_rank'][str(rank)]['requests']
    summary=dict(complete=True,source_results_sha256=sha(root/'results.csv'),
        source_summary_sha256=sha(root/'summary.json'),source_archive_sha256=hashes,
        grouping='cohort + deployed fit + SHA256 of the complete quantized proposed center array; public identity excluded from update counts. Different fits remain separate audit workloads, not independent draws per update.',
        native_only=True,new_proofs=0,all_recorded_intervals_contain_truth=True,
        results_by_rank=results,
        center_equivalence_groups=[dict(cohort=key[0],fit=key[1],center_sha256=key[2],rank=key[3],source_sets=g['source_sets'],public_identity=g['identity'],truth=g['representative']['truth'],decision=g['representative']['decision']) for key,g in groups.items()])
    save(root/'distinct_summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    result=run(a.root.resolve(),Path(__file__).resolve().parents[2]);print(json.dumps(result['results_by_rank'],indent=2))
