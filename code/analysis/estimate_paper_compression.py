"""Exploratory wall-time forecast from a completed full-size native pilot."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from campaigns.paper_compression import configuration
from experiments.laptop_comparison import atomic_json
from experiments.paper_compression import case_key,case_plan


WARM_STAGES={'compression','center_projection','numeric_preparation','full_numeric_preparation',
    'point_norm_preparation','full_point_norm_preparation','heldout_assignment','full_heldout_assignment',
    'knn','full_knn','kmeans','full_kmeans','kmeans_refine','full_kmeans_refine',
    'dbscan_pair_build','full_dbscan_pair_build','dbscan_fit','full_dbscan_fit'}


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def workload_seconds(w, repetitions, queries, ratio=1.):
    value=0.
    for stage,measurement in w.items():
        if not isinstance(measurement,dict) or 'samples' not in measurement: continue
        scale=1. if stage.startswith('full_') else ratio
        value+=measurement['median']*(repetitions+int(stage in WARM_STAGES))*scale
    for control,rows in w['distance_queries'].items():
        scale=1. if control=='full' else ratio
        value+=np.mean([r['seconds'] for r in rows])*queries*repetitions*scale
    for profile in w.get('dbscan_profiles',{}).values():
        for stage in ('dbscan_fit','full_dbscan_fit'):
            value+=profile[stage]['median']*(repetitions+1)*(1. if stage.startswith('full_') else ratio)
    return float(value)


def estimate(root,profile):
    root=Path(root);manifest=read(root/'manifest.json')
    if not manifest['complete']: raise ValueError('The pilot must be complete')
    pilot=manifest['identity']['configuration'];config=configuration(profile);records=[]
    if pilot.get('version')!=config['version']:raise ValueError('Pilot measurement version differs: run a fresh v2 pilot')
    for folder in sorted((root/'jobs').glob('*')):
        metadata=read(folder/'dataset.json');fits=read(folder/'fits.json')
        rows=[read(p) for p in (folder/'rows').glob('*.json')]
        observed_workloads={case_key(r):read(folder/r['workloads']/'timings.json') for r in rows if r['workloads']}
        known= sum(r['model_construction']['median']+r['screening_compression']['median'] for r in rows)
        known+=sum(workload_seconds(w,pilot['timing_repetitions'],pilot['queries']) for w in observed_workloads.values())
        known+=fits['pca']['median']*(pilot['fit_repetitions']+int(pilot['fit_warmup']))
        known+=sum(v['median']*(pilot['fit_repetitions']+int(pilot['fit_warmup'])) for v in fits.get('pca_randomized',{}).values())
        known+=sum(v['median']*(pilot['fit_repetitions']+int(pilot['fit_warmup'])) for v in fits['centroids'].values())
        known+=sum(v['median']*(pilot['timing_repetitions']+1) for v in fits['basis'].values())
        overhead=max(.02,(read(folder/'done.json')['wall_seconds']-known)/len(rows))
        train_n=metadata['actual_training_rows'];d=metadata['d'];max_rank=min(train_n-1,d)
        dims=sorted(set([m for m in config['m'] if m<d]+([max_rank] if max_rank<min(d,max(config['m'])) else [])))
        expected=[c for phase in ('validation','test') for k in config['k'] for start in range(config['starts'])
            for c in case_plan(config,dims,k,start,min(k-1,d),train_n,d,phase)]
        def closest(case, measured=False):
            candidates=[r for r in rows if r['method']==case['method'] and r['phase']==case['phase']
                and (not measured or r['workloads'])]
            if not candidates: return None
            return min(candidates,key=lambda r:abs(np.log(max(1,r['m'])/max(1,case['m']))))
        seconds=fits['pca']['median']*(config['fit_repetitions']+int(config['fit_warmup']))
        if config.get('randomized_pca',False):
            seconds+=np.mean([v['median'] for v in fits['pca_randomized'].values()])*config['seeds']*(config['fit_repetitions']+int(config['fit_warmup']))
        mean_center=np.mean([v['median'] for v in fits['centroids'].values()])
        mean_basis=np.mean([v['median'] for v in fits['basis'].values()])
        seconds+=sum((mean_center*(config['fit_repetitions']+int(config['fit_warmup']))+
            mean_basis*(config['timing_repetitions']+1))*k/5 for k in config['k'])*config['starts']
        selected=read(folder/'selection.json')['choices'];timed_count=0
        for case in expected:
            r=closest(case);ratio=case['m']/r['m'] if case['method']!='full' else 1.
            map_scale=max(1,case['q'])/max(1,r['q'])
            seconds+=r['model_construction']['median']*map_scale+ r['screening_compression']['median']*ratio
            seconds+=overhead*max(1,case['k']/r['k'])
            chosen=next((s for s in selected if s['method']==case['method']),None)
            benchmark=case['phase']=='test' and (case['method'] in ('full','protected_only') or
                case['m'] in config['timing_m'] or (chosen and chosen['selected'] and chosen['m']==case['m'] and chosen['q']==case['q']))
            if benchmark:
                timed_count+=1; candidate=closest(case,True)
                if candidate:
                    w=observed_workloads[case_key(candidate)]
                    seconds+=workload_seconds(w,config['timing_repetitions'],config['queries'],case['m']/candidate['m'])*max(1,case['k']/candidate['k'])
        records.append(dict(dataset=metadata['dataset'],variant=metadata['variant'],expected_quality_rows=len(expected),
            forecast_timed_cases=timed_count,forecast_hours=seconds/3600,
            planning_low_hours=seconds/7200,planning_high_hours=seconds/1800,
            pilot_peak_rss_gib=read(folder/'execution.json')['peak_tree_rss_gib']))
    total=sum(r['forecast_hours'] for r in records)
    output=dict(profile=profile,forecast_hours=total,planning_low_hours=total*.5,planning_high_hours=total*2,
        expected_quality_rows=sum(r['expected_quality_rows'] for r in records),jobs=records,
        scope='Exploratory interpolation, not a confidence interval or completion guarantee. '
        'Uses pilot-selected dimensions and linear m/q/k cost scaling; new selections, solver iterations, '
        'background load, thermal throttling, artifact I/O and memory limits can change runtime. '
        'RSS is observed pilot memory, not a guaranteed maximum for the full grid.',
        pilot_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        estimator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(root/'report'/f'runtime_forecast_{profile}.json',output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--profile',choices=['paper','balanced'],required=True);args=parser.parse_args()
    result=estimate(args.pilot,args.profile)
    print(json.dumps({k:v for k,v in result.items() if k!='jobs'}))


if __name__=='__main__':main()
