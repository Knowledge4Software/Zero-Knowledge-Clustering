"""Build common shape-cost profiles from earlier server measurements only."""
from dataclasses import asdict
from pathlib import Path
import statistics

from analysis.server_routing_calibration_20261102 import calibrate_case, sample_work
from experiments.pca_advantage_20261102 import read, sha


def collect_profiles(repo):
    repo = Path(repo)
    outputs = repo/'code/validation/outputs'
    pins, result = {}, {}
    def get(path):
        pins[str(path.relative_to(repo))] = sha(path)
        return read(path)
    def profile(cohort, case, method):
        root = outputs/cohort
        base = root/'matched'/case/(method+'_base')
        ref = root/'matched'/case/(method+'_refined')
        inp = get(base/'input.json');m=inp['model']
        bs = get(base/'status.json')['resources'];rs=get(ref/'status.json')['resources']
        group=get(base/'bundle/group_query.json');count=len(inp['reuse_centers'])-1
        body=sum(map(sample_work,group['samples']))
        correction=None;ref_overhead=0.
        if (ref/'bundle/refinement_prove.json').exists():
            rr=get(ref/'bundle/refinement_prove.json');samples=rr.get('samples',[])
            if samples:
                rb=sum(map(sample_work,samples));correction=rb/len(samples)
                ref_overhead=max(0.,rs['refinement_prove']['wall_seconds']-rb)+max(0.,rs['refinement_verify']['wall_seconds']-bs['group_verify']['wall_seconds'])
        warm_paths=sorted((root/'warm'/case).glob('repetition_*/*ordered_head*/checked.json'))
        if cohort=='server_pca_advantage_20261102':
            warm_paths=[root/'warm'/case/(case+'_ordered_head_warm01/checked.json')]
        assert warm_paths
        wc=get(warm_paths[0]);wp=warm_paths[0].parent/'baseline_full/private/full_reuse_8.json'
        fb=get(wp);fresh=[s for s in fb['samples'] if s.get('centre_set',0)>0]
        assert len(fresh)==wc['requests']*len(get(root/'matched'/case/method/'full/input.json')['queries'])
        fw=sum(map(sample_work,fresh));fr=wc['resources']
        full_per=fw/wc['requests'];full_fixed=max(0.,fr['baseline_full_prove']['wall_seconds']-fw)+fr['baseline_verify']['wall_seconds']
        cold=get(root/'matched'/case/'ordered_head/status.json')['cases']['full']['wall_seconds']
        full_init=max(0.,cold-(full_per*count+full_fixed))
        return dict(shape=[m['n'],m['d'],m['k'],len(m['basis']),m['input_bits'],m['basis_scale']],
          certificate_seconds=body/count,certificate_batch_seconds=max(0.,bs['group_query']['wall_seconds']-body)+bs['group_verify']['wall_seconds'],
          correction_seconds=correction,refinement_batch_seconds=ref_overhead,
          full_seconds=full_per,full_batch_seconds=full_fixed,
          prepare_seconds=bs['group_compact_prepare']['wall_seconds'],
          keys_rows_seconds=bs['group_keys']['wall_seconds']+bs['group_rows']['wall_seconds'],
          refinement_setup_seconds=rs.get('refinement_setup',{}).get('wall_seconds',0.),
          full_initialization_seconds=full_init)
    for dataset in ['fashion','har','mnist','text']:
        cohort='server_campaign_20261101' if dataset in ['fashion','text'] else 'server_diversity_20261101'
        pool=[profile(cohort,f'{dataset}_seed{s}_n128_q4',method)
              for s in [701,709,719] for method in ['ordered_head','pca']]
        assert all(p['shape']==pool[0]['shape'] for p in pool) and pool[0]['shape'][3]==79
        def med(key):
            values=[p[key] for p in pool if p[key] is not None and (p[key]>0 or key not in ['refinement_batch_seconds','refinement_setup_seconds'])]
            if not values:raise ValueError('Missing measured profile field '+key)
            return statistics.median(values)
        costs={k:med(k) for k in ['full_seconds','certificate_seconds','correction_seconds','full_batch_seconds','certificate_batch_seconds','refinement_batch_seconds']}
        costs['margin']=.10
        result[dataset]={'79':dict(costs=costs,shape=pool[0]['shape'],
            **{k:med(k) for k in ['prepare_seconds','keys_rows_seconds','refinement_setup_seconds','full_initialization_seconds']})}
    prior=outputs/'server_pca_advantage_20261102'
    for dataset,case in [('fashion','fashion_seed829_n128_q4'),('har','har_seed811_n128_q4')]:
        costs,source=calibrate_case(prior,case,'pca')
        for name,digest in source['files'].items():pins[str((prior/name).relative_to(repo))]=digest
        bs=get(prior/'matched'/case/'pca_base/status.json')['resources']
        rs=get(prior/'matched'/case/'pca_refined/status.json')['resources']
        shape=result[dataset]['79']['shape'].copy();shape[3]=16
        result[dataset]['16']=dict(costs=asdict(costs),shape=shape,
            prepare_seconds=bs['group_compact_prepare']['wall_seconds'],
            keys_rows_seconds=bs['group_keys']['wall_seconds']+bs['group_rows']['wall_seconds'],
            refinement_setup_seconds=rs['refinement_setup']['wall_seconds'],
            full_initialization_seconds=result[dataset]['79']['full_initialization_seconds'])
    # Identical N,D,K,precision,rank and Go circuits: shared hardware profile.
    assert result['fashion']['79']['shape']==result['mnist']['79']['shape']
    result['mnist']['16']=result['fashion']['16']
    return dict(profiles=result,source_sha256=pins,
         interpretation='Common measured circuit-shape profiles for both bases; rank16 unavailable for text. '
                        'Profiles predict costs, not guarantees. Medians across old rank79 arms/fits; '
                        'old rank16 PCA measurements shared fairly with both bases.')
