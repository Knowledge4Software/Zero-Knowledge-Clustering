"""Resource-bounded RQ2--RQ5 campaigns. See docs/rq2345_complete.md.

Each task is isolated in a monitored process. Proof timing tasks are serial.
No dataset-sized proof or paid campaign is launched without --run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading
import math
import shutil

from experiments.rq1_complete import DATASETS, write_json as atomic_json, save_npz, bootstrap_crossed
from experiments.common import campaign_lock

CODE = Path(__file__).resolve().parents[1]
STAGES = ['rq2_pilot', 'rq2_quality', 'rq2_selected', 'rq3_native', 'rq3_proof', 'rq3_grinding',
          'rq4_centroid', 'rq4_dbscan', 'rq4_phase', 'rq5_native', 'rq5_precision', 'rq5_proof','rq5_ablation']
PREREG = dict(eps=.15, pair_set_size=2000)
STOP = threading.Event()


def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    if hasattr(value,'item'): return clean_json(value.item())
    return value


def write_json(path,value):
    atomic_json(path,clean_json(value))


def spec(name):
    loader, kwargs = DATASETS[name]
    return dict(loader=loader, kwargs=dict(kwargs))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def plan(args):
    tasks = []
    def add(stage, **kw):
        if stage in args.stages:
            task = dict(stage=stage, **kw)
            backends=getattr(args,'backends',['groth16']) if stage in [
                'rq2_pilot','rq2_quality','rq2_selected','rq3_proof','rq5_proof','rq5_ablation'] and 'row' in task else [None]
            for backend in backends:
                item=copy.deepcopy(task)
                if backend: item['row']['backend']=backend
                tasks.append(dict(item, id=digest(item)))
    base = dict(dataset=dict(loader='gaussian_mixture', kwargs=dict(
        n=16, d=32, k=5, gamma=3., seed=12345)), k=5, m=16, q=2,
        seed=0, clustering_start=0, precision_bits=16, B=4096, backend='groth16')
    # A small engineering ladder, not a substitute for RQ1-selected workloads.
    for n, d in [(8,16), (16,32), (32,64), (64,128)]:
        row = copy.deepcopy(base)
        row['dataset']['kwargs'].update(n=n, d=d)
        row['m'] = min(16, d)
        for variant in ['full', 'sparse', 'dense', 'coord', 'pca-public']:
            add('rq2_pilot', row={**row, 'variant':variant}, scope='engineering_pilot')
    for name in args.datasets:
        directory = args.rq1 / name
        if 'rq4_centroid' in args.stages:
            add('rq4_centroid', name=name, directory=str(directory.resolve()))
        add('rq4_dbscan',name=name,row=dict(dataset=spec(name),projection='sparse_jl'),
            m_values=[32,64,128],q_values=[1,2,4],seeds=args.seeds,max_points=args.dbscan_points,
            directory=str(directory.resolve()) if (directory/'artifacts/dataset.npz').exists() else None)
        selected = directory / 'selected_operating_points.csv'
        if any(s in args.stages for s in ['rq2_selected','rq2_quality']) and selected.exists():
            import pandas as pd
            frame = pd.read_csv(selected)
            manifest=json.loads((directory/'manifest.json').read_text())
            variants = dict(sparse_jl='sparse', rademacher='dense', coordinate='coord', pca='pca-public')
            for record in frame.to_dict('records'):
                if record['method'] not in variants:
                    continue
                row = {**base, 'dataset':spec(name), 'k':int(record['k']),
                       'm':int(record['m']), 'q':max(1,int(record['q'])),
                       'proof_n_max':getattr(args,'proof_points',0)}
                add('rq2_selected', name=name, directory=str(directory.resolve()),
                    row={**row,'variant':variants[record['method']]}, scope='matched_declared_subset_requalified' if row['proof_n_max'] else 'exact_RQ1_dataset_new_quantized_witness')
                add('rq2_quality',name=name,directory=str(directory.resolve()),
                    row={**row,'variant':variants[record['method']]},seeds=args.seeds,starts=args.starts)
            for method in manifest['config']['methods']:
                for k in manifest['config']['k']:
                    if method not in variants:
                        add('rq2_selected',name=name,method=method,k=k,
                            excluded='No verified implemented proof circuit for this native method')
                    elif not ((frame.method==method)&(frame.k==k)).any():
                        add('rq2_selected',name=name,method=method,k=k,
                            excluded='No CI-eligible RQ1 operating point')
            for k in manifest['config']['k']:
                add('rq2_selected', name=name, directory=str(directory.resolve()),
                    row={**base,'dataset':spec(name),'k':int(k),'variant':'full',
                         'proof_n_max':getattr(args,'proof_points',0)},
                    scope='matched_declared_subset_requalified' if getattr(args,'proof_points',0) else 'exact_RQ1_dataset_new_quantized_witness')
        else:
            for stage in ['rq2_selected','rq2_quality']:
                add(stage, name=name, unavailable='Run and export RQ1 selection first')
    for seed in range(args.seeds):
        for source in ['nonce','beacon','commit_reveal']:
            row = copy.deepcopy(base)
            row['dataset']['kwargs'].update(n=16,d=32)
            row.update(m=16, seed=seed)
            add('rq3_native', row=row, source=source)
    for gamma in [.5,1.,2.,3.,4.]:
        for m in [16,32,64,128]:
            row=copy.deepcopy(base)
            row['dataset']['kwargs'].update(n=1000,d=256,gamma=gamma)
            row.update(m=m,q=2,projection='sparse_jl',sweep_factor='gamma',sweep_value=gamma)
            add('rq4_phase',row=row,seeds=args.seeds,starts=args.starts)
    add('rq3_proof', row=base, scope='same_witness_matched_authentication_ablation')
    add('rq3_grinding', row={**base,'projection':'sparse_jl'}, trials=args.grinding_trials)
    levels = dict(n=[500,1000,2000,4000], d=[128,256,512,1024], k=[5,10,20],
                  m=[16,32,64,128,256], q=[1,2,4], gamma=[.5,1.,2.,3.,4.])
    for factor, values in levels.items():
        for value in values:
            row = copy.deepcopy(base)
            row['dataset']['kwargs'].update(n=1000,d=256,k=5)
            row.update(m=64,q=2,projection='sparse_jl',sweep_factor=factor,sweep_value=value)
            if factor in ['n','d','gamma']:
                row['dataset']['kwargs'][factor] = value
            if factor in ['k','m','q']:
                row[factor] = value
            add('rq5_native',row=row,seeds=args.seeds,starts=args.starts)
    for name in args.datasets:
        for bits in [8,12,16,24]:
            add('rq5_precision', row={**base,'dataset':spec(name),'m':32,
                'projection':'sparse_jl','precision_bits':bits,'normalization_reference_bits':24},
                name=name,seeds=args.seeds,starts=args.starts)
    for factor, values in dict(n=[8,16,32,64],d=[16,32,64,128],m=[4,8,16,32],
                               k=[2,3,5],q=[1,2,4]).items():
        for value in values:
            row = copy.deepcopy(base)
            if factor in ['n','d']:
                row['dataset']['kwargs'][factor] = value
            if factor in ['m','k','q']:
                row[factor] = value
            row['m'] = min(row['m'],row['dataset']['kwargs']['d'])
            row.update(variant='sparse',sweep_factor=factor,sweep_value=value)
            add('rq5_proof', row=row, scope='engineering_one_factor_proof_scaling')
    for bits in [8,12,16,24]:
        add('rq5_proof', row={**base,'variant':'sparse','precision_bits':bits,
            'normalization_reference_bits':24,'sweep_factor':'precision_bits','sweep_value':bits},
            scope='fixed_physical_reference_quantization_proof_scaling')
    for n,d in [(16,32),(32,64)]:
        row=copy.deepcopy(base);row['dataset']['kwargs'].update(n=n,d=d)
        row.update(variant='sparse',precision_bits=24,normalization_reference_bits=24)
        add('rq5_ablation',row=row,scope='declared_representative_and_stress_engineering_ablation')
    return list({t['id']:t for t in tasks}.values())


def counts(mask, stable):
    import numpy as np
    total, qualified = len(mask), int(np.sum(mask))
    violations = int(np.sum(mask & ~stable))
    return dict(n_total=total,n_qualified=qualified,n_violations=violations,
        qualified_coverage=qualified/total if total else None,
        conditional_violation_rate=violations/qualified if qualified else None,
        uncertified_stability=float(np.sum(~mask & stable))/(total-qualified) if total>qualified else None)


def bootstrap_nested(values,draws=2000,seed=9876):
    """Resample starts, then fresh commitment/challenge draws within each start."""
    import numpy as np
    values=np.asarray(values,float)
    if values.ndim!=2 or not np.isfinite(values).all():
        raise ValueError('nested bootstrap requires finite draw x start values')
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(draws):
        starts=rng.integers(values.shape[1],size=values.shape[1])
        samples.append(np.mean([values[rng.integers(len(values),size=len(values)),start].mean() for start in starts]))
    lo,hi=np.quantile(samples,[.025,.975])
    return float(values.mean()),float(lo),float(hi)


def frozen_dataset(directory):
    import numpy as np
    from data.loaders import _sha256_array
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if not (directory/'completion.json').exists():
        raise ValueError('RQ1 run has no completion record')
    with np.load(directory/'artifacts/dataset.npz') as a:
        X=a['X'].copy()
    if _sha256_array(X)!=manifest['dataset']['sha256']:
        raise ValueError('RQ1 dataset artifact differs from recorded array hash')
    return dict(X=X,meta=manifest['dataset'])


def centroid_export(task, out):
    import numpy as np
    import pandas as pd
    directory = Path(task['directory'])
    manifest = json.loads((directory/'manifest.json').read_text())
    if not (directory/'completion.json').exists():
        raise ValueError('RQ1 dataset run is incomplete; finish it before centroid export')
    eps = manifest['config']['eps']
    rows = pd.read_parquet(directory/'rows.parquet')
    result = []
    for row in rows.to_dict('records'):
        if row['method'] != 'sparse_jl':
            continue
        with np.load(directory/row['baseline_artifact']) as b, np.load(directory/row['pointwise_artifact']) as a:
            start = int(row['clustering_start'])
            mask = b['margin'] & (a['max_point_centroid_distortion'][start] <= eps)
            stable = b['labels'] == a['labels'][start]
            result.append({**row,**counts(mask,stable), 'stage':'rq4_centroid','dataset':task['name'],
                'rq1_artifact_directory':str(directory.resolve()),
                'global_embedding_event':bool(np.all(a['max_point_centroid_distortion'][start]<=eps))})
    pd.DataFrame(result).to_parquet(out/'centroid_rows.parquet',index=False)
    return dict(rows=len(result), rq1_source_sha256=manifest['source_sha256'],
        scope='exact_RQ1_arrays_conditional_centroid_certificate', raw_file='centroid_rows.parquet')


def dbscan_measure(task, out, *, cache=None):
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from commitjl.certificates import dbscan_pair_certificate, pair_squared_distances
    from commitjl.clustering import dbscan_deterministic
    from commitjl.sparse_gen import sparse_matrix_gen
    from experiments.pipeline import load_row_dataset
    cache=cache if cache is not None else {}
    row = task['row']
    if 'data' not in cache:
        cache['data']=frozen_dataset(task['directory']) if task.get('directory') else load_row_dataset(row)
    data=cache['data'];X=np.asarray(data['X'],float)
    rng = np.random.default_rng(20261005)
    indices = np.sort(rng.choice(len(X), min(len(X),task['max_points']),replace=False))
    X = X[indices]; n,d = X.shape
    if row['m'] > d:
        return dict(status='excluded',reason='m exceeds input dimension')
    # Radius is selected once from original geometry, independent of projection.
    from sklearn.neighbors import NearestNeighbors
    min_samples = 5
    if 'radius' not in cache:
        cache['radius']=(float(row['dataset']['kwargs']['rho']) if row['dataset']['loader']=='dbscan_margin' else
              float(np.median(NearestNeighbors(n_neighbors=min_samples).fit(X).kneighbors(X)[0][:,-1])))
    radius=cache['radius']
    if not radius > 0:
        raise ValueError('nonpositive median fifth-neighbor radius')
    seed = hashlib.sha256(b'CommitJL-RQ1-native-v2'+row['seed'].to_bytes(8,'little')).digest()
    _, A = sparse_matrix_gen(seed,row['m'],d,row['q'])
    Z = X @ (A/np.sqrt(row['q'])).T
    if 'pc' not in cache: cache['pc']=dbscan_pair_certificate(X,radius,.15,max_pairs=None)
    pc=cache['pc']
    original = pc['orig_sqdist']; projected = pair_squared_distances(Z,pc['i'],pc['j'])
    event = np.abs(projected-original) <= .15*original + 1e-12
    stable = (original<=radius**2)==(projected<=radius**2)
    mask = pc['certified'] & event
    near = ~pc['certified']
    if 'labels' not in cache: cache['labels']=dbscan_deterministic(X,radius,min_samples)
    labels=cache['labels']
    projected_labels = dbscan_deterministic(Z,radius,min_samples)
    original_counts=np.ones(n,dtype=np.int64); projected_counts=np.ones(n,dtype=np.int64)
    for target, neighbors in [(original_counts,original<=radius**2),(projected_counts,projected<=radius**2)]:
        np.add.at(target,pc['i'][neighbors],1);np.add.at(target,pc['j'][neighbors],1)
    original_core=original_counts>=min_samples;projected_core=projected_counts>=min_samples
    save_npz(out/'dbscan.npz',X=X,Z=Z,source_indices=indices,i=pc['i'],j=pc['j'],
        original_sq=original,projected_sq=projected,band_certificate=pc['certified'],
        embedding_event=event,labels=labels,projected_labels=projected_labels,
        original_core=original_core,projected_core=projected_core)
    return {**row,**counts(mask,stable),'n':n,'d':d,'rho':radius,'min_samples':min_samples,
        'radius_policy':'generator_radius' if row['dataset']['loader']=='dbscan_margin' else 'median_original_fifth_neighbor_distance',
        'dataset_sha256':data['meta'].get('sha256'),
        'subset_seed':20261005,'original_n':len(data['X']),'pairs_evaluated':len(original),
        'global_embedding_event':bool(event.all()),'graph_stable':bool(stable.all()),
        'core_agreement':float(np.mean(original_core==projected_core)),
        'original_noise_fraction':float(np.mean(labels==-1)), 'projected_noise_fraction':float(np.mean(projected_labels==-1)),
        'band_coverage':float(pc['certified'].mean()),'near_band_pairs':int(near.sum()),
        'near_band_stability':float(stable[near].mean()) if near.any() else None,
        'ari':float(adjusted_rand_score(labels,projected_labels)),
        'nmi':float(normalized_mutual_info_score(labels,projected_labels)),
        'scope':'all_pairs_on_declared_fixed_subset_not_full_dataset', 'raw_file':'dbscan.npz'}


def ordering_measure(task, out):
    import numpy as np
    from commitjl.protocol import run_protocol,Prover,Verifier
    from commitjl.randomness import TaskLog
    from commitjl.clustering import kmeans_paired
    from commitjl.quantize import quantize,scale_for_precision
    from experiments.pipeline import load_row_dataset,normalize_for_quantization
    from experiments.rq3_overhead import _rho_for_source
    row=task['row']; data=load_row_dataset(row); S=scale_for_precision(16)
    X=quantize(normalize_for_quantization(data['X'],S,4096),S,4096)
    C=np.rint(kmeans_paired(X.astype(float),row['k'],seeds=[row['clustering_start']])[0][1]).astype('int64')
    # The untimed initial transcript freezes the SAME matrix for both arms.
    frozen=run_protocol(X,C,row['m'],row['q'],S,
        challenge_fn=lambda:_rho_for_source(task['source'],TaskLog(),row['seed']))
    samples=[]
    for repetition in range(11):
        for arm in (['postcommit','prepublished'] if repetition%2==0 else ['prepublished','postcommit']):
            t=time.perf_counter()
            if arm=='postcommit':
                def challenge_fixture():
                    # Time the source operation; replay the frozen value solely
                    # for this matched-matrix timing counterfactual.
                    _rho_for_source(task['source'],TaskLog(),row['seed']+repetition)
                    return frozen['rho']
                result=run_protocol(X,C,row['m'],row['q'],S,challenge_fn=challenge_fixture,
                    commitment_salt=frozen['r_X'])
                if not np.array_equal(result['A_dense'],frozen['A_dense']):
                    raise AssertionError('postcommit matrix differs from paired reference')
                stages=result['transcript']['stage_times_s']
                stages={**stages,**{f'substage_{k}':v for k,v in
                    result['transcript']['public']['expansion_substage_times_s'].items()}}
            else:
                prover=Prover(X,S); verifier=Verifier(TaskLog(),row['m'],row['q'],C)
                _,commit_s=prover.commit_data(salt=frozen['r_X']); _,theta_s=verifier.record_theta()
                _,assign,cost,cluster_s=prover.project_and_cluster(frozen['A_dense'],C)
                if not np.array_equal(assign,frozen['assignments']) or int(cost)!=int(frozen['Psi_hat']):
                    raise AssertionError('paired ordering witness mismatch')
                stages=dict(commit_X=commit_s,record_theta=theta_s,project_and_cluster=cluster_s)
            elapsed=time.perf_counter()-t
            if repetition:
                samples.append(dict(repetition=repetition,arm=arm,total_seconds=elapsed,**stages))
    write_json(out/'timings.json',samples)
    # Encoded audit transcript bytes, distinct from in-memory ndarray sizes.
    wire=json.dumps(frozen['transcript'],sort_keys=True,separators=(',',':')).encode()
    (out/'transcript.json').write_bytes(wire)
    save_npz(out/'witness.npz',X=X,C=C,A=frozen['A_dense'],assignments=frozen['assignments'])
    return {**row,'source':task['source'],'rho_hex':frozen['rho'].hex(),
        'projection_seed_hex':frozen['seed'].hex(),'transcript_json_bytes':len(wire),
        'randomness_scope':'simulated_local_beacon' if task['source']=='beacon' else 'local_'+task['source'],
        'scope':'isolated_same_matrix_ordering_counterfactual_replayed_challenge',
        'fresh_task_per_arm':True,'warmup':1,'measured_pairs':10,'raw_file':'timings.json'}


def prove_measure(task, out, settings):
    import numpy as np
    from commitjl import zk_bridge
    from experiments.rq2_proof_cost import build_zk_instance
    row=task['row']; data=None
    if task['stage'] in ['rq2_selected','rq2_quality']:
        directory=Path(task['directory'])
        data=frozen_dataset(directory)
        if row.get('proof_n_max'):
            from data.loaders import _sha256_array
            source_hash=data['meta']['sha256'];source_n=len(data['X'])
            indices=np.sort(np.random.default_rng(20261005).choice(source_n,
                min(source_n,row['proof_n_max']),replace=False))
            data=dict(X=data['X'][indices],meta={**data['meta'],'n':len(indices),
                'source_dataset_sha256':source_hash,'sha256':_sha256_array(data['X'][indices])})
            save_npz(out/'subset.npz',source_indices=indices,X=data['X'])
        if len(data['X']) < row['k']:
            return dict(status='excluded',reason='declared subset has fewer rows than fitted clusters',
                n=len(data['X']),k=row['k'])
        if row['variant']=='pca-public' and row['m']>min(len(data['X'])-1,data['X'].shape[1]):
            return dict(status='excluded',reason='selected PCA m exceeds subset rank; choose a new subset operating point')
        elements=int(data['X'].size)
        budget=settings.get('max_proof_input_elements',8192)
        if elements>budget:
            return dict(status='excluded',reason='declared proof feasibility budget exceeded before commitment/proving',
                n=len(data['X']),d=data['X'].shape[1],input_elements=elements,max_proof_input_elements=budget,
                dataset_sha256=data['meta']['sha256'],failure_kind='preflight_budget_exclusion_not_measured_OOM')
    qualification=None
    if task['stage']=='rq2_selected' and row['variant']!='full':
        quality=dict(stage='rq2_quality',name=task['name'],directory=task['directory'],
            row=row,seeds=settings['seeds'],starts=settings['starts'])
        path=out.parents[1]/'rq2_quality'/digest(quality)/'result.json'
        if not path.exists():
            return dict(status='unavailable',error='Run rq2_quality in this campaign before selected proofs')
        qualification=json.loads(path.read_text())
        if not qualification.get('eligible'):
            return dict(status='excluded',reason='quantized ensemble failed CI eligibility',qualification=qualification)
    if task['stage']=='rq2_quality':
        import pandas as pd
        samples=quantized_quality_samples(row,data,task['seeds'],task['starts'],out)
        frame=pd.DataFrame(samples);frame.to_parquet(out/'quality_rows.parquet',index=False)
        result={}
        for metric in ['ari','r_phi']:
            matrix=frame.pivot(index='seed',columns='clustering_start',values=metric)
            bootstrap=bootstrap_nested if row['variant']=='sparse' else bootstrap_crossed
            mean,lo,hi=bootstrap(matrix.to_numpy(),settings['bootstrap'])
            result.update({metric+'_mean':mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
        result['eligible']=bool((row['variant']=='pca-public' or task['seeds']>=2) and task['starts']>=2 and
            result['ari_ci_low']>=.8 and result['r_phi_ci_high']<=1.1)
        return {**result,'scope':'quantized_challenge_ensemble_fixed_dataset_pointwise_CI',
            'projection_randomness':'fresh_nonce_per_sparse_draw_with_one_fixed_input_opening; seeded_dense_or_coordinate; deterministic_PCA',
            'CI_design':'nested_draws_within_start' if row['variant']=='sparse' else 'crossed_seed_start',
            'witness_count':len(samples),
            'raw_file':'quality_rows.parquet'}
    inst,*_=build_zk_instance(row,PREREG,data=data)
    if qualification and (inst.benchmark_metrics['ari']<.8 or inst.benchmark_metrics['r_phi']>1.1):
        return dict(status='excluded',reason='fresh proof witness fails observed fidelity',
            qualification=qualification,**inst.benchmark_metrics)
    write_json(out/'witness.json',inst.to_json_dict())
    arms=[('reference',inst)]
    if task['stage'] in ['rq3_proof','rq5_ablation']:
        # Authentication removed, exact constants/assignments/cost retained.
        from commitjl.sparse_gen import sparse_matrix_gen
        _,A=sparse_matrix_gen(inst.projection_seed,inst.m,inst.d,inst.q)
        public=copy.deepcopy(inst); public.variant='prepublished'; public.constant_A=A
        public.sparse_records=None; public.projection_seed=None; public.challenge_rho=None
        unbatched=copy.deepcopy(inst); unbatched.no_range_batch=True
        arms += [('same_matrix_prepublished',public),('same_witness_unbatched',unbatched)]
        if task['stage']=='rq5_ablation':
            for variant in ['full','dense','coord','pca-public']:
                other,*_=build_zk_instance({**row,'variant':variant},PREREG)
                arms.append((variant,other))
            lower,*_=build_zk_instance({**row,'precision_bits':16},PREREG)
            arms.append(('lower_precision_16_fixed_reference_24',lower))
    results=[]
    write_json(out/'proof_arms.json',results)
    for arm,witness in arms:
        write_json(out/(arm+'_witness.json'),witness.to_json_dict())
        t=time.perf_counter()
        res=zk_bridge.run_prove(witness,timeout=settings['proof_timeout'],
            repetitions=settings['proof_reps'],warmup=1,
            max_rss_bytes=int(settings['proof_rss_gib']*1024**3))
        elapsed=time.perf_counter()-t
        write_json(out/(arm+'_proof.json'),dict(raw=res.raw,error=res.error,failure_kind=res.failure_kind,
            peak_rss_bytes=res.peak_rss_bytes_psutil,end_to_end_proof_call_seconds=elapsed))
        results.append(dict(arm=arm,ok=res.ok,num_constraints=res.num_constraints,
            variant=witness.variant,backend=witness.backend,n=witness.n,d=witness.d,k=witness.k,
            m=witness.m,q=witness.q,precision_bits=witness.benchmark_metrics.get('precision_bits'),
            no_range_batch=witness.no_range_batch,
            witness_quality={**witness.benchmark_metrics,'variant':witness.variant},
            prove_seconds=res.prove_seconds,verify_seconds=res.verify_seconds,proof_bytes=res.proof_bytes,
            peak_rss_bytes=res.peak_rss_bytes_psutil,compile_seconds=res.compile_seconds,
            setup_seconds=res.setup_seconds,witness_seconds=res.witness_seconds,
            successful_proofs=res.successful_proofs,failure_kind=res.failure_kind,error=res.error,
            attempted_measured_proofs=settings['proof_reps'],n_failure=settings['proof_reps']-res.successful_proofs,
            verify_ok=res.verify_ok,peak_go_heap_sys_bytes=res.peak_rss_bytes_go,
            peak_os_rss_bytes=res.peak_rss_bytes_os,
            proof_call_seconds=elapsed))
        write_json(out/'proof_arms.json',results)
    return {**inst.benchmark_metrics,'status':'ok' if all(r['ok'] for r in results) else 'proof_failed',
        'scope':task['scope'],'quality_scope':'observed_single_quantized_witness_not_CI_qualified',
        'native_selection_does_not_qualify_quantized_proof':True,
        'ensemble_qualification':qualification,
        'arms':results,'raw_file':'proof_arms.json'}


def quantized_quality_samples(row,data,seeds,starts,out):
    """Geometry of exact integer maps; authenticate the selected proof separately.

    Reuse one native input commitment so large X is not rehashed fifty times.
    Fresh rho makes independent commitment-bound matrices within each start.
    This is a fidelity ensemble, not a deployment TaskLog/Q benchmark.
    """
    import numpy as np
    from commitjl.commit import commit_matrix
    from commitjl.protocol import derive_seed
    from commitjl.sparse_gen import sparse_matrix_gen
    from commitjl.zk_bridge import field_hash_theta
    from commitjl.clustering import kmeans_paired,nearest_centroid_assign,phi_X
    from commitjl.quantize import quantize,scale_for_precision
    from commitjl.metrics import clustering_agreement,sample_pair_set,delta_uv,distortion_summary
    from experiments.pipeline import normalize_for_quantization,build_projection,_centroids_from_labels
    S=scale_for_precision(row['precision_bits']);Xq=quantize(normalize_for_quantization(data['X'],S,row['B']),S,row['B'])
    n,d=Xq.shape;k=row['k'];m=row['m'];q=row['q'];variant=row['variant']
    opening=commit_matrix(Xq)
    write_json(out/'quality_commitment.json',dict(c_x=str(opening.root),r_x=str(opening.salt),
        dataset_sha256=data['meta']['sha256'],scope='native_geometric_ensemble_reuses_input_opening'))
    pairs=sample_pair_set(n,min(2000,n*(n-1)//2),20261005)
    save_npz(out/'quality_input.npz',Xq=Xq,pairs=np.asarray(pairs))
    baselines=[]
    for start in range(starts):
        Cq=np.rint(kmeans_paired(Xq.astype(float),k,seeds=[start])[0][1]).astype('int64')
        labels=nearest_centroid_assign(Xq,Cq);cost=phi_X(Xq,Cq)
        save_npz(out/f'baseline_start{start}.npz',Cq=Cq,labels=labels)
        baselines.append((Cq,labels,cost,field_hash_theta(Cq)))
    samples=[]
    map_count=1 if variant=='pca-public' else seeds
    for seed in range(map_count):
        A_shared=None
        if variant!='sparse':
            name={'dense':'rademacher','coord':'coordinate','pca-public':'pca'}[variant]
            A_shared=build_projection(name,Xq.astype(float),m,q,seed,S)['int']
        for start,(Cq,baseline,full_cost,c_theta) in enumerate(baselines):
            rho=os.urandom(32) if variant=='sparse' else None
            projection_seed=derive_seed(opening.root,c_theta,rho) if rho is not None else None
            A=sparse_matrix_gen(projection_seed,m,d,q)[1] if projection_seed is not None else A_shared
            Z=(A@Xq.T).T;Cz=(A@Cq.T).T
            assigned=nearest_centroid_assign(Z,Cz)
            agreement=clustering_agreement(baseline,assigned)
            projected_cost=phi_X(Xq,_centroids_from_labels(Xq,assigned,k))
            factor=1/np.sqrt(q) if variant=='sparse' else 1/np.sqrt(m) if variant=='dense' else 1/S if variant=='pca-public' else 1.
            delta=delta_uv(Xq,Z.astype(float)*factor,pairs)
            distortion=distortion_summary(delta,.15)
            save_npz(out/f'map_seed{seed}_start{start}.npz',A=A,Z=Z,labels=assigned,pair_delta=delta)
            samples.append(dict(seed=seed,clustering_start=start,ari=agreement['ari'],nmi=agreement['nmi'],
                projection_normalization=factor,quantization_scale=S,
                r_phi=projected_cost/full_cost if full_cost else float('nan'),
                delta_q95=distortion['q95'],delta_exceedance=distortion['exceedance_rate'],n=n,d=d,k=k,m=A.shape[0],q=q,
                dataset_sha256=data['meta']['sha256'],c_x=str(opening.root),c_theta=str(c_theta),
                rho_hex=rho.hex() if rho is not None else None,
                projection_seed_hex=projection_seed.hex() if projection_seed is not None else None))
    return samples


def worker(task, out, settings):
    out.mkdir(parents=True,exist_ok=True)
    if task.get('unavailable'):
        return dict(status='unavailable',error=task['unavailable'])
    if task.get('excluded'):
        return dict(status='excluded',reason=task['excluded'])
    stage=task['stage']
    if stage in ['rq2_pilot','rq2_quality','rq2_selected','rq3_proof','rq5_proof','rq5_ablation']:
        return prove_measure(task,out,settings)
    if stage=='rq4_centroid':
        return centroid_export(task,out)
    if stage=='rq4_dbscan':
        import pandas as pd
        cache={};samples=[]
        for m in task['m_values']:
            for q in task['q_values']:
                for seed in range(task['seeds']):
                    directory=out/f'm{m}_q{q}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
                    result=dbscan_measure({**task,'row':{**task['row'],'m':m,'q':q,'seed':seed}},directory,cache=cache)
                    result.setdefault('status','ok');result['artifact_directory']=str(directory.resolve())
                    samples.append(result)
        pd.DataFrame(samples).to_parquet(out/'dbscan_rows.parquet',index=False)
        return dict(rows=len(samples),raw_file='dbscan_rows.parquet')
    if stage=='rq3_native':
        return ordering_measure(task,out)
    if stage=='rq3_grinding':
        from experiments.pipeline import load_row_dataset,build_projection
        from commitjl.metrics import sample_pair_set,delta_uv
        import numpy as np
        row=task['row']; X=load_row_dataset(row)['X']
        pairs=sample_pair_set(len(X),2000,20261005)
        trial_seeds=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in np.random.SeedSequence(20261005).spawn(task['trials'])]
        deltas=[];maps=[];flags=[]
        for seed in trial_seeds:
            P=build_projection('sparse_jl',X,row['m'],row['q'],seed,2**16)['real']
            delta=delta_uv(X,X@P.T,pairs);finite=delta[np.isfinite(delta)]
            if not len(finite): raise ValueError('no distinct pairs for grinding')
            deltas.append(delta);maps.append(P);flags.append(bool(np.max(np.abs(finite))>.15))
        rate=float(np.mean(flags))
        save_npz(out/'grinding_trials.npz',X=X,pairs=np.asarray(pairs),trial_seeds=np.asarray(trial_seeds,dtype=np.uint64),
                 pair_delta=np.asarray(deltas),P=np.asarray(maps),flags=np.asarray(flags))
        write_json(out/'trial_flags.json',flags)
        groups=[]
        for Q in [1,2,4,8,16]:
            used=np.array(flags[:len(flags)//Q*Q],dtype=bool)
            groups.append(dict(Q=Q,batches=len(used)//Q,empirical_success=float(used.reshape(-1,Q).any(axis=1).mean()),
                empirical_plugin_bound=min(1.,Q*float(used.mean()))))
        write_json(out/'grinding.json',groups)
        return dict(delta_geom=rate,trials=len(flags),scope='empirical_sampled_pairs_plugin_bound_not_theorem',raw_file='grinding.json')
    from experiments.pipeline import run_native,PipelineCache,load_row_dataset
    from experiments.rq5_precision import measure_precision_row
    from experiments.rq4_certificates import run_centroid_cert
    import pandas as pd
    cache=PipelineCache(); precision_cache={}
    data=load_row_dataset(task['row']) if stage=='rq5_precision' else None
    if data is None: data,_=cache.dataset(task['row'])
    if data is not None:
        save_npz(out/'dataset.npz',X=data['X'])
    samples=[]
    for seed in range(task['seeds']):
        for start in range(task['starts']):
            row={**task['row'],'seed':seed,'clustering_start':start}
            t=time.perf_counter()
            if stage=='rq4_phase':
                result=run_centroid_cert(row,PREREG,cache=cache,
                    artifact_path=out/f'centroid_seed{seed}_start{start}.npz')
            else:
                result=(measure_precision_row(row,PREREG,data=data,cache=precision_cache,
                    artifact_path=out/f'labels_seed{seed}_start{start}.npz') if stage=='rq5_precision'
                        else run_native(row,PREREG,cache=cache,
                            artifact_path=out/f'native_seed{seed}_start{start}.npz'))
            result['row_call_seconds']=time.perf_counter()-t
            samples.append(result)
    pd.DataFrame(samples).to_parquet(out/'native_rows.parquet',index=False)
    return dict(rows=len(samples),raw_file='native_rows.parquet',
        timing_scope='shared_baseline_and_projection_cache_for_native_scaling; row_call_seconds_is_not_cold_latency')


def kill_tree(proc):
    import psutil
    try:
        children=psutil.Process(proc.pid).children(recursive=True)
        for child in children:
            try: child.kill()
            except psutil.Error: pass
    except psutil.Error: pass
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def recover_proof_checkpoint(directory, result):
    checkpoint=directory/'proof_arms.json'
    if 'arms' not in result and checkpoint.exists():
        result['arms']=json.loads(checkpoint.read_text())
        result['completed_proof_arms']=len(result['arms'])
        result['proof_checkpoint_scope']='completed arms only; task failure remains in inventory'
    return result


def execute(task, root, settings):
    import psutil
    directory=root/task['stage']/task['id']; directory.mkdir(parents=True,exist_ok=True)
    result_file=directory/'result.json'
    if result_file.exists():
        return json.loads(result_file.read_text())
    if (directory/'proof_arms.json').exists():
        result=recover_proof_checkpoint(directory,dict(status='failed',error='previous_task_interrupted',
            task_id=task['id'],stage=task['stage']))
        write_json(result_file,result)
        return result
    write_json(directory/'task.json',task)
    if shutil.disk_usage(root).free < settings.get('min_free_gib',5)*1024**3:
        result=dict(status='failed',error='disk_space_limit',task_id=task['id'],stage=task['stage'])
        write_json(result_file,result);return result
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','GOMAXPROCS']:
        env[key]=str(settings['threads'])
    if task['stage'] in ['rq2_pilot','rq2_selected','rq3_proof','rq5_proof','rq5_ablation']:
        env['GOMAXPROCS']=str(settings.get('proof_threads',16))
    start=time.perf_counter(); peak=0; reason=None
    with (directory/'run.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-m','experiments.rq2345_complete',
            '--worker',str(directory/'task.json'),'--settings',str(root/'settings.json')],
            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                if STOP.is_set():
                    reason='interrupted';kill_tree(proc);break
                try:
                    parent=psutil.Process(proc.pid)
                    rss=0
                    for p in [parent]+parent.children(recursive=True):
                        try: rss+=p.memory_info().rss
                        except psutil.Error: pass
                    peak=max(peak,rss)
                except psutil.Error: pass
                if peak>settings['task_rss_gib']*1024**3:
                    reason='memory_limit'; kill_tree(proc); break
                if time.perf_counter()-start>settings['task_timeout']:
                    reason='timeout'; kill_tree(proc); break
                time.sleep(.1)
        except BaseException as exc:
            kill_tree(proc)
            result=recover_proof_checkpoint(directory,dict(status='failed',
                error='interrupted' if isinstance(exc,KeyboardInterrupt) else 'parent_exception',
                task_id=task['id'],stage=task['stage'],elapsed_seconds=time.perf_counter()-start,
                monitored_peak_rss_bytes=peak,child_exit_code=proc.returncode))
            write_json(result_file,result)
            raise
    result=json.loads(result_file.read_text()) if result_file.exists() else dict(status='failed',error=reason or 'child_process_failed')
    if reason or proc.returncode != 0:
        result.update(status='failed',error=reason or 'child_process_failed')
    result=recover_proof_checkpoint(directory,result)
    result.update(task_id=task['id'],stage=task['stage'],elapsed_seconds=time.perf_counter()-start,
        monitored_peak_rss_bytes=peak,child_exit_code=proc.returncode)
    write_json(result_file,result)
    return result


def export(root):
    import numpy as np
    import pandas as pd
    rows=[]
    tasks=json.loads((root/'plan.json').read_text())
    for task in tasks:
        path=root/task['stage']/task['id']/'result.json'
        row=json.loads(path.read_text()) if path.exists() else dict(status='not_run')
        rows.append({**task.get('row',{}),**row,'task_id':task['id'],'stage':task['stage'],
            'name':task.get('name','synthetic'),'artifact_directory':str(path.parent.resolve())})
    flat=pd.json_normalize(rows,sep='.')
    flat.to_csv(root/'inventory.csv',index=False)
    summaries=[]
    # Preserve all factor and dataset identities. Crossed bootstrap only for complete axes.
    expanded=[]
    for row in rows:
        filename={'rq4_centroid':'centroid_rows.parquet','rq4_dbscan':'dbscan_rows.parquet'}.get(row['stage'],'native_rows.parquet')
        raw=Path(row['artifact_directory'])/filename
        if row['status']=='ok' and raw.exists():
            for r in pd.read_parquet(raw).to_dict('records'):
                if row['stage']=='rq4_centroid':
                    r['seed']=r['projection_seed']
                expanded.append({**row,**r})
        else: expanded.append(row)
    frame=pd.DataFrame(expanded)
    pd.json_normalize(expanded,sep='.').to_csv(root/'all_rows.csv',index=False)
    for stage in ['rq5_native','rq5_precision','rq4_dbscan','rq4_phase','rq4_centroid']:
        subset=frame[(frame.stage==stage)&(frame.status=='ok')]
        if subset.empty: continue
        subset=subset.copy()
        groupcols=['name','m','q']+(['sweep_factor','sweep_value','k'] if stage in ['rq5_native','rq4_phase'] else
                                   ['precision_bits','k'] if stage=='rq5_precision' else ['k'] if stage=='rq4_centroid' else [])
        subset['group']=subset.apply(lambda r:json.dumps({c:r[c] for c in groupcols},sort_keys=True),axis=1)
        for key,group in subset.groupby('group'):
            record=dict(stage=stage,**json.loads(key),rows=len(group))
            for field in ['n','d','dataset_sha256']:
                if field in group and group[field].nunique(dropna=True)==1:
                    record[field]=group[field].dropna().iloc[0]
            for metric in ['ari','nmi','agreement','r_phi','delta_median','delta_q95','delta_q99','delta_max','delta_exceedance',
                           'margin_coverage','coverage','qualified_coverage','conditional_violation_rate','uncertified_stability',
                           'mean_abs_quantization_error','near_band_stability','core_agreement',
                           't_kmeans_s','t_proj_build_s','t_assign_s','row_call_seconds']:
                if metric not in group or group[metric].isna().any(): continue
                if 'clustering_start' in group and group.clustering_start.notna().all():
                    matrix=group.pivot(index='seed',columns='clustering_start',values=metric)
                else:
                    matrix=group.set_index('seed')[[metric]]
                settings=json.loads((root/'settings.json').read_text())
                complete=(len(matrix)==settings['seeds'] and
                    matrix.shape[1]==(settings['starts'] if stage!='rq4_dbscan' else 1) and
                    not matrix.isna().any().any())
                record['complete']=complete
                if complete:
                    mean,lo,hi=bootstrap_crossed(matrix.to_numpy(),settings['bootstrap'])
                    record.update({metric+'_mean':mean,metric+'_ci_low':lo,metric+'_ci_high':hi})
            summaries.append(record)
    summary=pd.DataFrame(summaries) if summaries else pd.DataFrame(columns=['stage','name','m','q','rows','complete'])
    summary.to_csv(root/'summary.csv',index=False)
    summary.to_latex(root/'summary.tex',index=False)
    plot_exports(root,summary,rows)
    scaling_exports(root,frame)
    artifacts=[]
    import zipfile
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix not in ['.npz','.parquet','.json','.jsonl','.log']: continue
        arrays=[]
        if path.suffix=='.npz':
            with zipfile.ZipFile(path) as archive: arrays=[Path(name).stem for name in archive.namelist()]
        artifacts.append(dict(relative_path=path.relative_to(root).as_posix(),bytes=path.stat().st_size,arrays=';'.join(arrays)))
    pd.DataFrame(artifacts,columns=['relative_path','bytes','arrays']).to_csv(root/'artifact_index.csv',index=False)
    return rows


def scaling_exports(root,frame):
    import numpy as np
    import pandas as pd
    settings=json.loads((root/'settings.json').read_text())
    planned=json.loads((root/'plan.json').read_text())
    def planned_levels(stage,factor,backend=None):
        return {t['row']['sweep_value'] for t in planned if t['stage']==stage
                and t.get('row',{}).get('sweep_factor')==factor
                and (backend is None or t['row'].get('backend')==backend)}
    slopes=[]
    if not frame.empty:
        native=frame[(frame.stage=='rq5_native')&(frame.status=='ok')]
        if not native.empty:
            for factor,group in native.groupby('sweep_factor'):
                if factor=='gamma': continue
                for metric in ['t_assign_s','t_proj_build_s','t_kmeans_s']:
                    estimates=[]
                    for (seed,start),g in group.groupby(['seed','clustering_start']):
                        valid=g[(g.sweep_value>0)&(g[metric]>0)]
                        if len(valid)>=3 and set(valid.sweep_value)==planned_levels('rq5_native',factor):
                            estimates.append(dict(seed=seed,start=start,slope=float(np.polyfit(
                                np.log(valid.sweep_value.to_numpy(float)),np.log(valid[metric].to_numpy(float)),1)[0])))
                    if estimates:
                        matrix=pd.DataFrame(estimates).pivot(index='seed',columns='start',values='slope')
                        complete=(matrix.shape==(settings['seeds'],settings['starts']) and not matrix.isna().any().any())
                        result=dict(scope='native_cached_components',factor=factor,metric=metric,complete=complete,
                            levels=int(group.sweep_value.nunique()))
                        if complete:
                            mean,lo,hi=bootstrap_crossed(matrix.to_numpy(),settings['bootstrap'])
                            result.update(slope=mean,ci_low=lo,ci_high=hi)
                        slopes.append(result)
                    else:
                        slopes.append(dict(scope='native_cached_components',factor=factor,metric=metric,
                            complete=False,levels=int(group.sweep_value.nunique()),
                            CI_note='missing planned levels or fewer than three valid levels; no slope reported'))
    path=root/'proof_summary.csv'
    if path.exists() and path.stat().st_size>1:
        try: proofs=pd.read_csv(path)
        except pd.errors.EmptyDataError: proofs=pd.DataFrame()
        if not proofs.empty:
            for (backend,factor),group in proofs[(proofs.stage=='rq5_proof')&proofs.ok].groupby(['backend','sweep_factor']):
                if factor=='precision_bits': continue
                for metric in ['prove_median','num_constraints']:
                    valid=group[(group.sweep_value>0)&(group[metric]>0)]
                    if valid.sweep_value.nunique()>=3:
                        slope=float(np.polyfit(np.log(valid.sweep_value.to_numpy(float)),np.log(valid[metric].to_numpy(float)),1)[0])
                        slopes.append(dict(scope='measured_proof_engineering_shapes',factor=factor,backend=backend,
                            metric=metric,slope=slope,levels=int(valid.sweep_value.nunique()),
                            complete=set(valid.sweep_value)==planned_levels('rq5_proof',factor,backend),
                            CI_note='descriptive fit; repeated proof timings use one witness per level'))
    (pd.DataFrame(slopes) if slopes else pd.DataFrame(columns=['scope','factor','metric','slope','ci_low','ci_high'])).to_csv(root/'scaling_slopes.csv',index=False)
    inventory=pd.read_csv(root/'inventory.csv')
    if not inventory.empty:
        inventory.groupby(['stage','status'],dropna=False).size().rename('tasks').reset_index().to_csv(root/'failure_counts.csv',index=False)


def plot_exports(root, summary, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    folder=root/'figures';folder.mkdir(exist_ok=True)
    if not summary.empty:
        for stage in summary.stage.unique():
            data=summary[summary.stage==stage]
            factors=data.sweep_factor.dropna().unique() if 'sweep_factor' in data and stage=='rq5_native' else [None]
            for factor in factors:
                subset=data[data.sweep_factor==factor] if factor else data
                if stage=='rq4_phase':
                    for metric in ['qualified_coverage','conditional_violation_rate']:
                        y=metric+'_mean'
                        if y not in subset: continue
                        cells=subset.pivot(index='sweep_value',columns='m',values=y)
                        fig,ax=plt.subplots(figsize=(6,4))
                        im=ax.imshow(cells.to_numpy(float),aspect='auto',origin='lower',vmin=0,vmax=1)
                        ax.set_xticks(range(len(cells.columns)),labels=cells.columns)
                        ax.set_yticks(range(len(cells.index)),labels=cells.index)
                        ax.set(xlabel='m',ylabel='gamma',title=metric)
                        fig.colorbar(im,ax=ax);fig.tight_layout()
                        fig.savefig(folder/f'phase_{metric}.pdf');fig.savefig(folder/f'phase_{metric}.png',dpi=150);plt.close(fig)
                    continue
                x='sweep_value' if factor else 'precision_bits' if stage=='rq5_precision' else 'm'
                for metric in sorted(c[:-5] for c in subset.columns if c.endswith('_mean')):
                    y=metric+'_mean'
                    if y not in subset or not subset[y].notna().any(): continue
                    fig,ax=plt.subplots(figsize=(6,4))
                    for name,group in subset.groupby('name'):
                        # q and k are separate series, never averaged away.
                        groups=[c for c in ['q','k'] if c in group and group[c].notna().any() and c!=factor]
                        iterator=group.groupby(groups,dropna=False) if groups else [('all',group)]
                        for keys,g in iterator:
                            g=g.dropna(subset=[y]).sort_values(x)
                            if g.empty: continue
                            ax.plot(g[x],g[y],marker='o',label=f'{name}, {keys}')
                            ax.fill_between(g[x].to_numpy(float),g[metric+'_ci_low'].to_numpy(float),
                                g[metric+'_ci_high'].to_numpy(float),alpha=.15)
                    ax.set(xlabel=factor or x,ylabel=metric);ax.legend(fontsize=7);fig.tight_layout()
                    stem=f'{stage}_{factor or x}_{metric}'
                    fig.savefig(folder/(stem+'.pdf'));fig.savefig(folder/(stem+'.png'),dpi=150);plt.close(fig)
    proofs=[];ordering=[]
    for row in rows:
        folder_path=Path(row['artifact_directory'])
        for arm in row.get('arms',[]):
            raw=folder_path/(arm['arm']+'_proof.json')
            details=json.loads(raw.read_text()) if raw.exists() else {}
            samples=details.get('raw',{}).get('prove_seconds_samples',[])
            record={'stage':row['stage'],'task_id':row['task_id'],'task_status':row['status'],'variant':row.get('variant'),
                **{key:row.get(key) for key in ['n','d','k','m','q','backend','sweep_factor','sweep_value','precision_bits','name']},
                **arm,'prove_median':float(np.median(samples)) if samples else None,
                'samples':samples}
            # Each arm can change dimensions, precision or statement variant.
            # Saved witnesses also repair metadata when exporting older raw runs.
            quality=arm.get('witness_quality',{})
            for key in ['n','d','k','m','q','backend','variant','precision_bits','normalization_reference_bits',
                        'ari','nmi','agreement','r_phi','ari_physical_reference','r_phi_physical_reference',
                        'dataset_sha256','source_dataset_sha256']:
                if key in quality: record[key]=quality[key]
            witness_path=folder_path/(arm['arm']+'_witness.json')
            if witness_path.exists():
                witness=json.loads(witness_path.read_text())
                record.update({key:witness[key] for key in ['n','d','k','m','q','backend','variant','no_range_batch'] if key in witness})
            proofs.append(record)
        path=folder_path/'timings.json'
        if path.exists():
            data=pd.DataFrame(json.loads(path.read_text()))
            pivot=data.pivot(index='repetition',columns='arm',values='total_seconds')
            for repetition,r in pivot.iterrows():
                ordering.append(dict(task_id=row['task_id'],source=row.get('source'),
                    repetition=repetition,postcommit=r.postcommit,prepublished=r.prepublished,
                    paired_difference=r.postcommit-r.prepublished))
    (pd.DataFrame(proofs) if proofs else pd.DataFrame(columns=['stage','task_id','variant','arm','ok','backend',
        'sweep_factor','sweep_value','prove_median','num_constraints'])).to_csv(root/'proof_summary.csv',index=False)
    (pd.DataFrame(ordering) if ordering else pd.DataFrame(columns=['task_id','source','repetition','postcommit',
        'prepublished','paired_difference'])).to_csv(root/'ordering_pairs.csv',index=False)
    if proofs:
        successful=[p for p in proofs if p['ok'] and p['samples']]
        if successful:
            fig,ax=plt.subplots(figsize=(13,max(4,len(successful)*.35)))
            import inspect
            orientation={'orientation':'horizontal'} if 'orientation' in inspect.signature(ax.boxplot).parameters else {'vert':False}
            ax.boxplot([p['samples'] for p in successful],**orientation)
            ax.set_yticks(range(1,len(successful)+1),labels=[f"{p['stage']}:{p['variant']}:{p['arm']}" for p in successful])
            ax.tick_params(axis='y',labelsize=8);ax.set_xlabel('Proof seconds');fig.tight_layout()
            fig.savefig(folder/'proof_times.pdf',bbox_inches='tight');fig.savefig(folder/'proof_times.png',dpi=150,bbox_inches='tight');plt.close(fig)


def main():
    run_start=time.perf_counter()
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',action='store_true'); ap.add_argument('--export',action='store_true')
    ap.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES)
    ap.add_argument('--rq1',type=Path,default=Path('results/rq1/runs/paper_balanced_v3'))
    ap.add_argument('--out',type=Path,default=Path('results/complete_rq2345/paper_balanced_v1'))
    ap.add_argument('--datasets',nargs='+',choices=list(DATASETS),default=list(DATASETS))
    ap.add_argument('--seeds',type=int,default=10); ap.add_argument('--starts',type=int,default=5)
    ap.add_argument('--jobs',type=int,default=4); ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--proof-threads',type=int,default=16)
    ap.add_argument('--backends',nargs='+',choices=['groth16','plonk'],default=['groth16'])
    ap.add_argument('--proof-reps',type=int,default=10); ap.add_argument('--proof-rss-gib',type=float,default=40)
    ap.add_argument('--proof-points',type=int,default=0,help='explicit fixed RQ2 row subset; 0 preserves all RQ1 rows, dimensions never truncated')
    ap.add_argument('--max-proof-input-elements',type=int,default=8192,
        help='RQ2 feasibility gate on n*d; exceeding it is an exclusion, never a silent subset or measured OOM')
    ap.add_argument('--task-rss-gib',type=float,default=48); ap.add_argument('--proof-timeout',type=float,default=7200)
    ap.add_argument('--task-timeout',type=float,default=14400)
    ap.add_argument('--min-free-gib',type=float,default=5)
    ap.add_argument('--dbscan-points',type=int,default=1000)
    ap.add_argument('--grinding-trials',type=int,default=256)
    ap.add_argument('--bootstrap',type=int,default=2000); ap.add_argument('--limit',type=int)
    ap.add_argument('--retry-failed',action='store_true',help='archive failed attempts and retry with identical settings')
    ap.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    ap.add_argument('--settings',type=Path,help=argparse.SUPPRESS)
    args=ap.parse_args()
    if args.worker:
        task=json.loads(args.worker.read_text()); settings=json.loads(args.settings.read_text())
        t=time.perf_counter()
        try:
            from experiments.common import source_fingerprint
            if settings.get('source_sha256') and settings['source_sha256']!=source_fingerprint():
                raise ValueError('Source changed during campaign; use a fresh output directory')
            result=worker(task,args.worker.parent,settings); result.setdefault('status','ok')
        except Exception as e:
            import traceback
            traceback.print_exc(); result=dict(status='failed',error=str(e))
        result['worker_seconds']=time.perf_counter()-t
        write_json(args.worker.parent/'result.json',result)
        return
    root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    lock_handle=campaign_lock(root)
    if args.export:
        export(args.out); return
    if min(args.seeds,args.starts,args.jobs,args.threads,args.proof_threads,args.proof_reps,args.bootstrap)<1 or args.dbscan_points<5 or args.grinding_trials<16:
        ap.error('positive repetition/resource counts, >=5 DBSCAN points and >=16 grinding trials required')
    if args.jobs>4 or args.jobs*args.threads>16:
        ap.error('this server preset supports <=4 jobs and jobs*threads <=16')
    if not 0<args.proof_rss_gib<args.task_rss_gib<=48 or min(args.proof_timeout,args.task_timeout,args.min_free_gib)<=0:
        ap.error('require 0 < proof RSS < task RSS <=48 GiB and positive timeout/disk budgets')
    if args.limit is not None and args.limit<1: ap.error('--limit must be positive')
    if args.proof_points<0: ap.error('--proof-points must be nonnegative')
    if args.max_proof_input_elements<1: ap.error('--max-proof-input-elements must be positive')
    from experiments.common import source_fingerprint,capture_env
    tasks=plan(args)
    if args.limit: tasks=tasks[:args.limit]
    settings={k:v for k,v in vars(args).items() if k not in ['worker','settings','run','export','retry_failed']}
    settings={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in settings.items()}
    settings['source_sha256']=source_fingerprint()
    settings['rq1_inputs']={}
    if any(s in args.stages for s in ['rq2_selected','rq2_quality','rq4_centroid','rq4_dbscan']):
        for name in args.datasets:
            for filename in ['manifest.json','selected_operating_points.csv','completion.json']:
                path=args.rq1/name/filename
                settings['rq1_inputs'][f'{name}/{filename}']=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    existing=root/'settings.json'
    if existing.exists() and json.loads(existing.read_text())!=settings:
        raise ValueError('campaign source/config changed; use a fresh --out directory')
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text())!=tasks:
        raise ValueError('RQ1 selection/task plan changed; use a fresh --out directory')
    write_json(existing,settings); write_json(root/'plan.json',tasks)
    write_json(root/'environment.json',capture_env())
    print(json.dumps(dict(tasks=len(tasks),output=str(root),stages={s:sum(t['stage']==s for t in tasks) for s in args.stages}),indent=2))
    if not args.run:
        print('Plan only. Add --run to execute.'); return
    if args.retry_failed:
        for task in tasks:
            directory=root/task['stage']/task['id'];path=directory/'result.json'
            failed=path.exists() and json.loads(path.read_text()).get('status') not in ['ok','excluded']
            interrupted=(directory/'proof_arms.json').exists() and not path.exists()
            if failed or interrupted:
                archive=directory/'attempts'/str(time.time_ns());archive.mkdir(parents=True)
                for old in list(directory.iterdir()):
                    if old.is_file(): old.replace(archive/old.name)
    # Timings are isolated; only untimed certificate/precision tasks overlap.
    isolated=sorted([t for t in tasks if t['stage'] in ['rq2_pilot','rq2_quality','rq2_selected','rq3_native','rq3_proof','rq5_native','rq5_proof','rq3_grinding','rq5_ablation']],
                    key=lambda t:STAGES.index(t['stage']))
    parallel=[t for t in tasks if t not in isolated]
    for task in isolated:
        result=execute(task,root,settings)
        print(task['stage'],task['id'],result['status'],flush=True)
    native_settings={**settings,'task_rss_gib':min(8.,settings['task_rss_gib'])}
    # Worker reads settings file; monitor uses the smaller aggregate-safe budget.
    pool=ThreadPoolExecutor(max_workers=min(args.jobs,4))
    try:
        for result in pool.map(lambda t:execute(t,root,native_settings),parallel):
            print(result['stage'],result['task_id'],result['status'],flush=True)
    except BaseException:
        STOP.set()
        raise
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    rows=export(root)
    elapsed=time.perf_counter()-run_start
    write_json(root/'run_resource_summary.json',dict(active_invocation_seconds=elapsed,
        quoted_hourly_rial=117150,estimated_active_invocation_rial=elapsed/3600*117150,
        billing_note='Arithmetic using user quoted rate; excludes server idle time and provider billing increments',
        max_observed_task_tree_rss_bytes=max((r.get('monitored_peak_rss_bytes',0) for r in rows),default=0),
        parallel_native_rss_limit_gib=8*min(args.jobs,4),serial_proof_rss_limit_gib=args.proof_rss_gib))
    bad=sum(r.get('status') not in ['ok','excluded'] for r in rows)
    print(f'Finished with {bad} failed/unavailable tasks; inventory.csv retains every attempt.')
    if bad: raise SystemExit(1)


if __name__=='__main__':
    main()
