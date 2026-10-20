"""Offline, checkpointed native paper comparison with explicit timing boundaries.

Worker entry point is launched by campaigns.paper_compression. This suite does
not replace ZK measurements or silently pool older campaigns with new results.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import threading
import time

import numpy as np
import psutil
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from commitjl.certificates import pair_squared_distances
from commitjl.metrics import sample_pair_set
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import NativeProjection, PreparedDataset, array_hash, file_hash, stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.laptop_comparison import atomic_json, save_arrays
from experiments.protected_jl_study import Archive
from experiments.paper_artifacts import artifact_issues

CODE = Path(__file__).resolve().parents[1]
METHODS = ('full', 'pca', 'pca_randomized', 'sparse_jl', 'protected_jl', 'protected_only')
STOCHASTIC = {'pca_randomized','sparse_jl','protected_jl'}


def bootstrap_crossed(values, draws=2000, seed=9876):
    """Vectorized independent resampling of the projection and start axes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError('Requires a complete finite seed by start matrix')
    rng = np.random.default_rng(seed)
    i = rng.integers(values.shape[0], size=(draws, values.shape[0]))
    j = rng.integers(values.shape[1], size=(draws, values.shape[1]))
    means = values[i[:,:,None],j[:,None,:]].mean(axis=(1,2))
    low, high = np.quantile(means, [.025,.975])
    return float(values.mean()), float(low), float(high)


def sqdist(x, c):
    return stable_squared_distances(x,c)


def assignment(z, centers, norms, center_norms):
    scores = 2*(z@centers.T)-center_norms
    labels = np.argmax(scores, axis=1)
    costs = norms-scores[np.arange(len(z)),labels]
    tol = 64*np.finfo(float).eps*(norms+center_norms.max(initial=0))
    top = np.sort(scores,axis=1)
    doubtful = (costs <= tol) | ((top[:,-1]-top[:,-2] <= tol) if len(centers)>1 else False)
    if np.any(doubtful):
        direct=sqdist(z[doubtful],centers)
        labels[doubtful]=direct.argmin(axis=1)
        costs[doubtful]=direct.min(axis=1)
    return labels, float(np.maximum(costs,0).sum())


def safe_ratio(numerator,denominator):
    # 0/0 is undefined, not evidence of perfect fidelity.
    return float(numerator/denominator) if denominator>0 else None


def reconstructed_centers(x, labels, k):
    counts = np.bincount(labels, minlength=k)
    sums = np.zeros((k, x.shape[1]))
    np.add.at(sums, labels, x)
    keep = counts > 0
    return sums[keep]/counts[keep,None]


def split_indices(n, seed=481516):
    order = np.random.default_rng(seed).permutation(n)
    train_n, valid_n = int(.6*n), int(.2*n)
    if min(train_n, valid_n, n-train_n-valid_n) < 2:
        raise ValueError('Insufficient rows for 60/20/20 split')
    return dict(train=np.sort(order[:train_n]), validation=np.sort(order[train_n:train_n+valid_n]),
                test=np.sort(order[train_n+valid_n:]))


def distortion(reference, projected, eps, tolerance):
    positive = reference > 0
    signed = projected[positive]/reference[positive]-1
    error = np.abs(signed)
    return dict(q50=float(np.quantile(error, .5)) if len(error) else None,
                q95=float(np.quantile(error, .95)) if len(error) else None,
                q99=float(np.quantile(error, .99)) if len(error) else None,
                maximum=float(error.max()) if len(error) else None, exceedance=float(np.mean(error > eps)) if len(error) else None,
                signed_mean=float(signed.mean()) if len(signed) else None, positive_count=int(positive.sum()),
                zero_count=int(np.count_nonzero(~positive)),
                zero_violations=int(np.count_nonzero(projected[~positive] > tolerance)))


def timed(function, repetitions=1, warmup=False):
    """RSS covers a shared worker; delta is not a standalone allocator measure."""
    if warmup:
        value = function()
        del value
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()
    def monitor():
        while not stop.wait(.02):
            try: peak[0] = max(peak[0], process.memory_info().rss)
            except psutil.Error: return
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    samples = []
    value = None
    try:
        for _ in range(repetitions):
            value = None
            begin = time.perf_counter(); value = function(); samples.append(time.perf_counter()-begin)
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set(); watcher.join(timeout=1)
    return value, dict(samples=samples, median=statistics.median(samples),
                       rss_baseline_bytes=baseline, peak_rss_bytes=peak[0],
                       peak_rss_delta_bytes=max(0, peak[0]-baseline))


def paired_timed(method, full, repetitions, rng, warmup=True):
    """Warm both controls, then interleave each measured repetition."""
    functions={'method':method,'full':full};values={};records={key:[] for key in functions};orders=[]
    if warmup:
        for key in rng.permutation(list(functions)): functions[key]()
    iterations={key:[] for key in functions}
    for _ in range(repetitions):
        order=list(rng.permutation(list(functions)));orders.append(order)
        for key in order:
            values[key]=None
            values[key],measure=timed(functions[key])
            records[key].append(measure)
            if hasattr(values[key],'n_iter_'): iterations[key].append(int(values[key].n_iter_))
    summaries={}
    for key,measures in records.items():
        samples=[m['samples'][0] for m in measures]
        summaries[key]=dict(samples=samples,median=statistics.median(samples),
            peak_rss_bytes=max(m['peak_rss_bytes'] for m in measures),
            rss_baseline_bytes=min(m['rss_baseline_bytes'] for m in measures),
            peak_rss_delta_bytes=max(m['peak_rss_delta_bytes'] for m in measures),
            paired_execution_order=orders,iterations=iterations[key])
    return values['method'],values['full'],summaries['method'],summaries['full']


def stable_topk(values,k):
    candidates=np.argpartition(values,k-1,axis=1)[:,:k]
    answer=np.empty((len(values),k),dtype=np.int64)
    for row,indices in enumerate(candidates):
        threshold=values[row,indices].max()
        lower=np.flatnonzero(values[row]<threshold)
        equal=np.flatnonzero(values[row]==threshold)[:k-len(lower)]
        chosen=np.concatenate((lower,equal))
        answer[row]=chosen[np.lexsort((chosen,values[row,chosen]))]
    return answer


def dbscan_profile(original,projected,radius,config,rng,tolerance):
    db,db_full,dt,fdt=paired_timed(
        lambda:DBSCAN(eps=radius**2,min_samples=5,metric='precomputed').fit(projected),
        lambda:DBSCAN(eps=radius**2,min_samples=5,metric='precomputed').fit(original),
        config['timing_repetitions'],rng)
    triangle=np.triu_indices(len(original),1);ref,actual=original[triangle],projected[triangle]
    truth,guess=ref<=radius**2,actual<=radius**2;stable=truth==guess;eps=config['eps']
    band=((1+eps)*ref<radius**2)|((1-eps)*ref>radius**2)
    event=np.abs(actual-ref)<=eps*ref+tolerance
    robust=((1+eps)*ref+tolerance<radius**2)|((1-eps)*ref-tolerance>radius**2)
    certified=robust&event;violations=int(np.count_nonzero(certified&~stable))
    if violations: raise ValueError('Conditional DBSCAN certificate violation')
    tp=int(np.sum(truth&guess));fp=int(np.sum(~truth&guess));fn=int(np.sum(truth&~guess));tn=int(np.sum(~truth&~guess))
    old_core=np.sum(original<=radius**2,axis=1)>=5;new_core=np.sum(projected<=radius**2,axis=1)>=5
    record=dict(dbscan_fit=dt,full_dbscan_fit=fdt,dbscan_radius=radius,dbscan_n=len(original),dbscan_pairs=len(ref),
        dbscan_ari=float(adjusted_rand_score(db_full.labels_,db.labels_)),dbscan_graph_agreement=float(stable.mean()),
        dbscan_graph_exact=bool(stable.all()),dbscan_certified_pairs=int(certified.sum()),
        dbscan_certified_violations=violations,dbscan_global_embedding_event=bool(event.all()),
        dbscan_core_agreement=float(np.mean(old_core==new_core)),dbscan_core_fraction=float(new_core.mean()),
        full_dbscan_core_fraction=float(old_core.mean()),dbscan_noise_fraction=float(np.mean(db.labels_==-1)),
        full_dbscan_noise_fraction=float(np.mean(db_full.labels_==-1)),
        dbscan_clusters=len(set(db.labels_)-{-1}),full_dbscan_clusters=len(set(db_full.labels_)-{-1}),
        dbscan_reference_partition_informative=len(np.unique(db_full.labels_))>1,
        graph_true_positive=tp,graph_false_positive=fp,graph_false_negative=fn,graph_true_negative=tn,
        graph_precision=safe_ratio(tp,tp+fp),graph_recall=safe_ratio(tp,tp+fn),graph_f1=safe_ratio(2*tp,2*tp+fp+fn),
        graph_balanced_accuracy=((tp/(tp+fn)+tn/(tn+fp))/2) if tp+fn and tn+fp else None)
    arrays=dict(original_labels=db_full.labels_,projected_labels=db.labels_,pair_i=triangle[0],pair_j=triangle[1],
        band_certificate=band,embedding_event=event,certified=certified,graph_stable=stable,
        original_core=old_core,projected_core=new_core)
    return record,arrays


def projection_seed(source_hash, centers, phase, seed, residual_m, q, start=0):
    payload = json.dumps(dict(version='paper-compression-native-v1', source=source_hash,
        centers=array_hash(centers) if centers is not None else None,phase=phase,draw=seed,
        start=start,residual_m=residual_m,q=q),sort_keys=True).encode()
    return hashlib.sha256(payload).digest()


def create_model(method, d, m, q, seed, centers, basis, components, mean):
    if method == 'full': return NativeProjection('full', d)
    if method in ('pca','pca_randomized'): return NativeProjection('pca', d, components=components[:m], origin=mean)
    if method == 'protected_only': return NativeProjection('pca', d, components=basis.T, origin=centers[0])
    u = basis if method == 'protected_jl' else np.empty((d, 0))
    residual = m-u.shape[1]
    _, a = sparse_matrix_gen_csr(seed, residual, d, q)
    a = a.astype(float); a.data /= np.sqrt(q)
    return NativeProjection('hybrid', d, basis=u, sparse_map=a,
                            origin=centers[0] if method == 'protected_jl' else np.zeros(d))


def case_plan(config, dimensions, k, start, rank, train_n, d, phase):
    for method in METHODS:
        if method=='pca_randomized' and not config.get('randomized_pca',False): continue
        if method == 'full': yield dict(method=method, m=d, q=0, seed=-1, k=k, start=start, phase=phase)
        elif method == 'protected_only':
            if rank: yield dict(method=method, m=rank, q=0, seed=-1, k=k, start=start, phase=phase)
        else:
            for m in dimensions:
                if m >= d or (method in ('pca','pca_randomized') and m > min(train_n-1, d)) or (method == 'protected_jl' and m <= rank): continue
                for q in (config['q'] if method in ('sparse_jl', 'protected_jl') else [0]):
                    residual = m-rank if method == 'protected_jl' else m
                    if q > residual: continue
                    for draw in range(config['seeds']) if method in STOCHASTIC else [-1]:
                        yield dict(method=method, m=m, q=q, seed=draw, k=k, start=start, phase=phase)


def case_key(case):
    return f"{case['phase']}_k{case['k']}_s{case['start']}_{case['method']}_m{case['m']}_q{case['q']}_seed{case['seed']}"


def select_validation(rows, config):
    """Selection uses validation quality only; no runtime/test fields are read."""
    import pandas as pd
    frame = pd.DataFrame(rows)
    if frame.empty: return [],[]
    if set(frame.phase)!={'validation'}: raise ValueError('Selection accepts validation rows only')
    selected, summaries = [], []
    for key, group in frame.groupby(['k', 'method', 'm', 'q']):
        entry = dict(zip(['k', 'method', 'm', 'q'], key))
        for field in ('k','m','q'): entry[field] = int(entry[field])
        expected_seeds = config['seeds'] if key[1] in STOCHASTIC else 1
        expected_draws = set(range(config['seeds'])) if key[1] in STOCHASTIC else {-1}
        complete = (len(group) == config['starts']*expected_seeds and
                    set(group.seed) == expected_draws and set(group.start) == set(range(config['starts'])) and
                    not group.duplicated(['seed','start']).any())
        entry['complete'] = complete
        entry['rows'] = len(group)
        if not complete:
            entry['qualified'] = False
            summaries.append(entry)
            continue
        if 'centroid_training_hit_iteration_cap' in group and group.centroid_training_hit_iteration_cap.any():
            entry.update(qualified=False,reason='Centroid training reached its iteration cap')
            summaries.append(entry);continue
        metrics=('ari', 'r_phi_train_reconstructed', 'pair_q95', 'centroid_q95', 'projected_cost_relative_error')
        if group[list(metrics)].isna().any().any():
            entry.update(qualified=False,reason='Undefined quality endpoint (zero reference objective or no positive distances)')
            summaries.append(entry);continue
        for metric in metrics:
            matrix = group.pivot(index='seed', columns='start', values=metric).to_numpy()
            mean, low, high = bootstrap_crossed(matrix, draws=config['bootstrap'])
            entry.update({metric+'_mean':mean, metric+'_ci_low':low, metric+'_ci_high':high})
        entry['qualified'] = bool(complete and entry['ari_ci_low'] >= config['thresholds']['ari'] and
            entry['r_phi_train_reconstructed_ci_high'] <= config['thresholds']['r_phi'] and
            entry['pair_q95_ci_high'] <= config['thresholds']['distortion'] and
            entry['centroid_q95_ci_high'] <= config['thresholds']['distortion'] and
            entry['projected_cost_relative_error_ci_high'] <= config['thresholds']['projected_cost_error'] and
            group.pair_zero_violations.sum() == 0 and group.centroid_zero_violations.sum() == 0)
        summaries.append(entry)
    for (k, method), group in pd.DataFrame(summaries).groupby(['k', 'method']):
        okay = group[group.qualified].sort_values(['m', 'q'])
        selected.append(dict(k=int(k), method=method, selected=bool(len(okay)),
                             m=int(okay.iloc[0].m) if len(okay) else None,
                             q=int(okay.iloc[0].q) if len(okay) else None))
    return selected, summaries


def quality_row(case, x, y, z, projected_c, projected_train, train_x, train_labels_full,
                full_sq, pairs, pair_reference, full_reconstructed_cost, full_train_reconstructed_cost, centers, config):
    norms = np.einsum('ij,ij->i', z, z)
    projected = sqdist(z, projected_c)
    labels = projected.argmin(axis=1)
    original_labels = full_sq.argmin(axis=1)
    train_labels = sqdist(projected_train, projected_c).argmin(axis=1)
    reconstruction = reconstructed_centers(x, labels, case['k'])
    train_reconstruction = reconstructed_centers(train_x, train_labels, case['k'])
    pair_values = pair_squared_distances(z, pairs[:,0], pairs[:,1])
    tolerance = 64*np.finfo(float).eps*max(1., float(norms.max(initial=0)), float(np.sum(projected_c*projected_c, axis=1).max(initial=0)))
    pe = distortion(pair_reference, pair_values, config['eps'], tolerance)
    ce = distortion(full_sq, projected, config['eps'], tolerance)
    positive = full_sq > 0
    per_point_error = np.max(np.where(positive, np.abs(projected-full_sq)/np.maximum(full_sq, 1e-300), 0), axis=1)
    embedding = np.all(np.abs(projected-full_sq) <= config['eps']*full_sq+tolerance, axis=1)
    ordered = np.sort(full_sq, axis=1)
    margin = (ordered[:,1] > (1+config['eps'])/(1-config['eps'])*ordered[:,0]) & (ordered[:,1] > 0)
    roundoff_margin = (1-config['eps'])*ordered[:,1]-(1+config['eps'])*ordered[:,0] > 2*tolerance
    certified = margin & embedding & roundoff_margin
    stable = labels == original_labels
    violations = int(np.count_nonzero(certified & ~stable))
    if violations: raise ValueError('Conditional assignment certificate violation')
    cost = float(full_sq.min(axis=1).sum())
    ratio = safe_ratio(projected.min(axis=1).sum(),cost)
    result = dict(case, n=len(x), d=x.shape[1], ari=float(adjusted_rand_score(original_labels, labels)),
        nmi=float(normalized_mutual_info_score(original_labels, labels)), agreement=float(stable.mean()),
        projected_cost_ratio=ratio, projected_cost_relative_error=abs(ratio-1) if ratio is not None else None,
        fixed_original_assignment_cost_ratio=safe_ratio(full_sq[np.arange(len(x)), labels].sum(),cost),
        r_phi=safe_ratio(sqdist(x, reconstruction).min(axis=1).sum(),full_reconstructed_cost),
        r_phi_train_reconstructed=safe_ratio(sqdist(x, train_reconstruction).min(axis=1).sum(),full_train_reconstructed_cost),
        original_fixed_cost=cost, original_reconstructed_cost=full_reconstructed_cost,
        original_train_reconstructed_cost=full_train_reconstructed_cost,
        centroid_q50=ce['q50'], centroid_q95=ce['q95'], centroid_q99=ce['q99'], centroid_max=ce['maximum'],
        centroid_zero_count=ce['zero_count'], centroid_zero_violations=ce['zero_violations'],
        pair_q50=pe['q50'], pair_q95=pe['q95'], pair_q99=pe['q99'], pair_max=pe['maximum'],
        pair_exceedance=pe['exceedance'], pair_signed_mean=pe['signed_mean'],
        pair_zero_count=pe['zero_count'], pair_positive_count=pe['positive_count'],
        pair_zero_violations=pe['zero_violations'], evaluated_pairs=len(pairs),
        margin_coverage=float(margin.mean()), certified_count=int(certified.sum()),
        certificate_coverage=float(certified.mean()), certified_violations=violations,
        uncertified_stability=float(stable[~certified].mean()) if np.any(~certified) else None,
        global_centroid_embedding_event=bool(embedding.all()),
        centroid_embedding_point_fraction=float(embedding.mean()),
        ari_ground_truth=float(adjusted_rand_score(y, labels)) if len(y) else None,
        nmi_ground_truth=float(normalized_mutual_info_score(y, labels)) if len(y) else None)
    artifacts = dict(labels=labels, train_labels=train_labels, projected_centroids=projected_c,
        projected_centroid_sq=projected, projected_pair_sq=pair_values,
        point_centroid_relative_error=per_point_error, margin_mask=margin,
        embedding_event=embedding, certified=certified, stable=stable,
        original_reconstructed_centers=reconstruction, original_train_reconstructed_centers=train_reconstruction)
    return result, artifacts


def measured_workloads(case, model, all_x, eval_indices, centers, full_sq, config, selection, out):
    """Projection, numerical preparation and every downstream task are separate."""
    repetitions = config['timing_repetitions']
    x = all_x
    if case['method'] == 'full':
        z = all_x
        compression = dict(samples=[0.]*repetitions, median=0., peak_rss_bytes=psutil.Process().memory_info().rss,
                           rss_baseline_bytes=psutil.Process().memory_info().rss, peak_rss_delta_bytes=0)
        projected_c = centers
        center_projection = dict(samples=[0.]*repetitions, median=0.)
    else:
        z, compression = timed(lambda:model.transform(all_x), repetitions, warmup=True)
        projected_c, center_projection = timed(lambda:model.transform(centers), repetitions, warmup=True)
    eval_z = z
    prepared, prep = timed(lambda:(np.einsum('ij,ij->i', eval_z, eval_z), np.einsum('ij,ij->i', projected_c, projected_c)), repetitions, warmup=True)
    norms, center_norms = prepared
    timings = dict(compression=compression, center_projection=center_projection, numeric_preparation=prep)
    (xn,cn), timings['full_numeric_preparation'] = timed(lambda:(np.einsum('ij,ij->i',x,x),
        np.einsum('ij,ij->i',centers,centers)), repetitions, warmup=True)
    _,timings['point_norm_preparation']=timed(lambda:np.einsum('ij,ij->i',z,z),repetitions,warmup=True)
    _,timings['full_point_norm_preparation']=timed(lambda:np.einsum('ij,ij->i',x,x),repetitions,warmup=True)
    # Interleave each method with its optimized Full control; norms are prepared
    # once for both. Do not treat arithmetic repetitions as geometric seeds.
    rng = np.random.default_rng(8001+case['start'])
    methods = dict(method=lambda:assignment(eval_z, projected_c, norms, center_norms),
                   full=lambda:assignment(x, centers, xn, cn))
    samples = {key:[] for key in methods}
    labels = None
    for _ in range(repetitions):
        for key in rng.permutation(list(methods)):
            begin = time.perf_counter(); value = methods[key](); samples[key].append(time.perf_counter()-begin)
            if key == 'method': labels = value[0]
    np.testing.assert_array_equal(labels, sqdist(eval_z, projected_c).argmin(axis=1))
    timings['assignment'] = dict(samples=samples['method'], median=statistics.median(samples['method']))
    timings['full_assignment'] = dict(samples=samples['full'], median=statistics.median(samples['full']))
    held_z,held_x=z[eval_indices],x[eval_indices]
    held_norms,held_full_norms=norms[eval_indices],xn[eval_indices]
    _,_,timings['heldout_assignment'],timings['full_heldout_assignment']=paired_timed(
        lambda:assignment(held_z,projected_c,held_norms,center_norms),
        lambda:assignment(held_x,centers,held_full_norms,cn),repetitions,rng)
    timings.update(workload_rows=len(x), heldout_rows=len(eval_indices),
                   measurement_version=2,
                   workload_scope='whole preserved corpus; geometric selection and quality use held-out folds')
    state = PreparedDataset(model, eval_z, norms, array_hash(x))
    full = PreparedDataset(NativeProjection('full', x.shape[1]), x, xn, array_hash(x))
    pair_rng = np.random.default_rng(7722)
    queries = []
    left_n = min(config['query_points'], max(1, len(x)//2))
    right_n = min(config['landmarks'], len(x)-left_n)
    for _ in range(config['queries']):
        query = pair_rng.choice(len(x), left_n+right_n, replace=False)
        queries.append((query[:left_n], query[left_n:]))
    save_arrays(out/'query_indices.npz',left_indices=np.stack([p[0] for p in queries]),right_indices=np.stack([p[1] for p in queries]))
    query_samples = {'method':[], 'full':[]}
    for rep in range(repetitions):
        for index, (left, right) in enumerate(queries):
            answers = {}
            for key in rng.permutation(['method', 'full']):
                begin = time.perf_counter(); answers[key] = (state if key == 'method' else full).squared_distances(left, right)
                seconds = time.perf_counter()-begin
                query_samples[key].append(dict(repetition=rep, query=index, seconds=seconds))
            if rep == 0 and index<config.get('query_artifact_samples',config['queries']):
                save_arrays(out/'queries'/f'query{index}.npz', left_indices=left, right_indices=right,
                            projected=answers['method'], original=answers['full'])
    timings['distance_queries'] = query_samples
    timings['pairs_per_query'] = left_n*right_n
    timings['query_answer_samples']=min(config.get('query_artifact_samples',config['queries']),config['queries'])
    # kNN over all evaluated points, not just the landmark set.
    knn_ids = np.sort(np.random.default_rng(188).choice(len(x), min(128, len(x)), replace=False))
    targets = np.arange(len(x)); neighbors = min(5, len(x)-1)
    def knn(cache):
        values = cache.squared_distances(knn_ids, targets)
        values[np.arange(len(knn_ids)), knn_ids] = np.inf
        return stable_topk(values,neighbors)
    new_neighbors,old_neighbors,kt,ft=paired_timed(lambda:knn(state),lambda:knn(full),repetitions,rng)
    timings.update(knn=kt, full_knn=ft,
        knn_recall=float(np.mean([len(set(a)&set(b))/neighbors for a,b in zip(new_neighbors, old_neighbors)])))
    save_arrays(out/'knn.npz', query_indices=knn_ids, original_neighbors=old_neighbors, projected_neighbors=new_neighbors)
    # This is new clustering in the representation, distinct from fixed-center
    # assignment. Both algorithms receive the same point-index initialization.
    initial = np.random.default_rng(900+case['start']).choice(len(x), case['k'], replace=False)
    def cluster(points):
        return KMeans(n_clusters=case['k'], init=points[initial].copy(), n_init=1,
                      max_iter=config['kmeans_max_iter'], tol=1e-4, algorithm='lloyd', random_state=case['start']).fit(points)
    fitted,original,kmt,fmt=paired_timed(lambda:cluster(eval_z),lambda:cluster(x),repetitions,rng)
    original_reconstruction = reconstructed_centers(x, original.labels_, case['k'])
    projected_reconstruction = reconstructed_centers(x, fitted.labels_, case['k'])
    km_ratio = safe_ratio(sqdist(x, projected_reconstruction).min(axis=1).sum(),sqdist(x, original_reconstruction).min(axis=1).sum())
    timings.update(kmeans=kmt, full_kmeans=fmt, kmeans_ari=float(adjusted_rand_score(original.labels_, fitted.labels_)),
        kmeans_r_phi=km_ratio, kmeans_iterations=int(fitted.n_iter_), full_kmeans_iterations=int(original.n_iter_),
        kmeans_hit_iteration_cap=any(i>=config['kmeans_max_iter'] for i in kmt['iterations']),
        full_kmeans_hit_iteration_cap=any(i>=config['kmeans_max_iter'] for i in fmt['iterations']))
    save_arrays(out/'kmeans.npz', initialization_indices=initial, original_labels=original.labels_,
        projected_labels=fitted.labels_, projected_centers=fitted.cluster_centers_,
        original_centers=original.cluster_centers_, reconstructed_original_centers=projected_reconstruction)
    def refine(points,initial_c):
        return KMeans(n_clusters=case['k'],init=initial_c.copy(),n_init=1,max_iter=config['kmeans_max_iter'],
            tol=1e-4,algorithm='lloyd',random_state=case['start']).fit(points)
    refined,old_refined,rt,frt=paired_timed(lambda:refine(z,projected_c),lambda:refine(x,centers),repetitions,rng)
    rc=reconstructed_centers(x,refined.labels_,case['k']);fc=reconstructed_centers(x,old_refined.labels_,case['k'])
    timings.update(kmeans_refine=rt,full_kmeans_refine=frt,
        kmeans_refine_ari=float(adjusted_rand_score(old_refined.labels_,refined.labels_)),
        kmeans_refine_r_phi=safe_ratio(sqdist(x,rc).min(axis=1).sum(),sqdist(x,fc).min(axis=1).sum()),
        kmeans_refine_hit_iteration_cap=any(i>=config['kmeans_max_iter'] for i in rt['iterations']),
        full_kmeans_refine_hit_iteration_cap=any(i>=config['kmeans_max_iter'] for i in frt['iterations']))
    save_arrays(out/'kmeans_refine.npz',original_labels=old_refined.labels_,projected_labels=refined.labels_,
        initial_original_centers=centers,initial_projected_centers=projected_c,
        original_centers=old_refined.cluster_centers_,projected_centers=refined.cluster_centers_,reconstructed_original_centers=rc)
    # DBSCAN radius is fixed using original validation geometry before test.
    db_ids = np.sort(np.random.default_rng(177).choice(len(x), min(config['dbscan_points'], len(x)), replace=False))
    db_matrix,full_matrix,pt,fpt=paired_timed(lambda:state.squared_distances(db_ids,db_ids),
        lambda:full.squared_distances(db_ids,db_ids),repetitions,rng)
    radius = selection['radius']
    tolerance = 64*np.finfo(float).eps*max(1., float(norms.max(initial=0)))
    primary,db_arrays=dbscan_profile(full_matrix,db_matrix,radius,config,rng,tolerance)
    timings.update(dbscan_pair_build=pt,full_dbscan_pair_build=fpt,**primary)
    save_arrays(out/'dbscan.npz', indices=db_ids, original_sq=full_matrix, projected_sq=db_matrix,
        **db_arrays)
    timings['dbscan_profiles']={}
    for name,other_radius in selection.get('radii',{}).items():
        if np.isclose(other_radius,radius,rtol=1e-12,atol=0): continue
        record,arrays=dbscan_profile(full_matrix,db_matrix,other_radius,config,rng,tolerance)
        timings['dbscan_profiles'][name]=record
        save_arrays(out/'dbscan_profiles'/f'{name}.npz',indices=db_ids,**arrays)
    return timings


def run_worker(job, root):
    config = job['config']; name = job['dataset']; variant = job['variant']
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    def budget():
        if time.monotonic()-began > job['max_seconds']:
            raise TimeoutError('Worker time budget; completed cases remain checkpointed')
    archive = Archive()
    data = archive.load(f'results/paper_server_20261006/rq1/{name}/artifacts/dataset.npz')
    original = data['X']; y = data['y']
    manifest = json.loads((CODE/f'results/server_paper_20261006/rq1/{name}/manifest.json').read_text())
    if array_hash(original) != manifest['dataset']['sha256']: raise ValueError('Original archive/manifest hash mismatch')
    source_indices = np.arange(len(original))
    if variant == 'nonzero':
        source_indices = source_indices[np.any(original != 0, axis=1)]
    if config['smoke']:
        source_indices = source_indices[:min(128, len(source_indices))]
    feature_indices = np.arange(original.shape[1])
    if config['smoke']:
        # Image corner pixels can all be constant. This is a correctness fixture,
        # not an estimate of publication quality on a truncated image border.
        varying = np.flatnonzero(np.ptp(original[source_indices], axis=0) > 0)
        feature_indices = varying[:min(32, len(varying))]
    x = original[np.ix_(source_indices, feature_indices)]
    y = y[source_indices] if len(y) else y
    x.setflags(write=False)
    del original, data
    folds = split_indices(len(x), config['split_seed'])
    train = x[folds['train']]; train.setflags(write=False)
    d = x.shape[1]
    max_rank = min(len(train)-1, d)
    dimensions = sorted(set([m for m in config['m'] if m < d]+
        ([max_rank] if max_rank < min(d,max(config['m'])) else [])))
    metadata = dict(dataset=name, variant=variant, original=manifest['dataset'],
        n=len(x), d=d, sha256=array_hash(x), actual_training_rows=len(train),
        split_counts={key:len(value) for key,value in folds.items()},
        constant_features=int(np.count_nonzero(x.min(axis=0) == x.max(axis=0))),
        zero_rows=int(np.count_nonzero(np.all(x == 0, axis=1))), dimensions=dimensions,
        class_counts={str(k):int(v) for k,v in zip(*np.unique(y, return_counts=True))} if len(y) else {},
        preprocessing='original coordinates unchanged; optional zero-row deletion only; no standardization', smoke=config['smoke'])
    metadata['dimension_exclusions'] = [dict(method=method, m=m, reason='m is not smaller than original d' if m >= d else 'm exceeds centered training sample rank')
        for method in ('pca','pca_randomized', 'sparse_jl', 'protected_jl') for m in config['m']
        if m >= d or (method in ('pca','pca_randomized') and m > max_rank)]
    worker_identity=dict(configuration=config,dataset_sha256=metadata['sha256'],worker_sha256=file_hash(__file__))
    identity_path=root/'worker_identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=worker_identity: raise ValueError('Worker checkpoint configuration/source changed')
    elif (root/'fits.json').exists() or any((root/'rows').glob('*.json')):
        raise ValueError('Legacy checkpoint: choose a fresh output directory')
    else:atomic_json(identity_path,worker_identity)
    atomic_json(root/'dataset.json', metadata)
    save_arrays(root/'indices.npz', source_indices=source_indices, feature_indices=feature_indices, **folds)
    fit_meta_path = root/'fits.json'
    if fit_meta_path.exists():
        fits = json.loads(fit_meta_path.read_text())
        if file_hash(root/'pca.npz')!=fits['pca_artifact_sha256']:raise ValueError('Corrupt shared PCA fit checkpoint')
        with np.load(root/'pca.npz', allow_pickle=False) as saved:
            components, mean = saved['components'], saved['mean']
    else:
        count = min(max(dimensions), max_rank)
        pca, measured = timed(lambda:PCA(n_components=count, svd_solver='full').fit(train),
                              config['fit_repetitions'], warmup=config['fit_warmup'])
        components, mean = pca.components_, pca.mean_
        save_arrays(root/'pca.npz', components=components, mean=mean, explained_variance=pca.explained_variance_)
        fits = dict(pca=measured, pca_solver='full', centroids={}, basis={},
                    pca_artifact_sha256=file_hash(root/'pca.npz'),pca_randomized={},
                    dataset_sha256=metadata['sha256'], fit_scope='training rows only')
        atomic_json(fit_meta_path, fits)
    if fits['dataset_sha256'] != metadata['sha256']: raise ValueError('Stale fit checkpoint')
    if config.get('randomized_pca',False):
        for draw in range(config['seeds']):
            budget();key=str(draw);destination=root/'randomized_pca'/f'{draw}.npz'
            if key in fits['pca_randomized']:
                if file_hash(destination)!=fits['pca_randomized'][key]['artifact_sha256']:
                    raise ValueError('Corrupt shared randomized PCA checkpoint')
                continue
            random_model,measure=timed(lambda:PCA(n_components=len(components),svd_solver='randomized',
                random_state=config['pca_fit_seed']+draw).fit(train),config['fit_repetitions'],warmup=config['fit_warmup'])
            save_arrays(destination,components=random_model.components_,mean=random_model.mean_,explained_variance=random_model.explained_variance_)
            fits['pca_randomized'][key]=dict(measure,artifact_sha256=file_hash(destination),solver='randomized',
                random_state=config['pca_fit_seed']+draw,n_components=len(components))
            atomic_json(fit_meta_path,fits);del random_model
    centers_by_key = {}
    for k in config['k']:
        for start in range(config['starts']):
            budget(); key = f'k{k}_s{start}'; destination = root/'fits'/f'{key}.npz'
            if destination.exists() and key in fits['centroids']:
                if file_hash(destination)!=fits['centroids'][key]['artifact_sha256']:raise ValueError('Corrupt shared centroid checkpoint')
                with np.load(destination, allow_pickle=False) as saved:
                    centers_by_key[key] = dict(centers=saved['centers'], basis=saved['basis'], train_labels=saved['train_labels'])
                continue
            km, measured = timed(lambda:KMeans(n_clusters=k, n_init=1, random_state=start,
                max_iter=config['kmeans_max_iter'], algorithm='lloyd', tol=1e-4).fit(train),
                config['fit_repetitions'], warmup=config['fit_warmup'])
            c = km.cluster_centers_
            u, basis_timing = timed(lambda:centroid_basis(c), config['timing_repetitions'], warmup=True)
            labels = sqdist(train, c).argmin(axis=1)
            save_arrays(destination, centers=c, basis=u, train_labels=labels)
            fits['centroids'][key] = dict(measured, iterations=int(km.n_iter_), hit_iteration_cap=bool(km.n_iter_ >= config['kmeans_max_iter']),
                artifact_sha256=file_hash(destination))
            fits['basis'][key] = basis_timing
            centers_by_key[key] = dict(centers=c, basis=u, train_labels=labels)
            atomic_json(fit_meta_path, fits)
    validation_rows = []
    selection = None
    artifact_cache={};model_files_checked=set()
    inventory = [dict(case, key=case_key(case)) for phase in ('validation', 'test') for k in config['k']
        for start in range(config['starts']) for case in case_plan(config, dimensions, k, start,
            centers_by_key[f'k{k}_s{start}']['basis'].shape[1], len(train), d, phase)]
    atomic_json(root/'case_inventory.json', dict(cases=inventory, expected_rows=len(inventory), exclusions=metadata['dimension_exclusions']))
    for phase in ('validation', 'test'):
        if phase == 'test':
            selection_path = root/'selection.json'
            fields=('phase','k','method','m','q','seed','start','ari','r_phi_train_reconstructed','pair_q95',
                'centroid_q95','projected_cost_relative_error','pair_zero_violations','centroid_zero_violations',
                'centroid_training_hit_iteration_cap')
            validation_signature=hashlib.sha256(json.dumps(sorted(
                [{p:r[p] for p in fields} for r in validation_rows],key=case_key),sort_keys=True).encode()).hexdigest()
            selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
            if selection is None or selection.get('validation_rows_sha256')!=validation_signature:
                choices, summaries = select_validation(validation_rows, config)
                # Radius independent of any projected geometry or test results.
                vx = x[folds['validation'][:config['dbscan_points']]]
                matrix = sqdist(vx, vx); np.fill_diagonal(matrix, 0)
                fifth = np.partition(matrix, min(4, len(vx)-1), axis=1)[:,min(4,len(vx)-1)]
                radius = 1. if name == 'dbscan_margin' else float(np.sqrt(np.median(fifth)))
                if radius <= 0:
                    positive = fifth[fifth > 0]
                    radius = float(np.sqrt(np.median(positive))) if len(positive) else 1e-6
                positive=fifth[fifth>0]
                radii={f'validation_q{p}':float(np.sqrt(np.quantile(positive,p/100)))
                    for p in (25,50,75)} if len(positive) else {}
                selection = dict(choices=choices, summaries=summaries, radius=radius,
                    radii=radii,validation_rows_sha256=validation_signature,
                    rule='validation only; no test fidelity or runtime used', thresholds=config['thresholds'])
                atomic_json(selection_path, selection)
            selection_fingerprint=hashlib.sha256(json.dumps({p:selection[p] for p in ('choices','radius','radii','thresholds')},sort_keys=True).encode()).hexdigest()
        ids = folds[phase]; eval_x = x[ids]; eval_y = y[ids] if len(y) else y
        pairs = np.asarray(sample_pair_set(len(eval_x), config['pairs'], config['pair_seed']), dtype=np.int64)
        pair_ref = pair_squared_distances(eval_x, pairs[:,0], pairs[:,1])
        for k in config['k']:
            for start in range(config['starts']):
                shared = centers_by_key[f'k{k}_s{start}']; c, u = shared['centers'], shared['basis']
                full_sq = sqdist(eval_x, c); labels_full = full_sq.argmin(axis=1)
                full_phi = float(sqdist(eval_x, reconstructed_centers(eval_x, labels_full, k)).min(axis=1).sum())
                full_train_phi = float(sqdist(eval_x, reconstructed_centers(train, shared['train_labels'], k)).min(axis=1).sum())
                save_arrays(root/'references'/f'{phase}_k{k}_s{start}.npz', pairs=pairs, original_pair_sq=pair_ref,
                    original_centroid_sq=full_sq, original_labels=labels_full, original_class_labels=eval_y)
                cases = list(case_plan(config, dimensions, k, start, u.shape[1], len(train), d, phase))
                np.random.default_rng(3001+k+start).shuffle(cases)
                for case in cases:
                    budget(); key = case_key(case); row_file = root/'rows'/f'{key}.json'
                    if row_file.exists():
                        row = json.loads(row_file.read_text())
                        same=(all(row.get(field)==value for field,value in case.items()) and row.get('dataset_sha256')==metadata['sha256'] and
                            (phase=='validation' or row.get('selection_fingerprint')==selection_fingerprint))
                        if same and not artifact_issues(root,row,artifact_cache):
                            if phase == 'validation': validation_rows.append(row)
                            continue
                        row_file.rename(row_file.with_suffix(f'.json.invalid-{time.time_ns()}'))
                    residual = case['m']-u.shape[1] if case['method'] == 'protected_jl' else case['m']
                    seed = projection_seed(metadata['sha256'],c if case['method']=='protected_jl' else None,
                        phase,case['seed'],residual,case['q'],start)
                    p_components,p_mean=components,mean
                    if case['method']=='pca_randomized':
                        with np.load(root/'randomized_pca'/f"{case['seed']}.npz",allow_pickle=False) as saved:
                            p_components,p_mean=saved['components'],saved['mean']
                    model, model_timing = timed(lambda:create_model(case['method'], d, case['m'], case['q'], seed,
                        c, u, p_components, p_mean), 1)
                    if case['method'] == 'full': z = x; transform_timing = dict(samples=[0.], median=0., peak_rss_bytes=psutil.Process().memory_info().rss)
                    else: z, transform_timing = timed(lambda:model.transform(x), 1)
                    projected_c = c if case['method'] == 'full' else model.transform(c)
                    row, artifacts = quality_row(case, eval_x, eval_y, z[ids], projected_c, z[folds['train']],
                        train, shared['train_labels'], full_sq, pairs, pair_ref, full_phi, full_train_phi, c, config)
                    row.update(dataset=name, variant=variant, dataset_sha256=metadata['sha256'], n_total=len(x),
                        protected_rank=u.shape[1] if case['method'] in ('protected_jl', 'protected_only') else 0,
                        residual_m=residual if case['method'] in ('protected_jl', 'sparse_jl') else 0,
                        map_seed=seed.hex() if case['q'] else None, model_sha256=model.fingerprint,
                        representation_bytes=z.nbytes, original_bytes=x.nbytes,
                        coordinate_reduction=d/case['m'], model_construction=model_timing,
                        screening_compression=transform_timing, centroid_fit=fits['centroids'][f'k{k}_s{start}'],
                        pca_fit=(fits['pca'] if case['method']=='pca' else fits['pca_randomized'][str(case['seed'])]
                            if case['method']=='pca_randomized' else None),
                        centers_sha256=array_hash(c),measurement_version=2,
                        centroid_training_hit_iteration_cap=fits['centroids'][f'k{k}_s{start}']['hit_iteration_cap'],
                        selection_fingerprint=selection_fingerprint if phase=='test' else None,
                        basis_fit=fits['basis'][f'k{k}_s{start}'] if case['method'] in ('protected_jl', 'protected_only') else None,
                        statistical_scope='held out from these model fits; fixed reused corpus, pointwise conditional intervals')
                    save_arrays(root/'arrays'/f'{key}.npz', **artifacts)
                    chosen = next((choice for choice in selection['choices'] if choice['k'] == k and choice['method'] == case['method']), None) if selection else None
                    benchmark = phase == 'test' and (case['method'] in ('full', 'protected_only') or
                        case['m'] in config['timing_m'] or (chosen and chosen['selected'] and chosen['m'] == case['m'] and chosen['q'] == case['q']))
                    if benchmark:
                        folder = root/'workloads'/key; folder.mkdir(parents=True, exist_ok=True)
                        workload = measured_workloads(case, model, x, ids, c, full_sq, config, selection, folder)
                        atomic_json(folder/'timings.json', dict(case, dataset=name, variant=variant, **workload))
                        row['workloads'] = str(folder.relative_to(root))
                    else: row['workloads'] = None
                    # Actual model parameters permit arbitrary new plots/replay.
                    model_path = root/'models'/f'{model.fingerprint}.npz'
                    parameters=model.parameters();valid=False
                    if model_path in model_files_checked:valid=True
                    elif model_path.exists():
                        try:
                            with np.load(model_path,allow_pickle=False) as stored:
                                valid=set(stored.files)==set(parameters) and all(np.array_equal(stored[p],v) for p,v in parameters.items())
                        except (OSError,ValueError):pass
                    if not valid:save_arrays(model_path,**parameters)
                    model_files_checked.add(model_path)
                    row['model_artifact'] = model_path.relative_to(root).as_posix()
                    bound_files=[root/'arrays'/f'{key}.npz',model_path]
                    if row['workloads']:
                        bound_files += [folder/'timings.json']+list(folder.rglob('*.npz'))
                    row['artifact_sha256']={p.relative_to(root).as_posix():file_hash(p) for p in bound_files}
                    atomic_json(row_file, row)
                    if phase == 'validation': validation_rows.append(row)
                    del z, artifacts, model
                    gc.collect()
        print(json.dumps(dict(dataset=name, variant=variant, phase=phase,
                              rows=len(list((root/'rows').glob('*.json'))), seconds=time.monotonic()-began)), flush=True)
    atomic_json(root/'done.json', dict(complete=True, rows=len(list((root/'rows').glob('*.json'))),
        wall_seconds=time.monotonic()-began, expected_phases=['validation', 'test']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    with threadpool_limits(limits=job['config']['threads']): run_worker(job, args.out)


if __name__ == '__main__': main()
