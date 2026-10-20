"""Offline worker for bounded laptop comparisons. No deployment protocol claim."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from commitjl import zk_bridge
from commitjl.clustering import _squared_distances, nearest_centroid_assign, psi_hat
from commitjl.protected_jl import centroid_basis, protected_project, integer_protected_project
from commitjl.sparse_gen import sparse_matrix_gen_csr
from experiments.protected_jl_study import Archive

CODE = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_arrays(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def distances(x, c):
    return _squared_distances(x, c)


def relative_errors(reference, projected):
    reference = np.asarray(reference, dtype=float)
    projected = np.asarray(projected, dtype=float)
    positive = reference > 0
    values = np.abs(projected[positive] / reference[positive] - 1)
    return dict(q50=float(np.quantile(values, .5)) if len(values) else 0.,
                q95=float(np.quantile(values, .95)) if len(values) else 0.,
                q99=float(np.quantile(values, .99)) if len(values) else 0.,
                maximum=float(values.max()) if len(values) else 0.,
                within_10=float(np.mean(values <= .1)) if len(values) else 1.,
                within_20=float(np.mean(values <= .2)) if len(values) else 1.,
                zero_reference_count=int((~positive).sum()),
                zero_reference_violations=int(np.count_nonzero(projected[~positive] != 0)))


def metric_record(reference, projected, pairs_original, pairs_projected, projected_labels=None):
    reference = np.asarray(reference, dtype=float)
    projected = np.asarray(projected, dtype=float)
    original_labels = reference.argmin(axis=1)
    labels = projected.argmin(axis=1) if projected_labels is None else np.asarray(projected_labels,dtype=np.int64)
    cost = float(reference.min(axis=1).sum())
    if cost <= 0: raise ValueError('Positive Full cost required for a relative objective')
    center_errors = relative_errors(reference, projected)
    pair_errors = relative_errors(pairs_original, pairs_projected)
    projected_min_ratio = float(projected.min(axis=1).sum()/cost)
    fixed_ratio = float(projected[np.arange(len(labels)), original_labels].sum()/cost)
    result = dict(ari=float(adjusted_rand_score(original_labels, labels)),
        agreement=float(np.mean(original_labels == labels)), full_cost=cost,
        projected_cost_ratio=projected_min_ratio,
        projected_cost_relative_error=abs(projected_min_ratio-1),
        projected_fixed_assignment_cost_ratio=fixed_ratio,
        original_assignment_cost_ratio=float(reference[np.arange(len(labels)), labels].sum()/cost))
    result.update({'centroid_'+key:value for key,value in center_errors.items()})
    result.update({'pair_'+key:value for key,value in pair_errors.items()})
    result['qualified_geometry'] = bool(result['ari'] >= .99 and
        result['centroid_q95'] <= .2 and result['pair_q95'] <= .2 and
        result['projected_cost_relative_error'] <= .1 and
        not result['centroid_zero_reference_violations'] and not result['pair_zero_reference_violations'])
    return result, labels


def nearest_neighbors(x):
    sq = np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(x*x,axis=1)[None,:]-2*x@x.T,0)
    np.fill_diagonal(sq, np.inf)
    return np.argsort(sq, axis=1, kind='stable')[:, :min(5,len(x)-1)]


def archived(name):
    archive = Archive()
    prefix = f'results/paper_server_20261006/rq1/{name}/artifacts/'
    dataset = archive.load(prefix+'dataset.npz')
    pca = archive.load(prefix+'pca.npz')['components']
    return archive, prefix, dataset, pca


def emit(root, key, spec, reference, projected, pairs_original, pairs_projected,
         labels_extra=None, exact_labels=None):
    metrics, labels = metric_record(reference, projected, pairs_original, pairs_projected,exact_labels)
    row = dict(spec, **metrics)
    if labels_extra is not None:
        row.update(labels_extra(labels))
    save_arrays(root/'arrays'/f'{key}.npz', labels=labels.astype('uint16'),
        projected_centroid_sq=np.asarray(projected,dtype=float),
        projected_pair_sq=np.asarray(pairs_projected,dtype=float))
    atomic_json(root/'rows'/f'{key}.json', row)
    return row


def geometry(job, root):
    archive,prefix,data,pca = archived(job['dataset'])
    x = data['X']; pairs = data['pairs']
    if job.get('smoke'): x=x[:64]; pairs=np.array([[i,i+1] for i in range(len(x)-1)])
    original_pair = np.sum((x[pairs[:,0]]-x[pairs[:,1]])**2,axis=1)
    subset = np.sort(np.random.default_rng(111).choice(len(x),min(256,len(x)),replace=False))
    neighbors = nearest_neighbors(x[subset])
    input_sha = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    save_arrays(root/'inputs.npz', pairs=pairs, original_pair_sq=original_pair,
                neighborhood_indices=subset, original_neighbors=neighbors)
    for k in job['k']:
        for start in job['starts']:
            old = archive.load(prefix+f'full_k{k}_start{start}.npz'); c=old['centroids']
            reference=distances(x,c); original_labels=reference.argmin(axis=1)
            begin=time.perf_counter(); u=centroid_basis(c); basis_seconds=time.perf_counter()-begin
            save_arrays(root/f'centers_k{k}_start{start}.npz', centroids=c,basis=u,
                        original_centroid_sq=reference,original_labels=original_labels)
            common=dict(dataset=job['dataset'],n=len(x),d=x.shape[1],k=k,start=start,
                dataset_sha256=input_sha,scope='all preserved paper rows' if not job.get('smoke') else 'first 64 rows smoke',
                saved_full_agreement=float(np.mean(original_labels==old['labels'][:len(x)])))
            def measure(key, spec, projector, setup_seconds):
                if (root/'rows'/f'{key}.json').exists(): return
                projector(x[:1],c)  # small warm-up, outside measured samples
                projection_samples=[]; assignment_samples=[]
                for _ in range(job.get('native_repetitions',3)):
                    begin=time.perf_counter(); z,cc=projector(x,c)
                    projection_samples.append(time.perf_counter()-begin)
                    begin=time.perf_counter(); projected=distances(z,cc); labels=projected.argmin(axis=1)
                    assignment_samples.append(time.perf_counter()-begin)
                pair=np.sum((z[pairs[:,0]]-z[pairs[:,1]])**2,axis=1)
                new_neighbors=nearest_neighbors(z[subset])
                recall=float(np.mean([len(set(a)&set(b))/len(a) for a,b in zip(neighbors,new_neighbors)]))
                timings=dict(projection_seconds_samples=projection_samples,assignment_seconds_samples=assignment_samples,
                    projection_seconds_median=statistics.median(projection_samples),
                    assignment_seconds_median=statistics.median(assignment_samples),
                    map_setup_seconds=setup_seconds,basis_seconds=basis_seconds if spec['method'].startswith('hybrid') or spec['method']=='protected_only' else 0.,
                    native_seconds_median=statistics.median([a+b for a,b in zip(projection_samples,assignment_samples)]),
                    pca_training_included=False,neighbor_recall_5=recall)
                emit(root,key,dict(common,**spec,**timings),reference,projected,original_pair,pair)
            measure(f'k{k}_s{start}_full',dict(method='full',total_m=x.shape[1],residual_m=0,rank=0,q=0,seed=-1),
                    lambda a,b:(a,b),0.)
            measure(f'k{k}_s{start}_protected',dict(method='protected_only',total_m=u.shape[1],residual_m=0,rank=u.shape[1],q=0,seed=-1),
                    lambda a,b:(protected_project(a,u,None,c[0]),protected_project(b,u,None,c[0])),0.)
            for total in job['m']:
                if total>x.shape[1]:continue
                if total<=len(pca):
                    matrix=pca[:total]
                    measure(f'k{k}_s{start}_pca{total}',dict(method='pca',total_m=total,residual_m=0,rank=0,q=0,seed=-1),
                            lambda a,b:(a@matrix.T,b@matrix.T),0.)
                for rank in sorted(set([0,min(2,u.shape[1]),u.shape[1]])):
                    residual=total-rank
                    if residual<1:continue
                    for q in job['q']:
                        if q>residual:continue
                        for seed in job['seeds']:
                            key=f'k{k}_s{start}_m{total}_r{rank}_q{q}_seed{seed}'
                            if (root/'rows'/f'{key}.json').exists():continue
                            # Fixed basis precedes this fresh native-simulation challenge.
                            rho=hashlib.sha256(f'laptop-confirmation-v1:{seed}'.encode()).digest()
                            map_seed=hashlib.sha256(input_sha.encode()+c.tobytes()+rho+f'{residual}:{q}'.encode()).digest()
                            begin=time.perf_counter(); _,a=sparse_matrix_gen_csr(map_seed,residual,x.shape[1],q)
                            a=a.astype(float);a.data/=np.sqrt(q);setup=time.perf_counter()-begin
                            basis=u[:,:rank]
                            measure(key,dict(method='sparse' if rank==0 else f'hybrid_r{rank}',total_m=total,
                                residual_m=residual,rank=rank,q=q,seed=seed,map_seed=map_seed.hex()),
                                lambda v,w:(protected_project(v,basis,a,c[0]),protected_project(w,basis,a,c[0])),setup)
            print(f"geometry {job['dataset']} k={k} start={start} complete",flush=True)
    atomic_json(root/'done.json',dict(complete=True,rows=len(list((root/'rows').glob('*.json')))))


def precision(job, root):
    archive,prefix,data,pca=archived(job['dataset']); all_x=data['X'];x=all_x[:job['n']]
    pairs=np.array([[i,i+1] for i in range(len(x)-1)])
    original_pairs=np.sum((x[pairs[:,0]]-x[pairs[:,1]])**2,axis=1)
    save_arrays(root/'inputs.npz',pairs=pairs,original_pair_sq=original_pairs)
    for k in job['k']:
        old=archive.load(prefix+f"full_k{k}_start{job['start']}.npz");c=old['centroids']
        factor=4096/max(float(np.max(np.abs(all_x))),float(np.max(np.abs(c))),1e-15)
        xq=np.rint(x*factor).astype(np.int64);cq=np.rint(c*factor).astype(np.int64)
        reference=distances(x,c);integer_reference=distances(xq,cq);integer_labels=integer_reference.argmin(axis=1)
        u=centroid_basis(cq)
        save_arrays(root/f'centers_k{k}.npz',centroids=c,basis=u,quantized_centroids=cq,
                    original_centroid_sq=reference,integer_centroid_sq=integer_reference,quantized_X=xq)
        common=dict(dataset=job['dataset'],n=len(x),d=x.shape[1],k=k,start=job['start'],input_factor=factor,
                    scope=f'first min({job["n"]},n) rows; saved full-subset centers')
        def record(key,spec,z,cc,normalization,rank,q):
            if (root/'rows'/f'{key}.json').exists():return
            weights=np.array([q]*rank+[1]*(z.shape[1]-rank),dtype=object)
            # Exact integer distances before conversion for descriptive ratios.
            projected=np.zeros((len(z),len(cc)),dtype=object)
            for j in range(len(cc)):
                diff=z.astype(object)-cc[j].astype(object)
                projected[:,j]=np.sum(diff*diff*weights,axis=1)
            diff=z[pairs[:,0]].astype(object)-z[pairs[:,1]].astype(object)
            pair=np.sum(diff*diff*weights,axis=1)
            float_sq=np.asarray(projected,dtype=float)/normalization
            float_pairs=np.asarray(pair,dtype=float)/normalization
            emit(root,key,dict(common,**spec),reference,float_sq/(factor*factor),original_pairs,float_pairs/(factor*factor),
                labels_extra=lambda labels:dict(ari_integer_full=float(adjusted_rand_score(integer_labels,labels)),
                    agreement_integer_full=float(np.mean(integer_labels==labels)),
                    projected_cost_ratio_integer_full=float(float_sq.min(axis=1).sum()/np.asarray(integer_reference.min(axis=1),dtype=float).sum())),
                    exact_labels=projected.argmin(axis=1))
        record(f'k{k}_integer_full',dict(method='integer_full',total_m=x.shape[1],scale=1,seed=-1,q=0),xq,cq,1,0,1)
        for total in job['m']:
            if total>x.shape[1]:continue
            for scale in job['scales']:
                if total<=len(pca):
                    matrix=np.rint(scale*pca[:total]).astype(np.int64)
                    record(f'k{k}_pca{total}_T{scale}',dict(method='integer_pca',total_m=total,scale=scale,seed=-1,q=0),
                           xq@matrix.T,cq@matrix.T,scale**2,0,1)
                for rank in [0,u.shape[1]]:
                    residual=total-rank
                    if residual<1:continue
                    b=np.rint(scale*u[:,:rank].T).astype(np.int64)
                    for seed in job['seeds']:
                        key=f'k{k}_m{total}_r{rank}_T{scale}_seed{seed}'
                        if (root/'rows'/f'{key}.json').exists():continue
                        map_seed=hashlib.sha256(f'precision-confirmation:{job["dataset"]}:{k}:{seed}:{residual}'.encode()).digest()
                        _,a=sparse_matrix_gen_csr(map_seed,residual,x.shape[1],1)
                        z=integer_protected_project(xq,b,a,scale);cc=integer_protected_project(cq,b,a,scale)
                        record(key,dict(method='integer_sparse' if rank==0 else 'integer_hybrid',total_m=total,
                            residual_m=residual,rank=rank,scale=scale,seed=seed,q=1,map_seed=map_seed.hex()),z,cc,scale**4,rank,1)
        print(f'precision {job["dataset"]} k={k} complete',flush=True)
    atomic_json(root/'done.json',dict(complete=True,rows=len(list((root/'rows').glob('*.json')))))


def proof_cases(job):
    n,d,total,k,q=job['shape']
    if job.get('dataset'):
        archive,prefix,data,components=archived(job['dataset']);raw=data['X']
        old=archive.load(prefix+f'full_k{k}_start3.npz');c=old['centroids']
        indices=np.sort(np.random.default_rng(222).choice(len(raw),n,replace=False))
        factor=4096/max(float(np.max(np.abs(raw))),float(np.max(np.abs(c))),1e-15)
        x=np.rint(raw[indices]*factor).astype(np.int64);c=np.rint(c*factor).astype(np.int64)
    else:
        rng=np.random.default_rng(123456+d+k)
        c=rng.integers(-50,51,size=(k,d),dtype=np.int64)
        x=c[np.arange(n)%k]+rng.integers(-6,7,size=(n,d),dtype=np.int64)
        train=c[np.arange(max(256,2*d))%k]+rng.integers(-12,13,size=(max(256,2*d),d),dtype=np.int64)
        _,_,components=np.linalg.svd(train-train.mean(axis=0),full_matrices=False)
    scale=job.get('scale',4096)
    seed=hashlib.sha256(f'laptop-proof-map:{job["shape"]}:{job.get("dataset", "synthetic")}'.encode()).digest()
    base=zk_bridge.build_sparse_instance(x,c,seed,q,total)
    base=replace(base,score_distances=True)
    def constant(matrix,variant='pca-public'):
        z=x@matrix.T;cc=c@matrix.T;labels=nearest_centroid_assign(z,cc)
        return replace(base,variant=variant,m=len(matrix),q=0,c_a=0,sparse_records=None,
                       constant_A=matrix,assign=labels.tolist(),psi_hat=int(psi_hat(z,cc,labels)))
    u=centroid_basis(c);b=np.rint(scale*u.T).astype(np.int64);residual=total-len(b)
    if residual<q:raise ValueError('Total output dimension too small for protected rank')
    from commitjl.protected_jl import weighted_integer_assignment
    from commitjl.zk_bridge import field_hash_sparse_list
    records,a=sparse_matrix_gen_csr(seed,residual,d,q)
    z=integer_protected_project(x,b,a,scale);cc=integer_protected_project(c,b,a,scale)
    labels,cost=weighted_integer_assignment(z,cc,len(b),q)
    l1=[sum(abs(int(v)) for v in row) for row in b]
    bound=4096*max(d*scale*scale+sum(v*v for v in l1),scale*max(l1,default=0))
    hybrid=replace(base,m=residual,sparse_records=records,c_a=field_hash_sparse_list(records),
        protected_basis=b,basis_scale=scale,assign=labels.tolist(),psi_hat=cost,
        range_bits=max(base.range_bits,bound.bit_length()+1))
    return dict(full=constant(np.eye(d,dtype=np.int64),'full'),
        pca=constant(np.rint(scale*components[:total]).astype(np.int64)),
        sparse_shared=base,sparse_lookup=replace(base,lookup_sparse=True),
        hybrid_shared=hybrid,hybrid_lookup=replace(hybrid,lookup_sparse=True),
        hybrid_specialized=replace(hybrid,specialize_sparse=True),
        protected_only=constant(b))


def restore_fixture_commitment(instances, root):
    """Reuse and authenticate a saved fixture opening across worker restarts."""
    paths=sorted((Path(root)/'instances').glob('*.json'))
    if not paths:return instances
    saved=json.loads(paths[0].read_text())
    first=next(iter(instances.values())).to_json_dict()
    keys=['n','d','k','x','centroids','c_theta']
    if any(saved[key]!=first[key] for key in keys):
        raise ValueError('Saved fixture input/model changed')
    for path in paths:
        prior=json.loads(path.read_text())
        if any(prior[key]!=saved[key] for key in keys+['c_x','r_x']):
            raise ValueError('Existing fixture checkpoints have inconsistent commitment openings')
    from commitjl.commit import commit_matrix
    from commitjl.hashing import BN254_FR
    x=np.array([[int(v) if int(v)<=BN254_FR//2 else int(v)-BN254_FR
        for v in row] for row in saved['x']],dtype=np.int64)
    if commit_matrix(x,salt=int(saved['r_x'])).root!=int(saved['c_x']):
        raise ValueError('Saved input commitment opening is invalid')
    return {name:replace(instance,c_x=int(saved['c_x']),r_x=int(saved['r_x']))
        for name,instance in instances.items()}


def circuits(job, root):
    zk_bridge.BIN_PATH=Path(job['binary'])
    instances=restore_fixture_commitment(proof_cases(job),root)
    for method in job['methods']:
        instance=instances[method]
        stages=['range','theta','data','matrix','projection',''] if job['kind']=='profile' else ['']
        for stage in stages:
            key=method+'_'+(stage or 'complete')
            row_path=root/'rows'/f'{key}.json'
            if row_path.exists():continue
            phase='prove' if job['kind']=='prove' else 'solve'
            instance=replace(instance,solve_only=phase=='solve',diagnostic_prefix=stage)
            inputs=root/'instances'/f'{key}.json';atomic_json(inputs,instance.to_json_dict())
            begin=time.perf_counter()
            result=zk_bridge.run_prove(instance,timeout=job['case_timeout'],build_if_missing=False,
                repetitions=job.get('repetitions',3),warmup=1 if phase=='prove' else 0,
                max_rss_bytes=int(job['rss_gib']*2**30))
            atomic_json(root/'records'/f'{key}.json',asdict(result))
            bounded=result.failure_kind in ['timeout','rss_limit','memory_limit','oom','oom_budget','memory_budget']
            if not result.ok and not bounded:
                # Recognize alternate RSS/budget failure names without masking solver errors.
                bounded='rss' in result.failure_kind or 'budget' in result.failure_kind
            row=dict(kind=job['kind'],shape=job['shape'],dataset=job.get('dataset','synthetic'),
                n=instance.n,d=instance.d,k=instance.k,total_m=instance.m+(len(instance.protected_basis) if instance.protected_basis is not None else 0),
                residual_m=instance.m if instance.variant=='sparse' else 0,method=method,stage=stage or 'complete',
                basis_scale=instance.basis_scale,coefficient_scale=job.get('scale',4096),
                status='ok' if result.ok else ('resource_limit' if bounded else 'error'),
                constraints=result.num_constraints,compile_seconds=result.compile_seconds,
                setup_seconds=result.setup_seconds,witness_seconds=result.witness_seconds,
                solve_seconds=result.raw.get('solve_seconds',0),
                prove_seconds_median=statistics.median(result.raw.get('prove_seconds_samples') or [0]),
                verify_seconds_median=statistics.median(result.raw.get('verify_seconds_samples') or [0]),
                successful_proofs=result.successful_proofs,verify_ok=result.verify_ok,
                peak_rss_gib=result.peak_rss_bytes_psutil/2**30,
                wall_seconds=time.perf_counter()-begin,failure_kind=result.failure_kind,error=result.error,
                rss_limit_gib=job['rss_gib'],go_memory_limit=os.environ.get('GOMEMLIMIT','default'),
                threads=job.get('threads',4))
            row['single_run_total']=row['compile_seconds']+row['setup_seconds']+row['witness_seconds']+row['prove_seconds_median']+row['verify_seconds_median']
            atomic_json(row_path,row)
            print(f"{job['kind']} {job['shape']} {method} {stage or 'complete'} {row['status']} constraints={row['constraints']}",flush=True)
            if row['status']=='error':raise RuntimeError(result.error)
    atomic_json(root/'done.json',dict(complete=True,rows=len(list((root/'rows').glob('*.json')))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();job=json.loads(args.job.read_text());args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=job.get('threads',4)):
        if job['kind']=='geometry':geometry(job,args.out)
        elif job['kind']=='precision':precision(job,args.out)
        else:circuits(job,args.out)


if __name__=='__main__':main()
