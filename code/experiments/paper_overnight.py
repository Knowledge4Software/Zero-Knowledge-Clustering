"""Sequential full-data, geometric-utility and authenticated-distance follow-up.

Run after the existing bound-audit campaign. Every child is bounded and writes
its own atomic receipt. Fixed-center accuracy and geometric utility stay separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.bound_audit import DATASETS,T,quality_case,native_stats,native_answer,save
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import array_hash,stable_squared_distances
from commitjl.sparse_gen import sparse_matrix_gen_csr
from data.loaders import load_dataset


def seed_for(*parts):return hashlib.sha256('/'.join(map(str,parts)).encode()).digest()


def dataset(name,full=False,smoke=False):
    if smoke:
        raw=load_dataset('gaussian_mixture',n=100,d=32,k=5,gamma=.5,seed=2)
    else:
        loader,kw=next((loader,dict(kw)) for label,loader,kw in DATASETS if label==name)
        if full:kw.pop('n_max',None)
        raw=load_dataset(loader,**kw)
    x=np.asarray(raw['X'],dtype=float);mask=np.linalg.norm(x,axis=1)>0
    if not mask.all():x=x[mask]
    return x,dict(name=name,n=len(x),d=x.shape[1],sha256=array_hash(x),source=raw['meta'],nonzero_filter=True,full_loader=full)


def stopped_tree(proc):
    # Only this job's descendants: never signal existing campaigns or services.
    try:children=psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:children=[]
    for p in children:
        try:p.terminate()
        except psutil.NoSuchProcess:pass
    proc.terminate()
    _,alive=psutil.wait_procs(children,timeout=3)
    for p in alive:
        try:p.kill()
        except psutil.NoSuchProcess:pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()


def run_bounded(command,log,seconds=2700):
    start=time.monotonic();peak=0;reason=None
    with Path(log).open('w',encoding='utf-8') as stream:
        p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            try:
                own=psutil.Process(p.pid);rss=sum(q.memory_info().rss for q in [own]+own.children(recursive=True));peak=max(peak,rss)
                if rss>5.3*2**30:reason='own process tree RSS exceeds 5.3 GiB'
            except psutil.NoSuchProcess:pass
            if psutil.virtual_memory().available<768*2**20:reason='host reserve below 768 MiB'
            if time.monotonic()-start>seconds:reason='finite job deadline'
            if reason:stopped_tree(p);break
            time.sleep(.25)
    return dict(returncode=p.returncode,wall_seconds=time.monotonic()-start,peak_rss_bytes=peak,stop=reason)


def full_worker(args):
    x,meta=dataset(args.dataset,full=True,smoke=args.smoke)
    rows,geometry,data=quality_case(x,args.dataset,args.k,args.start,args.out)
    xi,ci,scale,models,fits,tr,au,centers=data
    floating_dist=stable_squared_distances(x[au],centers)
    raw=dict(train_indices=tr,audit_indices=au,float_distances_to_centers=floating_dist,
             float_full_labels=floating_dist.argmin(1),quantized_centers=ci,float_centers=centers)
    for method in models:
        model=models[method]
        if method=='protected_tail':
            width=16-len(model['basis'])-1;seed=seed_for('geometry-fixed-corpus',args.dataset,args.k,args.start,0)
            _,a=sparse_matrix_gen_csr(seed,width,model['d'],1);a=a.tocsc()
            model=dict(model,tail_width=width,tail_rows=a.indices.tolist(),tail_signs=a.data.astype(int).tolist(),seed=seed.hex())
        vals=native_stats(method,model,xi[au],ci);labels,_=native_answer(method,vals)
        raw[method+'_labels']=labels
    np.savez_compressed(args.out/'raw_assignments.npz',**raw)
    save(args.out/'models.json',models)
    save(args.out/'result.json',dict(status='complete',complete=True,task='full',dataset=meta,k=args.k,start=args.start,
        quality=rows,geometry=geometry,scale=scale,note='Full cached loader rows, then zero-row removal; same fixed-center task as prior cohort'))


def neighbors(dist,query_ids):
    d=dist.copy();d[np.arange(len(query_ids)),query_ids]=np.inf
    return np.argsort(d,axis=1,kind='stable')[:,:min(10,d.shape[1]-1)]


def geometry_metrics(z,reference,queries,pair_i,pair_j,original_pairs):
    d=stable_squared_distances(z[queries],z);nn=neighbors(d,queries)
    recalls={}
    for count in [5,10]:
        count=min(count,nn.shape[1]);recalls[count]=np.asarray([len(set(a[:count])&set(b[:count]))/count for a,b in zip(nn,reference)])
    projected=np.sum((z[pair_i]-z[pair_j])**2,axis=1)
    error=projected/original_pairs-1
    metrics=dict(pair_q95_abs_relative_error=float(np.quantile(np.abs(error),.95)),pair_mean_abs_relative_error=float(np.abs(error).mean()),
                 pairs_exceed_015=float(np.mean(np.abs(error)>.15)),recall5=float(recalls[min(5,nn.shape[1])].mean()),
                 recall10=float(recalls[min(10,nn.shape[1])].mean()),pairs=len(error),queries=len(queries),
                 euclidean_coordinates=z.shape[1])
    return metrics,dict(neighbor_indices=nn,pair_signed_relative_errors=error,recall5_per_query=recalls[min(5,nn.shape[1])],
                        recall10_per_query=recalls[min(10,nn.shape[1])])


def geometry_worker(args):
    x,meta=dataset(args.dataset,smoke=args.smoke)
    rng=np.random.default_rng(10000+args.start);order=rng.permutation(len(x));tr=order[:int(.6*len(x))];au=order[int(.6*len(x)):]
    centers=KMeans(args.k,n_init=5,random_state=args.start).fit(x[tr]).cluster_centers_
    scale=4096/max(float(np.abs(x).max()),float(np.abs(centers).max()),1e-12)
    ci=np.rint(centers*scale).astype(np.int64);xa=x[au];wi=np.rint(xa*scale)-ci[0]
    u=np.rint(T*centroid_basis(ci.astype(float)).T)/T
    h=wi@u.T;residual=wi-h@u
    rng=np.random.default_rng(7700+args.start);queries=np.sort(rng.choice(len(au),min(128,len(au)),replace=False))
    reference=neighbors(stable_squared_distances(xa[queries],xa),queries)
    a=rng.integers(len(au),size=2000);b=rng.integers(len(au),size=2000)
    original_pairs=np.sum((xa[a]-xa[b])**2,axis=1);good=(a!=b)&(original_pairs>0);a,b,original_pairs=a[good],b[good],original_pairs[good]
    np.savez_compressed(args.out/'reference.npz',train_indices=tr,audit_indices=au,query_indices=queries,
        reference_neighbors=reference,pair_i=a,pair_j=b,original_pair_squared_distances=original_pairs,protected_basis=u,centers=centers)
    rows=[];exclusions=[];maps=2 if args.smoke else 10
    def record(method,budget,map_id,z,fit_time,apply_time,seed=''):
        metrics,raw=geometry_metrics(z/scale,reference,queries,a,b,original_pairs)
        label=f'{method}_m{budget}_map{map_id}'
        np.savez_compressed(args.out/(label+'.npz'),**raw)
        rows.append(dict(dataset=args.dataset,k=args.k,start=args.start,method=method,budget=budget,map=map_id,seed=seed,
            model_fit_seconds=fit_time,compression_seconds=apply_time,**metrics))
        save(args.out/'progress.json',dict(rows=rows,exclusions=exclusions))
    record('protected_head_only',len(u)+1,0,h,None,None)
    for budget in [16,32,64]:
        if budget<=min(len(tr),x.shape[1]):
            start=time.perf_counter();model=PCA(budget,svd_solver='randomized',random_state=args.start).fit(x[tr]);fit=time.perf_counter()-start
            p=np.rint(T*model.components_)/T;start=time.perf_counter();z=wi@p.T;elapsed=time.perf_counter()-start
            np.savez_compressed(args.out/f'pca_basis_m{budget}.npz',basis=p)
            record('pca',budget,0,z,fit,elapsed)
        else:exclusions.append(dict(method='pca',budget=budget,reason='training/dimension rank restriction'))
        for map_id in range(maps):
            for method,width,values in [('protected_tail',budget-len(u)-1,residual),('sparse_jl',budget,wi)]:
                if width<1:exclusions.append(dict(method=method,budget=budget,reason='no residual slots'));continue
                seed=seed_for('overnight-geometry-v1',args.dataset,args.k,args.start,budget,map_id,method)
                start=time.perf_counter();_,a_matrix=sparse_matrix_gen_csr(seed,width,x.shape[1],1);fit=time.perf_counter()-start
                start=time.perf_counter();tail=np.asarray(a_matrix@values.T).T
                z=np.column_stack((h,tail)) if method=='protected_tail' else tail
                elapsed=time.perf_counter()-start
                record(method,budget,map_id,z,fit,elapsed,seed.hex())
    save(args.out/'result.json',dict(status='complete',complete=True,task='geometry',dataset=meta,k=args.k,start=args.start,
        rows=rows,exclusions=exclusions,note='Native reproducible maps; energy excluded from Euclidean features; tail timings exclude already computed head/residual'))


def proof_worker(args):
    method=args.method;prefix=args.proof_case
    source=args.source/'cases'/f'{prefix}_{method}'
    if not (source/'verify.json').exists() or not json.loads((source/'verify.json').read_text()).get('complete'):
        save(args.out/'result.json',dict(status='excluded',complete=False,task='proof',reason='source bundle has no completed independent verification'));return
    original=json.loads((source/'input.json').read_text());context=json.loads((source/'native_context.json').read_text())
    src_prep=json.loads((source/'prepare.json').read_text());bundle=args.out/'bundle';events=[]
    def phase(name,inp):
        command=[str(args.bin),'-phase',name,'-instance',str(inp),'-out',str(bundle),'-reps','1' if args.smoke else '3',
            '-warmup','0' if args.smoke else '1','-max-constraints','1300000']
        if name=='verify':command+=['-trust',str(bundle/'trusted.json')]
        resource=run_bounded(command,args.out/(name+'.log'),seconds=2400)
        result=json.loads((bundle/(name+'.json')).read_text()) if (bundle/(name+'.json')).exists() else {}
        events.append(dict(phase=name,resource=resource,result=result));save(args.out/'events.json',events)
        if resource['returncode'] not in [0,3] or resource['stop']:raise RuntimeError(f'{prefix}/{method}/{name}: {resource}')
        return result
    existing=args.budget==16 or method in ['full','protected_energy']
    challenge=None;model=json.loads(json.dumps(original['model']));model_fit=0.
    if existing:
        shutil.copytree(source,bundle)
    else:
        bundle.mkdir()
        if method=='pca':
            x,_=dataset(args.dataset,smoke=args.smoke)
            order=np.random.default_rng(10000).permutation(len(x));tr=order[:int(.6*len(x))]
            if args.budget>min(len(tr),x.shape[1]):
                save(args.out/'result.json',dict(status='excluded',complete=False,task='proof',reason='PCA rank restriction'));return
            start=time.perf_counter();pca=PCA(args.budget,svd_solver='randomized',random_state=0).fit(x[tr])
            model['basis']=np.rint(T*pca.components_).astype(np.int64).tolist();model_fit=time.perf_counter()-start
            del x
        elif method=='protected_tail':
            width=args.budget-len(model['basis'])-1
            nonce=secrets.token_hex(32)
            params=hashlib.sha256(json.dumps(dict(centers=model['centers'],basis=model['basis'],n=model['n'],d=model['d'],tail_width=width),sort_keys=True).encode()).hexdigest()
            seed=seed_for('overnight-tail-v1',src_prep['statement']['cx'],params,nonce)
            start=time.perf_counter();_,a=sparse_matrix_gen_csr(seed,width,model['d'],1);a=a.tocsc();model_fit=time.perf_counter()-start
            model.update(tail_width=width,tail_rows=a.indices.tolist(),tail_signs=a.data.astype(int).tolist(),seed=seed.hex())
            challenge=dict(original_root=src_prep['statement']['cx'],parameters=params,nonce=nonce,seed=seed.hex(),
                note='verifier local CSPRNG after original root and width/model fixed; trusted host derivation, no in-circuit XOF')
            save(args.out/'challenge.json',challenge)
        else:raise RuntimeError('unexpected new model')
    n=model['n'];rng=np.random.default_rng(817);pairs=set()
    target=min(2 if args.smoke else 8,n*(n-1)//2)
    for _ in range(1000):
        pairs.add(tuple(sorted(rng.choice(n,2,replace=False).tolist())))
        if len(pairs)==target:break
    if len(pairs)!=target:raise RuntimeError('finite pair sampler did not reach target')
    inp=args.out/'geometry_input.json';save(inp,dict(original,model=model,queries=[list(p) for p in sorted(pairs)]))
    if existing:
        phase('verify',inp) # independently check reused Stage-1 and query proofs before adding a new key
    else:
        p=phase('prepare',inp)
        if p['statement']['cx']!=src_prep['statement']['cx']:raise RuntimeError('comparators do not share owner original root')
        s1=phase('stage1',inp)
        if not s1.get('complete'):
            save(args.out/'result.json',dict(status='excluded',complete=False,task='proof',events=events,reason='preprocessing budget'));return
    result=phase('geometry',inp)
    if not result.get('complete'):
        save(args.out/'result.json',dict(status='excluded',complete=False,task='proof',events=events,reason='geometry budget'));return
    verified=phase('verify',inp)
    # Pair fidelity is diagnostic: the proof establishes the stored embedding's distance, not an original-space approximation bound.
    x=np.asarray(context['original_float_points']);scale=context['input_scale'];power=T**4 if method!='full' else 1
    errors=[]
    for sample in result['samples']:
        i,j=sample['indices'];distance=float(np.sum((x[i]-x[j])**2));projected=int(sample['psi'])/scale**2/power
        errors.append(dict(indices=[i,j],original_squared_distance=distance,projected_squared_distance=projected,
                           relative_error=projected/distance-1 if distance>0 else None))
    save(args.out/'result.json',dict(status='complete',complete=True,task='proof',dataset=args.dataset,k=args.k,
        method=method,budget=args.budget,source=str(source),reused_preprocessing=existing,original_root=src_prep['statement']['cx'],
        model_fit_seconds=model_fit,geometry=result,independent_verification=verified,pair_fidelity=errors,
        note='Every distance uses authenticated head and optional tail rows. Energy is authenticated but excluded from Euclidean distance.'))


def summarize(root):
    quality=[];geometry=[];proofs=[];inventory=[]
    for path in sorted((root/'jobs').glob('*/result.json')):
        result=json.loads(path.read_text());inventory.append(dict(job=path.parent.name,status=result['status'],task=result['task']))
        if not result.get('complete'):continue
        if result['task']=='full':quality+=result['quality']
        elif result['task']=='geometry':geometry+=result['rows']
        elif result['task']=='proof':
            g=result['geometry'];proofs.append(dict(dataset=result['dataset'],k=result['k'],method=result['method'],budget=result['budget'],
                constraints=g['constraints'],compile_seconds=g['compile_seconds'],setup_seconds=g['setup_seconds'],
                prove_median_seconds=float(np.median([np.median(s['prove_seconds']) for s in g['samples']])),
                verify_median_seconds=float(np.median([np.median(s['verify_seconds']) for s in g['samples']])),
                pairs=len(g['samples']),reused_preprocessing=result['reused_preprocessing'],independently_verified=result['independent_verification']['complete']))
    for name,rows in [('full_native',quality),('geometry_native',geometry),('geometry_proofs',proofs),('inventory',inventory)]:
        if rows:pd.DataFrame(rows).to_csv(root/(name+'.csv'),index=False)
    text=['# Overnight paper follow-up','',
          'Full-corpus fixed-center fidelity, geometric utility and actual authenticated embedding-distance proofs are separate experiments.',
          'Full-native inputs use all available rows from the existing cached loaders, then remove zero rows; dataset manifests record actual sizes. '
          'Geometry uses the prior capped eight-corpus cohort, k=5/10, three starts, ten random maps, and storage budgets 16/32/64. '
          'Models and raw per-query neighbors, pair errors and assignments are retained under jobs/.',
          'Original floating Full and quantized Full references are reported separately. PCA projected cost is its own objective. '
          'Tail energy is excluded from Euclidean features. Geometry-tail timing is incremental after the head/residual; use full-native for complete compression comparisons.',
          'Authenticated geometry proofs establish exact arithmetic on committed embedding rows. They do not prove an original-space distance-error bound or nearest-neighbor accuracy. '
          'Width-16 bundles reuse already verified preprocessing; new widths receive fresh full preprocessing proofs. Never treat reuse as zero original preprocessing cost.',
          'A separate exact-affine baseline is included in full-native. It is not a general Euclidean embedding.',
          '',f'Completed job receipts: {len(inventory)}. See status.json for failures, exclusions, deadlines and overall completion.',
          'CSV exports: full_native.csv, geometry_native.csv, geometry_proofs.csv, inventory.csv. '
          'Any missing/failed/resource-excluded result is a coverage limitation, not a zero-time or successful measurement.']
    (root/'report.md').write_text('\n'.join(text)+'\n')


def campaign(args):
    root=args.out
    if (root/'status.json').exists():raise RuntimeError('Fresh output directory required')
    started=time.time();state=dict(status='running',complete=False,started_unix=started,host=platform.node(),jobs_done=0,failures=[],exclusions=[],
        resource_policy='sequential; 5.3 GiB own process tree; 768 MiB reserve; 45 minute per-job and 8 hour campaign limits',
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('bound_audit.py'),args.bin]})
    jobs=[]
    if args.smoke:
        jobs=[dict(task='full',dataset='gaussian_hard',k=5,start=0),dict(task='geometry',dataset='gaussian_hard',k=5,start=0)]
        for method in ['full','pca','protected_energy','protected_tail']:
            jobs.append(dict(task='proof',dataset='gaussian_hard',k=5,start=0,method=method,budget=16,proof_case='synthetic_smoke_k5'))
        for method in ['pca','protected_tail']:
            jobs.append(dict(task='proof',dataset='gaussian_hard',k=5,start=0,method=method,budget=32,proof_case='synthetic_smoke_k5'))
    else:
        # Put the JL utility test and its actual proof before the larger native expansion.
        jobs=[dict(task='geometry',dataset=name,k=k,start=start) for name,_,_ in DATASETS for k in [5,10] for start in range(3)]
        for name,k in [('gaussian_hard',5),('gaussian_hard',10),('mnist',5),('20ng_tfidf_nonzero',5)]:
            for method in ['full','protected_energy']:
                jobs.append(dict(task='proof',dataset=name,k=k,start=0,method=method,budget=16,proof_case=f'{name}_k{k}'))
            for budget in [16,32,64]:
                for method in ['pca','protected_tail']:
                    jobs.append(dict(task='proof',dataset=name,k=k,start=0,method=method,budget=budget,proof_case=f'{name}_k{k}'))
        jobs += [dict(task='full',dataset=name,k=k,start=start) for name in ['mnist','fashion_mnist','20ng_tfidf_nonzero'] for k in [5,10] for start in range(5)]
    state['jobs_planned']=len(jobs);save(root/'status.json',state);save(root/'plan.json',jobs)
    for index,job in enumerate(jobs):
        label=f'{index:03d}_{job["task"]}_{job["dataset"]}_k{job["k"]}_s{job["start"]}'
        if job['task']=='proof':label+=f'_{job["method"]}_m{job["budget"]}'
        folder=root/'jobs'/label;folder.mkdir(parents=True)
        state.update(current_job=label);save(root/'status.json',state)
        if time.time()-started>8*3600:
            state['exclusions'].append(dict(job=label,reason='overall eight-hour limit'));state['status']='deadline';break
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',job['task'],'--out',str(folder),'--bin',str(args.bin),
            '--source',str(args.source),'--dataset',job['dataset'],'--k',str(job['k']),'--start',str(job['start'])]
        for key in ['method','budget','proof_case']:
            if key in job:cmd+=['--'+key.replace('_','-'),str(job[key])]
        if args.smoke:cmd+=['--smoke']
        resource=run_bounded(cmd,folder/'worker.log',seconds=min(2700,max(1,8*3600-(time.time()-started))))
        save(folder/'resources.json',resource)
        receipt=folder/'result.json'
        if resource['returncode']!=0 or not receipt.exists():
            state['failures'].append(dict(job=label,resource=resource))
            # Repeated logic failures mean further jobs cannot be trusted; a resource skip remains explicit.
            if len(state['failures'])>=3 and not resource['stop']:
                state['status']='failed';save(root/'status.json',state);break
        elif json.loads(receipt.read_text())['status']=='excluded':state['exclusions'].append(dict(job=label,reason=json.loads(receipt.read_text()).get('reason')))
        state['jobs_done']=index+1;state['elapsed_seconds']=time.time()-started;save(root/'status.json',state)
        summarize(root)
    if state['jobs_done']==len(jobs):state.update(status='complete' if not state['failures'] else 'complete_with_failures',complete=not state['failures'])
    state['elapsed_seconds']=time.time()-started;save(root/'status.json',state);summarize(root)
    return state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--bin',type=Path,required=True);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--worker',choices=['full','geometry','proof']);parser.add_argument('--dataset',default='gaussian_hard')
    parser.add_argument('--k',type=int,default=5);parser.add_argument('--start',type=int,default=0);parser.add_argument('--budget',type=int,default=16)
    parser.add_argument('--method',default='protected_tail');parser.add_argument('--proof-case',default='gaussian_hard_k5');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve();args.bin=args.bin.resolve();args.source=args.source.resolve();args.out.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=4):
        if args.worker:
            if args.worker=='full':full_worker(args)
            elif args.worker=='geometry':geometry_worker(args)
            else:proof_worker(args)
        else:
            if not campaign(args)['complete']:raise SystemExit(2)


if __name__=='__main__':main()
