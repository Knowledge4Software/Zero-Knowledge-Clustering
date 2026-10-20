"""Finite, resource-guarded original-data binding / subset-query study.

Linux server runner. All methods use the NEW Poseidon2 row commitment and
fixed-point specification. Results must not be pooled with legacy MiMC times.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import signal
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from data.loaders import load_dataset
from commitjl.protected_jl import centroid_basis
from commitjl.sparse_gen import sparse_matrix_gen_csr
from commitjl.reusable_native import array_hash,stable_squared_distances

METHODS=('full','pca','protected_energy','protected_tail','affine')
DATASETS=[
 ('gaussian_easy','gaussian_mixture',dict(n=2000,d=256,k=10,gamma=1.,seed=0)),
 ('gaussian_hard','gaussian_mixture',dict(n=2000,d=512,k=10,gamma=.35,imbalance=3.,seed=0)),
 ('dbscan_margin','dbscan_margin',dict(n=1000,d=128)),
 ('mnist','mnist',dict(n_max=5000,seed=0)),
 ('fashion_mnist','fashion_mnist',dict(n_max=5000,seed=0)),
 ('gisette','gisette',dict(n_max=3000,seed=0)),
 ('arcene','arcene',dict()),
 ('20ng_tfidf_nonzero','20ng_tfidf',dict(n_max=3000,seed=0))]
CAPS={'gaussian_hard':32,'mnist':16,'fashion_mnist':16,'20ng_tfidf_nonzero':16,'gisette':8}
T=4096


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix(path.suffix+'.tmp').open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(path.with_suffix(path.suffix+'.tmp'),path)


def bounded(command,log,seconds=2400):
    start=time.monotonic();peak=0;stop=None
    with Path(log).open('w',encoding='utf-8') as out:
        proc=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
        while proc.poll() is None:
            try:
                p=psutil.Process(proc.pid);rss=sum(q.memory_info().rss for q in [p]+p.children(recursive=True));peak=max(peak,rss)
                if rss>5.3*2**30:stop='child RSS guard'
                if psutil.virtual_memory().available<768*2**20:stop='host RAM reserve'
            except psutil.NoSuchProcess:pass
            if time.monotonic()-start>seconds:stop='finite phase wall-time guard'
            if stop:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(.2)
    return dict(returncode=proc.returncode,wall_seconds=time.monotonic()-start,peak_rss_bytes=peak,stop=stop)


def measure(fn,reps=3):
    vals=[];answer=None
    for _ in range(reps):
        t=time.perf_counter();answer=fn();vals.append(time.perf_counter()-t)
    return answer,float(np.median(vals))


def build_models(x,centers,train,start,name,k):
    scale=4096/max(float(np.abs(x).max()),float(np.abs(centers).max()),1e-12)
    xi=np.rint(x*scale).astype(np.int64);ci=np.rint(centers*scale).astype(np.int64)
    t=time.perf_counter();u=centroid_basis(ci.astype(float));b=np.rint(T*u.T).astype(np.int64);protected_fit=time.perf_counter()-t
    t=time.perf_counter();pca=PCA(16,svd_solver='randomized',random_state=start).fit(x[train]);pb=np.rint(T*pca.components_).astype(np.int64);pca_fit=time.perf_counter()-t
    t=time.perf_counter();delta=ci[1:]-ci[0];_=(delta*delta).sum(1);affine_fit=time.perf_counter()-t
    common=dict(version='bound-audit-v1',n=1,d=x.shape[1],k=k,input_bits=12,basis_scale=T,
        input_scale=format(scale,'.17g'),centers=ci.tolist(),basis=[],tail_rows=[],tail_signs=[],tail_width=0,seed='')
    models={}
    for method in METHODS:
        model=dict(common,method=method)
        if method=='pca':model['basis']=pb.tolist()
        elif method.startswith('protected'):model['basis']=b.tolist()
        models[method]=model
    fits=dict(full=0.,pca=pca_fit,protected_energy=protected_fit,protected_tail=protected_fit,affine=affine_fit)
    return xi,ci,scale,models,fits


def add_tail(model,seed):
    width=16-len(model['basis'])-1
    _,a=sparse_matrix_gen_csr(seed,width,model['d'],1);ac=a.tocsc()
    assert np.all(np.diff(ac.indptr)==1)
    return dict(model,tail_width=width,tail_rows=ac.indices.astype(int).tolist(),
                tail_signs=ac.data.astype(int).tolist(),seed=seed.hex()),a


def native_stats(method,model,xa,ci):
    w=xa.astype(float)-ci[0];d=ci-ci[0]
    if method=='full':return xa.astype(float),ci.astype(float),None,None
    if method=='affine':
        wi=xa-ci[0];di=ci[1:]-ci[0]
        scores=2*(wi@di.T)-(di*di).sum(1)
        return scores,None,(wi*wi).sum(1),None
    u=np.asarray(model['basis'],float)/T
    h=w@u.T;hc=d@u.T
    if method=='pca':return h,hc,None,None
    residual=w-h@u
    energy=np.einsum('ij,ij->i',residual,residual)
    tail=None
    if model['tail_width']:
        from scipy.sparse import csc_matrix
        a=csc_matrix((model['tail_signs'],model['tail_rows'],np.arange(model['d']+1)),shape=(model['tail_width'],model['d']))
        tail=np.asarray(a@residual.T).T
    return h,hc,energy,tail


def native_answer(method,values):
    h,c,e,_=values
    if method=='affine':
        scores=np.column_stack((np.zeros(len(h),dtype=np.int64),h));labels=scores.argmax(1)
        return labels,float((e-scores[np.arange(len(h)),labels]).sum())
    distances=stable_squared_distances(h,c);labels=distances.argmin(1)
    return labels,float(distances[np.arange(len(h)),labels].sum()+(e.sum() if e is not None else 0.))


def quality_case(x,name,k,start,root):
    order=np.random.default_rng(10000+start).permutation(len(x));tr=order[:int(.6*len(x))];au=order[int(.6*len(x)):]
    t=time.perf_counter();centers=KMeans(k,n_init=5,random_state=start).fit(x[tr]).cluster_centers_;km=time.perf_counter()-t
    xi,ci,scale,models,fits=build_models(x,centers,tr,start,name,k)
    labels=stable_squared_distances(x[au],centers).argmin(1);phi=float(np.sum((x[au]-centers[labels])**2))
    qlabels=stable_squared_distances(xi[au].astype(float),ci.astype(float)).argmin(1)
    pair_rng=np.random.default_rng(701+start);ip=pair_rng.integers(len(au),size=1000);jp=pair_rng.integers(len(au),size=1000)
    keep=ip!=jp;ip,jp=ip[keep],jp[keep]
    original=np.sum((x[au[ip]]-x[au[jp]])**2,axis=1);good=original>0;ip,jp,original=ip[good],jp[good],original[good]
    rows=[];geometry=[]
    for method in METHODS:
        model=models[method]
        if method=='protected_tail':
            seed=hashlib.sha256(f'geometry-fixed-corpus/{name}/{k}/{start}/0'.encode()).digest()
            t=time.perf_counter();model,_=add_tail(model,seed);fits[method]+=time.perf_counter()-t
        vals,comp=measure(lambda:native_stats(method,model,xi[au],ci))
        (pred,estimated),down=measure(lambda:native_answer(method,vals),reps=5)
        original_assignment_cost=float(np.sum((x[au]-centers[pred])**2))
        rows.append(dict(dataset=name,k=k,start=start,n_audit=len(au),d=x.shape[1],method=method,
            ari=float(adjusted_rand_score(labels,pred)),agreement=float(np.mean(labels==pred)),
            ari_quantized_full=float(adjusted_rand_score(qlabels,pred)),
            agreement_quantized_full=float(np.mean(qlabels==pred)),r_phi=original_assignment_cost/phi,
            claimed_original_cost_error=estimated/scale**2/phi-1,
            shared_kmeans_seconds=km,model_fit_seconds=fits[method],compression_seconds=comp,
            downstream_seconds=down,stored_columns=x.shape[1] if method=='full' else 16 if method in ['pca','protected_tail'] else k,
            reference='floating original-space nearest fixed centers; quantized-Full agreement reported separately'))
        if method!='affine':
            h,_,_,tail=vals
            projected=np.sum((h[ip]-h[jp])**2,axis=1)
            if tail is not None:projected+=np.sum((tail[ip]-tail[jp])**2,axis=1)
            errors=np.abs(projected/scale**2/original-1)
            geometry.append(dict(dataset=name,k=k,start=start,map=0,method=method,q95=float(np.quantile(errors,.95)),
                mean_error=float(errors.mean()),exceed_015=float(np.mean(errors>.15)),pairs=len(errors),
                note='Energy is NOT treated as a Euclidean feature; no-tail geometry uses only the head'))
    return rows,geometry,(xi,ci,scale,models,fits,tr,au,centers)


def subsets(n,scope):
    sizes=sorted(set(q for q in ([1,4,8] if scope!='main' else [1,8,16,32]) if q<=n))
    rng=np.random.default_rng(618);out=[]
    for q in sizes:
        if q==n:out.append(list(range(n)));continue
        seen=set()
        for _ in range(100):
            value=tuple(sorted(rng.choice(n,q,replace=False).tolist()))
            if value not in seen:seen.add(value);out.append(list(value))
            target=2 if scope!='main' else 8 if q==8 else 4
            if len(seen)==target:break
    return sizes,out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--bin',type=Path,required=True)
    parser.add_argument('--scope',choices=['smoke','pilot','main'],default='pilot');parser.add_argument('--test-bin',type=Path)
    args=parser.parse_args();root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists():raise RuntimeError('New output directory required; old evidence is never mixed')
    binary=args.bin.resolve();start=time.time()
    state=dict(status='running',complete=False,scope=args.scope,started_unix=start,host=platform.node(),cpu_count=os.cpu_count(),
        memory_bytes=psutil.virtual_memory().total,threads=4,hash='Poseidon2/BN254; new cohort, not legacy MiMC',basis_scale=T,
        phases_done=0,failures=[],exclusions=[],source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__),binary,Path('commitjl/protected_jl.py'),Path('commitjl/sparse_gen.py'),Path('data/loaders.py')]})
    save(root/'status.json',state);quality=[];geometry=[];phases=[];manifest=[];pending=[]
    def run_phase(case,phase,inp,q=1):
        label=f'{phase}_{q}' if phase=='query' else phase;directory=root/'cases'/case;directory.mkdir(parents=True,exist_ok=True)
        state.update(current_case=case,current_phase=label);save(root/'status.json',state)
        cmd=[str(binary),'-phase',phase,'-instance',str(inp),'-out',str(directory),'-q',str(q),
             '-reps','3' if args.scope=='main' else '1','-warmup','1' if args.scope=='main' else '0','-max-constraints','1300000']
        if phase=='verify':cmd+=['-trust',str(directory/'trusted.json')]
        resource=bounded(cmd,directory/(label+'.process.log'));save(directory/(label+'.resources.json'),resource)
        file=directory/(f'query_{q}.json' if phase=='query' else phase+'.json')
        result=json.loads(file.read_text()) if file.exists() else dict(status='process_failed',error=resource)
        row=dict(case=case,phase=phase,q=q,**{k:v for k,v in resource.items() if k!='returncode'},
            returncode=resource['returncode'],result_status=result['status'],constraints=result.get('constraints',0),
            compile_seconds=result.get('compile_seconds',0),setup_seconds=result.get('setup_seconds',0),verified=result.get('verified',0))
        phases.append(row);pd.DataFrame(phases).to_csv(root/'phases.csv',index=False)
        state['phases_done']=len(phases)
        if resource['stop'] or resource['returncode'] not in [0,3]:
            state['failures'].append(row);save(root/'status.json',state);raise RuntimeError(f'{case}/{phase}: {result.get("error")}')
        if resource['returncode']==3:state['exclusions'].append(row)
        save(root/'status.json',state);print(json.dumps(dict(case=case,phase=label,status=result['status'],constraints=result.get('constraints'),wall=resource['wall_seconds'])),flush=True)
        return result
    try:
        if args.test_bin:
            resource=bounded([str(args.test_bin.resolve()),'-test.run','TestBound','-test.v'],root/'go_tests.log',seconds=600)
            save(root/'go_tests.json',resource)
            if resource['returncode']!=0:raise RuntimeError('Server circuit tests failed')
        sources=DATASETS if args.scope=='main' else [DATASETS[1]]
        if args.scope=='smoke':sources=[('synthetic_smoke','gaussian_mixture',dict(n=100,d=32,k=5,gamma=.5,seed=2))]
        with threadpool_limits(limits=4):
            for name,loader,kw in sources:
                x=np.asarray(load_dataset(loader,**kw)['X'],float);x=x[np.linalg.norm(x,axis=1)>0]
                manifest.append(dict(dataset=name,n=len(x),d=x.shape[1],sha256=array_hash(x)));save(root/'datasets.json',manifest)
                for k in ([5,10] if args.scope=='main' else [5]):
                    proof_data=None
                    for clustering_start in range(3 if args.scope=='main' else 1):
                        state.update(current_case=f'{name}_k{k}_native_s{clustering_start}',current_phase='native_quality');save(root/'status.json',state)
                        rr,gg,data=quality_case(x,name,k,clustering_start,root);quality+=rr;geometry+=gg
                        pd.DataFrame(quality).to_csv(root/'native_quality.csv',index=False);pd.DataFrame(geometry).to_csv(root/'native_geometry.csv',index=False)
                        if clustering_start==0:proof_data=data
                    eligible=args.scope!='main' or (name in CAPS and (k==5 or name=='gaussian_hard'))
                    if not eligible:continue
                    xi,ci,scale,models,fits,tr,au,centers=proof_data
                    n=CAPS.get(name,8) if args.scope=='main' else 4 if args.scope=='smoke' else 8
                    sizes,queries=subsets(n,args.scope);catalog=xi[au[:n]]
                    salt_x=str(secrets.randbelow(2**240)+1);salt_z=str(secrets.randbelow(2**240)+1)
                    owner_root=None;nonce=secrets.token_hex(32);model_cases=[]
                    for method in METHODS:
                        model=dict(models[method],n=n)
                        if method=='protected_tail':
                            assert owner_root is not None
                            param_digest=hashlib.sha256(json.dumps(dict(centers=model['centers'],basis=model['basis'],n=n,d=model['d']),sort_keys=True).encode()).hexdigest()
                            seed=hashlib.sha256(f'bound-tail-v1/{owner_root}/{param_digest}/{nonce}'.encode()).digest()
                            model,_=add_tail(model,seed)
                            save(root/f'{name}_k{k}_challenge.json',dict(original_root=owner_root,parameters=param_digest,nonce=nonce,seed=seed.hex(),
                                source='benchmark verifier local CSPRNG after owner commitment; no distributed beacon claim'))
                        case=f'{name}_k{k}_{method}';directory=root/'cases'/case;directory.mkdir(parents=True,exist_ok=True)
                        inp=directory/'input.json';save(inp,dict(model=model,x=catalog.tolist(),salt_x=salt_x,salt_z=salt_z,queries=queries))
                        save(directory/'native_context.json',dict(dataset=name,k=k,method=method,input_scale=scale,
                            model_fit_seconds=fits[method],n_catalog=n,train_rows=len(tr),heldout_rows=len(au),
                            original_float_points=x[au[:n]].tolist(),original_float_centers=centers.tolist(),native_split_seed=10000))
                        prepared_result=run_phase(case,'prepare',inp)
                        cx=prepared_result['statement']['cx']
                        if owner_root is None:owner_root=cx
                        assert cx==owner_root,'Comparators must share the same owner commitment'
                        pending.append((case,inp,sizes));model_cases.append(case)
                    save(root/f'{name}_k{k}_paired_cases.json',dict(original_root=owner_root,cases=model_cases))
                del x
            state['planned_bundles']=len(pending);state['bundles_complete']=0;save(root/'status.json',state)
            for case,inp,sizes in pending:
                p=run_phase(case,'stage1',inp)
                if not p['complete']:raise RuntimeError('Stage1 could not complete within declared budget')
                for q in sizes:run_phase(case,'query',inp,q)
                run_phase(case,'verify',inp)
                state['bundles_complete']+=1;save(root/'status.json',state)
        state.update(status='complete',complete=True)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        state['elapsed_seconds']=time.time()-start;save(root/'status.json',state)


if __name__=='__main__':main()
