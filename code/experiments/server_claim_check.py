"""Independent, bounded server check of fixed-center/Stage-2 paper claims.

Does not implement or benchmark Stage 1. Existing results remain untouched.
Run from code with --out a new directory and freshly built Linux binaries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import time

import numpy as np
import pandas as pd
import psutil
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from data.loaders import load_dataset
from commitjl.protected_jl import centroid_basis
from commitjl.reusable_native import array_hash
from experiments import rq2_query_cost as prior


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def bounded(command, log, seconds=600, max_gib=5.5):
    """Only terminate our own newly created process group on a resource limit."""
    start=time.monotonic();peak=0;stop=None
    with Path(log).open('w',encoding='utf-8') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                process=psutil.Process(child.pid)
                rss=sum(p.memory_info().rss for p in [process]+process.children(recursive=True))
                peak=max(peak,rss)
                if rss>max_gib*2**30:stop='RSS limit'
                if psutil.virtual_memory().available<768*2**20:stop='Host RAM reserve'
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            if time.monotonic()-start>seconds:stop='Wall-time limit'
            if stop:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(.1)
    return dict(returncode=child.returncode,wall_seconds=time.monotonic()-start,peak_rss_bytes=peak,stop=stop)


def binary_case(binary, root, inst):
    folder=root/'cases'/inst['label'];folder.mkdir(parents=True,exist_ok=True)
    inp=folder/'instance.json';out=folder/'result.json';save(inp,inst)
    resource=bounded([str(binary),'-instance',str(inp),'-out',str(out)],folder/'process.log')
    save(folder/'resources.json',resource)
    if resource['stop'] or resource['returncode']!=0:
        raise RuntimeError(f"{inst['label']}: {resource}")
    result=json.loads(out.read_text())
    if result.get('error'):raise RuntimeError(result['error'])
    if not inst['compile_only'] and not result['verify_ok']:raise RuntimeError('Proof verification failed')
    return result,resource


def native_controls(xa, centers, name, k):
    u=centroid_basis(centers);w=xa-centers[0];d=centers-centers[0]
    head=w@u;hc=d@u;energy=np.sum(w*w,axis=1)-np.sum(head*head,axis=1)
    distances=np.sum((xa[:,None,:]-centers[None,:,:])**2,axis=2)
    recovered=np.sum((head[:,None,:]-hc[None,:,:])**2,axis=2)+energy[:,None]
    np.testing.assert_allclose(recovered,distances,rtol=2e-11,atol=2e-9)
    scale=4096/max(np.abs(xa).max(),np.abs(centers).max(),1e-12)
    xi=np.rint(xa*scale).astype(np.int64);ci=np.rint(centers*scale).astype(np.int64)
    wi=xi-ci[0];di=ci-ci[0]
    table=np.sum(wi*wi,axis=1)[:,None]-2*(wi@di.T)+np.sum(di*di,axis=1)
    direct=np.sum((xi[:,None,:]-ci[None,:,:])**2,axis=2)
    np.testing.assert_array_equal(table,direct)
    # An attacker knowing a protected linear map can hide residual energy in
    # its kernel. Correct energy remains sufficient even without a fresh JL.
    from commitjl.sparse_gen import sparse_matrix_gen_csr
    _,a= sparse_matrix_gen_csr(b'claim-check-known-map',16-u.shape[1],xa.shape[1],1)
    a=a.toarray().astype(float);p=np.vstack((u.T,a-(a@u)@u.T))
    noise=np.random.default_rng(808).normal(size=w.shape)
    noise-=np.linalg.lstsq(p@p.T,p@noise.T,rcond=None)[0].T@p
    phi=distances.min(1).sum();noise*=np.sqrt(10*phi/np.sum(noise*noise))
    attacked=w+noise
    true_new=np.sum((attacked[:,None,:]-d[None,:,:])**2,axis=2).min(1).sum()
    def linear_cost(v):
        return np.sum(((v@p.T)[:,None,:]-(d@p.T)[None,:,:])**2,axis=2).min(1).sum()
    ah=attacked@u;ae=np.sum(attacked*attacked,axis=1)-np.sum(ah*ah,axis=1)
    exact_energy_cost=(np.sum((ah[:,None,:]-hc[None,:,:])**2,axis=2)+ae[:,None]).min(1).sum()
    np.testing.assert_allclose(exact_energy_cost,true_new,rtol=2e-11)
    return dict(dataset=name,k=k,rows=len(xa),head_rank=u.shape[1],
        float_head_energy_max_relative_error=float(np.max(np.abs(recovered-distances)/np.maximum(distances,1e-30))),
        exact_integer_affine_distances_equal=True,
        known_linear_map_claim_inflation=float(linear_cost(attacked)/linear_cost(w)),
        true_cost_inflation=float(true_new/phi),
        known_no_jl_head_energy_inflation=float(exact_energy_cost/phi),
        scope='Native algebra/one known-kernel perturbation; not an adversarial ZK proof or Stage-1 test')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--bin',type=Path,required=True)
    parser.add_argument('--old-bin',type=Path,required=True)
    parser.add_argument('--test-bin',type=Path,required=True)
    args=parser.parse_args();root=args.out.resolve();root.mkdir(parents=True,exist_ok=True)
    for field in ['bin','old_bin','test_bin']:setattr(args,field,getattr(args,field).resolve())
    source_paths=[Path(__file__),Path(prior.__file__),Path('commitjl/protected_jl.py'),Path('commitjl/sparse_gen.py'),
        Path('data/loaders.py'),args.bin,args.old_bin,args.test_bin]
    identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    state=dict(status='running',stage='tests',complete=False,cases=0,expected_cases=160,
        host=platform.node(),cpu_count=os.cpu_count(),memory_bytes=psutil.virtual_memory().total,
        numpy=np.__version__,threads=4,source_hashes=identity,started_unix=time.time())
    state_file=root/'status.json';save(state_file,state)
    if (root/'rows.csv').exists():raise RuntimeError('Choose a new output directory; this verification never mixes attempts')
    rows=[];controls=[];datasets=[]
    try:
        snapshots=root/'previous_results';snapshots.mkdir(exist_ok=True)
        previous_files=list(Path('results/protected_fixed_center_20261017').rglob('rows.csv'))
        native=pd.concat([pd.read_csv(p) for p in previous_files],ignore_index=True)
        native.to_csv(snapshots/'native_combined.csv',index=False)
        native_means=native[native.method.isin(['pca','sparse','protected','protected_int'])].groupby(
            ['dataset','k','method','m'])[['ari','r_phi']].mean().reset_index()
        native_means['passes_mean_rule']=native_means.ari.ge(.8)&native_means.r_phi.le(1.1)
        native_means.to_csv(snapshots/'native_means.csv',index=False)
        checks=dict(native_rows=len(native),m16_agreement=native[native.m.eq(16)].groupby('method').agreement.mean().to_dict(),
            m16_mean_rule_counts=native_means[native_means.m.eq(16)].groupby('method').passes_mean_rule.sum().astype(int).to_dict())
        for directory in ['rq2_query_cost_v2_20261017','adversarial_attack_20261017']:
            shutil.copyfile(Path('results')/directory/'rows.csv',snapshots/(directory+'.csv'))
        old_cost=pd.read_csv(snapshots/'rq2_query_cost_v2_20261017.csv')
        old_proved=old_cost[old_cost.error.fillna('').eq('')&old_cost.prove_seconds.notna()]
        checks['actual_proved_rows']=len(old_proved)
        checks['actual_proof_agreement']=old_proved.groupby('method').agreement_with_exact.mean().to_dict()
        save(root/'previous_result_checks.json',checks)
        test=bounded([str(args.test_bin),'-test.run','TestQuery','-test.v'],root/'go_tests.log',seconds=600)
        save(root/'go_tests.json',test)
        if test['returncode']!=0:raise RuntimeError('Circuit tests failed; see go_tests.log')
        with threadpool_limits(limits=4):
            for name,loader,kw in prior.DATASETS:
                x=np.asarray(load_dataset(loader,**kw)['X'],float);x=x[np.linalg.norm(x,axis=1)>0]
                datasets.append(dict(dataset=name,n=len(x),d=x.shape[1],sha256=array_hash(x)))
                for k in prior.KS:
                    order=np.random.default_rng(500+k).permutation(len(x));tr=order[:int(.6*len(x))];au=order[int(.6*len(x)):]
                    centers=KMeans(k,n_init=5,random_state=0).fit(x[tr]).cluster_centers_
                    xa=x[au[:8]];exact=prior.nearest(xa,centers)
                    true_cost=np.cumsum(np.sum((xa-centers[exact])**2,axis=1))
                    reps=prior.representations(x[tr],xa,centers,k,0)
                    energy_rep=reps['protected_energy'];ez,ec,escale,ebits=prior.quantize(energy_rep[0],energy_rep[1],energy_rep[3],energy_rep[2])
                    all_reps={}
                    for method in ['pca','protected','protected_energy']:
                        z,c,h,*energy=reps[method]
                        zi,ci,scale,bits=prior.quantize(z,c,energy[0] if energy else None,h)
                        all_reps[method]=(zi,ci,h,scale,bits,bool(energy))
                    h=energy_rep[2]
                    all_reps['head_energy_no_jl']=(ez[:,:h+1],ec[:,:h+1],h,escale,ebits,True)
                    controls.append(native_controls(xa,centers,name,k));save(root/'native_controls.json',controls)
                    for n in [1,2,4,8]:
                        for method,(zi,ci,h,scale,bits,energy) in all_reps.items():
                            label=f'{name}_k{k}_{method}_n{n}_cost_const'
                            inst=dict(label=label,n=n,w=zi.shape[1],k=k,assign_dims=h,coord_bits=16,
                                cost=True,const_centers=True,z=zi[:n].tolist(),centers=ci.tolist(),salt=str(10**30+n),
                                energy_col=energy,energy_bits=bits,compile_only=n not in [1,8],n_repetitions=3,n_warmup=1)
                            state.update(stage='proofs',current_case=label);save(state_file,state)
                            res,resource=binary_case(args.bin,root,inst)
                            row=dict(dataset=name,k=k,method=method,n=n,w=zi.shape[1],constraints=res['num_constraints'],
                                prove_seconds=float(np.median(res['prove_seconds_samples'])) if res['prove_seconds_samples'] else None,
                                verify_seconds=float(np.median(res['verify_seconds_samples'])) if res['verify_seconds_samples'] else None,
                                verify_ok=res['verify_ok'] if not inst['compile_only'] else None,
                                agreement=float(np.mean(np.asarray(res['assign'])==exact[:n])),
                                original_cost_rel_error=float(int(res['psi'])/scale**2/true_cost[n-1]-1),
                                **resource)
                            if method=='head_energy_no_jl':
                                full=next(v for v in rows if v['dataset']==name and v['k']==k and v['n']==n and v['method']=='protected_energy')
                                assert full['constraints']==row['constraints']
                                assert full['original_cost_rel_error']==row['original_cost_rel_error']
                            rows.append(row);pd.DataFrame(rows).to_csv(root/'rows.csv',index=False)
                            state['cases']=len(rows);save(state_file,state)
                            print(json.dumps(dict(case=label,done=len(rows),constraints=row['constraints'],prove=row['prove_seconds'])),flush=True)
                            if len(rows)==1:
                                check=dict(inst,label='original_binary_comparison',compile_only=True)
                                previous,_=binary_case(args.old_bin,root,check)
                                assert previous['num_constraints']==res['num_constraints'] and previous['psi']==res['psi'] and previous['assign']==res['assign']
                                save(root/'original_binary_comparison.json',dict(equal_constraints_cost_assignments=True))
                            if len(rows)==3:
                                fake=dict(inst,label='expected_stage2_acceptance_new_false_energy_root',n=1,compile_only=False)
                                fake['z']=[list(inst['z'][0])];fake['z'][0][h]=0
                                forged,_=binary_case(args.bin,root,fake)
                                save(root/'stage1_gap_demonstration.json',dict(verified=forged['verify_ok'],
                                    old_cost=res['psi'],new_cost=forged['psi'],
                                    note='Changed energy AND its commitment root. Expected Stage-2 acceptance; no fixed-root hash forgery. Stage 1 must reject a false link to original X.'))
                    save(root/'datasets.json',datasets)
                del x
        state.update(status='complete',stage='finished',complete=True)
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        state['elapsed_seconds']=time.time()-state['started_unix'];save(state_file,state)


if __name__=='__main__':main()
