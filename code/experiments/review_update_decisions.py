"""Independent-fit audit decisions on predeclared real-data Lloyd traces.

These are algorithmic update traces on public benchmarks, not production logs.
Future centers and bases never use the held-out audit records. Every requested
update, duplicate, PCA win, and inconclusive result is retained.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import kmeans_plusplus
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.protected_jl import centroid_basis
from commitjl.paired_certificates import paired_change
from commitjl.reuse_certificates import certificate
from data.loaders import load_dataset
from experiments.bound_audit import save
from experiments.review_decision_controls import exact_cost,decision

DATASETS=('mnist','fashion_mnist','20ng_tfidf')
SEEDS=(11,23,37,41,53)
KS=(3,5,10)
RANKS=(15,63)
# Selection made before this grid, independent of all audit outcomes.
PROOF_FITS={('mnist',3,11),('mnist',10,23),('fashion_mnist',5,11),
            ('fashion_mnist',3,23),('20ng_tfidf',10,11),('20ng_tfidf',5,23)}


def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def lloyd_snapshots(training,k,seed,steps=24):
    centers=kmeans_plusplus(training.astype(float),k,random_state=seed)[0]
    snapshots={}
    for step in range(1,steps+1):
        # Bound workspace even for sparse-source text represented as dense rows.
        labels=[]
        for start in range(0,len(training),128):
            x=training[start:start+128]
            labels.extend(((x[:,None,:]-centers[None,:,:])**2).sum(2).argmin(1).tolist())
        labels=np.asarray(labels)
        centers=np.asarray([training[labels==j].mean(0) if np.any(labels==j) else centers[j] for j in range(k)])
        snapshots[step]=np.rint(centers).astype(np.int64)
    return snapshots


def update_trace(snapshots,age):
    steps=[2,4,8,16] if age==1 else [9,12,16,24]
    base=snapshots[age]
    sets=[base]+[snapshots[s] for s in steps]
    roles=[dict(kind='deployed',iteration=age)]+[dict(kind='lloyd_update',iteration=s) for s in steps]
    reverse=np.clip(2*base-snapshots[steps[0]],-4096,4096)
    sets.append(reverse);roles.append(dict(kind='reverse_step_diagnostic',iteration=None))
    for i,(c,meta) in enumerate(zip(sets,roles)):
        meta.update(set=i,unchanged_centers=bool(np.array_equal(base,c)))
    return [c.tolist() for c in sets],roles


def protocol(datasets,seeds,ks):
    return dict(version='paired-update-decisions-v1',datasets=list(datasets),fit_seeds=list(seeds),k=list(ks),
        ranks=list(RANKS),basis_scale=16384,native_n=1024,proof_n=128,
        training='60% train / 40% audit; remove zero rows before split; fixed public source features; scales from training only, audit clipping counted.',
        prototype_updates='Lloyd iterations on training only. Early base iteration1 -> 2,4,8,16; mature base8 -> 9,12,16,24. Null and reverse-step diagnostic retained.',
        basis='Training-only rank63 learned and PCA; nested rank15 prefix at same precision.',
        decision='Certify strict decrease/increase of exact quantized nearest-center cost from sign of interval; otherwise inconclusive. No coverage threshold.',
        comparisons=['independent absolute intervals','paired change envelope with exact rounded-Gram correction','exact integer Full oracle'],
        proof_fit_selection=[dict(dataset=d,k=k,fit_seed=s,age=1) for d,k,s in sorted(PROOF_FITS)],
        selection='Proof workloads fixed before this grid; every update proved regardless of native outcome. Exploratory seed0 pilot excluded from confirmation seeds.',
        replication='Report fit seeds within each dataset/k stratum. Different k, ages, ranks and updates share each dataset/seed split and are dependent; proof counts are execution checks.',
        claim_boundary='No universal learned superiority or novelty-firstness assumed. Algorithmic benchmark traces are not production deployment traces.',
        python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__)


def run(a):
    a.out.mkdir(parents=True,exist_ok=False)
    p=protocol(a.datasets,a.seeds,a.ks);save(a.out/'protocol.json',p)
    rows=[];fits=[];owners=[]
    with threadpool_limits(limits=4):
        for dataset in a.datasets:
            data=load_dataset(dataset,n_max=5000,seed=0)
            x=np.asarray(data['X'],float);x=x[np.linalg.norm(x,axis=1)>0]
            for k in a.ks:
                for seed in a.seeds:
                    tick=time.monotonic();indices=np.random.default_rng(741+seed).permutation(len(x))
                    cut=int(.6*len(x));tr=x[indices[:cut]];audit=x[indices[cut:cut+1024]]
                    if len(audit)!=1024:raise ValueError('insufficient held-out catalog')
                    scale=4095/max(float(np.abs(tr).max()),1e-12)
                    training=np.rint(tr*scale).astype(np.int64);unclipped=np.rint(audit*scale)
                    audit=np.clip(unclipped,-4096,4096).astype(np.int64)
                    snapshots=lloyd_snapshots(training,k,seed)
                    prototype_seconds=time.monotonic()-tick
                    for age in (1,8):
                        center_sets,roles=update_trace(snapshots,age);base=center_sets[0]
                        fitted={};fit_times={}
                        for method in ('learned','pca'):
                            start=time.monotonic()
                            if method=='learned':
                                fitted[method]=enrich_centroid_basis(training.astype(float),base,
                                    extra=63-centroid_basis(base).shape[1],random_state=seed)[0].T
                            else:fitted[method]=PCA(63,svd_solver='randomized',random_state=seed).fit(training.astype(float)).components_
                            if len(fitted[method])!=63:raise ValueError('rank63 unavailable; do not silently substitute rank')
                            fit_times[method]=time.monotonic()-start
                        costs=[exact_cost(audit,c) for c in center_sets];base_cost=costs[0]
                        fit_id=f'{dataset}_k{k}_seed{seed}_age{age}'
                        meta=dict(dataset=dataset,k=k,fit_seed=seed,age=age,
                            training_sha256=array_sha(training),audit_sha256=array_sha(audit),
                            training_indices_sha256=array_sha(indices[:cut]),audit_indices_sha256=array_sha(indices[cut:cut+1024]),
                            source=data.get('meta',{}),input_scale=str(scale),prototype_trace_seconds=prototype_seconds,
                            audit_clipped_coordinates=int(np.count_nonzero(audit!=unclipped)),
                            representation_fit_seconds=fit_times,roles=roles)
                        fits.append(dict(id=fit_id,**meta));save(a.out/'fit_records.json',fits)
                        for method in ('learned','pca'):
                            for rank in RANKS:
                                basis=np.rint(fitted[method][:rank]*16384).astype(np.int64)
                                model=dict(version='bound-audit-v1',method='learned_energy',n=1024,d=audit.shape[1],k=k,
                                    input_bits=12,basis_scale=16384,input_scale=str(scale),centers=base,basis=basis.tolist(),
                                    tail_width=0,tail_rows=[],tail_signs=[],seed='')
                                old=certificate(model,audit,base)
                                for s,centers in enumerate(center_sets):
                                    absolute=old if s==0 else certificate(model,audit,centers)
                                    paired=paired_change(model,audit,base,centers)
                                    den=paired['cost_denominator'];truth=(costs[s]-base_cost)*den
                                    ilo=absolute['cost_lower_scaled']-old['cost_upper_scaled'];ihi=absolute['cost_upper_scaled']-old['cost_lower_scaled']
                                    plo,phi=paired['change_lower_scaled'],paired['change_upper_scaled']
                                    if not ilo<=truth<=ihi or not plo<=truth<=phi:raise AssertionError('certificate contradicts exact cost change')
                                    # Certified count is also checked against exact nearest-center labels.
                                    xx=np.asarray(audit,dtype=np.int64);cc=np.asarray(centers,dtype=np.int64)
                                    labels=((xx[:,None,:]-cc[None,:,:])**2).sum(2).argmin(1)
                                    wrong=sum(flag and label!=true for flag,label,true in zip(absolute['certified_flags'],absolute['labels'],labels))
                                    if wrong:raise AssertionError('incorrect assignment certificate')
                                    rows.append(dict(dataset=dataset,k=k,fit_seed=seed,age=age,method=method,rank=rank,n=1024,
                                        **roles[s],true_cost=costs[s],true_change=costs[s]-base_cost,
                                        paired_decision=decision(plo,phi),absolute_decision=decision(ilo,ihi),
                                        paired_lower_scaled=str(plo),paired_upper_scaled=str(phi),
                                        independent_lower_scaled=str(ilo),independent_upper_scaled=str(ihi),cost_denominator=str(den),
                                        paired_width_over_baseline=(phi-plo)/den/base_cost,
                                        absolute_change_width_over_baseline=(ihi-ilo)/den/base_cost,
                                        assignment_agreement=float(np.mean(np.asarray(absolute['labels'])==labels)),
                                        certified_fraction=absolute['certified_count']/len(audit),
                                        absolute_cost_relative_width=(absolute['cost_upper_scaled']-absolute['cost_lower_scaled'])/absolute['cost_lower_scaled'] if absolute['cost_lower_scaled'] else None,
                                        basis_sha256=array_sha(basis),false_certified_assignments=int(wrong)))
                                if (dataset,k,seed) in PROOF_FITS and age==1:
                                    name=f'{fit_id}_{method}_r{rank}';owner=copy.deepcopy(model);owner['n']=128
                                    context=f'{dataset}:{k}:{seed}'
                                    salt=lambda name:str(int.from_bytes(hashlib.sha256((context+name).encode()).digest(),'big')%(2**250))
                                    inp=dict(model=owner,x=audit[:128].tolist(),salt_x=salt('x'),salt_z=salt('z'),
                                        queries=[list(range(i,i+2)) for i in range(0,128,2)],reuse_centers=center_sets,
                                        comparison_base=base,experiment_metadata={**meta, 'method':method,'rank':rank,
                                            'basis_sha256':array_sha(basis),'proof_scope':'N128 prefix; native N1024 is separate'})
                                    path=a.out/'owner_inputs'/(name+'.json');save(path,inp)
                                    owners.append(dict(name=name,path=path.relative_to(a.out).as_posix(),dataset=dataset,k=k,fit_seed=seed,
                                        method=method,rank=rank,batch=2 if dataset=='20ng_tfidf' else 4,
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
                        pd.DataFrame(rows).to_csv(a.out/'cells.csv',index=False)
                        save(a.out/'proof_inputs.json',owners)
                        print(json.dumps(dict(completed=fit_id,cells=len(rows),proof_inputs=len(owners))),flush=True)
    frame=pd.DataFrame(rows);requests=frame[(frame.kind=='lloyd_update') & ~frame.unchanged_centers].copy()
    requests['paired_conclusive']=requests.paired_decision!='inconclusive';requests['absolute_conclusive']=requests.absolute_decision!='inconclusive'
    perfit=requests.groupby(['dataset','k','fit_seed','age','method','rank'],as_index=False).agg(
        requests=('set','count'),paired_decision_rate=('paired_conclusive','mean'),absolute_decision_rate=('absolute_conclusive','mean'),
        certified_fraction=('certified_fraction','mean'),agreement=('assignment_agreement','mean'),paired_width=('paired_width_over_baseline','mean'))
    perfit.to_csv(a.out/'fit_summary.csv',index=False)
    # Paired, descriptive comparisons. Do not treat updates as independent trials.
    paired=perfit.pivot(index=['dataset','k','fit_seed','age','rank'],columns='method',values=['paired_decision_rate','certified_fraction','agreement'])
    gaps=pd.DataFrame(index=paired.index)
    for metric in ['paired_decision_rate','certified_fraction','agreement']:
        gaps[metric+'_learned_minus_pca']=paired[(metric,'learned')]-paired[(metric,'pca')]
    gaps.reset_index().to_csv(a.out/'paired_fit_differences.csv',index=False)
    save(a.out/'summary.json',dict(complete=True,native_cells=len(rows),refitted_prototype_models=len(a.datasets)*len(a.ks)*len(a.seeds),
        randomized_split_seed_replications=len(a.datasets)*len(a.seeds),
        dependent_age_configurations=2*len(a.datasets)*len(a.ks)*len(a.seeds),proof_inputs=len(owners),
        numerical_false_decisions=0,scope='Native grid, not proof completion',
        protocol_sha256=hashlib.sha256((a.out/'protocol.json').read_bytes()).hexdigest()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',default=list(DATASETS));p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS));p.add_argument('--ks',type=int,nargs='+',default=list(KS))
    run(p.parse_args())


if __name__=='__main__':main()
