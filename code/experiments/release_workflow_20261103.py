"""Native release-gate simulation; no timing is labeled cryptographic service."""
from pathlib import Path
import argparse
import time
import numpy as np
from threadpoolctl import threadpool_limits
from commitjl.cost_aware import exact_distances
from commitjl.final_method_20261102 import fit_bases, validation_choice, RANKS
from commitjl.pilot_io import array_sha
from experiments.pca_advantage_20261102 import read, save, sha, evaluate, proposal_banks
from experiments.paper_completion_20261102 import ablate


def moments_upper(rows,old,new):
    distance=exact_distances(rows,old);labels=distance.argmin(1)
    newer=exact_distances(rows,new)
    upper=0
    for g in range(len(old)):
        members=labels==g
        upper+=min(sum(int(v) for v in newer[members,j])-sum(int(v) for v in distance[members,g])
                   for j in range(len(old)))
    truth=sum(map(int,newer.min(1)))-sum(map(int,distance.min(1)))
    assert truth<=upper
    return truth,upper,sum(map(int,distance.min(1)))


def candidate(batch,reference,rate=.25):
    labels=exact_distances(batch,reference).argmin(1)
    target=np.array([batch[labels==g].mean(0) if np.any(labels==g) else reference[g]
                     for g in range(len(reference))])
    return np.clip(np.rint((1-rate)*reference+rate*target),-4095,4095).astype(np.int64)


def run(original,out,profiles):
    started=time.perf_counter()
    with np.load(original/'arrays.npz') as z:a={k:z[k].copy() for k in z.files}
    provenance=read(original/'provenance.json');dataset=provenance['dataset'];seed=provenance['seed']
    old=a['old'].copy();bases={f:a[f] for f in ['centroid','pca']}
    choices=read(original/'selection.json')['choices'];episode=0;records=[];episodes=[]
    for step,batch in enumerate(np.array_split(a['test_incoming'],8)):
        # Candidate rate and order are fixed before outcomes. No favorable-update filtering.
        new=candidate(batch,old)
        if not episodes or episodes[-1]['episode']!=episode:
            folder=out/f'episode_{episode:02}';folder.mkdir(parents=True)
            np.savez_compressed(folder/'frozen.npz',old=old,**bases)
            save(folder/'selection.json',dict(choices=choices,old_sha256=array_sha(old),
                 frozen_unix=time.time(),before_next_candidate_test_evaluation=True))
            episodes.append(dict(episode=episode,first_step=step,old_sha256=array_sha(old),
                                  selection_sha256=sha(folder/'selection.json')))
        truth,upper,old_cost=moments_upper(a['rows'],old,new)
        methods={}
        for method,family in [('ordered_head','centroid'),('pca','pca')]:
            rank=choices[method]['rank']
            r=ablate(a['rows'],old,np.array([new]),bases[family],provenance['scale'],family,rank,
                     caps=(0,16),orders=('endpoint',))
            methods[method]=dict(rank=rank,without_corrections=r[0]['conclusive'],
                 with_cap16=r[1]['conclusive'],corrections=r[1]['corrections'],
                 lower=r[1]['lower'],upper=r[1]['upper'],denominator=r[1]['denominator'])
        accepted=truth<0
        records.append(dict(step=step,episode=episode,truth=truth,old_cost=old_cost,
             relative_change=truth/max(old_cost,1),sign='improvement' if truth<0 else 'worsening' if truth>0 else 'equal',
             moments_upper=upper,moments_conclusive=upper<0,accepted=accepted,
             candidate_sha256=array_sha(new),methods=methods))
        save(out/'progress.json',dict(complete=False,records=records,episodes=episodes))
        if accepted and step<7:
            old=new.copy();episode+=1
            # Installing a bank starts a new episode: refit and reselect on training/validation only.
            bases,_=fit_bases(a['past'],old,seed)
            validation_centers,_=proposal_banks(a['validation_incoming'],old,8)
            validation=[]
            for family,basis in bases.items():
                for rank in RANKS:
                    validation+=evaluate(basis,old,a['validation_rows'],validation_centers,
                                         provenance['scale'],family,rank)
            choices={m:validation_choice(validation,f,profiles[dataset])
                     for m,f in [('ordered_head','centroid'),('pca','pca')]}
    save(out/'result.json',dict(complete=True,dataset=dataset,seed=seed,records=records,episodes=episodes,
         original_arrays_sha256=sha(original/'arrays.npz'),native_seconds=time.perf_counter()-started,
         scope='Post-study native simulated release gate on inspected data. Fixed0.25 update rate; '
               'all eight incoming blocks retained. Exact integer oracle drives simulated installs. '
               'Authentication/reset/proof latency is not measured; no new cryptographic release is claimed. '
               'Within-fit stream steps and reused catalog decisions are dependent, not new fit replications.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--profiles',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):run(a.source,a.out,read(a.profiles)['profiles'])
