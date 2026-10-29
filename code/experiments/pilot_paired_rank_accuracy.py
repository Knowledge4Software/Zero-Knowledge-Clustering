"""Fixed-precision rank pilot for the original faster paired certificate.

Fresh nested prefixes share one training-only rank127 fit. A separate pinned
local rank63 basis anchors the preceding experiment. No proof-speed claim.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from data.loaders import load_dataset
from experiments.review_update_decisions import array_sha
from experiments.pilot_certificate_refinement import exact_rows, NATIVE
from commitjl.protected_jl import centroid_basis
from commitjl.enriched_jl import enrich_centroid_basis
from commitjl.reuse_certificates import certificate
from commitjl.paired_certificates import paired_change

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ordered-head-only', action='store_true')
    args = parser.parse_args();args.out.mkdir(exist_ok=False, parents=True)
    records, pins = [], []
    started = time.monotonic()
    source_dir = ROOT/'paper_results/review_controls/local_group_moments_extended_20261023'
    fits = json.loads((NATIVE/'fit_records.json').read_text())
    protocol = dict(workloads=[['mnist',5,11],['fashion_mnist',3,11]],ages=[1,8],n=1024,
        variants=['pinned63','nested63','nested95','nested127'],basis_scale=16384,
        fitting='Fresh training-only rank127 fit with nested prefixes; pinned63 is the preceding local basis, not necessarily that prefix.',
        scope='Native numerical pilot of original paired relation; no group moments, exact-row corrections or proof-speed claims.',
        selection='Same two development fits as earlier pilots, no independent confirmation.',threads=2,
        ordered_head_only=args.ordered_head_only)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    with threadpool_limits(limits=2):
        for dataset,k in [('mnist',5),('fashion_mnist',3)]:
            source=load_dataset(dataset,n_max=5000,seed=0);x=np.asarray(source['X'],float)
            x=x[np.linalg.norm(x,axis=1)>0];indices=np.random.default_rng(752).permutation(len(x));cut=int(.6*len(x))
            tr=x[indices[:cut]];scale=4095/float(np.abs(tr).max());training=np.rint(tr*scale).astype(np.int64)
            audit=np.clip(np.rint(x[indices[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
            for age in [1,8]:
                meta=next(f for f in fits if (f['dataset'],f['k'],f['fit_seed'],f['age'])==(dataset,k,11,age))
                assert array_sha(training)==meta['training_sha256'];assert array_sha(audit)==meta['audit_sha256']
                for method in (['learned'] if args.ordered_head_only else ['learned','pca']):
                    z=np.load(source_dir/f'{dataset}_k{k}_seed11_age{age}_{method}.npz')
                    sets=z['centers'];old=sets[0]
                    if method=='learned':
                        fitted=enrich_centroid_basis(training.astype(float),old,extra=127-centroid_basis(old).shape[1],random_state=11,preserve_pca_order=args.ordered_head_only)[0].T
                    else:
                        fitted=PCA(127,svd_solver='randomized',random_state=11).fit(training.astype(float)).components_
                    max_basis=np.rint(fitted*16384).astype(np.int64)
                    np.savez_compressed(args.out/f'{dataset}_age{age}_{method}_bases.npz',pinned63=z['basis'],max127=max_basis)
                    base_cost=sum(map(int,exact_rows(audit,old)))
                    for variant,rank in [('pinned63',63),('nested63',63),('nested95',95),('nested127',127)]:
                        basis=z['basis'] if variant=='pinned63' else max_basis[:rank]
                        model=dict(basis=basis.tolist(),basis_scale=16384,centers=old.tolist(),input_bits=12)
                        pins.append(dict(dataset=dataset,age=age,method=method,variant=variant,basis_sha256=array_sha(basis),audit_sha256=array_sha(audit)))
                        for s in range(1,5):
                            now=sets[s];cert=certificate(model,audit,now);bound=paired_change(model,audit,old,now)
                            exact=exact_rows(audit,now);true_cost=sum(map(int,exact));delta=true_cost-base_cost
                            labels=((audit[:,None,:]-now[None,:,:])**2).sum(2).argmin(1)
                            correct=np.asarray(cert['labels'])==labels
                            assert all(not flag or right for flag,right in zip(cert['certified_flags'],correct))
                            den=bound['cost_denominator'];assert bound['change_lower_scaled']<=delta*den<=bound['change_upper_scaled']
                            assert cert['cost_lower_scaled']<=true_cost*den<=cert['cost_upper_scaled']
                            truth='worsens' if delta>0 else 'improves' if delta<0 else 'equal'
                            assert bound['decision'] in [truth,'inconclusive']
                            estimate=cert['psi']/16384**4
                            records.append(dict(dataset=dataset,k=k,age=age,method=method,variant=variant,rank=rank,set=s,
                                agreement_percent=100*correct.mean(),assignment_coverage_percent=100*cert['certified_count']/len(audit),
                                absolute_cost_error_percent=100*abs(estimate/true_cost-1),truth=truth,decision=bound['decision'],
                                interval_width_percent=100*(bound['change_upper_scaled']-bound['change_lower_scaled'])/den/base_cost))
                        pd.DataFrame(records).to_csv(args.out/'results.csv',index=False)
                        print(json.dumps(dict(dataset=dataset,age=age,method=method,variant=variant,elapsed=time.monotonic()-started)),flush=True)
    data=pd.DataFrame(records)
    data.groupby(['age','method','variant']).agg(requests=('set','size'),agreement=('agreement_percent','mean'),
        coverage=('assignment_coverage_percent','mean'),cost_error=('absolute_cost_error_percent','mean'),
        resolved=('decision',lambda s:int((s!='inconclusive').sum()))).to_csv(args.out/'groups.csv')
    (args.out/'pins.json').write_text(json.dumps(pins,indent=2))
    summary=dict(complete=True,elapsed_seconds=time.monotonic()-started,unique_requests=16,method_variant_checks=len(records),false_certified_assignments=0,false_decisions=0,proof_timings_measured=False)
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
