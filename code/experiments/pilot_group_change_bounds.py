"""All frozen group-moment fits: bound switching corrections before exact work.

No new fits, update selection or cryptographic speed claim. All 24 historical
natural regressions and their companion updates remain in the evaluation.
"""
from pathlib import Path
import argparse,json,re,time
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from campaigns.local_fast_tradeoff import save,sha
from commitjl.group_change_bounds import correction_bounds,paired_correction_bounds,exact_old_correction_bounds,compressed_old_correction_bounds,refine
from commitjl.group_moment_reference import prepare_groups
from commitjl.paired_certificates import paired_change
from data.loaders import load_dataset
from experiments.pilot_certificate_refinement import exact_rows,sign
from experiments.review_update_decisions import array_sha

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'paper_results/review_controls/local_group_moments_extended_20261023'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--fresh-confirmation',action='store_true');p.add_argument('--exact-old',action='store_true');p.add_argument('--certificate-scale-bits',type=int,default=14,choices=[6,8,10,12,14]);p.add_argument('--direct-only',action='store_true');p.add_argument('--compressed-moments',action='store_true');p.add_argument('--regressions-only',action='store_true');a=p.parse_args()
    if a.compressed_moments and (not a.direct_only or not a.exact_old):p.error('compressed moments require --exact-old --direct-only')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    source_root=a.source.resolve()
    pins={x['fit']:x for x in json.loads((source_root/'pins.json').read_text())}
    previous=pd.read_csv(source_root/'results.csv');budgets=[0.] if a.direct_only else [0.,.005,.01,.02,.05,1.]
    protocol=dict(source=str(source_root),source_pins_sha256=sha(source_root/'pins.json'),fits=sorted(pins),rank=63,n=1024,
        variants=['ordinary_paired','group_separate','group_paired'],exact_row_budgets=budgets,
        selection='Every existing group-moment fit, both methods/ages, and all four actual updates retained. No new refits or update selection.',
        independent_cohort='Previously fixed seeds67/79; their existing native results contain no natural regressions.',
        group_interval='Exact assigned-group aggregate plus bounds on the difference of switching advantages. Paired version subtracts group movement before bounding residual norms.',
        refinement='Descending correction-interval width with row-index tie break; select before consulting exact corrections; fixed budget ceilings.',
        scope='Native feasibility only. Moment/membership binding and correction-bound proofs still require an efficient authenticated implementation. No speed claim from exact-row fractions.',threads=1)
    if a.fresh_confirmation:
        protocol.update(selection='All freshly predeclared confirmation fits and updates. No refitting or outcome selection after seeing these bounds.',independent_cohort='Fresh seeds167/193, both image datasets/k3/5/ages and representations; count natural regressions actually observed.')
    if a.exact_old:protocol.update(exact_old_profile=True,variants=['ordinary_paired','group_exact_old'],preprocessing='Exact old nearest labels computed from the fixed catalog before future centers. The proof profile must enforce this old argmin on every row.',fast_path='A<0 implies true improvement only after the exact-old preprocessing proof. No per-row query bounds needed on this path.')
    if a.certificate_scale_bits!=14:protocol.update(certificate_scale_bits=a.certificate_scale_bits,prediction_basis_unchanged=True,scope='Native screening of an auxiliary certificate basis rounded from the frozen basis. Original prediction outputs remain frozen. No proof-speed claim.')
    if a.compressed_moments:protocol.update(compressed_moments=True,variants=['ordinary_paired','group_compressed_moments'],scope='Native compressed-moment feasibility. No authenticated implementation or speed inferred from this screen.')
    if a.regressions_only:protocol.update(selection='All natural regressions in this frozen cohort and both representations. Stress test; not the full update coverage denominator.')
    save(out/'protocol.json',protocol);records=[];source_hashes={};dataset_cache={};audit_cache={}
    with threadpool_limits(limits=1):
        for ordinal,name in enumerate(sorted(pins)):
            if psutil.virtual_memory().available<650*2**20:raise RuntimeError('RAM reserve reached; keep partial results and resume only after freeing memory.')
            match=re.fullmatch(r'(.+)_k(\d+)_seed(\d+)_age(\d+)_(learned|pca)',name);assert match
            dataset,k,seed,age,method=match.groups();k,seed,age=map(int,[k,seed,age]);source=source_root/(name+'.npz')
            fit_rows=previous[(previous.dataset==dataset)&(previous.k==k)&(previous.seed==seed)&(previous.age==age)&(previous.method==method)]
            if a.regressions_only and not (fit_rows.true_change>0).any():continue
            frozen=np.load(source);source_hashes[source.name]=sha(source)
            if dataset not in dataset_cache:
                loaded=load_dataset(dataset,n_max=5000,seed=0);x=np.asarray(loaded['X'],float);dataset_cache[dataset]=x[np.linalg.norm(x,axis=1)>0]
            if (dataset,seed) not in audit_cache:
                x=dataset_cache[dataset];indices=np.random.default_rng(741+seed).permutation(len(x));cut=int(.6*len(x));tr=x[indices[:cut]]
                scale=4095/max(float(np.abs(tr).max()),1e-12);training=np.rint(tr*scale).astype(np.int64)
                audit=np.clip(np.rint(x[indices[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
                audit_cache[dataset,seed]=(audit,array_sha(training))
            rows,training_hash=audit_cache[dataset,seed];pin=pins[name]
            assert array_sha(rows)==pin['audit_sha256'] and training_hash==pin['training_sha256']
            assert array_sha(frozen['basis'])==pin['basis_sha256']
            centers=frozen['centers'];old=centers[0];groups=frozen['groups'].tolist();prepared=prepare_groups(rows,groups,k)
            assert prepared['counts']==frozen['counts'].tolist() and np.array_equal(prepared['sums'],frozen['sums'])
            model=dict(basis=frozen['basis'].tolist(),basis_scale=16384,centers=old.tolist(),input_bits=12)
            if a.certificate_scale_bits!=14:
                model['basis_scale']=2**a.certificate_scale_bits;model['basis']=np.rint(np.asarray(frozen['basis'],float)*model['basis_scale']/16384).astype(np.int64).tolist()
            if a.exact_old:
                groups=((rows[:,None,:]-old[None,:,:])**2).sum(2).argmin(1).tolist();prepared=prepare_groups(rows,groups,k)
            old_min=exact_rows(rows,old);old_g=old[np.asarray(groups)]
            for s in range(1,5):
                if a.regressions_only and not (fit_rows[fit_rows['set']==s].true_change>0).any():continue
                new=centers[s];new_min=exact_rows(rows,new);delta_rows=new_min-old_min;true_delta=sum(map(int,delta_rows))
                historical=previous[(previous.dataset==dataset)&(previous.k==k)&(previous.seed==seed)&(previous.age==age)&(previous.method==method)&(previous['set']==s)]
                assert len(historical)==1 and true_delta==int(historical.iloc[0].true_change)
                new_g=new[np.asarray(groups)]
                assigned_rows=2*(rows*(old_g-new_g)).sum(1)+(new_g*new_g-old_g*old_g).sum(1)
                den=model['basis_scale']**6;exact_correction=[(int(d)-int(v))*den for d,v in zip(delta_rows,assigned_rows)]
                common=dict(fit=name,dataset=dataset,k=k,seed=seed,age=age,method=method,set=s,
                    cohort='fresh_confirmation' if a.fresh_confirmation else ('confirmation' if seed in [67,79] else 'historical_development'),truth=sign(true_delta,true_delta),true_change=true_delta)
                ordinary=paired_change(model,rows,old,new)
                assert ordinary['change_lower_scaled']<=true_delta*den<=ordinary['change_upper_scaled']
                records.append(dict(**common,variant='ordinary_paired',budget_fraction=0.,exact_rows=0,uncertain_rows=None,
                    decision=sign(ordinary['change_lower_scaled'],ordinary['change_upper_scaled']),lower=ordinary['change_lower_scaled'],upper=ordinary['change_upper_scaled']))
                if a.exact_old:
                    separate=(compressed_old_correction_bounds if a.compressed_moments else exact_old_correction_bounds)(model,rows,prepared,old,new);joint=separate;variants=[('group_compressed_moments' if a.compressed_moments else 'group_exact_old',separate)]
                else:
                    separate=correction_bounds(model,rows,prepared,old,new);joint=paired_correction_bounds(model,rows,prepared,old,new)
                    assert all(int(l)<=int(a)<=v<=int(b)<=int(u) for l,a,v,b,u in zip(separate['row_lower'],joint['row_lower'],exact_correction,joint['row_upper'],separate['row_upper']))
                    variants=[('group_separate',separate),('group_paired',joint)]
                for variant,bounds in variants:
                    for budget in budgets:
                        result=refine(bounds,exact_correction,budget);lo,hi=result['change_lower_scaled'],result['change_upper_scaled']
                        assert lo<=true_delta*den<=hi
                        decision=sign(lo,hi);assert decision in ['inconclusive',common['truth']]
                        records.append(dict(**common,variant=variant,budget_fraction=budget,exact_rows=result['exact_rows'],uncertain_rows=result['uncertain_rows'],decision=decision,lower=lo,upper=hi,assigned_upper_bound_fast_path=bool(bounds.get('assigned_upper_bound_fast_path',False))))
                pd.DataFrame(records).to_csv(out/'results.csv',index=False)
                save(out/'status.json',dict(complete=False,phase='native_bounds',fits_complete=ordinal,total_fits=len(pins),active_fit=name,active_set=s,records=len(records),elapsed_seconds=time.monotonic()-started))
                print(json.dumps(dict(fit=name,set=s,truth=common['truth'],ordinary=ordinary['decision'],group_separate=sign(separate['change_lower_scaled'],separate['change_upper_scaled']),group_paired=sign(joint['change_lower_scaled'],joint['change_upper_scaled']))),flush=True)
    frame=pd.DataFrame(records);frame['conclusive']=frame.decision!='inconclusive';frame['correct']=frame.decision==frame.truth
    grouped=frame.groupby(['cohort','method','variant','budget_fraction','truth']).agg(requests=('set','count'),conclusive=('conclusive','sum'),correct=('correct','sum'),median_exact_rows=('exact_rows','median'),max_exact_rows=('exact_rows','max')).reset_index()
    grouped.to_csv(out/'groups.csv',index=False)
    save(out/'source_hashes.json',source_hashes)
    regressions=frame[(frame.method=='learned')&(frame.truth=='worsens')&(frame.variant=='ordinary_paired')]
    if not a.fresh_confirmation:assert len(regressions)==24
    summary=dict(complete=True,method_requests=len(frame[frame.variant=='ordinary_paired']),unique_requests=len(frame[frame.variant=='ordinary_paired'])//2,
        historical_natural_regressions=0 if a.fresh_confirmation else 24,independent_natural_regressions=len(regressions) if a.fresh_confirmation else None,false_conclusive_decisions=0,all_intervals_contain_truth=True,paired_correction_dominates_separate_row_bounds=None if a.exact_old else True,
        elapsed_seconds=time.monotonic()-started,scope=protocol['scope'])
    save(out/'summary.json',summary);save(out/'status.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
