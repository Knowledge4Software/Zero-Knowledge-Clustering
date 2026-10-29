"""Small frozen-data JL screens: random bases, work hints, aggregate envelopes."""
from pathlib import Path
import argparse,json,os,time
from math import ceil
import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits
from commitjl.pilot_io import save,sha,array_sha,sign,frozen_image_sample
from commitjl.group_change_bounds import exact_old_correction_bounds
from commitjl.group_moment_reference import prepare_groups
from commitjl.reuse_certificates import integer_stats
from commitjl.jl_certificate_pilots import jl_map,random_residual_basis,hint_context,sketch_hints,hint_order,refinement_work,aggregate_stats,compressed_aggregate_bounds

REPO=Path(__file__).resolve().parents[2]


def cases(suite):
    if suite=='matched':
        parent=REPO/'code/validation/outputs/local_fast_tradeoff_n128_20261024';source=parent/'head_r63/input.json';inp=json.loads(source.read_text());aux=REPO/'code/validation/outputs/group_compact_auxiliary_r79_n128_20261025/auxiliary_basis.json';learned=json.loads(aux.read_text())['certificate_basis'];old=np.asarray(inp['model']['centers'],dtype=np.int64)
        yield 'matched_n128',inp['model'],np.asarray(inp['x'],dtype=np.int64),[(s,np.asarray(c,dtype=np.int64)) for s,c in enumerate(inp['reuse_centers']) if c!=old.tolist()],learned,dict(input_sha256=sha(source),auxiliary_sha256=sha(aux))
        return
    sources={'historical':REPO/'paper_results/review_controls/local_group_moments_extended_20261023','fresh':REPO/'paper_results/review_controls/group_change_confirmation_20261024/source'};datasets={}
    for cohort,source in sources.items():
        previous=pd.read_csv(REPO/'code/validation/outputs/group_exact_confirmation_20261024'/cohort/'results.csv');selected=previous[(previous.method=='learned')&(previous.variant=='group_exact_old')&(previous.budget_fraction==0)]
        if suite in ['quick','regressions']:selected=selected[selected.truth=='worsens']
        if suite=='quick':selected=selected[selected.decision=='inconclusive']
        pins={r['fit']:r for r in json.loads((source/'pins.json').read_text())}
        for fit,requests in selected.groupby('fit',sort=True):
            item=requests.iloc[0];dataset=item.dataset;seed=int(item.seed);pin=pins[fit];archive=source/(fit+'.npz');frozen=np.load(archive)
            if dataset not in datasets:
                x=frozen_image_sample(REPO,dataset);datasets[dataset]=x[np.linalg.norm(x,axis=1)>0]
            x=datasets[dataset];idx=np.random.default_rng(741+seed).permutation(len(x));cut=int(.6*len(x));tr=x[idx[:cut]];scale=4095/max(float(np.abs(tr).max()),1e-12);rows=np.clip(np.rint(x[idx[cut:cut+1024]]*scale),-4096,4096).astype(np.int64)
            assert array_sha(rows)==pin['audit_sha256'] and array_sha(np.rint(tr*scale).astype(np.int64))==pin['training_sha256'];assert array_sha(frozen['basis'])==pin['basis_sha256']
            aux=REPO/'paper_results/review_controls/residual_certificate_rank_confirmation_20261025'/(fit+'.npz');learned=np.load(aux)['auxiliary_basis'][:79];assert np.array_equal(learned[:63],frozen['basis'])
            m=dict(basis=frozen['basis'].tolist(),basis_scale=16384,centers=frozen['centers'][0].tolist(),input_bits=12,n=len(rows),d=rows.shape[1],k=int(item.k),method='learned_energy',tail_width=0,tail_rows=[],tail_signs=[])
            yield fit,m,rows,[(int(r['set']),frozen['centers'][int(r['set'])]) for _,r in requests.iterrows()],learned.tolist(),dict(archive_sha256=sha(archive),pins_sha256=sha(source/'pins.json'),auxiliary_sha256=sha(aux))


def prediction_labels(model,h,new):
    hc,ec=integer_stats(model,new);hc=hc.astype(object);constant=np.asarray([sum(int(v)**2 for v in row)+int(e) for row,e in zip(hc,ec)],dtype=object)
    return (-2*h.astype(object)@hc.T+constant).argmin(1)


def distances(x,centers):
    result=np.empty((len(x),len(centers)),dtype=np.int64)
    for start in range(0,len(x),64):result[start:start+64]=((x[start:start+64,None,:]-centers[None,:,:])**2).sum(2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--suite',choices=['quick','matched','regressions','all'],default='quick');p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=a.resume);started=time.monotonic();records=[];sources={};requests=0;seeds=[20261026,20261027,20261028]
    if a.resume:
        records=pd.read_csv(out/'results.csv').to_dict('records');sources=json.loads((out/'source_hashes.json').read_text());previous=json.loads((out/'protocol.json').read_text());assert previous['suite']==a.suite
        save(out/'resume_receipt.json',dict(previous_runner_sha256=previous['source_sha256'],current_runner_sha256=sha(Path(__file__)),previous_status=json.loads((out/'status.json').read_text()),reason='Continue after RAM reserve stop. Stream identical canonical samples instead of materializing all70000 records; hashes pin exact data. Chunked Full checks; numerical helpers and fixed seeds unchanged. All completed case records retained.'))
    completed={(r['fit'],int(r['set'])) for r in records if r['idea']=='basis' and r['variant']=='base63'};requests=len(completed)
    save(out/'protocol.json',dict(suite=a.suite,selection={'quick':'All6 previously unresolved natural head regressions; post-hoc development stress cohort.','matched':'All19 original distinct N128 updates, unchanged catalog/model.','regressions':'All36 natural head regressions across both stored cohorts.','all':'All160 stored head updates, both cohorts.'}[a.suite],seeds=seeds,random_basis_extra=[16,32],hint_widths=[8,16,32,64],sparse_column_nonzeros=3,maximum_exact_refinement_fraction=.05,predictions_frozen=True,proofs_generated=False,scope='Native exploratory feasibility only. Numerical kernel time is not proof speed. Random-basis bounds retain exact rounded-Gram corrections; sparse-JL hints only order exact refinements and cannot authorize acceptance. Aggregate variants require a new binding proof, not implemented. All seeds and failed variants retained.',threads=1,source_sha256=sha(Path(__file__))))
    with threadpool_limits(limits=1):
        for fit,model,x,updates,learned,pin in cases(a.suite):
            updates=[(s,new) for s,new in updates if (fit,s) not in completed]
            if not updates:continue
            while psutil.virtual_memory().available<800*2**20:
                save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),suite=a.suite,requests_complete=requests,required_available_bytes=800*2**20,available_bytes=psutil.virtual_memory().available));time.sleep(15)
            sources[fit]=pin;old=np.asarray(model['centers'],dtype=np.int64);od=distances(x,old);groups=od.argmin(1);prepared=prepare_groups(x,groups.tolist(),len(old));h,_=integer_stats(model,x)
            models={'base63':model,'learned79':dict(model,basis=learned)}
            for extra in [16,32]:
                for seed in seeds:models[f'jl_orthogonal_r{63+extra}_s{seed}']=random_residual_basis(model,extra,seed)
            np.savez_compressed(out/(fit+'_bases.npz'),**{name:np.asarray(m['basis'],dtype=np.int64) for name,m in models.items()})
            aggregates={name:aggregate_stats(m,prepared) for name,m in models.items()};sketches={(width,seed):jl_map(x.shape[1],width,seed) for width in [8,16,32,64] for seed in seeds}
            for s,new in updates:
                while psutil.virtual_memory().available<750*2**20:
                    save(out/'status.json',dict(complete=False,status='waiting_for_ram',pid=os.getpid(),suite=a.suite,requests_complete=requests,required_available_bytes=750*2**20,available_bytes=psutil.virtual_memory().available));time.sleep(15)
                requests+=1;nd=distances(x,new);truth=int(nd.min(1).sum()-od.min(1).sum());target=sign(truth,truth);den=int(model['basis_scale'])**6;pred=prediction_labels(model,h,new);agreement=100*float(np.mean(pred==nd.argmin(1)))
                common=dict(fit=fit,set=s,n=len(x),true_change=truth,truth=target,prediction_agreement_percent=agreement)
                base=exact_old_correction_bounds(model,x,prepared,old,new);exact=[int(v)*den for v in nd.min(1)-nd[np.arange(len(x)),groups]];assert all(int(lo)<=v<=int(hi) for lo,v,hi in zip(base['row_lower'],exact,base['row_upper']))
                cap=ceil(.05*len(x));width_order=hint_order(base,None,'width');work=refinement_work(base,exact,width_order,cap);assert work['lower']<=truth*den<=work['upper']
                records.append(dict(**common,idea='refinement',variant='width',decision=work['decision'],exact_rows=work['exact_rows'],lower=work['lower'],upper=work['upper']))
                for name,m in models.items():
                    clock=time.monotonic();b=base if name=='base63' else exact_old_correction_bounds(m,x,prepared,old,new);elapsed=time.monotonic()-clock;decision=sign(b['change_lower_scaled'],b['change_upper_scaled']);assert b['change_lower_scaled']<=truth*den<=b['change_upper_scaled'];assert decision in ['inconclusive',target]
                    records.append(dict(**common,idea='basis',variant=name,decision=decision,lower=b['change_lower_scaled'],upper=b['change_upper_scaled'],native_seconds=elapsed))
                    clock=time.monotonic();compressed=compressed_aggregate_bounds(base,m,aggregates[name],old,new);elapsed=time.monotonic()-clock;assert compressed['lower']<=truth*den<=compressed['upper'];decision=sign(compressed['lower'],compressed['upper']);assert decision in ['inconclusive',target]
                    records.append(dict(**common,idea='aggregate',variant=name,decision=decision,lower=compressed['lower'],upper=compressed['upper'],private_aggregate_scalars=compressed['private_aggregate_scalars'],native_seconds=elapsed,exact_moment_fallback=decision=='inconclusive'))
                context=hint_context(model,x,new,groups)
                for (width,seed),sketch in sketches.items():
                    clock=time.monotonic();estimated,labels=sketch_hints(context,sketch);order=hint_order(base,estimated);work=refinement_work(base,exact,order,cap);elapsed=time.monotonic()-clock;assert work['lower']<=truth*den<=work['upper'];assert work['decision'] in ['inconclusive',target]
                    records.append(dict(**common,idea='refinement',variant=f'jl_m{width}_s{seed}',decision=work['decision'],exact_rows=work['exact_rows'],lower=work['lower'],upper=work['upper'],native_seconds=elapsed,hint_assignment_agreement_percent=100*float(np.mean(labels==nd.argmin(1)))))
                pd.DataFrame(records).to_csv(out/'results.csv',index=False);save(out/'source_hashes.json',sources);save(out/'status.json',dict(complete=False,pid=os.getpid(),suite=a.suite,requests_complete=requests,expected_requests={'quick':6,'matched':19,'regressions':36,'all':160}[a.suite],active_fit=fit,active_set=s,records=len(records),elapsed_seconds=time.monotonic()-started));print(json.dumps(dict(fit=fit,set=s,completed=requests,truth=target)),flush=True)
    assert requests=={'quick':6,'matched':19,'regressions':36,'all':160}[a.suite];frame=pd.DataFrame(records);summary=[]
    for (idea,variant),g in frame.groupby(['idea','variant'],sort=True):
        r=dict(idea=idea,variant=variant,requests=len(g),conclusive=int((g.decision!='inconclusive').sum()),regressions=int((g.truth=='worsens').sum()),regressions_detected=int(((g.decision=='worsens')&(g.truth=='worsens')).sum()))
        if idea=='refinement':r.update(total_exact_rows=int(g.exact_rows.sum()),maximum_exact_rows=int(g.exact_rows.max()))
        if idea=='aggregate':r.update(exact_moment_fallbacks=int(g.exact_moment_fallback.sum()),private_aggregate_scalars_min=int(g.private_aggregate_scalars.min()),private_aggregate_scalars_max=int(g.private_aggregate_scalars.max()))
        summary.append(r)
    result=dict(complete=True,suite=a.suite,requests=requests,all_intervals_contain_truth=True,false_conclusive=0,native_only=True,elapsed_seconds=time.monotonic()-started,results=summary);save(out/'summary.json',result);save(out/'status.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
